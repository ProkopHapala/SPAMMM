"""Systematic UFF/SPFF relaxation benchmarks on flat_1 (triptacene, 96 atoms).

Input: /home/prokop/svec_triptacene/flat_1.mol2 (no .xyz beside it; same geometry).

Vacuum pipelines compared:
  UFF  — multi-kernel/step | fused local | fused global
  SPFF — relax_batch       | relax_serial (local) | relax_global

Each case writes:
  debug/test_relax_flat1/<tag>_init_final.xyz   (start + end)
  debug/test_relax_flat1/<tag>_geometry.png     (before/after)
  debug/test_relax_flat1/<tag>.out

Phase 2 (optional): SPFF + GridFF substrate — see test_spff_gridff_substrate_flat1.
"""
import os
import time
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spammm.AtomicSystem import AtomicSystem
from spammm.forcefields.UFF_cl import UFF_cl
from spammm.forcefields.FFController import FFController
from spammm.forcefields.SPFF_cl import SPFF_cl
from tests.helpers.geometry import (
    distort, find_bonds, save_xyz_frames, plot_geometry, planarity,
)

FLAT1 = '/home/prokop/svec_triptacene/flat_1.mol2'
SUB_XYZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'substrates', 'NaCl_1x1_L3.xyz')
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

NSTEPS_VACUUM = 2000
NSTEPS_SUBSTRATE = 1500
DT = 0.01
DAMP = 0.9
FLIMIT = 100.0
FMAX_UFF = 100.0  # multi can plateau higher with damp=0.9; fused is bonds+angles only
FMAX_SPFF = 50.0
DAMP_UFF = 0.95   # matches prior flat_1 UFF vacuum (0.9 left fmax high)
MASS_MAP = {'H': 1.0, 'C': 12.0, 'O': 16.0, 'N': 14.0}


def _debug_dir():
    d = os.path.join('debug', 'test_relax_flat1')
    os.makedirs(d, exist_ok=True)
    return d


def _load_flat1(distort_amp=0.15):
    if not os.path.isfile(FLAT1):
        pytest.skip(f'Missing input molecule: {FLAT1}')
    mol = AtomicSystem(fname=FLAT1)
    mol.enames = np.array([str(e).split('.')[0].split('_')[0] for e in mol.enames], dtype=object)
    if mol.bonds is None:
        mol.findBonds()
    mol.neighs()
    # Aromatic PAH: treat all C as C_R for SPFF (no Kekule n_pi in headless mol2)
    mol.atom_types_spff = ['C_R' if e == 'C' else e for e in mol.enames]
    apos0 = np.asarray(mol.apos, dtype=np.float64).copy()
    if distort_amp > 0:
        mol.apos = distort(apos0, amplitude=distort_amp, seed=42)
    return mol, apos0


def _carbon_indices(enames):
    return [i for i, e in enumerate(enames) if e == 'C']


def _cc_ch_bond_stats(apos, enames, bonds):
    cc, ch = [], []
    for i, j in bonds:
        r = float(np.linalg.norm(apos[i] - apos[j]))
        ei, ej = enames[i], enames[j]
        if ei == 'C' and ej == 'C':
            cc.append(r)
        elif {ei, ej} == {'C', 'H'}:
            ch.append(r)
    return cc, ch


def _save_geometry_png(path, apos_init, apos_final, enames, title, proj='xy'):
    bonds = find_bonds(apos_final, enames, Rcut=1.8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_geometry(ax1, apos_init, enames, bonds, title=f'{title} initial', proj=proj)
    plot_geometry(ax2, apos_final, enames, bonds, title=f'{title} relaxed', proj=proj)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'REVIEW: {path}', flush=True)


def _fmax_from_forces(forces):
    f = np.asarray(forces)[:, :3]
    return float(np.max(np.linalg.norm(f, axis=1)))


def _write_summary(path, lines):
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'REVIEW: {path}', flush=True)


def _finish_case(outdir, tag, title, enames, apos_init, apos_final, lines, rev=None):
    xyz_path = os.path.join(outdir, f'{tag}_init_final.xyz')
    png_path = os.path.join(outdir, f'{tag}_geometry.png')
    save_xyz_frames(xyz_path, enames, [apos_init, apos_final],
                    comments=[f'{tag} start', f'{tag} end'])
    _save_geometry_png(png_path, apos_init, apos_final, enames, title)
    print(f'REVIEW: {xyz_path}', flush=True)
    lines = list(lines) + [f'REVIEW: {xyz_path}', f'REVIEW: {png_path}']
    _write_summary(os.path.join(outdir, f'{tag}.out'), lines)
    for L in lines:
        if rev is not None:
            rev.out(L)
        print(L, flush=True)
    return xyz_path, png_path


# ---------------------------------------------------------------------------
# Vacuum: UFF pipelines (multi / local fused / global fused)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
def test_uff_pipelines_vacuum_flat1(make_review):
    """UFF flat_1 vacuum: multi-kernel vs fused local vs fused global."""
    rev = make_review('uff_pipelines_vacuum')
    outdir = _debug_dir()
    mol, _ = _load_flat1()
    apos_init = np.asarray(mol.apos, dtype=np.float64).copy()
    enames = list(mol.enames)
    natoms = len(enames)
    assert natoms == 96
    masses = np.array([MASS_MAP.get(e, 12.0) for e in enames], dtype=np.float32)
    Cidx = _carbon_indices(enames)

    # Build once; re-upload positions for each pipeline
    uff = UFF_cl(bPrint=True)
    uff.toUFF(mol)
    print(f'[UFF] natoms={uff.natoms} nbonds={uff.nbonds} nangles={uff.nangles} '
          f'ndihedrals={uff.ndihedrals} device={uff.ctx.devices[0].name}', flush=True)
    assert 'nvidia' in uff.ctx.devices[0].name.lower() or 'geforce' in uff.ctx.devices[0].name.lower() \
        or 'rtx' in uff.ctx.devices[0].name.lower(), f'Expected NVIDIA GPU, got {uff.ctx.devices[0].name}'

    cases = [
        ('uff_multi',  'UFF multi-kernel', lambda: uff.relax(nsteps=NSTEPS_VACUUM, dt=DT, damp=DAMP_UFF, Flimit=FLIMIT)),
        ('uff_local',  'UFF fused local',  lambda: uff.relax_serial(nsteps=NSTEPS_VACUUM, dt=DT, damp=DAMP_UFF, Flimit=FLIMIT, wg=128)),
        ('uff_global', 'UFF fused global', lambda: uff.relax_global(nsteps=NSTEPS_VACUUM, dt=DT, damp=DAMP_UFF, Flimit=FLIMIT, wg=256)),
    ]
    summary = [
        f'UFF vacuum pipelines flat_1  device={uff.ctx.devices[0].name}',
        f'natoms={natoms} nbonds={uff.nbonds} nangles={uff.nangles} nsteps={NSTEPS_VACUUM} dt={DT} damp={DAMP_UFF}',
        f'Note: fused local/global = bonds+angles only (no dihedrals/inversions/NB)',
        '',
        f'{"tag":16s} {"t_s":>10s} {"E":>12s} {"fmax":>10s} {"plan_C":>8s}',
    ]
    failures = []

    for tag, title, runner in cases:
        uff.upload_positions(apos_init, masses=masses)
        t0 = time.perf_counter()
        try:
            E_raw = runner()
            uff.queue.finish()
        except Exception as e:
            failures.append(f'{tag} RUN FAILED: {type(e).__name__}: {e}')
            print(failures[-1], flush=True)
            continue
        t_relax = time.perf_counter() - t0
        E = float(np.asarray(E_raw).ravel()[0])
        apos_final = np.asarray(uff.get_positions(), dtype=np.float64)
        fmax = _fmax_from_forces(uff.get_forces()[0])
        plan = planarity(apos_final, Cidx)
        bonds = find_bonds(apos_final, enames, Rcut=1.8)
        cc, ch = _cc_ch_bond_stats(apos_final, enames, bonds)
        lines = [
            f'{title} flat_1 vacuum',
            f'device={uff.ctx.devices[0].name}',
            f'nsteps={NSTEPS_VACUUM} dt={DT} damp={DAMP_UFF}',
            f't_relax={t_relax:.4f}s  E={E:.6f}  fmax={fmax:.6f}',
            f'planarity_C={plan:.6f}',
            f'n_CC={len(cc)} mean_CC={np.mean(cc) if cc else np.nan:.4f}  '
            f'n_CH={len(ch)} mean_CH={np.mean(ch) if ch else np.nan:.4f}',
        ]
        _finish_case(outdir, tag, title, enames, apos_init, apos_final, lines, rev=rev)
        summary.append(f'{tag:16s} {t_relax:10.4f} {E:12.4f} {fmax:10.4f} {plan:8.4f}')
        if not (np.isfinite(apos_final).all() and np.isfinite(E)):
            failures.append(f'{tag}: non-finite geometry/energy')
        if fmax >= FMAX_UFF:
            failures.append(f'{tag}: fmax={fmax:.4f} >= {FMAX_UFF}')

    _write_summary(os.path.join(outdir, 'uff_pipelines_summary.out'), summary + ([''] + failures if failures else []))
    for L in summary:
        rev.out(L)
        print(L, flush=True)
    rev.finish()
    if failures:
        pytest.fail('; '.join(failures))


# ---------------------------------------------------------------------------
# Vacuum: SPFF pipelines (batch / serial local / global)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
def test_spff_pipelines_vacuum_flat1(make_review):
    """SPFF flat_1 vacuum: batch vs serial(local) vs global."""
    import pyopencl as cl

    rev = make_review('spff_pipelines_vacuum')
    outdir = _debug_dir()
    mol, _ = _load_flat1()

    ctrl = FFController(enable_nonbond=False)
    info = ctrl.build_ff(mol, ff_type='spff')
    md = ctrl.md
    device = md.ctx.devices[0].name
    print(f'[SPFF] natoms={info["natoms"]} nnode={info["nnode"]} nvecs={info["nvecs"]} '
          f'device={device} can_serial={ctrl._can_use_serial(False)}', flush=True)
    assert 'nvidia' in device.lower() or 'geforce' in device.lower() or 'rtx' in device.lower(), \
        f'Expected NVIDIA GPU, got {device}'

    # Snapshot full DOF buffer (atoms + pi) so each pipeline starts identical
    md.fromGPU('apos', md.atoms)
    init_atoms = md.atoms.copy()
    apos_init = init_atoms.reshape(-1, 4)[:info['natoms'], :3].astype(np.float64).copy()
    # After nodes-first reorder, mol.enames matches SPFF atom order
    enames = list(mol.enames)[:info['natoms']]
    if len(enames) != info['natoms']:
        enames = ['C'] * info['natoms']
    Cidx = _carbon_indices(enames)

    def _upload_init():
        md.atoms[:] = init_atoms
        md.toGPU('apos', md.atoms)
        cl.enqueue_fill_buffer(md.queue, md.buffer_dict['avel'], np.float32(0), 0, md.buffer_dict['avel'].size)
        md.queue.finish()

    cases = [
        ('spff_batch',  'SPFF batch',  lambda: md.relax_batch(nsteps=NSTEPS_VACUUM, do_nb=False)),
        ('spff_serial', 'SPFF serial local', lambda: md.relax_serial(nsteps=NSTEPS_VACUUM, dt=DT, damp=DAMP, Flimit=FLIMIT)),
        ('spff_global', 'SPFF fused global', lambda: md.relax_global(nsteps=NSTEPS_VACUUM, dt=DT, damp=DAMP, Flimit=FLIMIT, wg=256, do_faf=False)),
    ]
    summary = [
        f'SPFF vacuum pipelines flat_1  device={device}',
        f'natoms={info["natoms"]} nnode={info["nnode"]} nvecs={info["nvecs"]} '
        f'nsteps={NSTEPS_VACUUM} dt={DT} damp={DAMP}',
        f'WG_serial={SPFF_cl.SERIAL_WG_SIZE} MAX_NVEC={SPFF_cl.SERIAL_MAX_NVEC}',
        '',
        f'{"tag":16s} {"t_s":>10s} {"E":>12s} {"fmax":>10s} {"plan_C":>8s}',
    ]

    for tag, title, runner in cases:
        if tag == 'spff_serial' and not ctrl._can_use_serial(False):
            msg = f'{tag} SKIPPED: serial caps exceeded'
            print(msg, flush=True)
            rev.out(msg)
            continue
        _upload_init()
        md.set_md_params(dt=DT, damp=DAMP, Flimit=FLIMIT)
        t0 = time.perf_counter()
        runner()
        md.queue.finish()
        t_relax = time.perf_counter() - t0
        E = float(md.get_total_energy())
        apos_final = np.asarray(ctrl.get_positions(), dtype=np.float64)
        fmax = float(ctrl.get_fmax())
        plan = planarity(apos_final, Cidx)
        bonds = find_bonds(apos_final, enames, Rcut=1.8)
        cc, ch = _cc_ch_bond_stats(apos_final, enames, bonds)
        lines = [
            f'{title} flat_1 vacuum',
            f'device={device}',
            f'natoms={info["natoms"]} nnode={info["nnode"]} nvecs={info["nvecs"]}',
            f'nsteps={NSTEPS_VACUUM} dt={DT} damp={DAMP}',
            f't_relax={t_relax:.4f}s  E={E:.6f}  fmax={fmax:.6f}',
            f'planarity_C={plan:.6f}',
            f'n_CC={len(cc)} mean_CC={np.mean(cc) if cc else np.nan:.4f}  '
            f'n_CH={len(ch)} mean_CH={np.mean(ch) if ch else np.nan:.4f}',
        ]
        _finish_case(outdir, tag, title, enames, apos_init, apos_final, lines, rev=rev)
        summary.append(f'{tag:16s} {t_relax:10.4f} {E:12.4f} {fmax:10.4f} {plan:8.4f}')
        assert np.isfinite(apos_final).all() and np.isfinite(E)
        assert fmax < FMAX_SPFF, f'{tag} fmax too large: {fmax}'

    _write_summary(os.path.join(outdir, 'spff_pipelines_summary.out'), summary)
    for L in summary:
        rev.out(L)
        print(L, flush=True)
    rev.finish()
    ctrl.teardown()


# ---------------------------------------------------------------------------
# Phase 2: SPFF + GridFF on NaCl
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
def test_spff_gridff_substrate_flat1(make_review):
    """SPFF bonded + GridFF substrate (batch loop). Serial has no GridFF."""
    from spammm.surfaces.GridFFRelaxedScan import (
        ensure_gridff_file, load_gridff_array, download_state,
    )
    from spammm.forcefields.SPFFbuilder import SPFF
    from spammm.topology.FFparams import SPFFparams
    import pyopencl as cl

    rev = make_review('spff_substrate')
    outdir = _debug_dir()
    if not os.path.isfile(SUB_XYZ):
        pytest.skip(f'Missing substrate: {SUB_XYZ}')

    mol, apos0 = _load_flat1(distort_amp=0.05)
    enames = list(mol.enames)
    sub = AtomicSystem(fname=SUB_XYZ, bPreinit=False)
    z_top = float(np.max(sub.apos[:, 2]))
    mol.apos = np.asarray(mol.apos, dtype=np.float32)
    mol_xy = mol.apos[:, :2].mean(axis=0)
    sub_xy = sub.apos[:, :2].mean(axis=0)
    mol.apos[:, 0] += sub_xy[0] - mol_xy[0]
    mol.apos[:, 1] += sub_xy[1] - mol_xy[1]
    mol.apos[:, 2] += (z_top + 3.5) - float(mol.apos[:, 2].min())
    apos_init = mol.apos[:, :3].copy()

    grid_cache = os.path.join(outdir, 'gridff_cache')
    os.makedirs(grid_cache, exist_ok=True)
    atom_types = os.path.join(DATA, 'AtomTypes.dat')
    elem_types = os.path.join(DATA, 'ElementTypes.dat')
    # Prefer cached GridFF if present
    cached = os.path.join(grid_cache, 'data', 'double3', 'Bspline_PLQd.npy')
    print('[GridFF] ensuring NaCl GridFF (may take a while on first run)...', flush=True)
    t_g0 = time.perf_counter()
    gridff_path = ensure_gridff_file(
        src_xyz=os.path.abspath(SUB_XYZ),
        out_dir=grid_cache,
        gridff_path=cached if os.path.isfile(cached) else None,
        dg=(0.2, 0.2, 0.2),
        sigma=0.0,
        alpha_morse=1.5,
        atom_types_path=atom_types,
        element_types_path=elem_types,
        nmaxiter=200,
        nPerStep=25,
    )
    t_grid = time.perf_counter() - t_g0
    print(f'[GridFF] path={gridff_path} t_build={t_grid:.2f}s', flush=True)
    gridff = load_gridff_array(gridff_path)

    g0 = (
        float(sub.apos[:, 0].min()) - 1.0,
        float(sub.apos[:, 1].min()) - 1.0,
        float(sub.apos[:, 2].min()) - 0.5,
    )
    dg = (0.2, 0.2, 0.2)

    params = SPFFparams(DATA + '/')
    mol.atypes = np.array([params.getAtomType(t, bErr=False) for t in mol.atom_types_spff], dtype=np.int32)
    spff = SPFF()
    spff.toSPFFsp3_loc(mol, params.atom_types_map)
    for ia in range(spff.natoms):
        spff.apos[ia, 3] = MASS_MAP.get(enames[ia] if ia < len(enames) else 'C', 12.0)

    md = SPFF_cl(enable_nonbond=True)
    md.realloc(spff, nSystems=1)
    md.upload_all_systems()
    md.setup_kernels()
    md.initGridFF(
        grid_shape=tuple(int(x) for x in gridff.shape[:3]),
        bspline_data=np.ascontiguousarray(gridff),
        grid_p0=g0,
        grid_step=dg,
        use_texture=False,
        r_damp=0.0,
        alpha_morse=1.5,
        bKernels=True,
    )
    device = md.ctx.devices[0].name
    print(f'[SPFF+GridFF] device={device}', flush=True)
    if getattr(md, 'kernel_args_getNonBond_GridFF_Bspline_ex2', None) is None:
        pytest.fail('GridFF ex2 kernel not initialized')
    cl.enqueue_fill_buffer(md.queue, md.buffer_dict['avel'], np.float32(0), 0, md.buffer_dict['avel'].size)
    md.set_md_params(dt=DT, damp=DAMP, Flimit=FLIMIT)
    md.queue.finish()

    t0 = time.perf_counter()
    for _ in range(NSTEPS_SUBSTRATE):
        md.run_cleanForceSPFFf4()
        md.run_getNonBond_GridFF_Bspline_ex2()
        md.run_getSPFFf4()
        md.run_updateAtomsSPFFf4()
    md.queue.finish()
    t_relax = time.perf_counter() - t0

    apos_final, F, Eper = download_state(md, spff.natoms, spff.nvecs)
    fmax = float(np.max(np.linalg.norm(F, axis=1)))
    E = float(np.sum(Eper))
    z_com = float(apos_final[:, 2].mean())
    z_min = float(apos_final[:, 2].min())
    z_rel = z_min - z_top

    xyz_path = os.path.join(outdir, 'spff_substrate_init_final.xyz')
    sub_e = [str(e) for e in sub.enames]
    sub_p = np.asarray(sub.apos, dtype=np.float64)[:, :3]
    with open(xyz_path, 'w') as f:
        for tag, pos in [('init', apos_init), ('relaxed', apos_final)]:
            n = len(enames) + len(sub_e)
            f.write(f'{n}\nflat_1+NaCl {tag} z_rel={z_rel:.3f} fmax={fmax:.4f}\n')
            for e, p in zip(sub_e, sub_p):
                f.write(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')
            for e, p in zip(enames, pos):
                f.write(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')
    print(f'REVIEW: {xyz_path}', flush=True)

    png_path = os.path.join(outdir, 'spff_substrate_geometry.png')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, pos, title in ((ax1, apos_init, 'init'), (ax2, apos_final, 'relaxed')):
        ax.scatter(sub_p[:, 0], sub_p[:, 2], c='lightblue', s=20, label='NaCl', zorder=1)
        ax.scatter(pos[:, 0], pos[:, 2], c=['black' if e == 'C' else 'gray' for e in enames], s=40, label='mol', zorder=2)
        ax.axhline(z_top, color='k', ls='--', lw=0.8)
        ax.set_xlabel('x [A]'); ax.set_ylabel('z [A]')
        ax.set_title(f'SPFF+GridFF {title} (xz)')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f'REVIEW: {png_path}', flush=True)

    lines = [
        f'SPFF + GridFF NaCl substrate flat_1',
        f'device={device}',
        f'z_top={z_top:.4f} g0={g0} dg={dg}',
        f'gridff={gridff_path} t_grid={t_grid:.2f}s shape={gridff.shape}',
        f'nsteps={NSTEPS_SUBSTRATE} t_relax={t_relax:.4f}s E={E:.6f} fmax={fmax:.6f}',
        f'z_com={z_com:.4f} z_min={z_min:.4f} z_rel(min-top)={z_rel:.4f}',
        f'REVIEW: {xyz_path}',
        f'REVIEW: {png_path}',
        'Note: UFF+GridFF not wired — SPFF+GridFF is the substrate case.',
    ]
    _write_summary(os.path.join(outdir, 'spff_substrate.out'), lines)
    for L in lines:
        rev.out(L)
        print(L, flush=True)
    rev.finish()

    assert np.isfinite(apos_final).all()
    assert z_rel > 0.5, f'molecule crashed into surface: z_rel={z_rel}'
    assert fmax < 50.0, f'substrate fmax too large: {fmax}'
