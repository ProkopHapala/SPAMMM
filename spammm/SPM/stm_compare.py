"""STM / frontier-MO compare — DFTB vs pySCF (orbitals, current, vacuum panels).

SSOT for ``run_spm.py stm *`` and ``tests/SPM/testplot_stm_basis_compare.py``.

Orbital maps: signed ψ (phase, RdBu).  STM maps: current I≥0 (viridis); tip picks φ_t.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import time

import numpy as np

from spammm import atomicUtils as au
from spammm.quantum.DFTB_utils import WFC_HSD_PATHS
from spammm.SPM import AFM_utils as afm_utils

_REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_OUT = os.path.join(_REPO_ROOT, 'debug', 'stm_orbital_compare')
SA_PTCDA = os.path.join(_REPO_ROOT, 'debug', 'dftb_basis_sa_ptcda', 'PTCDA_sa_params.json')
ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15, 'Br': 35, 'I': 53}
HAU2EV = 27.211396641308

MOLECULES = {
    'pentacene': {'xyz': 'data/xyz/pentacene.xyz'},
    'PTCDA': {'xyz': 'data/xyz/PTCDA.xyz', 'sa_params': SA_PTCDA},
    'benzene': {'xyz': 'data/xyz/benzene.xyz'},
    'pyridine': {'xyz': 'data/xyz/pyridine.xyz'},
}


def _load_pu():
    spec = importlib.util.spec_from_file_location(
        'pySCF_utils_new', os.path.join(_REPO_ROOT, 'spammm', 'quantum', 'pySCF_utils-new.py'))
    pu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pu)
    return pu


def _resolve_xyz_path(xyz: str) -> str:
    return xyz if os.path.isabs(xyz) else os.path.join(_REPO_ROOT, xyz)


def load_mol(name: str):
    if name not in MOLECULES:
        raise KeyError(f'Unknown molecule {name!r}; known={list(MOLECULES)}')
    info = dict(MOLECULES[name])
    path = _resolve_xyz_path(info['xyz'])
    return _load_xyz(path, name, info)


def load_mol_xyz(xyz_path: str, name: str | None = None):
    path = _resolve_xyz_path(xyz_path)
    mol_name = name or os.path.splitext(os.path.basename(path))[0]
    return _load_xyz(path, mol_name, {})


def _load_xyz(path: str, name: str, info: dict):
    pos, _, names, _, _ = au.load_xyz(path)
    pos = np.asarray(pos, dtype=np.float64)
    names = list(names)
    types = np.array([ELEM_Z.get(e, 6) for e in names], dtype=np.int32)
    return name, pos, names, types, info


def resolve_molecules(molecule_args, *, xyz: str | None = None):
    """Return list of (name, pos, names, types, info)."""
    if xyz:
        return [load_mol_xyz(xyz)]
    out = []
    for raw in molecule_args:
        for part in str(raw).split(','):
            part = part.strip()
            if part:
                out.append(load_mol(part))
    if not out:
        raise ValueError('No molecule specified')
    return out


def load_zeta_override(info):
    sp = info.get('sa_params')
    if not sp or not os.path.isfile(sp):
        return None
    js = json.load(open(sp))
    best = js.get('best_params') or {}
    return {el: float(v[1]) for el, v in best.items()}


def make_scan_grid(pos, step, margin):
    lo = pos[:, :2].min(axis=0) - margin
    hi = pos[:, :2].max(axis=0) + margin
    xs = np.arange(lo[0], hi[0] + 1e-9, step)
    ys = np.arange(lo[1], hi[1] + 1e-9, step)
    return xs.astype(np.float64), ys.astype(np.float64)


def _mo_label(i, homo):
    rel = int(i) - int(homo)
    if rel == 0:
        return f'HOMO #{i}'
    if rel == 1:
        return f'LUMO #{i}'
    if rel < 0:
        return f'HOMO{rel} #{i}'
    return f'LUMO+{rel-1} #{i}'


def _basis_key(bases: str) -> str:
    bl = [b.strip() for b in bases.split(',') if b.strip()]
    if any('3ob' in b for b in bl):
        return '3ob-3-1'
    return 'mio-1-1'


def run_frontier_orbitals(mol_name, pos, names, types, args):
    """Signed ψ at z≈z_above; vertical + horizontal spectrum↔orbital plots."""
    out_dir = os.path.join(args.outdir, mol_name, 'frontier_diag')
    os.makedirs(out_dir, exist_ok=True)
    z_mol = float(np.mean(pos[:, 2]))
    z_plane = z_mol + float(args.z_above)
    scan_xs, scan_ys = make_scan_grid(pos, args.scan_step, args.margin)
    n_near = int(args.n_near)
    basis_key = _basis_key(args.bases)
    basis_hsd = WFC_HSD_PATHS[basis_key]
    work = os.path.join(out_dir, f'dftb_work_{basis_key}')
    field = 'psi' if str(getattr(args, 'field', 'psi')).lower() == 'psi' else 'psi'

    lines = [
        f'Frontier orbital diagnostic: {mol_name}',
        f'z_plane = z_mol({z_mol:.3f}) + {args.z_above} = {z_plane:.3f} Å',
        f'scan={len(scan_xs)}x{len(scan_ys)} step={args.scan_step}  n_near=±{n_near}',
        f'field={field} (signed phase, RdBu)',
        f'DFTB basis={basis_key}  pySCF={args.pyscf_xc}/{args.pyscf_basis}',
        '',
        'NOTE: HOMO from valence electron count (NOT eigvals<0).',
        '',
    ]

    t0 = time.perf_counter()
    d = afm_utils.get_density_from_dftb_dense(
        pos, types, basis_hsd, work, step=0.5, margin=0.5, z_extra=0.5,
        verbosity=0, project_density=False)
    t_scf_dftb = time.perf_counter() - t0
    eigvecs, eigvals = d['eigvecs'], np.asarray(d['eigvals'], dtype=np.float64)
    n_elec = afm_utils.dftb_n_valence_electrons(enames=names)
    homo_d, lumo_d = afm_utils.dftb_frontier_mo_indices(eigvals, n_elec=n_elec)
    E_d_eV = eigvals * HAU2EV
    lo = max(0, homo_d - n_near)
    hi = min(len(eigvals) - 1, lumo_d + n_near)
    mo_list_d = list(range(lo, hi + 1))
    lines += [
        f'[DFTB] HOMO#{homo_d} E={E_d_eV[homo_d]:.3f} eV  gap={E_d_eV[lumo_d]-E_d_eV[homo_d]:.3f} eV  '
        f'SCF={t_scf_dftb:.3f} s',
    ]

    projector = d['projector']
    atoms_dict = d['atoms_dict']
    afm_utils._set_projector_species_basis(projector, atoms_dict, d['basis_ang'], rc_max=6.0)

    maps_d, t_proj_d = {}, 0.0
    for imo in mo_list_d:
        lab = _mo_label(imo, homo_d)
        t1 = time.perf_counter()
        psi = afm_utils.project_mo_xy_slice(
            projector, eigvecs[imo], d['norb_per_atom'], d['orb_offsets'], atoms_dict,
            scan_xs, scan_ys, z_plane, use_exp_basis=False)
        t_proj_d += time.perf_counter() - t1
        maps_d[lab] = psi
    lines.append(
        f'[DFTB] projection ({len(mo_list_d)} MOs) = {t_proj_d:.3f} s  '
        f'({1e3*t_proj_d/max(len(mo_list_d), 1):.1f} ms/MO)')

    pu = _load_pu()
    backend = pu.resolve_backend(args.pyscf_backend)
    t0 = time.perf_counter()
    r = pu.run_scf_geometry(names, pos, basis=args.pyscf_basis, xc=args.pyscf_xc,
                            backend=backend, release=False, max_memory_mb=20000)
    t_scf_py = time.perf_counter() - t0
    mf = r['mf']
    homo_p, lumo_p = pu.homo_lumo_indices(mf)
    E_p_eV = np.asarray(mf.mo_energy, dtype=np.float64) * HAU2EV
    lo_p = max(0, homo_p - n_near)
    hi_p = min(len(E_p_eV) - 1, lumo_p + n_near)
    mo_list_p = list(range(lo_p, hi_p + 1))
    lines.append(
        f'[pySCF] HOMO#{homo_p} E={E_p_eV[homo_p]:.3f} eV  gap={E_p_eV[lumo_p]-E_p_eV[homo_p]:.3f} eV  '
        f'SCF={t_scf_py:.3f} s')

    maps_p, t_proj_p = {}, 0.0
    for imo in mo_list_p:
        lab = _mo_label(imo, homo_p)
        t1 = time.perf_counter()
        psi = pu.eval_mo_on_xy_slice(mf, imo, scan_xs, scan_ys, z_plane)
        t_proj_p += time.perf_counter() - t1
        maps_p[lab] = np.asarray(psi)
    lines.append(
        f'[pySCF] projection ({len(mo_list_p)} MOs) = {t_proj_p:.3f} s  '
        f'({1e3*t_proj_p/max(len(mo_list_p), 1):.1f} ms/MO)')
    pu.release_scf(mf)

    labels, maps_d2, maps_p2 = [], {}, {}
    maps_d_abs, maps_p_abs, rels = {}, {}, []
    for k in range(-n_near, n_near + 2):
        id_, ip_ = homo_d + k, homo_p + k
        if id_ < 0 or id_ >= len(eigvals) or ip_ < 0 or ip_ >= len(E_p_eV):
            continue
        if k == 0:
            lab = f'HOMO  D#{id_}/P#{ip_}'
        elif k == 1:
            lab = f'LUMO  D#{id_}/P#{ip_}'
        elif k < 0:
            lab = f'HOMO{k}  D#{id_}/P#{ip_}'
        else:
            lab = f'LUMO+{k-1}  D#{id_}/P#{ip_}'
        labels.append(lab)
        maps_d2[lab] = maps_d[_mo_label(id_, homo_d)]
        maps_p2[lab] = maps_p[_mo_label(ip_, homo_p)]
        maps_d_abs[id_] = maps_d2[lab]
        maps_p_abs[ip_] = maps_p2[lab]
        rels.append(k)
    labels_eup = list(reversed(labels))

    spec_png = os.path.join(out_dir, f'spectrum_{mol_name}.png')
    afm_utils.plot_eigspectrum_compare(
        E_d_eV, homo_d, E_p_eV, homo_p, spec_png, n_near=n_near,
        title=f'{mol_name} eigenvalue spectrum  DFTB {basis_key} vs pySCF {args.pyscf_xc}/{args.pyscf_basis}')
    lines.append(f'REVIEW: {spec_png}')

    gal_png = os.path.join(out_dir, f'orbitals_z{args.z_above:.1f}_{mol_name}.png')
    afm_utils.plot_frontier_orbital_gallery(
        maps_d2, maps_p2, labels_eup, scan_xs, scan_ys, z_plane, gal_png,
        atom_pos=pos, field=field,
        title=f'{mol_name}  ψ  z={z_plane:.2f}Å  E↑  |  DFTB {basis_key} vs pySCF')
    lines.append(f'REVIEW: {gal_png}')

    base_title = (f'{mol_name}  spectrum↔ψ  z={z_plane:.2f}Å  |  DFTB {basis_key} vs pySCF '
                  f'{args.pyscf_xc}/{args.pyscf_basis}')
    layouts = [('vertical', 'E↑'), ('horizontal', 'E→')]
    if getattr(args, 'layout', 'both') == 'vertical':
        layouts = [('vertical', 'E↑')]
    elif getattr(args, 'layout', 'both') == 'horizontal':
        layouts = [('horizontal', 'E→')]

    for lay, tag in layouts:
        combo_png = os.path.join(out_dir, f'spectrum_orbitals_{lay}_z{args.z_above:.1f}_{mol_name}.png')
        afm_utils.plot_spectrum_with_orbitals(
            E_d_eV, homo_d, maps_d_abs, E_p_eV, homo_p, maps_p_abs, rels,
            scan_xs, scan_ys, combo_png, atom_pos=pos, field=field, layout=lay,
            title=f'{base_title}  ({tag}, {lay})')
        lines.append(f'REVIEW: {combo_png}')
    if 'vertical' in [l[0] for l in layouts]:
        legacy = os.path.join(out_dir, f'spectrum_orbitals_z{args.z_above:.1f}_{mol_name}.png')
        shutil.copy2(
            os.path.join(out_dir, f'spectrum_orbitals_vertical_z{args.z_above:.1f}_{mol_name}.png'),
            legacy)
        lines.append(f'REVIEW: {legacy}  (alias of vertical)')

    summary = os.path.join(out_dir, 'SUMMARY.out')
    open(summary, 'w').write('\n'.join(lines) + '\n')
    lines.append(f'REVIEW: {summary}')
    print('\n'.join(lines))
    return {
        't_scf_dftb': t_scf_dftb, 't_proj_dftb': t_proj_d,
        't_scf_pyscf': t_scf_py, 't_proj_pyscf': t_proj_p,
        'homo_dftb': homo_d, 'homo_pyscf': homo_p,
        'E_homo_dftb': float(E_d_eV[homo_d]), 'E_homo_pyscf': float(E_p_eV[homo_p]),
    }


def run_frontier_stm_current(mol_name, pos, names, types, args):
    """MO-resolved STM current I≥0 at z≈stm_z_above; per tip (s|pz|py); vertical + horizontal."""
    z_mol = float(np.mean(pos[:, 2]))
    z_plane = z_mol + float(args.stm_z_above)
    scan_xs, scan_ys = make_scan_grid(pos, args.scan_step, args.margin)
    n_near = int(args.n_near)
    tips = [t.strip().lower() for t in args.stm_tips.split(',') if t.strip()]
    for t in tips:
        if t not in afm_utils.STM_TIP_ORBITALS:
            raise ValueError(f'Unknown stm tip {t!r}; use s,pz,py')
    basis_key = _basis_key(args.bases)
    basis_hsd = WFC_HSD_PATHS[basis_key]
    root_dir = os.path.join(args.outdir, mol_name, 'frontier_stm_diag')
    os.makedirs(root_dir, exist_ok=True)
    work = os.path.join(root_dir, f'dftb_work_{basis_key}')

    lines = [
        f'Frontier STM diagnostic: {mol_name}',
        f'z_stm = z_mol({z_mol:.3f}) + {args.stm_z_above} = {z_plane:.3f} Å',
        f'tips={tips}  scan={len(scan_xs)}x{len(scan_ys)} step={args.scan_step}  n_near=±{n_near}',
        'STM current I≥0 (viridis); tip selects φ_t, I~|⟨φ_s|H\'|φ_t⟩|²',
        f'DFTB basis={basis_key}  pySCF={args.pyscf_xc}/{args.pyscf_basis}',
        '',
    ]

    t0 = time.perf_counter()
    d = afm_utils.get_density_from_dftb_dense(
        pos, types, basis_hsd, work, step=0.5, margin=0.5, z_extra=0.5,
        verbosity=0, project_density=False)
    t_scf_dftb = time.perf_counter() - t0
    eigvecs, eigvals = d['eigvecs'], np.asarray(d['eigvals'], dtype=np.float64)
    n_elec = afm_utils.dftb_n_valence_electrons(enames=names)
    homo_d, lumo_d = afm_utils.dftb_frontier_mo_indices(eigvals, n_elec=n_elec)
    E_d_eV = eigvals * HAU2EV
    lo = max(0, homo_d - n_near)
    hi = min(len(eigvals) - 1, lumo_d + n_near)
    mo_list_d = list(range(lo, hi + 1))
    projector = d['projector']
    atoms_dict = d['atoms_dict']
    basis_ang = d['basis_ang']
    species_per_atom = list(range(len(names)))
    afm_utils._set_projector_species_basis(projector, atoms_dict, basis_ang, rc_max=8.0)
    lines.append(
        f'[DFTB] HOMO#{homo_d} E={E_d_eV[homo_d]:.3f} eV  gap={E_d_eV[lumo_d]-E_d_eV[homo_d]:.3f} eV  '
        f'SCF={t_scf_dftb:.3f} s')

    pu = _load_pu()
    backend = pu.resolve_backend(args.pyscf_backend)
    t0 = time.perf_counter()
    r = pu.run_scf_geometry(names, pos, basis=args.pyscf_basis, xc=args.pyscf_xc,
                            backend=backend, release=False, max_memory_mb=20000)
    t_scf_py = time.perf_counter() - t0
    mf = r['mf']
    homo_p, lumo_p = pu.homo_lumo_indices(mf)
    E_p_eV = np.asarray(mf.mo_energy, dtype=np.float64) * HAU2EV
    lo_p = max(0, homo_p - n_near)
    hi_p = min(len(E_p_eV) - 1, lumo_p + n_near)
    mo_list_p = list(range(lo_p, hi_p + 1))
    lines.append(
        f'[pySCF] HOMO#{homo_p} E={E_p_eV[homo_p]:.3f} eV  gap={E_p_eV[lumo_p]-E_p_eV[homo_p]:.3f} eV  '
        f'SCF={t_scf_py:.3f} s')

    rels = [k for k in range(-n_near, n_near + 2)
            if (homo_d + k) >= 0 and (homo_d + k) < len(eigvals)
            and (homo_p + k) >= 0 and (homo_p + k) < len(E_p_eV)]

    layouts = [('vertical', 'E↑'), ('horizontal', 'E→')]
    if getattr(args, 'layout', 'both') == 'vertical':
        layouts = [('vertical', 'E↑')]
    elif getattr(args, 'layout', 'both') == 'horizontal':
        layouts = [('horizontal', 'E→')]

    for tip in tips:
        tip_dir = os.path.join(root_dir, f'tip_{tip}')
        os.makedirs(tip_dir, exist_ok=True)
        maps_d_abs, maps_p_abs = {}, {}
        t_proj_d = t_proj_p = 0.0
        for imo in mo_list_d:
            t1 = time.perf_counter()
            arr = afm_utils.project_mo_stm_sk_slice(
                projector, eigvecs[imo], atoms_dict, basis_ang, names, species_per_atom,
                scan_xs, scan_ys, z_plane, tip_orbital=tip, intensity=True)
            t_proj_d += time.perf_counter() - t1
            maps_d_abs[imo] = arr
        for imo in mo_list_p:
            t1 = time.perf_counter()
            arr = pu.eval_mo_stm_pyscf_slice(mf, imo, scan_xs, scan_ys, z_plane, tip_orbital=tip)
            t_proj_p += time.perf_counter() - t1
            maps_p_abs[imo] = np.asarray(arr)

        tip_label = {'s': 's-tip', 'pz': 'p_z-tip', 'py': 'p_y-tip'}[tip]
        base_title = (f'{mol_name}  STM {tip_label}  z={z_plane:.2f}Å  |  DFTB {basis_key} vs pySCF '
                      f'{args.pyscf_xc}/{args.pyscf_basis}')
        for lay, tag in layouts:
            combo_png = os.path.join(
                tip_dir, f'spectrum_stm_{lay}_z{args.stm_z_above:.1f}_{mol_name}.png')
            afm_utils.plot_spectrum_with_orbitals(
                E_d_eV, homo_d, maps_d_abs, E_p_eV, homo_p, maps_p_abs, rels,
                scan_xs, scan_ys, combo_png, atom_pos=pos, field='psi2', layout=lay,
                title=f'{base_title}  I≥0  ({tag}, {lay})')
            lines.append(f'REVIEW: {combo_png}')
        lines.append(
            f'[tip={tip}] DFTB proj {t_proj_d:.3f}s ({1e3*t_proj_d/len(mo_list_d):.1f} ms/MO)  '
            f'pySCF proj {t_proj_p:.3f}s ({1e3*t_proj_p/len(mo_list_p):.1f} ms/MO)')

    pu.release_scf(mf)
    summary = os.path.join(root_dir, 'SUMMARY.out')
    open(summary, 'w').write('\n'.join(lines) + '\n')
    lines.append(f'REVIEW: {summary}')
    print('\n'.join(lines))
    n_mo = len(mo_list_d)
    t_stm_d = sum(
        float(line.split('DFTB proj ')[1].split('s')[0])
        for line in lines if line.startswith('[tip=') and 'DFTB proj' in line)
    t_stm_p = sum(
        float(line.split('pySCF proj ')[1].split('s')[0])
        for line in lines if line.startswith('[tip=') and 'pySCF proj' in line)
    return {
        't_scf_dftb': t_scf_dftb, 't_scf_pyscf': t_scf_py,
        't_proj_dftb': t_stm_d, 't_proj_pyscf': t_stm_p,
        'homo_dftb': homo_d, 'homo_pyscf': homo_p,
        'tips': tips, 'z_plane': z_plane, 'n_mo': n_mo,
        'scan': f'{len(scan_xs)}x{len(scan_ys)}',
    }


def _run_dftb_channel(mol_name, pos, types, basis_key, out_dir, scan_xs, scan_ys, heights,
                      zeta_override, field, verbosity):
    basis_hsd = WFC_HSD_PATHS[basis_key]
    work = os.path.join(out_dir, f'dftb_work_{basis_key}')
    t0 = time.time()
    res = afm_utils.compute_stm_basis_variants(
        pos, types, basis_hsd, work, scan_xs, scan_ys, heights,
        projection_variants=('stock', 'prolonged'),
        zeta_override=zeta_override, field=field, verbosity=verbosity)
    print(f"  [{basis_key}] HOMO={res['homo']} LUMO={res['lumo']}  "
          f"E=[{res['E_homo']:.3f},{res['E_lumo']:.3f}] Ha "
          f"({res['E_homo']*HAU2EV:.2f},{res['E_lumo']*HAU2EV:.2f} eV)  wall={time.time()-t0:.1f}s")
    np.savez(os.path.join(out_dir, f'dftb_{basis_key}_stm.npz'),
             scan_xs=scan_xs, scan_ys=scan_ys, heights=heights,
             homo=res['homo'], lumo=res['lumo'],
             E_homo=res['E_homo'], E_lumo=res['E_lumo'],
             **{f'{v}_{lab}': res['maps'][v][lab]
                for v in res['maps'] for lab in res['maps'][v]})
    return res


def _run_pyscf_channel(mol_name, pos, names, out_dir, scan_xs, scan_ys, heights,
                       field, prefer_backend='auto', basis='def2-SVP', xc='PBE'):
    pu = _load_pu()
    backend = pu.resolve_backend(prefer_backend)
    pyscf_dir = os.path.join(out_dir, 'pyscf')
    os.makedirs(pyscf_dir, exist_ok=True)
    t0 = time.time()
    print(f"  [pySCF] SCF backend={backend} {xc}/{basis} ...")
    r = pu.run_scf_geometry(names, pos, basis=basis, xc=xc, backend=backend, release=False)
    mf = r['mf']
    meta = pu.write_frontier_mo_cubes(mf, pyscf_dir, names, pos, step_A=0.25, margin_A=4.0,
                                      z_extra_A=5.0, prefix=f'{mol_name}_{xc}_{basis}'.replace('*', 's'))
    print(f"  [pySCF] HOMO={meta['homo']} LUMO={meta['lumo']}  "
          f"E_H={meta['E_homo_eV']:.3f} E_L={meta['E_lumo_eV']:.3f} eV  wall={time.time()-t0:.1f}s")
    maps = {}
    for lab, imo in (('HOMO', meta['homo']), ('LUMO', meta['lumo'])):
        stack = [pu.eval_mo_on_xy_slice(mf, imo, scan_xs, scan_ys, h) for h in heights]
        if field == 'psi2':
            stack = [p ** 2 for p in stack]
        maps[lab] = np.stack(stack, axis=2).astype(np.float32)
    pu.release_scf(mf)
    np.savez(os.path.join(pyscf_dir, 'stm_slices.npz'),
             scan_xs=scan_xs, scan_ys=scan_ys, heights=heights,
             HOMO=maps['HOMO'], LUMO=maps['LUMO'],
             homo=meta['homo'], lumo=meta['lumo'],
             E_homo_eV=meta['E_homo_eV'], E_lumo_eV=meta['E_lumo_eV'])
    for p in meta['paths'].values():
        print(f"  REVIEW: {p}")
    return maps, meta


def _make_stm_panel(mol_name, pos, out_dir, scan_xs, scan_ys, heights, channels, field):
    labels = ['HOMO', 'LUMO']
    lines = [f'STM orbital compare: {mol_name}', f'field={field}', f'heights={list(map(float, heights))}', '']
    for ih, h in enumerate(heights):
        maps_by_col = []
        titles = []
        for title, maps in channels:
            titles.append(title)
            maps_by_col.append({lab: maps[lab][:, :, ih:ih + 1] for lab in labels})
        png = os.path.join(out_dir, f'panel_{mol_name}_z{h:.1f}.png')
        afm_utils.plot_stm_basis_compare_panel(
            maps_by_col, scan_xs, scan_ys, h, labels, titles, png,
            field=field, atom_pos=pos,
            title=f'{mol_name}  STM {field}  z={h:.1f} Å  |  stock vs prolonged STO')
        print(f'REVIEW: {png}')
        lines.append(f'REVIEW: {png}')
    summary = os.path.join(out_dir, 'SUMMARY.out')
    open(summary, 'w').write('\n'.join(lines) + '\n')
    print(f'REVIEW: {summary}')
    return summary


def run_stm_vacuum_panel(mol_name, pos, names, types, info, args):
    """HOMO/LUMO vacuum STM panel: DFTB stock/prolonged vs pySCF."""
    out_dir = os.path.join(args.outdir, mol_name)
    os.makedirs(out_dir, exist_ok=True)
    scan_xs, scan_ys = make_scan_grid(pos, args.scan_step, args.margin)
    heights = np.array([float(x) for x in args.heights.split(',')], dtype=np.float64)
    zeta = load_zeta_override(info)
    field = str(args.field).lower()
    if field not in ('psi', 'psi2', 'ldos'):
        field = 'psi2'

    print(f"\n=== {mol_name}: {len(names)} atoms  scan={len(scan_xs)}x{len(scan_ys)}  "
          f"heights={list(heights)}  field={field} ===")

    channels = []
    bases = [b.strip() for b in args.bases.split(',') if b.strip()]
    if 'mio' in bases:
        r_mio = _run_dftb_channel(mol_name, pos, types, 'mio-1-1', out_dir, scan_xs, scan_ys,
                                  heights, zeta, field, args.verbosity)
        channels.append(('mio stock', r_mio['maps']['stock']))
        channels.append(('mio prolonged', r_mio['maps']['prolonged']))
    if '3ob' in bases:
        r_3ob = _run_dftb_channel(mol_name, pos, types, '3ob-3-1', out_dir, scan_xs, scan_ys,
                                  heights, zeta, field, args.verbosity)
        channels.append(('3ob stock', r_3ob['maps']['stock']))
        channels.append(('3ob prolonged', r_3ob['maps']['prolonged']))

    if not getattr(args, 'skip_pyscf', False):
        maps_py, _ = _run_pyscf_channel(
            mol_name, pos, names, out_dir, scan_xs, scan_ys, heights, field,
            prefer_backend=args.pyscf_backend, basis=args.pyscf_basis, xc=args.pyscf_xc)
        channels.append((f'pySCF {args.pyscf_xc}/{args.pyscf_basis}', maps_py))

    gallery = [(t, m) for t, m in channels if t.startswith('3ob') or t.startswith('pySCF')]
    if not gallery:
        gallery = channels
    _make_stm_panel(mol_name, pos, out_dir, scan_xs, scan_ys, heights, gallery, field)


def add_stm_common_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group('geometry')
    g.add_argument('--molecule', nargs='+', default=['pentacene'],
                   help='Registry name(s), comma-separated ok (pentacene, PTCDA, …)')
    g.add_argument('--xyz', default=None, help='Override geometry (.xyz); ignores --molecule')
    g.add_argument('--outdir', default=DEFAULT_OUT, help='Output root')
    g.add_argument('--scan-step', type=float, default=0.25, help='XY scan step [Å]')
    g.add_argument('--margin', type=float, default=3.0, help='XY margin beyond atoms [Å]')
    b = p.add_argument_group('DFTB / pySCF')
    b.add_argument('--bases', default='3ob', help='DFTB bases for panel: mio,3ob')
    b.add_argument('--pyscf-backend', default='auto', choices=('auto', 'gpu', 'cpu', 'smalldft'))
    b.add_argument('--pyscf-basis', default='def2-SVP')
    b.add_argument('--pyscf-xc', default='PBE')
    b.add_argument('--skip-pyscf', action='store_true')
    b.add_argument('--verbosity', type=int, default=0)


def add_orbital_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--z-above', type=float, default=0.5, help='Slice height above molecular plane [Å]')
    p.add_argument('--n-near', type=int, default=5, help='±N MOs around HOMO/LUMO')
    p.add_argument('--layout', default='both', choices=('both', 'vertical', 'horizontal'),
                   help='Spectrum↔orbital layout (default: both)')


def add_stm_current_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--stm-z-above', type=float, default=3.0, help='STM height above molecular plane [Å]')
    p.add_argument('--stm-tips', default='s,pz,py', help='Tip orbitals: s,pz,py')
    p.add_argument('--n-near', type=int, default=5, help='±N MOs around HOMO/LUMO')
    p.add_argument('--layout', default='both', choices=('both', 'vertical', 'horizontal'))


def add_panel_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--heights', default='2.5,3.0,3.5', help='Comma probe heights [Å]')
    p.add_argument('--field', default='psi2', choices=('psi', 'psi2', 'ldos'),
                   help='psi=orbital phase; psi2/ldos=STM intensity')


def run_fgr_transfer_compare(mol_name, pos, names, types, info, args):
    """Compare legacy exp-overlap STM vs long-tail FGR I_S / I_H / I_τ (Level B EH).

    Columns: overlap_exp | I_S=|c†Sc|² | I_H=|c†Hc|² | I_τ=|c†(H−ES)c|²
    Rows: HOMO/LUMO × tip (s, pz by default).
    Artifacts: debug/stm_fgr_compare/<mol>/
    """
    out_dir = os.path.join(args.outdir, mol_name)
    os.makedirs(out_dir, exist_ok=True)
    z_mol = float(np.mean(pos[:, 2]))
    z_plane = z_mol + float(args.stm_z_above)
    scan_xs, scan_ys = make_scan_grid(pos, args.scan_step, args.margin)
    tips = [t.strip().lower() for t in str(args.stm_tips).split(',') if t.strip()]
    for t in tips:
        if t not in afm_utils.STM_TIP_ORBITALS:
            raise ValueError(f'Unknown stm tip {t!r}; use s,pz,py')
    basis_key = _basis_key(args.bases)
    basis_hsd = WFC_HSD_PATHS[basis_key]
    work = os.path.join(out_dir, f'dftb_work_{basis_key}')
    K = float(getattr(args, 'eh_K', 1.75))
    tip_elem = str(getattr(args, 'tip_elem', 'C'))
    rcut = float(getattr(args, 'rcut', 15.0))
    taper_w = float(getattr(args, 'taper_w', 2.0))

    lines = [
        f'FGR transfer compare: {mol_name}',
        f'z = z_mol({z_mol:.3f}) + {args.stm_z_above} = {z_plane:.3f} Å',
        f'scan={len(scan_xs)}x{len(scan_ys)} step={args.scan_step}  tips={tips}',
        f'DFTB={basis_key}  Level-B EH K={K}  tip_elem={tip_elem}  rcut={rcut}  taper_w={taper_w}',
        'Methods: overlap_exp (legacy) | I_S | I_H | I_τ=H−E·S (recommended)',
        '',
    ]

    t0 = time.perf_counter()
    d = afm_utils.get_density_from_dftb_dense(
        pos, types, basis_hsd, work, step=0.5, margin=0.5, z_extra=0.5,
        verbosity=0, project_density=False)
    t_scf = time.perf_counter() - t0
    eigvecs, eigvals = d['eigvecs'], np.asarray(d['eigvals'], dtype=np.float64)
    n_elec = afm_utils.dftb_n_valence_electrons(enames=names)
    homo, lumo = afm_utils.dftb_frontier_mo_indices(eigvals, n_elec=n_elec)
    E_eV = eigvals * HAU2EV
    projector = d['projector']
    atoms_dict = d['atoms_dict']
    basis_ang = d['basis_ang']
    species_per_atom = list(range(len(names)))
    afm_utils._set_projector_species_basis(projector, atoms_dict, basis_ang, rc_max=max(8.0, rcut))
    lines.append(
        f'[DFTB] HOMO#{homo} E={E_eV[homo]:.3f} eV  LUMO#{lumo} E={E_eV[lumo]:.3f} eV  '
        f'SCF={t_scf:.3f}s')

    zeta = load_zeta_override(info)
    t0 = time.perf_counter()
    tables, tip_type0, name_to_smp, prol, _sto = afm_utils._stm_fgr_prepare_tables(
        projector, work, basis_ang, tip_elem=tip_elem, zeta_override=zeta, K=K,
        rcut_table=rcut, sample_elems=sorted(set(names)))
    t_tab = time.perf_counter() - t0
    lines.append(
        f'[tables] pairs={tables["n_pair"]} n_r={tables["n_r"]} '
        f'r0={tables["r_grid0"]} dr={tables["dr"]}  wall={t_tab:.2f}s  '
        f'smp_types={tables["smp_names"]}')

    # Degenerate-cluster aware MO selection: sum I over near-degenerate manifold.
    # DFTB returns arbitrary rotations within a degenerate subspace → asymmetric
    # single-MO maps; I_total = Σ_i |M_i|² is the physically correct STM current
    # (all degenerate states contribute at finite bias/T) and restores symmetry.
    degen_thresh_eV = float(getattr(args, 'degen_thresh', 0.0))
    HAU2EV_loc = HAU2EV
    def degen_cluster(center, direction):
        """Return list of MO indices in the degenerate cluster around `center`.
        direction=-1 → search downward (HOMO side); +1 → upward (LUMO side)."""
        if degen_thresh_eV <= 0.0:
            return [center]
        E0 = E_eV[center]
        cluster = [center]
        i = center + direction
        while 0 <= i < len(E_eV) and abs(E_eV[i] - E0) < degen_thresh_eV:
            cluster.append(i)
            i += direction
        return sorted(cluster)
    homo_cluster = degen_cluster(homo, -1)
    lumo_cluster = degen_cluster(lumo, +1)
    if len(homo_cluster) > 1:
        lines.append(f'[degen] HOMO cluster {homo_cluster} '
                     f'E={[f"{E_eV[i]:.4f}" for i in homo_cluster]} eV  '
                     f'(thresh={degen_thresh_eV} eV) → summing I over cluster')
    if len(lumo_cluster) > 1:
        lines.append(f'[degen] LUMO cluster {lumo_cluster} '
                     f'E={[f"{E_eV[i]:.4f}" for i in lumo_cluster]} eV  '
                     f'(thresh={degen_thresh_eV} eV) → summing I over cluster')
    mo_groups = [('HOMO', homo_cluster), ('LUMO', lumo_cluster)]
    methods = [
        ('overlap_exp', None),
        ('I_S', 'S'),
        ('I_H', 'H'),
        ('I_tau', 'tau'),
    ]
    row_labels = []
    # maps_by_col[method_idx][row_label] = (nx,ny)
    maps_by_col = [{}, {}, {}, {}]
    for tip in tips:
        for lab, cluster in mo_groups:
            suffix = f'×{len(cluster)}' if len(cluster) > 1 else ''
            row = f'{lab}{suffix} tip={tip}'
            row_labels.append(row)
            E_tun = float(eigvals[cluster[0]])  # sample partition (cluster avg ≈ same)
            for ic, (mtitle, mode) in enumerate(methods):
                t1 = time.perf_counter()
                arr_sum = None
                for imo in cluster:
                    if mode is None:
                        arr = afm_utils.project_mo_stm_sk_slice(
                            projector, eigvecs[imo], atoms_dict, basis_ang, names,
                            species_per_atom, scan_xs, scan_ys, z_plane,
                            tip_orbital=tip, rcut=min(8.0, rcut), intensity=True)
                    else:
                        arr = afm_utils.project_mo_stm_fgr_slice(
                            projector, eigvecs[imo], atoms_dict, basis_ang, names,
                            species_per_atom, scan_xs, scan_ys, z_plane,
                            tables, tip_type0, name_to_smp, E_tun,
                            tip_orbital=tip, tip_elem=tip_elem, mode=mode,
                            rcut=rcut, taper_w=taper_w, intensity=True)
                    arr_sum = arr if arr_sum is None else arr_sum + arr
                dt = time.perf_counter() - t1
                maps_by_col[ic][row] = arr_sum
                lines.append(
                    f'  [{mtitle:11s}] {row:16s}  max={float(arr_sum.max()):.3e}  '
                    f'mean={float(arr_sum.mean()):.3e}  {dt*1e3:.1f} ms'
                    + (f'  ({len(cluster)} MOs)' if len(cluster) > 1 else ''))

    col_titles = [m[0] for m in methods]
    png = os.path.join(out_dir, f'fgr_compare_z{args.stm_z_above:.1f}_{mol_name}.png')
    afm_utils.plot_stm_fgr_method_panel(
        maps_by_col, scan_xs, scan_ys, z_plane, row_labels, col_titles, png,
        atom_pos=pos,
        title=(f'{mol_name}  FGR vs overlap  z={z_plane:.2f}Å  DFTB {basis_key}  '
               f'EH K={K}  E_tunnel=ε_sample'))
    lines.append(f'REVIEW: {png}')

    # Per-row contrast metric: centre vs perimeter (rough halo diagnostic)
    nx, ny = len(scan_xs), len(scan_ys)
    cx0, cx1 = nx // 4, 3 * nx // 4
    cy0, cy1 = ny // 4, 3 * ny // 4
    lines.append('')
    lines.append('Contrast = mean(centre)/mean(border)  (>1 ⇒ brighter molecular core)')
    for row in row_labels:
        parts = []
        for ic, (mtitle, _) in enumerate(methods):
            a = maps_by_col[ic][row]
            core = float(a[cx0:cx1, cy0:cy1].mean())
            border = float(a.mean()) - 0.5 * core  # crude
            # better: frame mean
            frame = np.concatenate([a[:nx // 8].ravel(), a[-nx // 8:].ravel(),
                                    a[:, :ny // 8].ravel(), a[:, -ny // 8:].ravel()])
            fr = float(frame.mean()) + 1e-30
            parts.append(f'{mtitle}={core/fr:.2f}')
        lines.append(f'  {row}: ' + '  '.join(parts))

    np.savez(os.path.join(out_dir, f'fgr_compare_z{args.stm_z_above:.1f}.npz'),
             scan_xs=scan_xs, scan_ys=scan_ys, z_plane=z_plane,
             homo=homo, lumo=lumo, E_homo=float(eigvals[homo]), E_lumo=float(eigvals[lumo]),
             homo_cluster=np.array(homo_cluster), lumo_cluster=np.array(lumo_cluster),
             degen_thresh_eV=degen_thresh_eV,
             **{f'{mtitle}_{row.replace(" ", "_")}': maps_by_col[ic][row]
                for ic, (mtitle, _) in enumerate(methods) for row in row_labels})
    summary = os.path.join(out_dir, 'SUMMARY.out')
    open(summary, 'w').write('\n'.join(lines) + '\n')
    lines.append(f'REVIEW: {summary}')
    print('\n'.join(lines))
    return {'homo': homo, 'lumo': lumo, 'png': png, 'summary': summary}


def run_br_stm_fgr_compare(mol_name, atomPos, enames, args):
    """BR-STM compare: overlap vs FGR I_τ with PP-AFM tip displacement.

    Uses the same Fz height stack as ``run_br_stm_afm_panel`` (where PP bending is
    significant). Two columns: BR-overlap (legacy, displaced) | BR-I_τ (FGR H−E·S, displaced).
    Rows: HOMO/LUMO × tip (s, pz). One panel with all Fz heights as columns.

    Artifacts: debug/stm_br_fgr_compare/<mol>/
    """
    import matplotlib
    matplotlib.use('Agg')
    from spammm.SPM import AFM as afm_mod
    from spammm.SPM.ModularPipeline import ModularAFMPipeline
    from spammm.config_utils import get_dftb_basis_path
    from spammm.quantum.DFTB.DFTBplusParser import (
        parse_wfc_hsd, convert_wfc_to_species_list_ang, make_slater_tail_species_list,
    )

    atomPos = np.asarray(atomPos, dtype=np.float64)
    enames = list(enames)
    out_dir = os.path.join(args.outdir, mol_name)
    os.makedirs(out_dir, exist_ok=True)
    tips = [t.strip().lower() for t in str(args.stm_tips).split(',') if t.strip()]
    for t in tips:
        if t not in afm_utils.STM_TIP_ORBITALS:
            raise ValueError(f'Unknown stm tip {t!r}; use s,pz,py')
    basis_key = _basis_key(args.bases)
    K = float(getattr(args, 'eh_K', 1.75))
    tip_elem = str(getattr(args, 'tip_elem', 'C'))
    rcut = float(getattr(args, 'rcut', 15.0))
    taper_w = float(getattr(args, 'taper_w', 2.0))
    degen_thresh_eV = float(getattr(args, 'degen_thresh', 0.005))

    # ── PP-AFM pipeline (Stages 1-4) — same h_Fz stack as stm br ──────────────
    step = float(getattr(args, 'step', 0.1))
    margin = float(getattr(args, 'margin', 4.0))
    z_extra = float(getattr(args, 'z_extra', 6.0))
    scan_range = float(getattr(args, 'scan_range', 3.0))
    scan_step = float(getattr(args, 'scan_step', step))
    h_min = float(getattr(args, 'h_min', 3.7))
    h_max = float(getattr(args, 'h_max', 4.7))
    h_step = float(getattr(args, 'h_step', 0.1))
    amp = float(getattr(args, 'amp', 1.0))
    amp_align = not bool(getattr(args, 'no_amp_align', False))
    K_LAT_Nm = float(getattr(args, 'K_LAT', 0.5))
    K_RAD = float(getattr(args, 'K_RAD', 20.0))
    bond_length = float(getattr(args, 'bond_length', 3.0))
    tip_mode = str(getattr(args, 'tip_mode', 'co'))
    projection = str(getattr(args, 'projection', 'prolonged'))
    force_recompute = bool(getattr(args, 'force', False))

    h_df, h_Fz, h_scan = afm_utils.afm_df_height_stacks(h_min, h_max, h_step, amp=amp, amp_align=amp_align)

    lines = [
        f'BR-STM FGR compare: {mol_name}',
        f'Fz heights (BR-STM): {list(h_Fz)}',
        f'scan_step={scan_step}  tips={tips}',
        f'DFTB={basis_key}  Level-B EH K={K}  tip_elem={tip_elem}  rcut={rcut}  taper_w={taper_w}',
        f'PP-AFM: tip={tip_mode} K_LAT={K_LAT_Nm} N/m  projection={projection}',
        'Methods: BR-overlap_exp (legacy, displaced) | BR-I_τ=H−E·S (FGR, displaced)',
        '',
    ]
    print('\n'.join(lines))

    pipe = ModularAFMPipeline(
        xyz_file=None, output_dir=out_dir, atomPos=atomPos, enames=enames,
        basis=basis_key, slako_prefix=basis_key, tip_mode=tip_mode,
        step=step, margin=margin, z_extra=z_extra,
        scan_range=scan_range, scan_step=scan_step,
        height_range=(float(h_scan[0]), float(h_scan[-1]) + 0.5 * h_step),
        height_step=h_step,
        work_dir=os.path.join(out_dir, 'dftb_work'), backend='dftb',
    )
    pipe.heights = np.asarray(h_scan, dtype=np.float64)
    pipe.cache_stage3 = os.path.join(out_dir, f'cache_stage3_potentials_{projection}.npz')
    pipe.cache_stage4 = os.path.join(out_dir, f'cache_stage4_relax_{projection}.npz')

    pa = dict(afm_mod.PAULI_FITTED_DEFAULTS.get(basis_key, afm_mod.PAULI_FITTED_DEFAULTS['3ob-3-1']))
    dm, eigvecs, eigvals = pipe.stage1_scf(force_recompute=force_recompute)
    rho_scf_stock, rho_na, rho_diff_stock = pipe.stage2_project(dm, force_recompute=force_recompute)

    basis_hsd_path = get_dftb_basis_path(basis_key)
    basis_data = parse_wfc_hsd(basis_hsd_path)
    basis_ang = convert_wfc_to_species_list_ang(basis_data, resolution_bohr=0.04)
    prol_ang = make_slater_tail_species_list(basis_ang)
    sto_ang = prol_ang if projection == 'prolonged' else basis_ang

    if projection == 'prolonged':
        afm_utils._set_projector_species_basis(pipe.projector, pipe.atoms_dict, prol_ang, rc_max=6.0)
        rho_scf_pauli = pipe.projector.project_density_dense(
            dm.astype(np.float32), pipe.norb_per_atom, pipe.orb_offsets, pipe.atoms_dict, pipe.grid_spec)
        if hasattr(pipe.projector, 'queue'):
            pipe.projector.queue.finish()
    else:
        rho_scf_pauli = rho_scf_stock
        afm_utils._set_projector_species_basis(pipe.projector, pipe.atoms_dict, basis_ang, rc_max=6.0)

    V_ES, E_pauli, E_ES, E_vdw, F_total = pipe.stage3_potentials(
        rho_scf_pauli, rho_na, rho_diff_stock, force_recompute=force_recompute,
        pauli_params={'A': pa['A'], 'beta': pa['beta']},
        vdw_params={'C6_CO': 30.0})
    K_LAT = afm_mod.stiffness_Nm_to_eVA2(K_LAT_Nm)
    s4_force = bool(force_recompute) or (not os.path.exists(pipe.cache_stage4))
    if (not s4_force) and os.path.exists(pipe.cache_stage4):
        if int(np.load(pipe.cache_stage4)['FEs_relax'].shape[2]) != int(len(h_scan)):
            s4_force = True
    _df_legacy, tip_disp, FEs_relax = pipe.stage4_relax(
        F_total, force_recompute=s4_force,
        relax_params={'K_LAT': K_LAT, 'K_RAD': K_RAD, 'bond_length': bond_length},
        ppm_mode=True)
    del _df_legacy, V_ES, E_pauli, E_ES, E_vdw

    # Extract tip displacement at Fz heights (same as stm br Stage 3)
    idx_Fz = [int(np.argmin(np.abs(h_scan - h))) for h in h_Fz]
    tip_disp_Fz = {'dx': tip_disp['dx'][:, :, idx_Fz], 'dy': tip_disp['dy'][:, :, idx_Fz]}
    dxy_Fz = np.hypot(tip_disp_Fz['dx'], tip_disp_Fz['dy']).astype(np.float32)
    scan_xs, scan_ys = pipe.scan_xs, pipe.scan_ys
    print(f'[BR-FGR] |dxy| at Fz heights: {[float(dxy_Fz[:,:,i].max()) for i in range(len(h_Fz))]} Å')

    # ── FGR tables ────────────────────────────────────────────────────────────
    eigvals = np.asarray(eigvals, dtype=np.float64)
    E_eV = eigvals * HAU2EV
    homo, lumo = afm_utils.dftb_frontier_mo_indices(eigvals, atomTypes=pipe.atoms_dict['type'])
    species_per_atom = list(range(len(enames)))
    afm_utils._set_projector_species_basis(pipe.projector, pipe.atoms_dict, basis_ang, rc_max=max(8.0, rcut))
    tables, tip_type0, name_to_smp, _prol, _sto = afm_utils._stm_fgr_prepare_tables(
        pipe.projector, os.path.join(out_dir, 'dftb_work'), basis_ang,
        tip_elem=tip_elem, K=K, rcut_table=rcut, sample_elems=sorted(set(enames)))
    print(f'[BR-FGR] HOMO#{homo} E={E_eV[homo]:.3f} eV  LUMO#{lumo} E={E_eV[lumo]:.3f} eV')

    # Degenerate clusters
    def degen_cluster(center, direction):
        if degen_thresh_eV <= 0.0:
            return [center]
        E0 = E_eV[center]
        cluster = [center]
        i = center + direction
        while 0 <= i < len(E_eV) and abs(E_eV[i] - E0) < degen_thresh_eV:
            cluster.append(i)
            i += direction
        return sorted(cluster)
    homo_cluster = degen_cluster(homo, -1)
    lumo_cluster = degen_cluster(lumo, +1)
    if len(homo_cluster) > 1:
        print(f'[degen] HOMO cluster {homo_cluster} E={[f"{E_eV[i]:.4f}" for i in homo_cluster]} eV')
    if len(lumo_cluster) > 1:
        print(f'[degen] LUMO cluster {lumo_cluster} E={[f"{E_eV[i]:.4f}" for i in lumo_cluster]} eV')
    mo_groups = [('HOMO', homo_cluster), ('LUMO', lumo_cluster)]
    # Only 2 methods: BR-overlap (displaced) and BR-I_τ (FGR, displaced)
    methods = [('BR-overlap', None), ('BR-I_tau', 'tau')]

    # ── Compute BR-STM for each method × MO × tip ─────────────────────────────
    XX, YY = np.meshgrid(scan_xs, scan_ys, indexing='ij')
    nx, ny = len(scan_xs), len(scan_ys)
    # maps_by_col[method_idx][row_label] = (nx, ny, nz_Fz)
    maps_by_col = [{}, {}]
    row_labels = []
    for tip in tips:
        for lab, cluster in mo_groups:
            suffix = f'×{len(cluster)}' if len(cluster) > 1 else ''
            row = f'{lab}{suffix} tip={tip}'
            row_labels.append(row)
            E_tun = float(eigvals[cluster[0]])
            for ic, (mtitle, mode) in enumerate(methods):
                t1 = time.perf_counter()
                arr_3d = np.zeros((nx, ny, len(h_Fz)), dtype=np.float32)
                if mode is None:
                    # Legacy overlap BR-STM with displacement (compute_bond_resolved_stm)
                    # Note: compute_bond_resolved_stm uses exp basis; set it first.
                    # It only accepts 1 MO at a time → loop over cluster and sum I.
                    afm_utils._set_projector_species_basis(pipe.projector, pipe.atoms_dict, sto_ang, rc_max=6.0)
                    arr_3d = np.zeros((nx, ny, len(h_Fz)), dtype=np.float32)
                    for imo in cluster:
                        arr_3d += afm_utils.compute_bond_resolved_stm(
                            pipe.projector, eigvecs, eigvals, scan_xs, scan_ys, h_Fz,
                            tip_disp_Fz, pipe.norb_per_atom, pipe.orb_offsets, pipe.atoms_dict,
                            mo_indices=[imo], field='psi2', use_exp_basis=True, exp_beta=1.0, exp_r0=3.0)
                else:
                    # FGR I_τ BR-STM with displacement
                    for iz, h in enumerate(h_Fz):
                        X_disp = XX + tip_disp_Fz['dx'][:, :, iz]
                        Y_disp = YY + tip_disp_Fz['dy'][:, :, iz]
                        for imo in cluster:
                            arr = afm_utils.project_mo_stm_fgr_points(
                                pipe.projector, eigvecs[imo], pipe.atoms_dict, basis_ang, enames,
                                species_per_atom, X_disp, Y_disp, float(h),
                                tables, tip_type0, name_to_smp, E_tun,
                                tip_orbital=tip, tip_elem=tip_elem, mode=mode,
                                rcut=rcut, taper_w=taper_w, intensity=True)
                            arr_3d[:, :, iz] += arr
                dt = time.perf_counter() - t1
                maps_by_col[ic][row] = arr_3d
                lines.append(
                    f'  [{mtitle:11s}] {row:16s}  max={float(arr_3d.max()):.3e}  '
                    f'mean={float(arr_3d.mean()):.3e}  {dt*1e3:.1f} ms'
                    + (f'  ({len(cluster)} MOs)' if len(cluster) > 1 else ''))

    # ── Plot: single panel with Fz heights as columns ┐
    # Use plot_afm_variant_height_strip (same as 03_brstm_vs_stm.png)
    # Build variants: one per method, each with rows keyed by row_label
    extent = afm_utils.scan_extent(scan_xs, scan_ys)
    variants = {}
    for ic, (mtitle, _) in enumerate(methods):
        variants[mtitle] = {row: maps_by_col[ic][row] for row in row_labels}
    # row_specs: (key, variant_key, y_label, cmap)
    row_specs = []
    for row in row_labels:
        row_specs.append((row, 'BR-overlap', row, 'viridis'))
        row_specs.append((row, 'BR-I_tau', row, 'viridis'))
    # Actually plot_afm_variant_height_strip expects (field, variant_key, label, cmap)
    # where field is the key into the variant dict. We need a different layout:
    # 2 columns (methods) × N rows (MO×tip). Use the same function as 03_brstm_vs_stm.
    # Build a combined variant dict: each method is a variant, each row is a field.
    # But the function plots (field, variant) → so we need field=row_label, variant=method.
    # Rebuild: variants_by_method[method][row] = arr_3d
    # row_specs: for each row, two entries (one per method column)
    row_specs_flat = []
    for row in row_labels:
        row_specs_flat.append((row, 'BR-overlap', row, 'viridis'))
        row_specs_flat.append((row, 'BR-I_tau', row, 'viridis'))
    png = os.path.join(out_dir, f'br_fgr_compare_{mol_name}.png')
    afm_utils.plot_afm_variant_height_strip(
        variants, row_specs_flat, h_Fz, png, scale='per_image',
        title=(f'{mol_name}  BR-STM: overlap vs FGR I_τ  @ Fz heights  '
               f'|dxy|_max={float(dxy_Fz.max()):.2f}Å  DFTB {basis_key}  EH K={K}  tip={tip_mode}'),
        dpi=140, apos=atomPos, show_atoms=True,
        extent=extent, amp=None, amp_align=False, figsize_row=1.4,
    )
    lines.append(f'REVIEW: {png}')

    # Save npz
    np.savez(os.path.join(out_dir, 'br_fgr_compare.npz'),
             scan_xs=scan_xs, scan_ys=scan_ys, h_Fz=h_Fz,
             homo=homo, lumo=lumo, E_homo=float(eigvals[homo]), E_lumo=float(eigvals[lumo]),
             homo_cluster=np.array(homo_cluster), lumo_cluster=np.array(lumo_cluster),
             degen_thresh_eV=degen_thresh_eV, dxy_Fz=dxy_Fz,
             tip_disp_dx_Fz=tip_disp_Fz['dx'], tip_disp_dy_Fz=tip_disp_Fz['dy'],
             **{f'{mtitle}_{row.replace(" ", "_")}': maps_by_col[ic][row]
                for ic, (mtitle, _) in enumerate(methods) for row in row_labels})
    summary = os.path.join(out_dir, 'SUMMARY.out')
    open(summary, 'w').write('\n'.join(lines) + '\n')
    lines.append(f'REVIEW: {summary}')
    print('\n'.join(lines))
    return {'homo': homo, 'lumo': lumo, 'png': png, 'summary': summary}


def add_fgr_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--stm-z-above', type=float, default=3.0, help='STM height above molecular plane [Å]')
    p.add_argument('--stm-tips', default='s,pz', help='Tip orbitals: s,pz,py')
    p.add_argument('--tip-elem', default='C', help='Phantom tip atom element for SK tables')
    p.add_argument('--eh-K', type=float, default=1.75, dest='eh_K', help='Extended-Hückel K')
    p.add_argument('--rcut', type=float, default=15.0, help='Atom-pair cutoff [Å]')
    p.add_argument('--taper-w', type=float, default=2.0, dest='taper_w',
                   help='Cosine taper width at rcut [Å] (eliminates hard-cutoff ring artifacts)')
    p.add_argument('--degen-thresh', type=float, default=0.005, dest='degen_thresh',
                   help='Degeneracy threshold [eV]: sum I over MOs within this gap of HOMO/LUMO (0=off)')

