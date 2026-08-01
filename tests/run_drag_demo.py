"""Drag-and-bounce demo: 4 PTCDA on NaCl(FAF), one dragged by end-O through others.

Shows:
  - PairFF molecule-molecule collisions (Pauli repulsion + Coulomb)
  - FAF substrate interaction (O→Na+ electrostatic attraction, stick-slip)
  - Anchor on an end oxygen atom (not COM)
  - Linear drag path through the other molecules

Outputs (debug/drag_demo/):
  - frames.xyz       — full atom trajectory (real atoms only)
  - cogs.xyz          — COG trajectory (4 atoms/frame)
  - anchor_path.xyz   — anchor target + actual trajectory
  - drag_demo.gif     — 2-panel animation (top: xy, side: xz)

Run:  PYTHONUNBUFFERED=1 python3 tests/run_drag_demo.py
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from spammm.forcefields.RigidBodyDynamics import RigidBodyPairFF
from spammm.forcefields.RigidBodyUtils import load_molecule
from spammm.surfaces.FoldedRigid import load_fit

PTCDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'xyz', 'PTCDA.xyz')
FIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'fits', 'ptcda_nacl_factorized.npz')
OUTDIR = os.path.join('debug', 'drag_demo')

# ─── Parameters ────────────────────────────────────────────────────────────
NMOL = 2               # 1 dragged + 1 obstacle (simpler, clearer stick-slip)
Z_INIT = 3.0            # Å above surface — closer = stronger corrugation (0.75 eV vs 0.24 at 3.6)
DT = 0.02
LIN_DAMP = 0.95         # moderate damping — let molecule settle on surface
ANG_DAMP = 0.92
N_STEPS = 16000
RELAX_STEPS = 500       # relaxation steps before dragging (settle on surface)
STEPS_PER_CHUNK = 5
FRAME_EVERY = 40        # save frame every 40 MD steps → 400 frames
ANCHOR_K = 0.2          # moderate spring — stick-slip when k*lag > Faf_Fx_max (0.59)
K_Z = 5.0              # strong z-confinement — hold molecule ON surface against Fz repulsion
# Anchor on O[25] — right-end anhydride oxygen (x=+5.73 in body frame)
ANCHOR_ATOM_LOCAL = 25
# Linear drag: from O atom initial position, total displacement DRAG_X_END along x
DRAG_X_END = 40.0       # Å total drag (passes through 1 obstacle at x=24)
DRAG_Y = 0.0            # straight line along x

# Obstacle positions: 1 molecule in the path
OBSTACLE_X = [24.0]     # Å
OBSTACLE_Y = [0.0]


def _outdir():
    os.makedirs(OUTDIR, exist_ok=True)
    return OUTDIR


def build_system():
    """Build 4 PTCDA on NaCl with FAF.  Mol 0 is dragged, mols 1-3 are obstacles."""
    fit = load_fit(FIT)
    # Load with qeq=False — the FAF fix now uses fit's atom_plqh (correct charges)
    apos, enames, REQs, _ = load_molecule(PTCDA, qeq=False, name='PTCDA')
    # Mol 0 starts to the left; mols 1+ are obstacles along x
    n_obstacle = len(OBSTACLE_X)
    pos_list = [[0.0, 0.0, Z_INIT]]
    for i in range(n_obstacle):
        pos_list.append([OBSTACLE_X[i], OBSTACLE_Y[i], Z_INIT])
    pos = np.array(pos_list, dtype=np.float32)
    # All molecules unrotated — long axis along x (same as drag direction)
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (NMOL, 1))
    rbd = RigidBodyPairFF.from_molecules([(apos, enames, REQs)] * NMOL, pos, quats=quat)
    rbd.attach_pairff_faf(fit, z_init=Z_INIT, k_z=K_Z, enable=True)
    # Element names per real site (repeat for each molecule)
    types_flat = np.concatenate([p['types'] for p in rbd._mb_packs])
    real_mask = types_flat == 0
    enames_flat = np.tile(enames, NMOL)
    return rbd, real_mask, enames_flat


def anchor_target(step, base_pos):
    """Linear drag from base_pos to base_pos + DRAG_X_END along y=DRAG_Y.

    Starts at the O atom's initial position — no initial spring stretch.
    """
    frac = step / N_STEPS
    x = base_pos[0] + frac * DRAG_X_END
    y = base_pos[1] + DRAG_Y
    return np.array([x, y, base_pos[2]], dtype=np.float32)


def download_atoms(rbd, real_mask, enames_flat):
    atoms = np.empty((rbd.total_atoms, 4), dtype=np.float32)
    rbd.fromGPU('apos_world', atoms)
    rbd.queue.finish()
    return enames_flat.copy(), atoms[real_mask, :3].copy(), atoms


def download_cogs(rbd):
    pos4 = np.empty((rbd.n_bodies, 4), dtype=np.float32)
    rbd.fromGPU('poss', pos4)
    rbd.queue.finish()
    return pos4[:, :3].copy()


def write_xyz_frame(fh, names, pos, comment):
    pos = np.atleast_2d(pos)
    fh.write(f"{len(names)}\n{comment}\n")
    for n, p in zip(names, pos):
        fh.write(f"{n} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


def run_demo():
    outdir = _outdir()
    print(f"[demo] building {NMOL}×PTCDA on NaCl with FAF ...", flush=True)
    rbd, real_mask, enames_flat = build_system()
    n_per_mol = len(rbd._mb_packs[0]['types'])
    n_real_per_mol = int((rbd._mb_packs[0]['types'] == 0).sum())
    mol_real_indices = [np.arange(m * n_per_mol, m * n_per_mol + n_real_per_mol) for m in range(NMOL)]
    print(f"[demo] total_atoms={rbd.total_atoms}, real={int(real_mask.sum())}, per_mol={n_real_per_mol}", flush=True)

    # Global index of the anchored atom (molecule 0, local index ANCHOR_ATOM_LOCAL)
    anchor_ia = 0 * n_per_mol + ANCHOR_ATOM_LOCAL
    print(f"[demo] anchored atom: mol=0, O[{ANCHOR_ATOM_LOCAL}], global={anchor_ia}", flush=True)

    # Initial world position of the anchored atom (computed from body-frame + CoM)
    # No MD step needed — just rotate body-frame O position by identity quaternion + CoM
    from spammm.surfaces.FoldedRigid import Z_SURF_TOP
    z_world = Z_SURF_TOP + Z_INIT
    # Get body-frame positions from the rbd's apos_body buffer
    apos_body = np.empty((rbd.total_atoms, 4), dtype=np.float32)
    rbd.fromGPU('apos_body', apos_body); rbd.queue.finish()
    anchor_base = np.array([
        apos_body[anchor_ia, 0] + 0.0,  # body x + CoM x (CoM at origin for mol 0)
        apos_body[anchor_ia, 1] + 0.0,  # body y + CoM y
        z_world,                            # world z (all PTCDA atoms planar at z=0)
    ], dtype=np.float32)
    atoms0 = np.empty((rbd.total_atoms, 4), dtype=np.float32)
    rbd.fromGPU('apos_world', atoms0); rbd.queue.finish()
    print(f"[demo] anchor base position: {anchor_base}", flush=True)
    print(f"[demo] drag path: O atom from x={anchor_base[0]:.1f} to x={anchor_base[0]+DRAG_X_END:.1f}, y={DRAG_Y}", flush=True)

    # Verify FAF charges are present
    plqh = rbd.folded_params['atom_type']
    o_plqh = plqh[anchor_ia]
    print(f"[demo] anchored O PLQH: P={o_plqh[0]:.2f} L={o_plqh[1]:.4f} Q={o_plqh[2]:.4f} H={o_plqh[3]:.1f}", flush=True)
    assert abs(o_plqh[2]) > 0.1, f"O charge is {o_plqh[2]} — FAF Coulomb will be zero! Fix needed."

    # Open XYZ trajectory files
    fh_frames = open(os.path.join(outdir, 'frames.xyz'), 'w')
    fh_cogs = open(os.path.join(outdir, 'cogs.xyz'), 'w')
    fh_anchor = open(os.path.join(outdir, 'anchor_path.xyz'), 'w')

    # Storage for animation
    all_positions, all_cogs, all_targets, all_anchor_actual = [], [], [], []

    # Initial anchor (no spring during relaxation)
    anchors = np.zeros((rbd.total_atoms, 4), dtype=np.float32)
    rbd.toGPU('anchors', anchors)

    # Relaxation phase: let molecules settle on the surface (no anchor, high damping)
    print(f"[demo] relaxing {RELAX_STEPS} steps ...", flush=True)
    rbd.run_multimol_md(RELAX_STEPS, dt=DT, lin_damp=0.5, ang_damp=0.5, faf=True)
    atoms_relax = np.empty((rbd.total_atoms, 4), dtype=np.float32)
    rbd.fromGPU('apos_world', atoms_relax); rbd.queue.finish()
    E_relax = float(atoms_relax[:, 3].sum())
    anchor_base_relaxed = atoms_relax[anchor_ia, :3].copy()
    print(f"[demo] after relaxation: E={E_relax:.4f}  anchor_O={anchor_base_relaxed}", flush=True)

    # Set anchor at the RELAXED O position
    anchor_base = anchor_base_relaxed
    target0 = anchor_target(0, anchor_base)
    anchors[anchor_ia] = [target0[0], target0[1], target0[2], ANCHOR_K]
    rbd.toGPU('anchors', anchors)

    # Save initial frame
    names, pos, atoms = download_atoms(rbd, real_mask, enames_flat)
    cogs = download_cogs(rbd)
    E0 = float(atoms[:, 3].sum())
    write_xyz_frame(fh_frames, names, pos, f"step=0 E={E0:.4f}")
    write_xyz_frame(fh_cogs, ['C']*NMOL, cogs, "step=0 COGs")
    write_xyz_frame(fh_anchor, ['C'], target0, "step=0 anchor_target")
    all_positions.append(pos); all_cogs.append(cogs)
    all_targets.append(target0); all_anchor_actual.append(atoms0[anchor_ia, :3].copy())
    print(f"[demo] step=0  E={E0:.4f}  anchor={atoms0[anchor_ia,:3]}", flush=True)

    # MD loop
    step = 0
    t0 = time.perf_counter()
    while step < N_STEPS:
        chunk = min(STEPS_PER_CHUNK, N_STEPS - step)
        target = anchor_target(step + chunk, anchor_base)
        anchors[anchor_ia] = [target[0], target[1], target[2], ANCHOR_K]
        rbd.toGPU('anchors', anchors)
        rbd.run_multimol_md(chunk, dt=DT, lin_damp=LIN_DAMP, ang_damp=ANG_DAMP, faf=True)
        step += chunk
        if step % FRAME_EVERY == 0 or step >= N_STEPS:
            names, pos, atoms = download_atoms(rbd, real_mask, enames_flat)
            cogs = download_cogs(rbd)
            E = float(atoms[:, 3].sum())
            anchor_act = atoms[anchor_ia, :3].copy()
            write_xyz_frame(fh_frames, names, pos, f"step={step} E={E:.4f}")
            write_xyz_frame(fh_cogs, ['C']*NMOL, cogs, f"step={step} COGs")
            write_xyz_frame(fh_anchor, ['C'], target, f"step={step} anchor_target")
            all_positions.append(pos); all_cogs.append(cogs)
            all_targets.append(target); all_anchor_actual.append(anchor_act)
            if step % (FRAME_EVERY * 5) == 0 or step >= N_STEPS:
                print(f"[demo] step={step:4d}  E={E:.4f}  anchor=({anchor_act[0]:.2f},{anchor_act[1]:.2f},{anchor_act[2]:.2f})  target_x={target[0]:.2f}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"[demo] MD done in {elapsed:.1f}s ({N_STEPS/elapsed:.0f} steps/s)", flush=True)
    fh_frames.close(); fh_cogs.close(); fh_anchor.close()

    # ─── GIF ─────────────────────────────────────────────────────────────
    print(f"[demo] rendering GIF ({len(all_positions)} frames) ...", flush=True)
    make_gif(outdir, all_positions, all_cogs, all_targets, all_anchor_actual,
             mol_real_indices, enames_flat, n_real_per_mol)
    print(f"[demo] done.", flush=True)
    print(f"REVIEW: {os.path.join(outdir, 'frames.xyz')}", flush=True)
    print(f"REVIEW: {os.path.join(outdir, 'cogs.xyz')}", flush=True)
    print(f"REVIEW: {os.path.join(outdir, 'anchor_path.xyz')}", flush=True)
    print(f"REVIEW: {os.path.join(outdir, 'drag_demo.gif')}", flush=True)


# ─── Colors ───────────────────────────────────────────────────────────────
MOL_COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
ELEM_SIZES = {'C': 20, 'O': 35, 'H': 12}
ELEM_EDGE  = {'C': '#666', 'O': '#B71C1C', 'H': '#999'}


def make_gif(outdir, all_positions, all_cogs, all_targets, all_anchor_actual,
             mol_real_indices, enames_flat, n_real_per_mol):
    """2-panel animation: top=xy (top-down), bottom=xz (side view)."""
    n_frames = len(all_positions)
    all_xy = np.vstack([p[:, :2] for p in all_positions])
    all_xz = np.vstack([p[:, [0, 2]] for p in all_positions])
    pad = 3.0
    xmin, xmax = all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad
    ymin, ymax = all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad
    zmin, zmax = all_xz[:, 1].min() - 0.5, all_xz[:, 1].max() + 0.5

    # Per-atom colors and sizes
    mol_col = []
    mol_sz = []
    mol_edge = []
    for m in range(NMOL):
        for j in range(n_real_per_mol):
            en = enames_flat[m * n_real_per_mol + j]
            mol_col.append(MOL_COLORS[m])
            mol_sz.append(ELEM_SIZES.get(en, 20))
            mol_edge.append(ELEM_EDGE.get(en, '#666'))

    fig, (ax_xy, ax_xz) = plt.subplots(2, 1, figsize=(10, 10), dpi=80,
                                       gridspec_kw={'height_ratios': [1.5, 1]})
    ax_xy.set_xlim(xmin, xmax); ax_xy.set_ylim(ymin, ymax)
    ax_xy.set_aspect('equal'); ax_xy.set_facecolor('#FAFAFA')
    ax_xy.set_ylabel('y (Å)'); ax_xy.set_title('Top view (xy) — drag & bounce')
    ax_xz.set_xlim(xmin, xmax); ax_xz.set_ylim(zmin, zmax)
    ax_xz.set_aspect('auto'); ax_xz.set_facecolor('#FAFAFA')
    ax_xz.set_xlabel('x (Å)'); ax_xz.set_ylabel('z (Å)')
    ax_xz.set_title('Side view (xz) — surface bouncing')
    ax_xz.axhline(0, color='#BDBDBD', linewidth=0.5, linestyle='--')  # surface top reference

    scat_xy = ax_xy.scatter([], [], s=20, c=[], zorder=5, edgecolors='k', linewidths=0.3)
    scat_xz = ax_xz.scatter([], [], s=20, c=[], zorder=5, edgecolors='k', linewidths=0.3)
    anchor_xy, = ax_xy.plot([], [], '*', color='red', markersize=16, zorder=10)
    target_xy, = ax_xy.plot([], [], 'x', color='red', markersize=10, markeredgewidth=2, zorder=9)
    anchor_xz, = ax_xz.plot([], [], '*', color='red', markersize=16, zorder=10)
    target_xz, = ax_xz.plot([], [], 'x', color='red', markersize=10, markeredgewidth=2, zorder=9)

    cog_lines_xy, cog_lines_xz = [], []
    for m in range(NMOL):
        l1, = ax_xy.plot([], [], '--', color=MOL_COLORS[m], alpha=0.4, linewidth=1.0)
        l2, = ax_xz.plot([], [], '--', color=MOL_COLORS[m], alpha=0.4, linewidth=1.0)
        cog_lines_xy.append(l1); cog_lines_xz.append(l2)
    anchor_trail_xy, = ax_xy.plot([], [], '-', color='red', alpha=0.5, linewidth=1.5, label='anchor path')
    anchor_trail_xz, = ax_xz.plot([], [], '-', color='red', alpha=0.5, linewidth=1.5)
    target_trail_xy, = ax_xy.plot([], [], ':', color='red', alpha=0.3, linewidth=1.0)
    target_trail_xz, = ax_xz.plot([], [], ':', color='red', alpha=0.3, linewidth=1.0)

    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MOL_COLORS[0], markersize=8, label='mol 0 (dragged)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MOL_COLORS[1], markersize=8, label='mol 1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MOL_COLORS[2], markersize=8, label='mol 2'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MOL_COLORS[3], markersize=8, label='mol 3'),
        Line2D([0], [0], marker='*', color='red', markersize=12, label='anchored O atom', linestyle=''),
        Line2D([0], [0], marker='x', color='red', markersize=8, label='anchor target', linestyle=''),
        Line2D([0], [0], color='red', alpha=0.5, label='anchor path'),
        Line2D([0], [0], color='gray', alpha=0.4, linestyle='--', label='COG path'),
    ]
    ax_xy.legend(handles=legend_elems, loc='upper left', fontsize=7, framealpha=0.9)
    step_text = ax_xy.text(0.98, 0.02, '', transform=ax_xy.transAxes, ha='right', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def update(frame):
        pos = all_positions[frame]
        cogs = all_cogs[frame]
        target = all_targets[frame]
        anchor_act = all_anchor_actual[frame]
        scat_xy.set_offsets(pos[:, :2]); scat_xy.set_color(mol_col); scat_xy.set_sizes(mol_sz)
        scat_xz.set_offsets(pos[:, [0, 2]]); scat_xz.set_color(mol_col); scat_xz.set_sizes(mol_sz)
        anchor_xy.set_data([anchor_act[0]], [anchor_act[1]])
        target_xy.set_data([target[0]], [target[1]])
        anchor_xz.set_data([anchor_act[0]], [anchor_act[2]])
        target_xz.set_data([target[0]], [target[2]])
        for m in range(NMOL):
            trail_xy = np.array([all_cogs[f][m, [0, 1]] for f in range(frame + 1)])
            trail_xz = np.array([all_cogs[f][m, [0, 2]] for f in range(frame + 1)])
            cog_lines_xy[m].set_data(trail_xy[:, 0], trail_xy[:, 1])
            cog_lines_xz[m].set_data(trail_xz[:, 0], trail_xz[:, 1])
        a_trail_xy = np.array([all_anchor_actual[f][:2] for f in range(frame + 1)])
        a_trail_xz = np.array([[all_anchor_actual[f][0], all_anchor_actual[f][2]] for f in range(frame + 1)])
        t_trail_xy = np.array([all_targets[f][:2] for f in range(frame + 1)])
        t_trail_xz = np.array([[all_targets[f][0], all_targets[f][2]] for f in range(frame + 1)])
        anchor_trail_xy.set_data(a_trail_xy[:, 0], a_trail_xy[:, 1])
        anchor_trail_xz.set_data(a_trail_xz[:, 0], a_trail_xz[:, 1])
        target_trail_xy.set_data(t_trail_xy[:, 0], t_trail_xy[:, 1])
        target_trail_xz.set_data(t_trail_xz[:, 0], t_trail_xz[:, 1])
        step_text.set_text(f'step {frame * FRAME_EVERY}/{N_STEPS}')
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=60, blit=False, repeat=True)
    gif_path = os.path.join(outdir, 'drag_demo.gif')
    anim.save(gif_path, writer=PillowWriter(fps=15), dpi=80)
    plt.close(fig)
    print(f"[demo] GIF saved: {gif_path} ({os.path.getsize(gif_path) / 1e6:.1f} MB)", flush=True)


if __name__ == '__main__':
    run_demo()
