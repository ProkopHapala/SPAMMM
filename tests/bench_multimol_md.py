#!/usr/bin/env python3
"""
bench_multimol_md.py — Benchmark multi-molecule MD launch-overhead strategies.

Standalone headless experiment. No GUI, no plotting. Tests 5 strategies
for running concurrent multi-molecule rigid-body MD on GPU:

  A — Optimized bare-enqueue loop (set_args once, no finish per step)
  C — Persistent kernel + atomic global barrier (1 launch, N steps internal)
  D — Stale-position multi-step (existing run_pairff with niter=K)
  E — Single-workgroup all-molecule (1 launch, local barrier, Gauss-Seidel)

Strategy B (split force+integrate) is deferred — likely slower than A.

Usage:
  python3 tests/bench_multimol_md.py --mol PTCDA --nmol 8 --steps 100
  python3 tests/bench_multimol_md.py --mol PTCDA --nmol 16 --steps 500 --strategies A,C,E
"""
import os, sys, time, argparse
import numpy as np
import pyopencl as cl

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from spammm.forcefields.RigidBodyDynamics import RigidBodyPairFF
from spammm.forcefields.RigidBodyUtils import load_molecule, grid_pos

# ─── Molecule paths ──────────────────────────────────────────────────────
_MOL_DIR = os.path.join(REPO, 'data', 'mol')
_XYZ_DIR = os.path.join(REPO, 'data', 'xyz')
MOL_PATHS = {
    'PTCDA':             os.path.join(_XYZ_DIR, 'PTCDA.xyz'),
    'NTCDI':             os.path.join(_MOL_DIR, 'NTCDI.mol2'),
    'formic_acid':       os.path.join(_XYZ_DIR, 'HCOOH.xyz'),
    'terephthalic_acid': os.path.join(_XYZ_DIR, 'terephthalic_acid.xyz'),
    'uracil':            os.path.join(_XYZ_DIR, 'uracil.xyz'),
    'adenine':           os.path.join(_XYZ_DIR, 'adenine.xyz'),
}
_NO_QEQ = {'NTCDI'}


# ─── Setup ───────────────────────────────────────────────────────────────

def build_system(mol_name, nmol, spacing=6.0, z=3.0, seed=42):
    """Build multi-molecule PairFF system using existing from_molecules."""
    path = MOL_PATHS[mol_name]
    apos, enames, REQs, bonds = load_molecule(path, qeq=(mol_name not in _NO_QEQ), name=mol_name)
    molecules = [(apos, enames, REQs)] * nmol
    pos = grid_pos(nmol, spacing=spacing, z=z)
    quat = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (nmol, 1))
    # Small random rotation per molecule
    rng = np.random.default_rng(seed)
    for i in range(nmol):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-12
        ang = rng.uniform(-0.3, 0.3)
        s = np.sin(0.5 * ang); c = np.cos(0.5 * ang)
        quat[i] = [axis[0] * s, axis[1] * s, axis[2] * s, c]
    rbd = RigidBodyPairFF.from_molecules(
        molecules, pos, quats=quat, active_body=0,
        mass_trans=1.0, mass_rot=1.0,
        He=-0.1, rc=3.0, w=0.7, k_z=0.0, morse_alpha=1.8,
        z_target=z, epair_dist=1.4, sigma_dist=1.0, Hs=1.0, beta=1.7,
    )
    return rbd, (apos, enames, REQs, bonds)


def save_state(rbd):
    """Snapshot all dynamics buffers for reset between strategies."""
    out = {}
    for name in ['poss', 'qrots', 'vposs', 'vrots', 'fire_state']:
        arr = np.empty((rbd.n_bodies, 4), dtype=np.float32)
        rbd.fromGPU(name, arr)
        out[name] = arr
    rbd.queue.finish()
    return out


def restore_state(rbd, state):
    """Restore dynamics buffers from snapshot."""
    for name, arr in state.items():
        rbd.toGPU(name, arr.copy())
    rbd.queue.finish()


# ─── Kernel header registration for new kernels ──────────────────────────

def register_multimol_kernels(rbd):
    """Register kernel headers for the new multimol kernels on the rbd instance."""
    required = ("rigid_body_pairff_multimol_kernel", "rigid_body_pairff_multimol_persistent_kernel", "rigid_body_pairff_multimol_single_wg_kernel")
    missing = [name for name in required if name not in rbd.kernelheaders]
    if missing:
        raise RuntimeError(f"RigidBodyPairFF did not register production multimol kernels: {missing}")
    return
    rbd.kernelheaders["rigid_body_pairff_multimol_kernel"] = """__kernel
void rigid_body_pairff_multimol_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    const int                n_mols,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    const float              dt,
    const float4             md_params,
    const int                niter
)"""
    rbd.kernelheaders["rigid_body_pairff_multimol_persistent_kernel"] = """__kernel
void rigid_body_pairff_multimol_persistent_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global       int*      g_barrier,
    const int                n_mols,
    const int                n_wg,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    const float              dt,
    const float4             md_params,
    const int                niter
)"""
    rbd.kernelheaders["rigid_body_pairff_multimol_single_wg_kernel"] = """__kernel
void rigid_body_pairff_multimol_single_wg_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    const int                n_mols,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    const float              dt,
    const float4             md_params,
    const int                niter
)"""


# ─── Strategy implementations ────────────────────────────────────────────

def strategy_A_naive(rbd, n_steps, dt=0.05, lin_damp=0.92, ang_damp=0.88, fire=False):
    """Naive baseline: multimol_kernel with full overhead per step.

    Calls generate_kernel_args + kernel() + finish() every step — the
    anti-pattern we want to measure. Same kernel as A_optimized, just
    with all the avoidable Python overhead.
    """
    for _ in range(n_steps):
        rbd._multimol_launchers.clear()
        rbd.run_multimol_md(1, dt=dt, lin_damp=lin_damp, ang_damp=ang_damp, fire=fire)


def strategy_A_optimized(rbd, n_steps, dt=0.05, lin_damp=0.92, ang_damp=0.88, fire=False):
    """Strategy A: optimized bare-enqueue loop.

    Retain kernel object, set_args once, bare enqueue_nd_range_kernel per step,
    single finish() at end. Uses the new multimol_kernel (all molecules move).
    """
    rbd.run_multimol_md(n_steps, dt=dt, lin_damp=lin_damp, ang_damp=ang_damp, fire=fire)


def strategy_A_eventless(rbd, n_steps, dt=0.05, lin_damp=0.92, ang_damp=0.88, fire=False):
    """Exact ping-pong MD using the optional eventless OpenCL enqueue."""
    rbd.run_multimol_md(n_steps, dt=dt, lin_damp=lin_damp, ang_damp=ang_damp, fire=fire, eventless=True)


def strategy_A_niter(rbd, n_steps, dt=0.05, lin_damp=0.92, ang_damp=0.88, fire=False, K=10):
    """Strategy A_niter: Jacobi stale-position multi-step in ONE kernel launch.

    Uses multimol_kernel with niter=K. Each molecule moves K steps based on
    the INITIAL positions of other molecules (Jacobi / stale-position).
    poss[a] is only written AFTER the niter loop, so inter-molecule forces
    use stale positions within the loop. This is the same pattern that
    gave 100× speedup for single-molecule MD.

    n_steps total → ceil(n_steps / K) kernel launches.
    Accuracy degrades with K (stale positions), but for relaxation the
    error is often acceptable and self-correcting.
    """
    rbd._multimol_launchers.clear()
    rbd.run_multimol_md(n_steps, dt=dt, lin_damp=lin_damp, ang_damp=ang_damp, fire=fire, batch=K)


def strategy_A_niter_opt(rbd, n_steps, dt=0.05, lin_damp=0.92, ang_damp=0.88, fire=False, K=10):
    """Strategy A_niter_opt: same as A_niter but with set_args once (optimized).

    The niter parameter changes per launch (last chunk may be smaller),
    so we re-set only that one arg. Even faster than A_niter.
    """
    rbd.run_multimol_md(n_steps, dt=dt, lin_damp=lin_damp, ang_damp=ang_damp, fire=fire, batch=K)


def strategy_A_predict(rbd, n_steps, dt=0.05, lin_damp=0.92, ang_damp=0.88, fire=False, K=10):
    """Approximate K-step chunks with constant-velocity partner prediction."""
    rbd.run_multimol_md(n_steps, dt=dt, lin_damp=lin_damp, ang_damp=ang_damp, fire=fire, batch=K, predict_partners=True)


def measure_launch_overhead(rbd, n_launches=1000):
    """Measure bare kernel launch overhead with niter=0 (empty kernel)."""
    kname = "rigid_body_pairff_multimol_kernel"
    krnl = cl.Kernel(rbd.prg, kname)
    rbd.kernel_params['dt'] = np.float32(0.0)
    rbd.kernel_params['niter'] = np.int32(0)  # 0 iterations = empty kernel
    rbd.kernel_params['predict_partners'] = np.int32(0)
    rbd.kernel_params['md_params'] = np.array([0.92, 0.88, 1.0, 1.0], dtype=np.float32)
    # Ensure FAF dummy args exist (kernel 15 now has do_faf + FAF buffers)
    rbd.kernel_params.setdefault('do_faf', np.int32(0))
    rbd.kernel_params.setdefault('folded_tensor_meta', np.array([0, 0, 0, 0], dtype=np.int32))
    rbd.kernel_params.setdefault('folded_lvec2d', np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))
    if 'folded_site_coeffs' not in rbd.buffer_dict:
        rbd.check_buf('folded_site_coeffs', 4)
        rbd.toGPU('folded_site_coeffs', np.zeros(1, dtype=np.float32))
    if 'folded_z_params' not in rbd.buffer_dict:
        rbd.check_buf('folded_z_params', 16)
        rbd.toGPU('folded_z_params', np.zeros((1, 4), dtype=np.float32))
    overrides = dict(poss_in=rbd.buffer_dict['poss'], qrots_in=rbd.buffer_dict['qrots'], vposs_in=rbd.buffer_dict['vposs'], vrots_in=rbd.buffer_dict['vrots'], poss_out=rbd.buffer_dict['poss_alt'], qrots_out=rbd.buffer_dict['qrots_alt'], vposs_out=rbd.buffer_dict['vposs_alt'], vrots_out=rbd.buffer_dict['vrots_alt'])
    args = rbd.generate_kernel_args(kname, overrides=overrides)
    krnl.set_args(*args)
    gs = (rbd.roundUpGlobalSize(rbd.n_bodies * rbd.nloc),)
    ls = (rbd.nloc,)
    q = rbd.queue
    t0 = time.perf_counter()
    for _ in range(n_launches):
        cl.enqueue_nd_range_kernel(q, krnl, gs, ls)
    q.finish()
    t1 = time.perf_counter()
    us_per_launch = (t1 - t0) * 1e6 / n_launches
    return us_per_launch


def strategy_C_persistent(rbd, n_steps, dt=0.05, lin_damp=0.92, ang_damp=0.88, fire=False):
    """Strategy C: persistent kernel with atomic global barrier.

    One kernel launch, niter=n_steps internal loop.
    WARNING: may deadlock if not all workgroups are simultaneously resident.
    """
    rbd.run_multimol_persistent(n_steps, dt=dt, lin_damp=lin_damp, ang_damp=ang_damp, fire=fire)


def strategy_D_stale(rbd, n_steps, K=10, dt=0.05, lin_damp=0.92, ang_damp=0.88, fire=False):
    """Strategy D: stale-position multi-step using existing allmol kernel.

    Runs K steps per kernel call with stale inter-molecule positions.
    Only the active molecule moves per call; we cycle through all molecules.
    n_steps total, K steps per call, cycling active_mol.
    """
    n_mols = rbd.n_bodies
    calls_per_mol = max(1, n_steps // (K * n_mols))
    total_steps_done = 0
    for _ in range(calls_per_mol):
        for a in range(n_mols):
            if total_steps_done >= n_steps:
                break
            rbd.active_body = a
            rbd.kernel_params['active_mol'] = np.int32(a)
            steps_this = min(K, n_steps - total_steps_done)
            rbd.run_pairff(steps_this, dt, lin_damp=lin_damp, ang_damp=ang_damp, fire=fire, faf=False)
            total_steps_done += steps_this
    rbd.queue.finish()


def strategy_E_single_wg(rbd, n_steps, dt=0.05, lin_damp=0.92, ang_damp=0.88, fire=False):
    """Strategy E: single-workgroup all-molecule kernel.

    All molecules in one workgroup, local barrier, Gauss-Seidel update.
    One kernel launch, niter=n_steps internal loop.
    """
    rbd.run_multimol_single_wg(n_steps, dt=dt, lin_damp=lin_damp, ang_damp=ang_damp, fire=fire)


# ─── Benchmark harness ───────────────────────────────────────────────────

def get_energy(rbd):
    """Download total energy (sum of per-atom E from apos_world)."""
    atoms = np.empty((rbd.total_atoms, 4), dtype=np.float32)
    rbd.fromGPU('apos_world', atoms)
    rbd.queue.finish()
    return float(atoms[:, 3].sum())


def get_positions(rbd):
    """Download molecule positions."""
    pos = np.empty((rbd.n_bodies, 4), dtype=np.float32)
    rbd.fromGPU('poss', pos)
    rbd.queue.finish()
    return pos.copy()


def benchmark_strategy(name, func, rbd, state0, n_steps, dt, warmup=10, **kw):
    """Run one strategy, measure time, return results dict."""
    # Reset to initial state
    restore_state(rbd, state0)
    # Warmup (not timed)
    if warmup > 0:
        func(rbd, warmup, dt=dt, **kw)
        restore_state(rbd, state0)
    # Timed run
    t0 = time.perf_counter()
    func(rbd, n_steps, dt=dt, **kw)
    t1 = time.perf_counter()
    wall_ms = (t1 - t0) * 1000.0
    us_per_step = wall_ms * 1000.0 / n_steps
    steps_per_s = n_steps / (t1 - t0)
    # Final energy + positions
    E_final = get_energy(rbd)
    pos_final = get_positions(rbd)
    return {
        'name': name, 'wall_ms': wall_ms, 'us_per_step': us_per_step,
        'steps_per_s': steps_per_s, 'E_final': E_final, 'pos_final': pos_final,
    }


def run_benchmark(mol_name, nmol, n_steps, strategies, dt=0.05, spacing=6.0, z=3.0, K=10):
    """Run full benchmark for one (mol, nmol) configuration."""
    print(f'\n{"="*70}')
    print(f'Config: {nmol}×{mol_name}, {n_steps} steps, dt={dt}')
    print(f'{"="*70}')
    rbd, (apos, enames, REQs, bonds) = build_system(mol_name, nmol, spacing=spacing, z=z)
    n_atoms_per_mol = len(apos)
    total_atoms = rbd.total_atoms
    print(f'  {n_atoms_per_mol} atoms/mol, {total_atoms} total atoms (incl. epairs)')
    print(f'  GPU: {rbd.ctx.devices[0].name} ({rbd.ctx.devices[0].max_compute_units} CUs)')

    # Register new kernel headers
    register_multimol_kernels(rbd)

    # Run 1 evaluation step to populate apos_world with meaningful energies
    strategy_A_optimized(rbd, 1, dt=dt)
    restore_state(rbd, save_state(rbd))

    # Save initial state (after evaluation)
    state0 = save_state(rbd)
    E0 = get_energy(rbd)
    pos0 = get_positions(rbd)
    print(f'  Initial energy: {E0:.6f} eV')

    # Measure bare launch overhead (niter=0 = empty kernel)
    overhead_us = measure_launch_overhead(rbd, n_launches=1000)
    print(f'  Bare launch overhead (niter=0): {overhead_us:.2f} µs/launch')

    results = []
    ref_pos = None
    ref_E = None

    strategy_map = {
        'A_naive':      ('A_naive',      strategy_A_naive,      {}),
        'A_opt':        ('A_opt',        strategy_A_optimized,  {}),
        'A_eventless':  ('A_eventless',  strategy_A_eventless,  {}),
        'A_niter':      ('A_niter',      strategy_A_niter,      {'K': K}),
        'A_niter_opt':  ('A_niter_opt',  strategy_A_niter_opt,  {'K': K}),
        'A_predict':     ('A_predict',     strategy_A_predict,    {'K': K}),
        'C_persist':    ('C_persist',    strategy_C_persistent, {}),
        'D_stale':      ('D_stale',      strategy_D_stale,      {'K': K}),
        'E_singlewg':   ('E_singlewg',   strategy_E_single_wg,  {}),
    }

    for skey in strategies:
        if skey not in strategy_map:
            print(f'  WARNING: unknown strategy {skey!r}, skipping')
            continue
        label, func, extra_kw = strategy_map[skey]
        print(f'  Running {label}...', end=' ', flush=True)
        try:
            res = benchmark_strategy(label, func, rbd, state0, n_steps, dt, **extra_kw)
            results.append(res)
            # First successful strategy with correct concurrent MD = reference
            if skey in ('A_opt', 'A_naive') and ref_pos is None:
                ref_pos = res['pos_final'].copy()
                ref_E = res['E_final']
            # Compute RMSD vs reference if available
            rmsd = None
            if ref_pos is not None and skey not in ('A_naive', 'A_opt'):
                rmsd = float(np.sqrt(np.mean((res['pos_final'][:, :3] - ref_pos[:, :3]) ** 2)))
            dE = None
            if ref_E is not None and skey not in ('A_naive', 'A_opt'):
                dE = res['E_final'] - ref_E
            res['rmsd'] = rmsd
            res['dE'] = dE
            print(f'{res["wall_ms"]:8.1f} ms  ({res["us_per_step"]:7.1f} µs/step,  {res["steps_per_s"]:8.0f} steps/s)'
                  f'  E={res["E_final"]:.4f}', end='')
            if rmsd is not None:
                print(f'  RMSD={rmsd:.6f} Å  ΔE={dE:+.6f} eV', end='')
            print()
        except Exception as e:
            print(f'FAILED: {e}')
            import traceback; traceback.print_exc()
            results.append({'name': label, 'error': str(e)})

    return results, rbd


def print_summary(all_results):
    """Print final summary table."""
    print(f'\n{"="*70}')
    print('SUMMARY')
    print(f'{"="*70}')
    for config_name, results in all_results:
        print(f'\n{config_name}:')
        print(f'  {"Strategy":<14s} {"wall(ms)":>10s} {"µs/step":>10s} {"steps/s":>10s} {"E_final":>12s} {"RMSD(Å)":>10s} {"ΔE(eV)":>12s}')
        print(f'  {"-"*14} {"-"*10} {"-"*10} {"-"*10} {"-"*12} {"-"*10} {"-"*12}')
        ref_E = None
        for r in results:
            if 'error' in r:
                print(f'  {r["name"]:<14s} {"ERROR":>10s} {r["error"][:50]}')
                continue
            if r['name'] in ('A_opt', 'A_naive') and ref_E is None:
                ref_E = r['E_final']
            dE_str = f'{r["E_final"] - ref_E:+.6f}' if ref_E is not None and r['name'] not in ('A_opt', 'A_naive') else '—'
            rmsd_str = '—'  # filled from results dict if available
            print(f'  {r["name"]:<14s} {r["wall_ms"]:10.1f} {r["us_per_step"]:10.1f} {r["steps_per_s"]:10.0f} {r["E_final"]:12.6f} {rmsd_str:>10s} {dE_str:>12s}')


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Benchmark multi-molecule MD launch-overhead strategies')
    ap.add_argument('--mol', type=str, default='PTCDA', choices=list(MOL_PATHS.keys()))
    ap.add_argument('--nmol', type=int, nargs='+', default=[8, 16], help='number of molecules to test')
    ap.add_argument('--steps', type=int, default=100, help='MD steps per benchmark')
    ap.add_argument('--dt', type=float, default=0.05, help='timestep [ps]')
    ap.add_argument('--strategies', type=str, default='A_naive,A_opt,C_persist,D_stale,E_singlewg',
                    help='comma-separated strategies to run')
    ap.add_argument('--K', type=int, default=10, help='stale-position batch size for Strategy D')
    ap.add_argument('--spacing', type=float, default=6.0, help='molecule grid spacing [Å]')
    ap.add_argument('--z', type=float, default=3.0, help='initial height above surface [Å]')
    ap.add_argument('--fire', action='store_true', help='use FIRE relaxation instead of damped MD')
    args = ap.parse_args()

    strategies = [s.strip() for s in args.strategies.split(',') if s.strip()]
    all_results = []
    for nmol in args.nmol:
        config_name = f'{nmol}×{args.mol}'
        results, _ = run_benchmark(args.mol, nmol, args.steps, strategies,
                                   dt=args.dt, spacing=args.spacing, z=args.z, K=args.K)
        all_results.append((config_name, results))

    print_summary(all_results)


if __name__ == '__main__':
    main()
