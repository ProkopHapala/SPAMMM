#!/usr/bin/env python3
"""
bench_fdbm.py — End-to-end FDBM AFM pipeline timing (diagnostic only).

Runs ModularAFMPipeline S1–S4 with SPAMMM_AFM_BENCH segmented timers.
No algorithm changes — measures where time goes (CPU vs GPU vs IO vs INIT).

Usage:
  SPAMMM_AFM_BENCH=1 SPAMMM_AFM_BENCH_NO_IO=1 SPAMMM_VERBOSITY=0 \\
    python tests/SPM/bench_fdbm.py --mol data/xyz/benzene.xyz --tip gaussian --repeats 2

  # optional cProfile dump
  ... --profile

Env:
  SPAMMM_AFM_BENCH=1         enable [BENCH] segment table
  SPAMMM_AFM_BENCH_NO_IO=1   skip stage cache np.savez
  AFM_DEBUG_PLOT_LEVEL=0     no plots
  SPAMMM_VERBOSITY=0         quiet pipeline prints
  SPAMMM_AFM_CPU_TASKS=1     CPU build_tasks (parity)
  SPAMMM_AFM_CPU_FFT=1       NumPy FFT (parity)
  SPAMMM_AFM_NA_ORBITAL_LOOP=1  legacy per-AO rho_na (slow; parity)
"""
import os
import sys
import argparse
import shutil
import tempfile

# Gate BEFORE importing spammm (globals read env at import)
os.environ.setdefault('PYOPENCL_CTX', '0')
os.environ['SPAMMM_AFM_BENCH'] = '1'
os.environ.setdefault('SPAMMM_AFM_BENCH_NO_IO', '1')
os.environ.setdefault('SPAMMM_VERBOSITY', '0')
os.environ.setdefault('AFM_DEBUG_PLOT_LEVEL', '0')
os.environ.setdefault('AFM_DEBUG_SAVE_LEVEL', '0')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _run_once(mol, out_dir, tip_mode, step, scan_range, scan_step, height_range, height_step, margin, force_recompute, label):
    from spammm.SPM.AFM import AFMBench
    from spammm.SPM.ModularPipeline import ModularAFMPipeline
    from spammm.quantum import DFTB_utils as du

    bench = AFMBench.reset()
    bench.start_run()

    basis = 'mio-1-1'
    slako = du.SK_PATHS.get(basis, basis)
    pipe = ModularAFMPipeline(
        xyz_file=mol,
        output_dir=out_dir,
        basis=basis, slako_prefix=slako,
        step=step, margin=margin, z_extra=6.0,
        scan_range=scan_range, scan_step=scan_step,
        height_range=height_range, height_step=height_step,
        tip_mode=tip_mode, backend='dftb',
    )
    print(f"\n=== {label}  mol={os.path.basename(mol)} tip={tip_mode} "
          f"step={step} scan=±{scan_range}/{scan_step} h={height_range}/{height_step} "
          f"ngrid will be set in S2; scan_n=({len(pipe.scan_xs)},{len(pipe.scan_ys)},{len(pipe.heights)}) ===")

    dm, eigvecs, eigvals = pipe.stage1_scf(force_recompute=force_recompute)
    rho_scf, rho_na, rho_diff = pipe.stage2_project(dm, force_recompute=force_recompute)
    print(f"  grid ngrid={pipe.ngrid} origin={pipe.origin} step={pipe.step}")
    V_ES, E_p, E_es, E_vdw, F_total = pipe.stage3_potentials(
        rho_scf, rho_na, rho_diff, force_recompute=force_recompute)
    df, tip_disp, FEs = pipe.stage4_relax(F_total, force_recompute=force_recompute, ppm_mode=True)

    result = bench.report(title=label)
    print(f"  df shape={df.shape}  finite={bool(__import__('numpy').isfinite(df).all())}  "
          f"|df|max={float(__import__('numpy').nanmax(__import__('numpy').abs(df))):.4g}")
    return result


def main():
    ap = argparse.ArgumentParser(description='FDBM AFM pipeline benchmark')
    ap.add_argument('--mol', default=os.path.join(ROOT, 'data', 'xyz', 'benzene.xyz'))
    ap.add_argument('--tip', choices=['gaussian', 'co'], default='gaussian', help='gaussian = interactive-speed tip; co = GUI default real tip')
    ap.add_argument('--step', type=float, default=0.15, help='density grid step (Å)')
    ap.add_argument('--scan-range', type=float, default=3.0)
    ap.add_argument('--scan-step', type=float, default=0.2, help='lateral scan step (Å); GUI often 0.1')
    ap.add_argument('--hmin', type=float, default=2.8)
    ap.add_argument('--hmax', type=float, default=3.6)
    ap.add_argument('--hstep', type=float, default=0.2)
    ap.add_argument('--margin', type=float, default=4.0)
    ap.add_argument('--repeats', type=int, default=2, help='timed runs after optional cold')
    ap.add_argument('--no-cold', action='store_true', help='skip cold (kernel-compile) run')
    ap.add_argument('--profile', action='store_true', help='also dump cProfile stats')
    ap.add_argument('--cpu-tasks', action='store_true',  help='SPAMMM_AFM_CPU_TASKS=1: CPU build_tasks (parity backup)')
    ap.add_argument('--cpu-fft', action='store_true', help='SPAMMM_AFM_CPU_FFT=1: NumPy FFT (parity backup)')
    ap.add_argument('--out', default=None, help='output/workdir (default: temp)')
    args = ap.parse_args()

    if args.cpu_tasks:
        os.environ['SPAMMM_AFM_CPU_TASKS'] = '1'
    if args.cpu_fft:
        os.environ['SPAMMM_AFM_CPU_FFT'] = '1'
    # Re-bind AFM module flags if already imported
    import spammm.SPM.AFM as afm_mod
    afm_mod.AFM_CPU_TASKS = int(os.environ.get('SPAMMM_AFM_CPU_TASKS', '0'))
    afm_mod.AFM_CPU_FFT = int(os.environ.get('SPAMMM_AFM_CPU_FFT', '0'))
    print(f"backends: tasks={'CPU' if afm_mod.AFM_CPU_TASKS else 'GPU'}  fft={'CPU' if afm_mod.AFM_CPU_FFT else 'GPU'}")

    mol = args.mol if os.path.isabs(args.mol) else os.path.join(ROOT, args.mol)
    if not os.path.isfile(mol):
        raise SystemExit(f"Molecule not found: {mol}")

    out_dir = args.out or tempfile.mkdtemp(prefix='bench_fdbm_')
    os.makedirs(out_dir, exist_ok=True)
    print(f"REVIEW: {out_dir}")
    print(f"mol={mol} tip={args.tip} out={out_dir}")

    height_range = (args.hmin, args.hmax)
    kwargs = dict(
        mol=mol, out_dir=out_dir, tip_mode=args.tip, step=args.step,
        scan_range=args.scan_range, scan_step=args.scan_step,
        height_range=height_range, height_step=args.hstep, margin=args.margin,
    )

    def _clear_cache():
        for fn in ('cache_stage1_scf.npz', 'cache_stage2_grids.npz',
                   'cache_stage3_potentials.npz', 'cache_stage4_relax.npz'):
            p = os.path.join(out_dir, fn)
            if os.path.isfile(p):
                os.remove(p)

    def run_all():
        # Cold: includes OpenCL compile / AFMulator ctor / DFTB projector
        if not args.no_cold:
            _clear_cache()
            _run_once(force_recompute=True, label='COLD (incl. OpenCL compile)', **kwargs)

        # Warm repeats: same process, force recompute but kernels already compiled
        walls = []
        for i in range(args.repeats):
            _clear_cache()
            r = _run_once(force_recompute=True, label=f'WARM recompute #{i+1}', **kwargs)
            walls.append(r['wall'])
        if walls:
            import numpy as np
            print(f"WARM wall times: {[f'{w:.3f}' for w in walls]}  "
                  f"median={float(np.median(walls)):.3f}s  min={min(walls):.3f}s")

    if args.profile:
        import cProfile, pstats
        prof_path = os.path.join(out_dir, 'bench_fdbm.cprofile')
        pr = cProfile.Profile()
        pr.enable()
        run_all()
        pr.disable()
        pr.dump_stats(prof_path)
        stats = pstats.Stats(pr).sort_stats('cumulative')
        print("\n========== [cProfile top 40 cumulative] ==========")
        stats.print_stats(40)
        print(f"REVIEW: {prof_path}")
    else:
        run_all()

    print(f"\nDone. Artifacts under: {out_dir}")


if __name__ == '__main__':
    main()
