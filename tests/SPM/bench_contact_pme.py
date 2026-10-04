#!/usr/bin/env python3
"""
bench_contact_pme.py — Phase-0 baseline for contact_pme optimization
(doc/Tasks/PME_ContactSurface_Opt.md).

Measures, on the NVIDIA GPU:
  --fit   fit_contact_pme wall time (mesh raster + prefilter + core fit)
  --eval  evaluator microbenchmark: ns/eval for evalContactPME (bucket)
          vs evalContactPMELocal (all-atom local), kernel-only loop
  --scan  golden scan: run_scan_contact_pme timing + golden FEs/iters saved
          to debug/bench_contact_pme/ for later A-track parity

Usage:
  python tests/SPM/bench_contact_pme.py                          # all stages, PTCDA
  python tests/SPM/bench_contact_pme.py --replica 4 4            # 4x4 PTCDA grid (~600 atoms)
  python tests/SPM/bench_contact_pme.py --eval --nq 1000000
"""
import os
import sys
import argparse
import time
import numpy as np

os.environ.setdefault('PYOPENCL_CTX', '0')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, 'debug', 'bench_contact_pme')
DATA = os.path.join(ROOT, 'data')
PARAMS_PATH = os.path.join(DATA, 'ElementTypes.dat')


def make_replica_xyz(mol_path, nx, ny, out_path, margin=8.0):
    """Tile molecule on a regular nx*ny grid; returns path to written xyz."""
    from spammm.surfaces.ContactSurface import load_atom_data, write_assembly_xyz
    apos, reqs, enames, _, qs = load_atom_data(mol_path)
    apos = apos - apos.mean(axis=0)
    span = apos[:, :2].max(axis=0) - apos[:, :2].min(axis=0)
    step = span.max() + 4.0
    all_pos, all_en, all_q = [], [], []
    for iy in range(ny):
        for ix in range(nx):
            all_pos.append(apos + np.array([ix * step, iy * step, 0.0]))
            all_en.extend(enames)
            all_q.extend(qs)
    all_pos = np.vstack(all_pos)
    all_pos[:, :2] += margin
    lvec = np.array([[nx * step + 2 * margin, 0, 0], [0, ny * step + 2 * margin, 0], [0, 0, 20.0]])
    write_assembly_xyz(out_path, all_pos, all_en, np.asarray(all_q, dtype=np.float64), lvec)
    return out_path


def make_afm(xyz_path):
    from spammm.SPM.AFM import AFMulator
    afm = AFMulator(use_morse=True, use_fire=False)
    afm.load_molecule(xyz_path)
    afm.assign_params(params_path=PARAMS_PATH)
    afm.tipQs[:] = 0.0          # contact_pme MVP: radial oracle only (test l.1722)
    return afm


def bench_fit(afm, h_mesh=1.0):
    t0 = time.perf_counter()
    params = afm.fit_contact_pme(h_mesh=h_mesh, bPrint=True)
    t1 = time.perf_counter()
    print(f"[FIT] fit_contact_pme total: {t1 - t0:.3f} s  na={params.na} mesh={params.mesh_shape}", flush=True)
    return params


def bench_eval(afm, nq=1_000_000, wg=32, iters=5, seed=1):
    """Kernel-only ns/eval for bucket vs local evaluator on interior queries."""
    import pyopencl as cl
    params = afm.cpm
    lo, hi = params.query_interior
    origin, h = params.mesh_origin, params.mesh_h
    box_lo = origin + (np.asarray(lo) + 0.5) * h
    box_hi = origin + (np.asarray(hi) - 0.5) * h
    rng = np.random.default_rng(seed)
    q = rng.uniform(box_lo, box_hi, size=(nq, 3)).astype(np.float64)
    E, F = afm.eval_contact_pme(q, core_backend='auto')          # buffers now resident
    print(f"[EVAL] nq={nq} sanity: E[{E.min():.3f},{E.max():.3f}] |F|max={np.abs(F).max():.3f}", flush=True)

    na = params.na
    nx, ny, nz = params.mesh_shape
    nbuckets = int(params.bucket_nbx) * int(params.bucket_nby)
    mesh_meta = np.array([nx, ny, nz, 0], dtype=np.int32)
    mesh_origin_h = np.array([origin[0], origin[1], origin[2], h], dtype=np.float32)
    core_meta = np.array([na, params.bucket_nbx, params.bucket_nby, nbuckets], dtype=np.int32)
    x0, y0, _, _ = params.bucket_bounds
    core_bucket_meta = np.array([x0, y0, params.bucket_cell_size, params.core_d_span], dtype=np.float32)
    nG = afm._roundup(nq, wg)

    lmem = int(afm.ctx.devices[0].get_info(cl.device_info.LOCAL_MEM_SIZE))
    for backend in ('bucket', 'local'):
        if backend == 'local':
            if na * 36 > lmem:
                print(f"[EVAL] local : SKIP na={na} needs {na * 36} B > {lmem} B local mem", flush=True)
                continue
            LATOMS = cl.LocalMemory(na * 16)
            LCOEFFS = cl.LocalMemory(na * 5 * 4)
            def launch():
                afm.prg.evalContactPMELocal(afm.queue, (nG,), (wg,),
                    afm.cpm_queries_cl, afm.cpm_out_fe_cl, afm.cpm_status_cl,
                    afm.cpm_min_r_cl, afm.cpm_offender_cl, afm.cpm_overflow_cl,
                    afm.cpm_mesh_cl, mesh_meta, mesh_origin_h,
                    afm.cpm_atoms_cl, afm.cpm_core_coeffs_cl,
                    core_meta, core_bucket_meta, np.int32(nq), LATOMS, LCOEFFS)
        else:
            def launch():
                afm.prg.evalContactPME(afm.queue, (nG,), (wg,),
                    afm.cpm_queries_cl, afm.cpm_out_fe_cl, afm.cpm_status_cl,
                    afm.cpm_min_r_cl, afm.cpm_offender_cl, afm.cpm_overflow_cl,
                    afm.cpm_mesh_cl, mesh_meta, mesh_origin_h,
                    afm.cpm_atoms_cl, afm.cpm_core_coeffs_cl,
                    afm.cpm_buckets_cl, afm.cpm_offsets_cl,
                    core_meta, core_bucket_meta, np.int32(nq))
        launch(); afm.queue.finish()                             # warm-up
        t0 = time.perf_counter()
        for _ in range(iters):
            launch()
        afm.queue.finish()
        dt = (time.perf_counter() - t0) / iters
        print(f"[EVAL] {backend:6s}: {dt * 1e3:9.3f} ms  = {dt * 1e9 / nq:8.3f} ns/eval", flush=True)


def bench_scan(afm, nxy=(48, 48), nz=60, dtip=-0.1, modes=('fire', 'sph'), tag='',
               backend='auto'):
    for mode in modes:
        t0 = time.perf_counter()
        FEs, pts = afm.run_scan_contact_pme(nxy=nxy, nz=nz, dtip=dtip, relax_mode=mode,
                                          core_backend=backend)
        afm.queue.finish()
        dt = time.perf_counter() - t0
        nev = nxy[0] * nxy[1] * nz
        iters = getattr(afm, 'last_relax_iters', None)
        msg = f"[SCAN] {mode:4s} nxy={nxy} nz={nz}: {dt:.3f} s = {dt * 1e9 / nev:.1f} ns/slice-px"
        if iters is not None:
            msg += f"  iters mean={float(iters[iters > 0].mean() if (iters > 0).any() else -1):.1f} max={int(iters.max())}"
        print(msg, flush=True)
        np.savez_compressed(os.path.join(OUT, f'golden_{mode}{tag}.npz'),
                            FEs=FEs, pts=pts, iters=iters if iters is not None else np.array([]),
                            nxy=np.array(nxy), nz=np.int32(nz), dtip=np.float64(dtip),
                            backend=np.array(backend))
        print(f"  saved {OUT}/golden_{mode}{tag}.npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mol', default=os.path.join(DATA, 'xyz', 'PTCDA.xyz'))
    ap.add_argument('--replica', type=int, nargs=2, metavar=('NX', 'NY'), default=None)
    ap.add_argument('--fit', action='store_true')
    ap.add_argument('--eval', action='store_true')
    ap.add_argument('--scan', action='store_true')
    ap.add_argument('--nq', type=int, default=1_000_000)
    ap.add_argument('--nxy', type=int, nargs=2, default=(48, 48))
    ap.add_argument('--nz', type=int, default=60)
    ap.add_argument('--h-mesh', type=float, default=1.0)
    args = ap.parse_args()
    if not (args.fit or args.eval or args.scan):
        args.fit = args.eval = args.scan = True

    os.makedirs(OUT, exist_ok=True)
    mol = args.mol
    tag = ''
    if args.replica is not None:
        nx, ny = args.replica
        mol = os.path.join(OUT, f'replica_{nx}x{ny}.xyz')
        if not os.path.exists(mol):
            make_replica_xyz(args.mol, nx, ny, mol)
        tag = f'_r{nx}x{ny}'
    print(f"[BENCH] mol={mol}", flush=True)

    afm = make_afm(mol)
    print(f"[BENCH] device: {afm.ctx.devices[0].name}", flush=True)
    if args.fit:
        bench_fit(afm, h_mesh=args.h_mesh)
    elif afm.cpm is None:
        afm.fit_contact_pme(h_mesh=args.h_mesh, bPrint=False)
    if args.eval:
        bench_eval(afm, nq=args.nq)
    if args.scan:
        bench_scan(afm, nxy=tuple(args.nxy), nz=args.nz, tag=tag)


if __name__ == '__main__':
    main()
