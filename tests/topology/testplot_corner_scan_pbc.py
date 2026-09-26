#!/usr/bin/env python3
"""Periodic four-corner square scan for 1D H-bond chain cells (DFTB+ PBC).

Cell = build_pbc_cell(name) from PBC_CHAIN_ARTS — one junction internal, one
crossing the cell boundary (HbondRecord.a_shift/d_shift).  Corners relaxed with
`run_pbc` (pinned scan-H + bond partner), edges+diagonals SP on interpolated
frames; J = E_RR+E_LL-E_RL-E_LR.

Outputs: debug/test_corner_scan_pbc/corner_<name>.{png,xyz} + _overlay.png
Run: python tests/topology/testplot_corner_scan_pbc.py --name quinolone4
     python tests/topology/testplot_corner_scan_pbc.py --name all
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from spammm.topology.ascii_art_heterocycle import build_pbc_cell, PBC_CHAIN_ARTS
from spammm.quantum.coordinate_scan import (run_corner_scan_pbc, write_corner_scan_jobs_pbc, plot_corner_scan, plot_corner_overlay, plot_corner_diagram, write_corner_scan_xyz, load_scan, save_scan,
                                            CORNER_US, CORNER_NAMES, SQUARE_PATHS, _axis_grid, interpolate_all_atoms, junction_bond_lengths, HAU2EV)

DEBUG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'test_corner_scan_pbc')


def recover_scan(name, work_dir, atoms, lvs, hbonds, dx=0.25, sk_set=None, nk=(1, 8, 1), filling_temp=300.0):
    """Rebuild the scan dict (and scan.pkl) from an existing run's DFTB output
    dirs — parses energies from OUT files, corner geometries from geom.out.gen.
    For rescuing runs done before save_scan existed, or after manual edits.
    Rigid path SPs missing from the bundle (compact bakes don't include them)
    are computed locally — cheap single-points — into the same path_* dirs."""
    from spammm.quantum.DFTB_utils import parse_energy_out, read_relaxed_geometry, run_pbc

    def _job_done(d):
        out = os.path.join(d, 'OUT')
        if not os.path.exists(out):
            return False
        with open(out) as f:
            txt = f.read()
        if 'Driver' in open(os.path.join(d, 'dftb_in.hsd')).read():
            return 'Geometry converged' in txt     # geo-opt: partial runs still print Total Energy
        return 'Total Energy' in txt               # single point

    def _energy(d):
        if not _job_done(d):
            return np.nan
        try:
            return parse_energy_out(os.path.join(d, 'OUT')) * HAU2EV
        except Exception:
            return np.nan

    corners = {}
    scaffold = atoms.apos
    for u in CORNER_US:
        cname = CORNER_NAMES[u]
        d = os.path.join(work_dir, f'corner_{cname}')
        cwd = os.getcwd()
        os.chdir(d)
        apos_r = read_relaxed_geometry(atoms.apos, do_relax=True)
        os.chdir(cwd)
        corners[u] = dict(name=cname, apos=apos_r, e_ev=_energy(d), bonds=junction_bond_lengths(apos_r, hbonds, lvs))
        if u == CORNER_US[0]:
            scaffold = apos_r
        bs = '  '.join(f"DH={d1:.3f} HA={d2:.3f}" for d1, d2 in corners[u]['bonds'])
        print(f"    {cname} junctions: {bs}  E={corners[u]['e_ev']:.4f} eV")

    fr = _axis_grid(0.0, 1.0, dx)
    paths = []
    cwd = os.getcwd()
    for ip, (uA, uB, lab) in enumerate(SQUARE_PATHS):
        stack = interpolate_all_atoms(corners[uA]['apos'], corners[uB]['apos'], fr)
        energies_ev = np.full(len(fr), np.nan)
        energies_ev[0], energies_ev[-1] = corners[uA]['e_ev'], corners[uB]['e_ev']   # SP at relaxed corner = corner energy
        n_local = 0
        for i in range(1, len(fr) - 1):
            d = os.path.join(work_dir, f'path_{ip:02d}_{i:03d}')
            e = _energy(d)
            if not np.isfinite(e):                 # not baked/ran -> cheap local SP (self-heals into path_* dir)
                E_h, _, _ = run_pbc(stack[i], atoms.enames, lvs, sk_set=sk_set, do_relax=False, nk=nk, workdir=d, Temperature=filling_temp)
                e = E_h * HAU2EV; n_local += 1
            energies_ev[i] = e
        if n_local:
            print(f"    {lab}: {n_local} rigid path SPs computed locally")
        pth = dict(uA=uA, uB=uB, label=lab, fracs=fr, energies_ev=energies_ev, apos_frames=stack)
        e_rel = np.full(len(fr), np.nan); e_rel[0], e_rel[-1] = corners[uA]['e_ev'], corners[uB]['e_ev']
        apos_rel = [corners[uA]['apos']] + [None] * (len(fr) - 2) + [corners[uB]['apos']]
        have_r = False
        for i in range(1, len(fr) - 1):
            d = os.path.join(work_dir, f'pathr_{ip:02d}_{i:03d}')
            if os.path.exists(os.path.join(d, 'OUT')):
                e = _energy(d)
                if np.isfinite(e):
                    os.chdir(d)
                    apos_rel[i] = read_relaxed_geometry(stack[i], do_relax=True)
                    os.chdir(cwd)
                    e_rel[i] = e
                    have_r = True
        if have_r:
            pth['energies_ev_relaxed'] = e_rel
            pth['apos_frames_relaxed'] = apos_rel
            print(f"    {lab}: recovered {np.isfinite(e_rel[1:-1]).sum()}/{len(fr)-2} relaxed path points")
        paths.append(pth)

    E00, E10, E01, E11 = (corners[u]['e_ev'] for u in CORNER_US)
    J = E11 + E00 - E10 - E01 if np.isfinite([E00, E10, E01, E11]).all() else np.nan
    meta = dict(name=name, scan_type='corner_square_pbc', dx=dx, lvs=[list(v) for v in lvs], hbond_records=[h.to_dict() for h in hbonds], J_ev=J, recovered=True, relax_paths=any('energies_ev_relaxed' in p for p in paths))
    scan = dict(corners=corners, corner_us=CORNER_US, paths=paths, J_ev=J, hbonds=hbonds, enames=atoms.enames, apos_ref=atoms.apos, lvs=lvs, meta=meta)
    save_scan(scan, os.path.join(work_dir, 'scan.pkl'))
    print(f"  recovered scan -> {work_dir}/scan.pkl   J = {J:+.4f} eV")
    return scan


def run_one(name, dx=0.25, relax_corners=True, relax_paths=False, sk_set=None, work_root=None, nk=(1, 8, 1), filling_temp=300.0, rescue_temp=600.0, hbond_length=2.8, tilt=None, zigzag=None, slant=None, jkink=None, plot_only=False, recover=False, prepare=None, rigid_jobs=False):
    atoms, lvs, hbonds = build_pbc_cell(name, hbond_length=hbond_length, tilt=tilt, zigzag=zigzag, slant=slant, jkink=jkink)
    if len(hbonds) < 2:
        print(f"  SKIP {name}: only {len(hbonds)} junctions (need 2 for corner square)")
        return None
    work_dir = os.path.join(work_root or DEBUG_DIR, name)
    if prepare is not None:
        work_dir = os.path.join(prepare, name)
        print(f"\n=== {name}: bake cluster job tree -> {work_dir} ===")
        write_corner_scan_jobs_pbc(atoms.enames, atoms.apos, lvs, hbonds, [0, 1], dx=dx, sk_set=sk_set, work_dir=work_dir, nk=nk, filling_temp=filling_temp, relax_paths=relax_paths, rigid_jobs=rigid_jobs)
        return None
    if recover:
        print(f"\n=== {name}: recover scan from {work_dir} ===")
        scan = recover_scan(name, work_dir, atoms, lvs, hbonds, dx=dx, sk_set=sk_set, nk=nk, filling_temp=filling_temp)
    elif plot_only:
        scan = load_scan(os.path.join(work_dir, 'scan.pkl'))
        print(f"\n=== {name}: replot from {work_dir}/scan.pkl ===")
    else:
        mapping = [0, 1]
        print(f"\n=== {name}: PBC corner square ({len(hbonds)} junctions, Ly={lvs[1, 1]:.2f} A, nk={nk}, dx={dx}) ===")
        scan = run_corner_scan_pbc(atoms.enames, atoms.apos, lvs, hbonds, mapping, dx=dx, relax_corners=relax_corners, relax_paths=relax_paths, sk_set=sk_set, work_dir=work_dir, nk=nk, filling_temp=filling_temp, rescue_temp=rescue_temp, verbose=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    for u in scan['corner_us']:
        c0 = scan['corners'][u]
        print(f"  {c0['name']} u={u}:  E = {c0['e_ev']:.4f} eV")
    print(f"  J = E_RR+E_LL-E_RL-E_LR = {scan['J_ev']:+.4f} eV  ({scan['J_ev']*1000:+.1f} meV)")
    png = os.path.join(DEBUG_DIR, f'corner_{name}.png')
    png_ov = os.path.join(DEBUG_DIR, f'corner_{name}_overlay.png')
    xyz = os.path.join(DEBUG_DIR, f'corner_{name}.xyz')
    plot_corner_scan(scan, atoms, f'{name} PBC corner square', png)
    plot_corner_overlay(scan, atoms, png_ov)
    svg = os.path.join(DEBUG_DIR, f'diagram_{name}.svg')
    plot_corner_diagram(scan, atoms, svg)
    plot_corner_diagram(scan, atoms, svg.replace('.svg', '.png'))
    write_corner_scan_xyz(scan, atoms, xyz)
    print(f"REVIEW: {png}")
    print(f"REVIEW: {png_ov}")
    print(f"REVIEW: {svg}")
    print(f"REVIEW: {xyz}")
    return scan


def main():
    parser = argparse.ArgumentParser(description='PBC 4-corner square scan for H-bond chain cells')
    parser.add_argument('--name', default='quinolone4', help=f'PBC_CHAIN_ARTS key (default: quinolone4; "all" = every entry; choices: {sorted(PBC_CHAIN_ARTS)})')
    parser.add_argument('--dx', type=float, default=0.25, help='Path fraction step (default: 0.25)')
    parser.add_argument('--sk_set', default=None, help='DFTB SK set (default: from config)')
    parser.add_argument('--nk', type=int, default=8, help='k-points along chain (default: 8)')
    parser.add_argument('--hbond', type=float, default=2.8, help='D..A junction gap [A] (default: 2.8)')
    parser.add_argument('--tilt', type=float, default=None, help='herringbone tilt [deg], alternate molecules +/-tilt (default: per-system, 45 except hq2q=0)')
    parser.add_argument('--zigzag', type=float, default=None, help='in-plane zigzag angle [deg] (default: per-system, hq2q=60)')
    parser.add_argument('--slant', type=float, default=None, help='oblique junction lean off y-axis [deg], alternating per junction (default: per-system, 0)')
    parser.add_argument('--jkink', type=float, default=None, help='outward donor-OH splay at junction apexes [deg] (default: per-system, 0)')
    parser.add_argument('--no-relax', action='store_true', help='Skip corner relax (rigid corners, SP only)')
    parser.add_argument('--relax-paths', action='store_true', help='Also relax interior path frames (scan-H pinned at interpolated position) -> relaxed reaction-path curves')
    parser.add_argument('--filling-temp', type=float, default=300.0, help='Fermi smearing T [K] for all PBC runs (default: 300)')
    parser.add_argument('--rescue-temp', type=float, default=600.0, help='Retry failed SCC at this Fermi T [K] (0 disables)')
    parser.add_argument('--plot-only', action='store_true', help='Skip DFTB; replot figures from saved <name>/scan.pkl')
    parser.add_argument('--recover', action='store_true', help='Rebuild scan.pkl from existing DFTB output dirs (no new DFTB runs), then replot')
    parser.add_argument('--work-root', default=None, help='Root dir containing <name>/ job dirs for --recover (default: debug/test_corner_scan_pbc)')
    parser.add_argument('--prepare', default=None, metavar='BUNDLE_DIR', help='Bake cluster job tree into BUNDLE_DIR/<name>/ (dftb_in.hsd only, no runs) + run_all.sh/jobs.txt at top level')
    parser.add_argument('--rigid-jobs', action='store_true', help='With --prepare, also bake interior rigid path SP jobs (default: skip — --recover recomputes them locally, they are ~1s each)')
    args = parser.parse_args()
    names = sorted(PBC_CHAIN_ARTS) if args.name == 'all' else args.name.split(',')
    if args.prepare:
        os.makedirs(args.prepare, exist_ok=True)
    for name in names:
        run_one(name, dx=args.dx, relax_corners=not args.no_relax, relax_paths=args.relax_paths, sk_set=args.sk_set, work_root=args.work_root, nk=(1, args.nk, 1), filling_temp=args.filling_temp, rescue_temp=args.rescue_temp or None, hbond_length=args.hbond, tilt=args.tilt, zigzag=args.zigzag, slant=args.slant, jkink=args.jkink, plot_only=args.plot_only, recover=args.recover, prepare=args.prepare, rigid_jobs=args.rigid_jobs)
    if args.prepare:
        write_run_script(args.prepare)


def write_run_script(bundle_dir):
    """Top-level runner + job manifest + PBS chunks for a prepared bundle.

    run_all.sh executes `dftb+ > OUT 2> ERR` in every job dir listed in the
    given list file(s) (default: jobs.txt), skipping finished jobs — one
    invocation = one PBS job running many sub-jobs sequentially.
    jobs.txt lists all job dirs (corners first).  chunks/<name>.txt holds one
    system's dirs — each PBS submission runs a single system end-to-end as one
    long sequential job.  SK_PREFIX env rewrites the baked slako prefix.
    """
    script = r"""#!/usr/bin/env bash
# Corner-scan bundle runner — sequential, many sub-jobs per invocation.
#   ./run_all.sh                          -> all jobs (jobs.txt)
#   ./run_all.sh dftb+                    -> explicit binary
#   ./run_all.sh dftb+ chunks/chunk_02.txt [more lists...]   -> one PBS job
#   SK_PREFIX=/path/to/3ob-3-1/ ./run_all.sh   rewrite SK prefix first
set -u
DFTB=${1:-dftb+}
[ $# -ge 1 ] && shift
ROOT=$(cd "$(dirname "$0")" && pwd)

# A geo-opt that died mid-run can still contain 'Total Energy' lines (one per
# finished step) -> Driver jobs are done only on 'Geometry converged'.
job_done() {
  [ -f "$ROOT/$1/OUT" ] || return 1
  if grep -q "Driver" "$ROOT/$1/dftb_in.hsd"; then
    grep -q "Geometry converged" "$ROOT/$1/OUT"
  else
    grep -q "Total Energy" "$ROOT/$1/OUT"
  fi
}
run_job() { (cd "$ROOT/$1" && "$DFTB" > OUT 2> ERR < /dev/null); }

if [ -n "${SK_PREFIX:-}" ]; then
  find "$ROOT" -name dftb_in.hsd -exec sed -i "s|Prefix = .*|Prefix = $SK_PREFIX|" {} +
fi
{ if [ $# -gt 0 ]; then cat "$@"; else cat "$ROOT/jobs.txt"; fi; } | \
while IFS= read -r d; do
  [ -z "$d" ] && continue
  job_done "$d" && continue
  echo "[$(date +%H:%M:%S)] $d"
  run_job "$d"
  # rescue tiers (same as in-session): hotter Fermi + more iters, then weaker mixing
  if ! job_done "$d"; then
    sed -i 's/Temperature \[K\] = [0-9.]*/Temperature [K] = 600/; s/MaxSccIterations = [0-9]*/MaxSccIterations = 500/' "$ROOT/$d/dftb_in.hsd"
    run_job "$d"
  fi
  if ! job_done "$d"; then
    sed -i 's/MixingParameter = [0-9.]*/MixingParameter = 0.05/' "$ROOT/$d/dftb_in.hsd"
    run_job "$d"
  fi
  job_done "$d" || echo "FAILED $d"
done
echo "ALL DONE"
"""
    with open(os.path.join(bundle_dir, 'run_all.sh'), 'w') as f:
        f.write(script)
    os.chmod(os.path.join(bundle_dir, 'run_all.sh'), 0o755)
    jobs = sorted(os.path.relpath(os.path.join(r, d), bundle_dir)
                  for r, dirs, _ in os.walk(bundle_dir) for d in dirs
                  if d.startswith(('corner_', 'path_', 'pathr_')))
    jobs.sort(key=lambda s: (0 if '/corner_' in s else 1 if '/path_' in s else 2, s))
    with open(os.path.join(bundle_dir, 'jobs.txt'), 'w') as f:
        f.write('\n'.join(jobs) + '\n')

    # one job-list per system: each PBS submission runs a single system end-to-end
    systems = sorted({j.split('/')[0] for j in jobs})
    cdir = os.path.join(bundle_dir, 'chunks')
    os.makedirs(cdir, exist_ok=True)
    for name in systems:
        with open(os.path.join(cdir, f'{name}.txt'), 'w') as f:
            f.write('\n'.join(j for j in jobs if j.split('/')[0] == name) + '\n')
    print(f"wrote {bundle_dir}/run_all.sh + jobs.txt ({len(jobs)} jobs) + chunks/<system>.txt ({len(systems)} system lists)")


if __name__ == '__main__':
    main()
