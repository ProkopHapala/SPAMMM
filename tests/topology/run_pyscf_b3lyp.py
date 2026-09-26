#!/usr/bin/env python3
"""B3LYP recalculation driver for the mol-tip-switch job set (cluster-ready).

Reads a job manifest (jobs.csv: key,xyz,charge,spin) exported by
    python tests/topology/testplot_mol_flakes_dftb.py --export-pyscf <dir>

Per job writes:
  mats/<key>.npz   enames, apos(Ang), charge, spin, E_eV, S2,
                   S      = overlap matrix,
                   h1e    = 1-electron (core) Hamiltonian,
                   fock   = converged Fock  (RKS: (nao,nao); UKS: (2,nao,nao)),
                   dm     = density matrix  (same shape; UKS [0]=alpha [1]=beta),
                   mo_energy, mo_occ
  results.csv      key, E_eV, S2, converged, note        (resumable)

Geometry optimization via `geometric` when installed (pip install geometric),
else single point on the exported geometry.  Tip atoms in the exported mol/flake
xyz carry a small +/-z deflection so puckered (pyrrole-like) minima can be found.

Conventions:
  charge = total charge (0, -1, +1)
  spin   = number of unpaired electrons (PySCF Mole(spin=...)):
           0 = singlet, 1 = doublet, 2 = triplet
  UKS for spin>0 or odd Ne, RKS otherwise.

Usage:
  python3 run_pyscf_b3lyp.py <jobdir> [--basis def2-svp] [--nopt] [--nodf]
                                      [--manifest jobs_bonus.csv]
                                      [--states k1,k2] [--j0 N --j1 M]
Example:
  python3 tests/topology/run_pyscf_b3lyp.py debug/mol_flakes_dftb/pyscf
  # cluster array job:  --j0/--j1 slice the manifest for per-array-task runs
"""
import os, csv, argparse
import numpy as np

HAU2EV = 27.211386245988


def run_one(xyz, charge, spin, basis, do_opt, do_df=True):
    """Returns (mf, mol, opt_ok) at the final (optimized) geometry.
    UKS for open shell / odd Ne, RKS else; density fitting (RI) when do_df.
    (No frozen core: KS-DFT is all-electron; that saves nothing here.)"""
    from pyscf import gto, dft

    def make_mf(mol):
        mf = dft.UKS(mol) if (spin or mol.nelectron % 2) else dft.RKS(mol)
        if do_df:
            mf = mf.density_fit()                  # RI Coulomb/exchange aux fit
        mf.xc = 'b3lyp'
        mf.conv_tol = 1e-9
        mf.grids.level = 3
        return mf

    opt_ok = True
    mol = gto.M(atom=xyz, basis=basis, charge=int(charge), spin=int(spin), verbose=0)
    mf = make_mf(mol)
    if do_opt and mol.natm > 1:
        try:
            from pyscf.geomopt.geometric_solver import kernel as opt_kernel
            opt_ok, mol = opt_kernel(mf, maxsteps=150)   # Mole at relaxed geom; mf.mol unchanged
            mf = make_mf(mol)                          # fresh mf bound to the relaxed geometry
            if not opt_ok:
                print('    ! geom opt NOT converged in 150 steps', flush=True)
        except ImportError:
            print('    (geometric not installed -> single point)', flush=True)
    mf.kernel()
    return mf, mol, opt_ok


def save_mats(path, key, mf, mol, basis):
    """Hamiltonian pieces + DM + overlap for bond-order analysis.
    apos = the OPTIMIZED geometry (mol carries the relaxed coords)."""
    en = [mol.atom_symbol(i) for i in range(mol.natm)]
    apos = mol.atom_coords()                         # Bohr -> Ang
    from pyscf.lib import param as pyscf_param
    apos = apos * pyscf_param.BOHR
    dm = np.asarray(mf.make_rdm1())                  # RKS (nao,nao); UKS (2,nao,nao)
    fk = np.asarray(mf.get_fock())
    np.savez_compressed(path, key=key, enames=np.asarray(en), apos=np.asarray(apos),
                        charge=mol.charge, spin=mol.spin, basis=basis,
                        E_eV=mf.e_tot * HAU2EV,
                        S2=mf.spin_square()[0], S=mf.get_ovlp(), h1e=mf.get_hcore(),
                        fock=fk, dm=dm, mo_energy=np.asarray(mf.mo_energy),
                        mo_occ=np.asarray(mf.mo_occ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jobdir', help='dir with jobs.csv + xyz/ (from --export-pyscf)')
    ap.add_argument('--manifest', default='jobs.csv', help='jobs.csv or jobs_bonus.csv')
    ap.add_argument('--basis', default='def2-svp', help='double-zeta default; def2-tzvp for final')
    ap.add_argument('--nodf', action='store_true', help='disable density fitting (RI)')
    ap.add_argument('--nopt', action='store_true', help='single point only (no geom opt)')
    ap.add_argument('--states', default=None, help='comma list of keys')
    ap.add_argument('--j0', type=int, default=0, help='first job index (array jobs)')
    ap.add_argument('--j1', type=int, default=None, help='last job index (exclusive)')
    args = ap.parse_args()

    jobs = list(csv.DictReader(open(os.path.join(args.jobdir, args.manifest))))
    if args.states:
        wanted = set(args.states.split(','))
        jobs = [j for j in jobs if j['key'] in wanted]
    jobs = jobs[args.j0:args.j1]
    mdir = os.path.join(args.jobdir, 'mats')
    os.makedirs(mdir, exist_ok=True)
    rfile = os.path.join(args.jobdir, 'results.csv')
    done = {r['key'] for r in csv.DictReader(open(rfile))} if os.path.isfile(rfile) else set()
    todo = [j for j in jobs if j['key'] not in done
            or not os.path.isfile(os.path.join(mdir, j['key'] + '.npz'))]
    print(f'{len(done)} done, {len(todo)} to run (basis {args.basis}, opt={not args.nopt})', flush=True)

    new = not os.path.isfile(rfile)
    with open(rfile, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['key', 'E_eV', 'S2', 'converged', 'note'])
        for n, j in enumerate(todo):
            key = j['key']
            print(f'[{n + 1}/{len(todo)}] {key}  q={j["charge"]} spin={j["spin"]}', flush=True)
            try:
                mf, mol, opt_ok = run_one(os.path.join(args.jobdir, j['xyz']),
                                          j['charge'], j['spin'], args.basis,
                                          not args.nopt, not args.nodf)
                save_mats(os.path.join(mdir, key + '.npz'), key, mf, mol, args.basis)
                note = ';'.join(s for s in ('' if mf.converged else 'scf_not_converged',
                                            '' if opt_ok else 'opt_not_converged') if s)
                w.writerow([key, f'{mf.e_tot * HAU2EV:.6f}', f'{mf.spin_square()[0]:.4f}',
                            int(mf.converged), note])
            except Exception as e:
                w.writerow([key, '', '', 0, f'FAILED: {e}'])
            f.flush()
    print(f'done -> {rfile} (+ {mdir}/)', flush=True)


if __name__ == '__main__':
    main()
