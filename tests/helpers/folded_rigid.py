"""
folded_rigid.py — Reference data system for regression testing.

Purpose: Test-specific utilities for saving, loading, and comparing reference
data (physical properties like distances, heights, convergence) as
human-readable JSON + XYZ files.

Core simulation and plotting functions have been moved to:
  - spammm.surfaces.FoldedRigid  — core workflow functions
  - spammm.surfaces.surface_plots — visualization functions

This module re-exports those functions for backward compatibility with
existing test imports, and provides the reference data system on top.
"""

import os, sys, json
import numpy as np

_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

# Re-export core functions for backward compatibility
from spammm.surfaces.FoldedRigid import (
    random_quaternion, replicate_substrate, load_substrate,
    fit_folded_for_molecule, setup_rigid_folded, relax_folded,
    nearest_substrate_distance, find_bonds,
    lateral_scan, relaxed_scan, manipulation_trajectory,
    save_xyz_trajectory,
    NACL_SUBSTRATE, Z_SURF_TOP, LATTICE_A,
    MORSE_ALPHAS, COULOMB_ALPHAS, COMBINED_ALPHAS,
)

# Re-export plotting functions for backward compatibility
from spammm.surfaces.surface_plots import (
    plot_relaxation, plot_molecule_substrate_xy, plot_molecule_substrate_xz,
    plot_relax_overview, plot_force_map, plot_manipulation,
    plot_relaxed_scan, plot_manipulation_trail,
    ELEMENT_COLORS, ELEMENT_SIZES,
)

# =============================================================================
# Reference data system for regression testing
# =============================================================================
# Stores physically meaningful properties (distances, heights, convergence)
# as human-readable JSON + XYZ text files. Comparison uses physical tolerances
# so small forcefield parameter changes don't break tests.
#
# File layout per test:
#   tests/ref_data/{ref_name}.ref.json   — physical properties (includes test_func field)
#   tests/ref_data/{ref_name}.ref.xyz    — final geometry (molecule + nearby substrate)

REF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ref_data')


def _xyz_to_string(enames, apos):
    """Format atoms as XYZ string (no count/comment line)."""
    lines = []
    for e, p in zip(enames, apos):
        lines.append(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}')
    return '\n'.join(lines)


def _parse_xyz_string(s):
    """Parse XYZ body (no count line) into (enames, apos)."""
    enames, apos = [], []
    for line in s.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        enames.append(parts[0])
        apos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return enames, np.array(apos, dtype=np.float64)


def _first_layer_substrate(sub_apos, sub_enames, z_top=Z_SURF_TOP, z_tol=0.5):
    """Filter substrate to only 1st layer atoms (within z_tol of z_top)."""
    apos = np.asarray(sub_apos[:, :3], dtype=np.float64)
    mask = np.abs(apos[:, 2] - z_top) < z_tol
    return apos[mask], [sub_enames[i] for i in range(len(sub_enames)) if mask[i]]


def _substrate_near_molecule(sub_apos, sub_enames, mol_apos, margin=3.0, z_top=Z_SURF_TOP, z_tol=0.5):
    """1st-layer substrate atoms within bbox around molecule (XY + margin)."""
    apos1, enames1 = _first_layer_substrate(sub_apos, sub_enames, z_top, z_tol)
    mol = np.asarray(mol_apos[:, :3], dtype=np.float64)
    xmin, ymin = mol[:, 0].min() - margin, mol[:, 1].min() - margin
    xmax, ymax = mol[:, 0].max() + margin, mol[:, 1].max() + margin
    mask = (apos1[:, 0] >= xmin) & (apos1[:, 0] <= xmax) & (apos1[:, 1] >= ymin) & (apos1[:, 1] <= ymax)
    return apos1[mask], [enames1[i] for i in range(len(enames1)) if mask[i]]


def save_reference(ref_name, mol_enames, final_atoms, final_pos, final_force, final_torque,
                   sub_apos, sub_enames, z_rel, test_func=None, extra_distances=None):
    """Save reference data as JSON + XYZ files.

    Args:
        ref_name: e.g. 'ptcda_nacl' → writes tests/ref_data/ptcda_nacl.ref.json
        mol_enames: list of molecule element names
        final_atoms: (natoms, 3) final molecule atom positions
        final_pos: (3,) final COM position
        final_force: float |F|
        final_torque: float |τ|
        sub_apos: (N, 3) replicated substrate positions
        sub_enames: list of substrate element names
        z_rel: float, adsorption height above surface
        test_func: name of the test function this reference belongs to (e.g. 'test_relax_ptcda_nacl')
        extra_distances: list of dicts with {element, d_Na, d_Cl} per atom
    """
    os.makedirs(REF_DIR, exist_ok=True)
    json_path = os.path.join(REF_DIR, f'{ref_name}.ref.json')
    xyz_path = os.path.join(REF_DIR, f'{ref_name}.ref.xyz')

    atom_distances = []
    for ia, e in enumerate(mol_enames):
        d_Na, _ = nearest_substrate_distance(final_atoms[ia], sub_apos, sub_enames, 'Na')
        d_Cl, _ = nearest_substrate_distance(final_atoms[ia], sub_apos, sub_enames, 'Cl')
        atom_distances.append({'element': e, 'd_Na': round(d_Na, 3), 'd_Cl': round(d_Cl, 3)})

    sub_near_pos, sub_near_names = _substrate_near_molecule(sub_apos, sub_enames, final_atoms)

    ref = {
        'description': f'Reference for {ref_name}: relaxed molecule on NaCl(100) folded basis',
        'test_func': test_func or '',
        'test_module': 'tests.test_folded_relax',
        'ref_name': ref_name,
        'z_rel': round(z_rel, 4),
        'force': round(final_force, 6),
        'torque': round(final_torque, 6),
        'atom_distances': atom_distances,
        'n_atoms': len(mol_enames),
    }

    with open(json_path, 'w') as f:
        json.dump(ref, f, indent=2)
        f.write('\n')

    with open(xyz_path, 'w') as f:
        total = len(sub_near_names) + len(mol_enames)
        f.write(f'{total}\n')
        f.write(f'{ref_name} reference: {len(sub_near_names)} substrate + {len(mol_enames)} molecule atoms\n')
        for e, p in zip(sub_near_names, sub_near_pos):
            f.write(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')
        for e, p in zip(mol_enames, final_atoms):
            f.write(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')

    print(f'[REF] Saved {json_path}')
    print(f'[REF] Saved {xyz_path}')


def load_reference(ref_name):
    """Load reference data. Returns dict or None if not found."""
    json_path = os.path.join(REF_DIR, f'{ref_name}.ref.json')
    xyz_path = os.path.join(REF_DIR, f'{ref_name}.ref.xyz')
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        ref = json.load(f)
    if os.path.exists(xyz_path):
        with open(xyz_path) as f:
            lines = f.readlines()
        xyz_body = ''.join(lines[2:])
        ref['xyz_enames'], ref['xyz_apos'] = _parse_xyz_string(xyz_body)
    return ref


def compare_to_reference(ref_name, mol_enames, final_atoms, final_pos, final_force, final_torque,
                         sub_apos, sub_enames, z_rel,
                         tol_z=0.5, tol_dist=0.3, force_thresh=0.05, torque_thresh=0.05):
    """Compare current results to saved reference.

    Uses physical tolerances:
    - z_rel within ±tol_z Å
    - per-atom d_Na, d_Cl within ±tol_dist Å
    - "O closer to Na than Cl" boolean must match
    - force < force_thresh, torque < torque_thresh

    Returns (passed: bool, messages: list[str]).
    """
    ref = load_reference(ref_name)
    if ref is None:
        return False, [f'Reference {ref_name} not found in {REF_DIR}']

    msgs = []
    passed = True

    ref_z = ref['z_rel']
    dz = abs(z_rel - ref_z)
    if dz > tol_z:
        passed = False
        msgs.append(f'FAIL z_rel: {z_rel:.3f} vs ref {ref_z:.3f} (Δ={dz:.3f} > {tol_z})')
    else:
        msgs.append(f'OK   z_rel: {z_rel:.3f} vs ref {ref_z:.3f} (Δ={dz:.3f})')

    if final_force > force_thresh:
        passed = False
        msgs.append(f'FAIL |F|={final_force:.6f} > {force_thresh}')
    else:
        msgs.append(f'OK   |F|={final_force:.6f} < {force_thresh}')

    if final_torque > torque_thresh:
        passed = False
        msgs.append(f'FAIL |τ|={final_torque:.6f} > {torque_thresh}')
    else:
        msgs.append(f'OK   |τ|={final_torque:.6f} < {torque_thresh}')

    ref_dists = ref['atom_distances']
    if len(ref_dists) != len(mol_enames):
        passed = False
        msgs.append(f'FAIL atom count: {len(mol_enames)} vs ref {len(ref_dists)}')
    else:
        for ia, (e, ref_d) in enumerate(zip(mol_enames, ref_dists)):
            d_Na, _ = nearest_substrate_distance(final_atoms[ia], sub_apos, sub_enames, 'Na')
            d_Cl, _ = nearest_substrate_distance(final_atoms[ia], sub_apos, sub_enames, 'Cl')
            ref_Na, ref_Cl = ref_d['d_Na'], ref_d['d_Cl']
            dNa = abs(d_Na - ref_Na)
            dCl = abs(d_Cl - ref_Cl)
            ref_bool = ref_Na < ref_Cl
            cur_bool = d_Na < d_Cl
            bool_match = ref_bool == cur_bool
            dist_ok = dNa < tol_dist and dCl < tol_dist
            if not bool_match:
                passed = False
                msgs.append(f'FAIL atom {ia} ({e}): Na<Cl changed (ref={ref_bool}, cur={cur_bool})')
            elif not dist_ok:
                passed = False
                msgs.append(f'FAIL atom {ia} ({e}): d(Na) Δ={dNa:.3f} d(Cl) Δ={dCl:.3f} > {tol_dist}')
            else:
                msgs.append(f'OK   atom {ia} ({e}): d(Na)={d_Na:.3f} (ref {ref_Na:.3f}) d(Cl)={d_Cl:.3f} (ref {ref_Cl:.3f})')

    return passed, msgs
