"""
Vibrations.py — Normal-mode analysis from molecular Hessians (DFTB or GPU force-field FD).

Rigid-body modes are projected from the Hessian before diagonalization — required for
floating molecules in the GUI where translations/rotations would otherwise appear as
near-zero "vibrations". In-plane vs out-of-plane classification uses displacement
energy fractions in the lab xy/z axes (suited to planar adsorbates on z≈0).

- **Backends:** DFTB+ `SecondDerivatives`; UFF/SPFF via `FFEvaluator.make_ff_eval_fn` + central FD
- **Entry point:** `run_vibrations(mol, backend=...)`
- **Units:** internal SSOT is cm⁻¹; display via `freq_cm1_to_unit` (meV, THz, kcal/mol)

Open issues:
- SPFF: pi-orbitals frozen during FD (nuclear-position modes only)
- Absolute frequencies from UFF/DFTB not calibrated to experiment
- Phonon bands / PBC: not implemented (see doc/Topics/Vibrations.md)
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .. import elements
from ..AtomicSystem import AtomicSystem
from ..forcefields.FFEvaluator import make_ff_eval_fn
from ..quantum.DFTB_utils import (
    write_dftb_input_hessian,
    read_hessian,
    hessian_hartree_bohr_to_eV_angstrom,
    DFTB_EXE,
    DEFAULT_SK_SET,
)

# h*c in eV·cm (for nu in cm^-1): E = HC_EV_CM * nu; E_zpe = 0.5 * HC_EV_CM * nu
HC_EV_CM = 1.239841984e-4
C_CM_S = 2.99792458e10
EV_TO_KCAL_MOL = 23.060548025

Backend = Literal['uff', 'spff', 'dftb']
FreqUnit = Literal['cm-1', 'meV', 'THz', 'kcal/mol']

FREQ_UNIT_LABELS = {
    'cm-1': 'cm⁻¹',
    'meV': 'meV',
    'THz': 'THz',
    'kcal/mol': 'kcal/mol',
}


def freq_cm1_to_unit(nu_cm1: float, unit: FreqUnit = 'cm-1') -> float:
    """Convert wavenumber (cm⁻¹) to display unit. Energy units use quantum E = h c ν."""
    nu = float(nu_cm1)
    if unit == 'cm-1':
        return nu
    if unit == 'THz':
        return nu * C_CM_S / 1e12
    E_eV = HC_EV_CM * abs(nu)
    sign = 1.0 if nu >= 0 else -1.0
    if unit == 'meV':
        return sign * E_eV * 1000.0
    if unit == 'kcal/mol':
        return sign * E_eV * EV_TO_KCAL_MOL
    raise ValueError(f"Unknown freq unit: {unit!r}")


def format_freq(nu_cm1: float, unit: FreqUnit = 'cm-1') -> str:
    v = freq_cm1_to_unit(nu_cm1, unit)
    if unit == 'cm-1':
        return f'{v:.2f}'
    if unit == 'THz':
        return f'{v:.4f}'
    if unit == 'meV':
        return f'{v:.2f}'
    return f'{v:.3f}'


def format_E_zpe(nu_cm1: float, unit: FreqUnit = 'cm-1') -> str:
    """Zero-point energy ½ h c ν in display-consistent units."""
    nu = float(nu_cm1)
    if unit in ('meV', 'kcal/mol'):
        return format_freq(0.5 * abs(nu), unit) if nu >= 0 else '—'
    return f'{0.5 * HC_EV_CM * abs(nu):.5f}'


@dataclass
class ModeInfo:
    index: int
    freq_cm1: float
    E_zpe_eV: float
    f_xy: float
    f_z: float
    character: str = ''


@dataclass
class VibrationResult:
    enames: list
    pos: np.ndarray           # (N, 3) Å
    masses: np.ndarray        # (N,) amu
    hessian: np.ndarray       # (3N, 3N) eV/Å² after rigid projection
    frequencies_cm1: np.ndarray  # (n_modes,) sorted ascending
    modes: np.ndarray         # (3N, n_modes) mass-normalized displacement vectors
    mode_info: list           # list[ModeInfo]
    backend: str
    n_rigid_removed: int = 6
    perm: Optional[list] = None

    def softest_indices(self, n=6):
        return np.argsort(np.abs(self.frequencies_cm1))[:n]

    def format_table(self, unit: FreqUnit = 'cm-1') -> str:
        flab = FREQ_UNIT_LABELS[unit]
        elab = flab if unit in ('meV', 'kcal/mol') else 'eV'
        lines = [f"{'#':>4} {('freq/' + flab):>14} {('E_zpe/' + elab):>14} {'f_xy':>8} {'f_z':>8}  character"]
        for m in self.mode_info:
            lines.append(f"{m.index:4d} {format_freq(m.freq_cm1, unit):>14} {format_E_zpe(m.freq_cm1, unit):>14} {m.f_xy:8.3f} {m.f_z:8.3f}  {m.character}")
        return '\n'.join(lines)

    def mode_freq_label(self, mode_index: int, unit: FreqUnit = 'cm-1') -> str:
        m = self.mode_info[mode_index]
        return f"{format_freq(m.freq_cm1, unit)} {FREQ_UNIT_LABELS[unit]}"


def atomic_masses(enames) -> np.ndarray:
    return np.array([elements.ELEMENT_DICT[e][elements.index_mass] for e in enames], dtype=np.float64)


def hessian_fd_forces(eval_fn, pos0, delta=1e-4):
    """Central finite-difference Hessian from eval_fn(pos) -> (E, F). F in eV/Å."""
    pos0 = np.asarray(pos0, dtype=np.float64)
    natoms = pos0.shape[0]
    ndof = 3 * natoms
    H = np.zeros((ndof, ndof), dtype=np.float64)
    for i in range(ndof):
        ia, c = divmod(i, 3)
        pos_p = pos0.copy(); pos_p[ia, c] += delta
        pos_m = pos0.copy(); pos_m[ia, c] -= delta
        _, Fp = eval_fn(pos_p.astype(np.float32))
        _, Fm = eval_fn(pos_m.astype(np.float32))
        H[i, :] = -(Fp - Fm).ravel() / (2.0 * delta)
    return 0.5 * (H + H.T)


def rigid_body_basis(pos, masses):
    """Mass-weighted translation + rotation basis vectors, shape (3N, 6)."""
    pos = np.asarray(pos, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    N = len(masses)
    com = np.average(pos, axis=0, weights=masses)
    r = pos - com
    R = np.zeros((3 * N, 6), dtype=np.float64)
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        for i in range(N):
            R[3 * i:3 * i + 3, k] = np.sqrt(masses[i]) * e
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        for i in range(N):
            R[3 * i:3 * i + 3, 3 + k] = np.sqrt(masses[i]) * np.cross(r[i], e)
    Q, _ = np.linalg.qr(R)
    return Q


def project_rigid_modes(hessian, pos, masses):
    """Remove translation/rotation: H' = P H P, P = I - Q Q^T."""
    Q = rigid_body_basis(pos, masses)
    P = np.eye(hessian.shape[0]) - Q @ Q.T
    Hp = P @ hessian @ P
    return 0.5 * (Hp + Hp.T), P


def hessian_to_modes(hessian, masses):
    """Diagonalize mass-weighted Hessian. Returns freq_cm1 (n_modes,), modes (3N, n_modes) physical."""
    masses = np.asarray(masses, dtype=np.float64)
    im = np.repeat(masses ** -0.5, 3)
    D = im[:, None] * hessian * im[None, :]
    omega2, eigvec_mw = np.linalg.eigh(D)
    eV_to_J = 1.602176634e-19
    amu_to_kg = 1.66053906660e-27
    ang_to_m = 1e-10
    c_cm = 2.99792458e10
    omega = np.sqrt(np.maximum(omega2, 0.0) * eV_to_J / (amu_to_kg * ang_to_m ** 2))
    freq_cm1 = omega / c_cm
    # imaginary modes: mark negative
    freq_cm1 = np.where(omega2 < 0, -omega / c_cm, freq_cm1)
    modes_phys = im[:, None] * eigvec_mw
    for k in range(modes_phys.shape[1]):
        norm = np.linalg.norm(modes_phys[:, k])
        if norm > 0:
            modes_phys[:, k] /= norm
    return freq_cm1, modes_phys


def mode_plane_fractions(mode_3n: np.ndarray):
    """Return (f_xy, f_z) displacement energy fractions for one mode vector (3N,)."""
    u = mode_3n.reshape(-1, 3)
    E_xy = float(np.sum(u[:, 0] ** 2 + u[:, 1] ** 2))
    E_z = float(np.sum(u[:, 2] ** 2))
    tot = E_xy + E_z
    if tot < 1e-30:
        return 0.5, 0.5
    return E_xy / tot, E_z / tot


def _character_label(f_xy, f_z):
    if f_z > 0.75:
        return 'out-of-plane'
    if f_xy > 0.75:
        return 'in-plane'
    return 'mixed'


def hessian_from_ff(mol, ff='uff', delta=1e-4, do_nonbond=False):
    eval_fn, pos0, natoms, perm, enames = make_ff_eval_fn(mol, ff=ff, do_nonbond=do_nonbond)
    H = hessian_fd_forces(eval_fn, pos0, delta=delta)
    masses = atomic_masses(enames)
    return H, pos0.astype(np.float64), masses, enames, perm


def hessian_from_dftb(enames, apos, workdir=None, sk_set=None, delta=1e-4):
    """Run DFTB+ SecondDerivatives; return H in eV/Å²."""
    sk_set = sk_set or DEFAULT_SK_SET
    apos = np.asarray(apos, dtype=np.float64)
    cwd = os.getcwd()
    own_tmp = workdir is None
    if own_tmp:
        workdir = tempfile.mkdtemp(prefix='spammm_vib_')
    try:
        os.chdir(workdir)
        with open('geo.xyz', 'w') as f:
            f.write(f"{len(enames)}\n")
            f.write("vibrations\n")
            for e, p in zip(enames, apos):
                f.write(f"{e} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        write_dftb_input_hessian(enames, gname='geo.xyz', sk_set=sk_set, delta=delta)
        ierr = os.system(f'{DFTB_EXE} > OUT 2> ERR')
        if ierr != 0:
            raise RuntimeError(f"DFTB+ Hessian failed with code {ierr} in {workdir}")
        H_hb = read_hessian('hessian.out', n_atoms=len(enames))
        H = hessian_hartree_bohr_to_eV_angstrom(H_hb)
        return H
    finally:
        os.chdir(cwd)
        if own_tmp:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)


def analyze_vibrations(hessian, pos, masses, enames, backend='uff', perm=None, rigid_freq_tol=20.0):
    Hp, _ = project_rigid_modes(hessian, pos, masses)
    freq_all, modes_all = hessian_to_modes(Hp, masses)
    vibr_mask = np.abs(freq_all) >= rigid_freq_tol
    freq = freq_all[vibr_mask]
    modes = modes_all[:, vibr_mask]
    order = np.argsort(np.abs(freq))
    freq = freq[order]
    modes = modes[:, order]
    mode_info = []
    for i, k in enumerate(range(modes.shape[1])):
        f = float(freq[k])
        f_xy, f_z = mode_plane_fractions(modes[:, k])
        mode_info.append(ModeInfo(index=i, freq_cm1=f, E_zpe_eV=0.5 * HC_EV_CM * abs(f), f_xy=f_xy, f_z=f_z, character=_character_label(f_xy, f_z)))
    return VibrationResult(enames=list(enames), pos=np.asarray(pos), masses=np.asarray(masses), hessian=Hp, frequencies_cm1=freq, modes=modes, mode_info=mode_info, backend=backend, perm=perm)


def run_vibrations(mol, backend: Backend = 'uff', delta=1e-4, do_nonbond=False, sk_set=None, workdir=None, rigid_freq_tol=20.0):
    """End-to-end vibrational analysis.

    Args:
        mol: AtomicSystem or path to .xyz
        backend: 'uff', 'spff', or 'dftb'
    """
    if isinstance(mol, str):
        mol = AtomicSystem(fname=mol)
    enames = list(mol.enames)
    pos = np.asarray(mol.apos[:, :3], dtype=np.float64)
    masses = atomic_masses(enames)
    perm = None
    if backend in ('uff', 'spff'):
        H, pos, masses, enames, perm = hessian_from_ff(mol, ff=backend, delta=delta, do_nonbond=do_nonbond)
    elif backend == 'dftb':
        H = hessian_from_dftb(enames, pos, workdir=workdir, sk_set=sk_set, delta=delta)
    else:
        raise ValueError(f"Unknown backend={backend!r}")
    return analyze_vibrations(H, pos, masses, enames, backend=backend, perm=perm, rigid_freq_tol=rigid_freq_tol)
