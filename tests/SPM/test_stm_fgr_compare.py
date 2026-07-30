"""L0 smoke: FGR STM (H−ES) vs legacy overlap STM.

Covers the open "L0 pytest: missing" item in
`doc/TopicalAudit/STM_FGR_Transfer.md` and the report
`doc/Reports/STM_FGR_Transfer_H_ES_2026-07-29.md`.

Two levels:
  - table smoke (no GPU): directed SK antisymmetry Sps ≈ −Ssp for identical STOs
  - scan smoke (GPU): benzene HOMO, I_τ=|c†(H−ES)c|² vs overlap_exp, finite & distinct
"""
import os
import numpy as np
import pytest

_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))

ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


def test_stm_fgr_table_sps_minus_ssp():
    """Directed SK antisymmetry: for identical s/p STOs, Sps ≈ −Ssp.

    Axis u = (R_sample − R_tip)/R = +z; tip at z=0, sample at z=R.
    <p_T,u|s_S> = −<s_T|p_S,u> by reflection z → R−z (zB = ZZ−R vs ZZ).
    This is the "store signed sp and ps, do not hard-code a second minus"
    invariant from the FGR wiring note.
    """
    from spammm.quantum.DFTB.DFTBplusParser import sto_two_center_sk_channels

    sto_s = (1.0, 1.6)   # (N, zeta) — synthetic, identical on both centres
    sto_p = (1.0, 1.6)
    R = 3.0
    Sss, Ssp, Sps, Spp_s, Spp_pi = sto_two_center_sk_channels(
        R, sto_s, sto_p, sto_s, sto_p)

    # Sanity: s-s and pπ (px-px) overlaps are positive at finite R; pσ (pz-pz,
    # same lab orientation) is negative at R~3 (positive lobe of tip overlaps
    # negative lobe of sample) — sign is not the invariant here.
    assert Sss > 0.0, f"Sss={Sss} should be > 0"
    assert Spp_pi > 0.0, f"Spp_pi={Spp_pi} should be > 0"
    assert abs(Spp_s) > 0.0, f"|Spp_s|={abs(Spp_s)} should be non-zero"
    # Directed antisymmetry (the FGR invariant): Sps ≈ −Ssp
    assert abs(Sps + Ssp) < 1e-6 * max(1.0, abs(Ssp)), \
        f"Sps={Sps} should equal -Ssp={-Ssp} for identical STOs"
    # sp channels are non-trivial at this R
    assert abs(Ssp) > 1e-6, f"|Ssp|={abs(Ssp)} too small to test antisymmetry"


@pytest.mark.gpu
@pytest.mark.slow
def test_stm_fgr_vs_overlap_benzene(tmp_path):
    """Benzene HOMO: FGR I_τ=|c†(H−ES)c|² vs legacy overlap_exp.

    Asserts: maps finite & ≥0; I_τ differs from overlap_exp (the whole point
    of FGR); selected device is NVIDIA (not PoCL) per opencl-nvidia-gpu policy.
    """
    from spammm import atomicUtils as au
    from spammm.quantum.DFTB_utils import WFC_HSD_PATHS
    from spammm.SPM import AFM_utils as afm_utils

    xyz = os.path.join(_ROOT, 'data', 'xyz', 'benzene.xyz')
    pos, _, names, _, _ = au.load_xyz(xyz)
    pos = np.asarray(pos, dtype=np.float64)
    types = np.array([ELEM_Z[e] for e in names], dtype=np.int32)

    lo = pos[:, :2].min(axis=0) - 2.0
    hi = pos[:, :2].max(axis=0) + 2.0
    xs = np.arange(lo[0], hi[0], 0.4)
    ys = np.arange(lo[1], hi[1], 0.4)
    z_mol = float(np.mean(pos[:, 2]))
    z_plane = z_mol + 3.0

    d = afm_utils.get_density_from_dftb_dense(
        pos, types, WFC_HSD_PATHS['3ob-3-1'], str(tmp_path / 'dftb'),
        step=0.5, margin=0.5, z_extra=0.5, verbosity=0, project_density=False)
    eigvecs = d['eigvecs']
    eigvals = np.asarray(d['eigvals'], dtype=np.float64)
    n_elec = afm_utils.dftb_n_valence_electrons(enames=names)
    homo, _lumo = afm_utils.dftb_frontier_mo_indices(eigvals, n_elec=n_elec)
    projector = d['projector']
    atoms_dict = d['atoms_dict']
    basis_ang = d['basis_ang']
    species_per_atom = list(range(len(names)))
    afm_utils._set_projector_species_basis(projector, atoms_dict, basis_ang, rc_max=10.0)

    # NVIDIA device policy: fail loud if we silently fell back to PoCL/CPU
    dev_name = str(projector.ctx.devices[0].name).lower()
    assert 'nvidia' in dev_name, \
        f"OpenCL device is {projector.ctx.devices[0].name!r}, expected NVIDIA (not PoCL/CPU)"

    tables, tip_type0, name_to_smp, _prol, _sto = afm_utils._stm_fgr_prepare_tables(
        projector, str(tmp_path / 'dftb'), basis_ang, tip_elem='C',
        zeta_override=None, K=1.75, rcut_table=10.0, sample_elems=sorted(set(names)))

    E_tun = float(eigvals[homo])  # elastic, E_tunnel = ε_sample

    overlap = afm_utils.project_mo_stm_sk_slice(
        projector, eigvecs[homo], atoms_dict, basis_ang, names, species_per_atom,
        xs, ys, z_plane, tip_orbital='s', rcut=8.0, intensity=True)
    i_tau = afm_utils.project_mo_stm_fgr_slice(
        projector, eigvecs[homo], atoms_dict, basis_ang, names, species_per_atom,
        xs, ys, z_plane, tables, tip_type0, name_to_smp, E_tun,
        tip_orbital='s', tip_elem='C', mode='tau', rcut=10.0, intensity=True)

    assert overlap.shape == i_tau.shape == (len(xs), len(ys))
    assert np.isfinite(overlap).all() and np.isfinite(i_tau).all()
    assert (overlap >= 0).all() and (i_tau >= 0).all(), "STM intensity must be ≥ 0"
    assert float(overlap.max()) > 0.0, "overlap_exp map is empty"
    assert float(i_tau.max()) > 0.0, "I_τ map is empty"
    # FGR H−ES must differ from the artificial-exp overlap (different operators)
    assert float(np.max(np.abs(i_tau - overlap))) > 0.0, \
        "I_τ identical to overlap_exp — FGR transfer not applied"
