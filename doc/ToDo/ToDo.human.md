
1) stable cassette rots for coarse grid ptcda (https://graphics.cs.utah.edu/research/projects/stable-cosserat-rods/) → `doc/Tasks/Import_CosseratRods_PTCDA.md`
2) implement PPAFM Kriging interactions (`/home/prokop/git/ppafm/docs/export/interpolation.export.md`) → `doc/Tasks/Import_KrigingGridFF.md`
3) Monte Carlo + Pauli master equations / charge rings (`/home/prokop/git/ppafm/docs/export/charge_rings.export.md`, `/home/prokop/git/FireCore/doc/Topics/ManyBody/MQCA_Hubbard_Ising.export.md`) → `doc/Tasks/Import_ChargeRings_PME.md`
4) prolonged radial basis with DFTB+ (both STM and AFM) — tests live under `tests/SPM/testplot_3ob_basis_tails.py` etc. → `doc/Tasks/ProlongedRadialBasis_DFTB.md`
5) Dyson orbital with DFTB+? (`doc/Dyson_orbitals_STM.chat.md`) → `doc/Tasks/DysonOrbitals_DFTB_STM.md`
6) make a little pyOpenCL weave / jit / codon driver for forcefield optimization → `doc/Tasks/FF_Optimizer_OpenCL_Driver.md`

Consolidation strategy: `doc/Tasks/RepoConsolidation.md` (eat FireCore/ppafm into SPAMMM pyOpenCL + plugin GUI).


# GUI

- [ ] 3D view
- [ ] Molecuar Browser


# Problems Found

- [ ] Relaxation is still slow, why ? (SPFF/UFF)
- [ ] Need to make sure I can open GUI in specific folder (save/load in that folder)
- [ ] AFM panel in SPAMM_GUI does not show any wiggets in sub-panes Parameters, Visualization, STM/Orbitals




# QUICK

- [x] export kekule structures
- [ ] after load of molecule hydrogens are not connected
- [x] QEq
- [x] Molecule fragments/groups for easy edit and manipulation.  Automatic search for bridges
- [x] decouple `ascii_art_heterocycle.py` from `KekulePure.py`


# longer

- [ ] reoranize the editor menu - a but overcomplicated and slow, key-sortcuts
- [ ] 3D view