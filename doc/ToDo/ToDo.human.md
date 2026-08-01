
1) stable cassette rots for coarse grid ptcda (https://graphics.cs.utah.edu/research/projects/stable-cosserat-rods/) → `doc/Tasks/Import_CosseratRods_PTCDA.md`
2) implement PPAFM Kriging interactions (`/home/prokop/git/ppafm/docs/export/interpolation.export.md`) → `doc/Tasks/Import_KrigingGridFF.md`
   - follow-on: site-resolved Pauli \(A,\beta\) maps + transferability → `doc/Tasks/Pauli_A_beta_KrigingTransferability.md`
3) ~~Monte Carlo + Pauli master equations / charge rings~~ **Done (PME A+D+F)** — Hubbard/MQCA/MC-fit Later (`doc/Tasks/Import_ChargeRings_PME.md`, audit `doc/TopicalAudit/ChargeRings_PME.md`)
   - follow-on (design): sites = rigid molecules via shared pose SSOT — `doc/Tasks/RigidMoleculePose_SSOT.md`, inventory `doc/TopicalAudit/RigidBody.md`
   - was: `/home/prokop/git/ppafm/docs/export/charge_rings.export.md`, `/home/prokop/git/FireCore/doc/Topics/ManyBody/MQCA_Hubbard_Ising.export.md`
4) prolonged radial basis with DFTB+ (both STM and AFM) — tests live under `tests/SPM/testplot_3ob_basis_tails.py` etc. → `doc/Tasks/ProlongedRadialBasis_DFTB.md`
   - FDBM refs: `/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/{pentacene,PTCDA,azaindol_dimer,azaindol_isodimer,benzoicacid_dimer,benzoicamid_dimer}_PBE_def2-SVP/` (`rho_N`/`esp_N`) vs DFTB stock + extended
5) Dyson orbital with DFTB+? (`doc/Dyson_orbitals_STM.chat.md`) → `doc/Tasks/DysonOrbitals_DFTB_STM.md`
6) make a little pyOpenCL weave / jit / codon driver for forcefield optimization → `doc/Tasks/FF_Optimizer_OpenCL_Driver.md`
7) **STM** systematic HOMO/LUMO: mio vs 3ob vs extended Slater vs pySCF cubes (pentacene, PTCDA, …) → `doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md`
8) **Kekulé π → exponential RI density** (atom + bond centers) vs DFT/DFTB → `doc/Tasks/Kekule_ExponentialDensityFit.md`
9) **Fast 2.5D AFM** (contact height + z-modes; PIC; hybrid?) → task `doc/Tasks/Fast_2p5D_AFM_ContactSurface.md` · parity report `doc/Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md` (sphere h₀ / coarse dx; USER visual pending)
   - **Open (2026-07-24):** helicene assembly showed contact-sep too long-ranged vs Morse+Coulomb GridFF — report `doc/Reports/Assembly_ContactSurface_AFM_helicene_2026-07-24.md`; next = PTCDA re-check + 1/2-atom toys (do not reinvent parity stack)
   - Assembly pipeline task: `doc/Tasks/Assembly_AFM_Pipeline.md`

Consolidation strategy: `doc/Tasks/RepoConsolidation.md` (eat FireCore/ppafm into SPAMMM pyOpenCL + plugin GUI).


# GUI

- [x] 3D view (ortho editor mode: `Enter` / `b2Dview`; Ring/atom/bond work; hex 2D-only) — `doc/Tasks/GUI_Editor_3D_ViewMode.md`
- [ ] Molecuar Browser
- [ ] add menu to load example moleucles form assiart of SMILES


## GUI Laptop

- [x] zoom-in / zoom out when mouse is not accessible (add sime slidebar), +/- button also not accesible on laptop, 3D viweport buttons, maybe make special pannel like "laptop accesability" — implemented: "Laptop Accessibility" collapsible section (zoom slider + Zoom In/Out/Reset buttons), `tests/GUI/test_accessibility_section.py`
- [x] The panel have fixed size - allow user to change size of side panel, also make sure if panel does not fit there is horizonal scrollbar — implemented: QSplitter (resizable 200–600px), horizontal scrollbar AsNeeded


## GUI Help + Shortcuts

- [ ] Help/cheatsheet panel in GUI + centralized shortcut registry (buttons auto-show key, help auto-generates) → `doc/Tasks/GUI_HelpPanel_ShortcutRegistry.md`
  - USER decisions (2026-08-01): Unicode modifiers `⌃⇧⌥`; both side-panel section + Help menu dialog; auto-generate cheatsheet from registry; mouse actions stay hardwired (not in registry) but documented in cheatsheet + help panel
  - Architecture (CORRECTED 2026-08-01): Registry = generic mechanism only (conflict detection, encoding, label sync, help gen). Extensions register their own shortcuts — NO central hardcoded action list. Fail-loud on conflict.
- [ ] Reorganize user-facing docs: move `doc/KekuleSolverVisualization.md` → `user_guide/` → `doc/Tasks/GUI_HelpPanel_ShortcutRegistry.md` §5




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

- [ ] reoranize the editor menu - a but overcomplicated and slow, key-sortcuts → `doc/Tasks/GUI_HelpPanel_ShortcutRegistry.md`
- [x] 3D view