# Reports

Validated and review-ready scientific results, with exact methods, artifacts, reproduction commands, and unresolved caveats. Task documents may evolve during investigation; reports here are the durable handoff.

- **MoleculeExtraction_PME8_2026-08-18.md** — CONTCAR molecule extraction (PBC bonds, PCA, approximate symmetry P2₁2₁2₁) + **PME8** 8-site/256-state charge-ring kernel (sparse iterative Euler solver, parity vs PME4 = 4.14e-08); 7-molecule cluster STM scan. Caveat: CONTCAR z-coords must be flattened for PME.
- **ContactPME_PAW_AFM_MemSpeed_2026-08-11.md** — contact_pme **PAW** soft-replacement split; CLI AFM USER-confirmed; **~900–1200×** field memory vs Morse@0.1Å; PP-scan still ~12–18× slower; kernel WG/local-mem gaps and speed-up ideas. Task: `ContactSurface_PME_ParallelPlan.md`.
- **PairFF_TipPull_PTCDI_QEq_2026-07-28.md** — Tip-pull on NaCl with PTCDI + physical QEq; API `tip_pull_scan`; **map display not Vispy SSOT yet** (tomorrow task `PairFF_MapDisplay_SSOT.md`).
- **ContactSurface_2p5D_vs_GridFF_2026-07-24.md** — **SSOT** coarse 2.5D contact-sep vs Morse+Coulomb GridFF: sphere `h₀`, `h0_R_scale=0.75`, atom-scale `bspl/scan` dx; PTCDA + helicene artifacts; status investigating.
- **Assembly_ContactSurface_AFM_helicene_2026-07-24.md** — Helicene SAM assembly → contact-sep AFM; PBC/overlay fixes; points at 2.5D vs GridFF report for parity.
- **PTCDA_NaCl_Rigid_Newton_FIRE_Relaxation.md** — Why aggressive rigid Newton creates noisy basin selection, how staged LM damping restores Newton–FIRE consistency, and practical solver tradeoffs.
- **StaticObstacle_DragDemo_2026-08-03.md** — Dimer split via connected components, static/dynamic/deleted body-state gating (kernels 14+15), combined PairFF(static)+FAF probe map, mid-drag toggle demo; caveats: graph rebuild enames check, FAF for editor builds.
- **PTCDA_FDBM_prolonged_basis.md** — Stock 3ob versus SA-prolonged PTCDA density tails, Pauli fitting, and AFM images.
- **PySCF_GPU_CO_zscan_PTCDA.md** — Site-correct GPU pySCF CO–PTCDA interaction scans, timing, reference data, and limitations.
- **Kriging_DFT_vs_DFTB_FDBM_pyridine.md** — Pyridine+CO: Kriging DFT GridFF vs FDBM from cubes and DFTB; Δρ recipes vs basis tails; dual-basis Slater rules; plotting SSOT; session history (investigating).
- **Kriging_FDBM_PauliFit_pyridine_2026-07-21.md** — Day log: contact vs residual Pauli \(A,\beta\) fits; N+C vs N+C+H; tip×sample matrix; open cube-ES caveats.
- **STM_ExtendedBasis_OrbitalCompare.md** — STM HOMO/LUMO: mio/3ob stock vs prolonged vs pySCF (awaiting USER review).
- **STM_FGR_Transfer_H_ES_2026-07-29.md** — FGR \(M=H-ES\) vs legacy `overlap_exp`; Level-B long-tail tables; pentacene/PTCDA panels (`debug/stm_fgr_compare/`; awaiting USER review).

- **Cube_ES_DeltaRho_NA_dipole.md** — **SSOT** for cube Δρ/NA dipole asymmetry (full investigation).
- **Cube_ES_DeltaRho_NA_Codex_handoff_2026-07-24.md** — Short handoff: element-invariant clamp + node project; awaiting USER visual.
- **Fukui_FDBM_panel_notes_2026-07-23.md** — Fukui panel: cube vs DFTB stock/prolonged; ES asymmetry; df↔Fz height shift. Status: investigating.
- **(Global) Caveats.md** — recurring traps: aniso cubes, all-e Δρ/NA, contact-surface `h₀`/sampling, corner vs center, sample vs project.

**Planned / task-backed:** site-resolved Pauli maps (`doc/Tasks/Pauli_A_beta_KrigingTransferability.md`); Kekulé RI density (`doc/Tasks/Kekule_ExponentialDensityFit.md`); 2.5D contact-surface parity + assembly AFM (`doc/Tasks/Fast_2p5D_AFM_ContactSurface.md`, `doc/Tasks/Assembly_AFM_Pipeline.md`; report `doc/Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md`).

**Pending FDBM molecule panel (densities ready):** pySCF PBE/def2-SVP under `/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/` — pentacene, PTCDA, azaindol_(iso)dimer, benzoicacid/amid dimers → cube FDBM vs DFTB stock vs prolonged (`doc/Tasks/ProlongedRadialBasis_DFTB.md`).
