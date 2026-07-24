# Reports

Validated and review-ready scientific results, with exact methods, artifacts, reproduction commands, and unresolved caveats. Task documents may evolve during investigation; reports here are the durable handoff.

- **Assembly_ContactSurface_AFM_helicene_2026-07-24.md** — Helicene SAM assembly → contact-sep AFM; PBC/overlay fixes; contact-sep vs GridFF Morse+Coulomb mismatch (open); links PTCDA parity harnesses.
- **PTCDA_NaCl_Rigid_Newton_FIRE_Relaxation.md** — Why aggressive rigid Newton creates noisy basin selection, how staged LM damping restores Newton–FIRE consistency, and practical solver tradeoffs.
- **PTCDA_FDBM_prolonged_basis.md** — Stock 3ob versus SA-prolonged PTCDA density tails, Pauli fitting, and AFM images.
- **PySCF_GPU_CO_zscan_PTCDA.md** — Site-correct GPU pySCF CO–PTCDA interaction scans, timing, reference data, and limitations.
- **Kriging_DFT_vs_DFTB_FDBM_pyridine.md** — Pyridine+CO: Kriging DFT GridFF vs FDBM from cubes and DFTB; Δρ recipes vs basis tails; dual-basis Slater rules; plotting SSOT; session history (investigating).
- **Kriging_FDBM_PauliFit_pyridine_2026-07-21.md** — Day log: contact vs residual Pauli \(A,\beta\) fits; N+C vs N+C+H; tip×sample matrix; open cube-ES caveats.
- **STM_ExtendedBasis_OrbitalCompare.md** — STM HOMO/LUMO: mio/3ob stock vs prolonged vs pySCF (awaiting USER review).

- **Fukui_FDBM_panel_notes_2026-07-23.md** — Fukui panel: cube vs DFTB stock/prolonged; ES asymmetry bisect (Δρ/NA origin, dipole_origin_bisect); df↔Fz height shift. Status: investigating.
- **(Global) Caveats.md** — recurring traps: all-e ρ−crude NA → fake multipoles; corner vs center; scipy sample vs project.

**Planned / task-backed:** site-resolved Pauli maps (`doc/Tasks/Pauli_A_beta_KrigingTransferability.md`); Kekulé RI density (`doc/Tasks/Kekule_ExponentialDensityFit.md`); 2.5D contact-surface parity bisect + assembly AFM (`doc/Tasks/Fast_2p5D_AFM_ContactSurface.md`, `doc/Tasks/Assembly_AFM_Pipeline.md`; report `doc/Reports/Assembly_ContactSurface_AFM_helicene_2026-07-24.md`).

**Pending FDBM molecule panel (densities ready):** pySCF PBE/def2-SVP under `/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/` — pentacene, PTCDA, azaindol_(iso)dimer, benzoicacid/amid dimers → cube FDBM vs DFTB stock vs prolonged (`doc/Tasks/ProlongedRadialBasis_DFTB.md`).
