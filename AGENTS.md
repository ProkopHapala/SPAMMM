SPAMMM (Scanning Probe Accelerated Modeling of Microscopy and Manipulation) is a scientific software project where numerical correctness and stability, as well as performance and simplicity are paramount.

## Core Principles

- **KISS** (Keep It Simple), Simplest solution that works. One-liner > ten-liner.
- **AHA** (Avoid Hasty Abstractions), avoid boilerplate
- **YAGNI** : **Surgical Edits** — Touch only what's needed. No unrelated cleanup. Comment out, don't delete. Ask if ambiguous.
- **DRY** : Inventory existing code before writing new. Generalize rather than duplicate. See skill:`code-reuse`.
- **SoC** (Separation of Concerns), separate module for Compute, plotting, Backend, CLI, GUI. Thin test scripts call general workhorse function from shared modules.
- **SSOT** : Authoritative single source of truth must be defined to avoid ambiguity and confusion. For molecular topology, `AtomicGraph` is the SSOT — all other representations (rendering, export, FF) derive from it. See skill:`molecular-structure-sync`.
- **TDD** : Define verification before coding. Parity checks vs reference/analytical/physical invariants. Run tests after every change.
- **Fail Fast** : No silent fallbacks (try-catch). Crashes with stack traces > masked bugs. Look for root cause, not symptoms.
- Compact code, unlimited line lengh (function call must be one line).  Short names for math symbols (`E_tot`, `T_ij`).

## Never Do This

- Never delete or rearrange existing code without explicit permission
- Never perform unrelated aesthetic/style edits
- Never apply quick-fixes that hide root causes (e.g. hard-coded outputs)
- Never reinvent functionality already implemented. Inventory fist, check the provided examples and references, base classs etc.
- Never copy-paste between apps — extract to shared lib and include.
- **Ask, don't guess** — when you encounter problem which where you are not sure, ask the user instead of trying to infer it.
- **NEVER mark an issue as "fixed", "resolved", or "done" without explicit USER confirmation.** This applies to bug reports, task documents, ToDo items, and any status tracking. A code change is NOT proof of a fix. You must: (1) run a test or verification that demonstrates the fix, (2) show the result to the USER, (3) wait for USER confirmation before updating any status field. Violating this rule is considered a critical error. When in doubt, leave the status as "investigating" or "unverified".
- **in devin always use deving tools to adit files, never** (`python3 << 'PYEOF'` heredocs, `sed -i`, `cat >`, `echo >>`, shell redirects). ALWAYS use the Devin `edit`/`write`/`read` tools so changes appear in the IDE diff viewer for USER review. Shell-based edits are invisible to the user and cannot be reviewed or reverted easily.

## Debugging & Testing

**Fail loud** — crashes with stack traces > masked bugs. Debug prints gated by verbosity. Tests in `tests/` with ref_data regression. Numerical correctness via parity checks vs reference/analytical/physical invariants. See skill:`running-tests`, skill:`visual-debugging`, skill:`reference-data`, skill:`gpu-debug`, skill:`numerical-parity`, skill:`forcefield-validation`.

**Test system SSOT:** `doc/TEST_DESIGN.md` — three review levels (L0 assert, L1 agent `.out`/`.log`, L2 human `.png`).

- **Routine:** `pytest -m "not slow"` — L0 only, fast regression.
- **Develop:** `pytest path --develop -s` — L0+L1+L2; agent must read `.out` artifacts and **never filter test stdout** (`| tail`, `| grep`, `&` forbidden).
- **Artifacts:** `debug/<script>/` (gitignored except `debug/README.md`). Agents read by explicit `REVIEW:` path from stdout.
- **Test scripts**: `test_*.py` (pytest), `testplot_*.py` (visual demos), `run_*.py` (CLI). See skill:`running-tests`.
- **Refactoring discipline**: Before refactoring, run each old file, show plots/results to USER for review. Identify useful features from each version. Reproduce carefully. Only delete old files after explicit USER approval. Never delete plots — they are the main results the USER reviews.

## Performance & Languages

* **GUI layout: maximally tight.** Zero margins, minimal spacing, widgets only as wide as their content (Maximum size policy). Side panel width-limited, does not push canvas. As many controls as possible in limited area — no wasted space. See `doc/Tasks/GUI_TightLayout.md`.

* Minimize Python orchestration; push compute to OpenCL kernels. Flat arrays, cache-aware, preallocate. See skill:`python-perf`, skill:`port-to-opencl`.
* GPU/OpenCL : memory latency, gather over scatter, local memory, minimize host-device transfers. See skill:`gpu-optimize`.
* **OpenCL device:** always prefer **NVIDIA GPU** (`OpenCLBase.select_device(preferred_vendor='nvidia')`). Never report PoCL/CPU timings as GPU. Agents must run OpenCL Shell commands unrestricted (`all`) so the NVIDIA ICD is visible — see `doc/AGENTS/notes/opencl-nvidia-device.md` and `.cursor/rules/opencl-nvidia-gpu.mdc`.
* **Python is the harness, not the engine** (skill:`python-perf`). NEVER write hot loops in Python — batch via NumPy or push to OpenCL. Per-atom `ax.scatter` loops, `for r in range(n_trial)` quaternion math, per-trial distance checks are all violations. If a loop is unavoidable, profile (`cProfile`) and confirm <1% of runtime.
* **Simulation code lives in pyOpenCL kernels** (skill:`gpu-optimize`, skill:`port-to-opencl`). Kernels must be well-parallelized: use workgroups, local memory, minimize host-device transfers and kernel launch overhead. Python only orchestrates.
* **Fuse secondary checks into existing kernels** (skill:`gpu-optimize` § "Fusing Secondary Checks"). If a kernel already computes a distance/overlap, add clash/collision flags in the same loop — never recompute on host. Reuse reserved `float4.w` channels for secondary results. Only check active-vs-partner pairs (frozen-frozen is invariant). Use active-set incremental updates (`E += ΔE_active`) on host — see skill:`python-perf` §6.
* **Long-running scripts MUST print unbuffered progress** (`flush=True` or `PYTHONUNBUFFERED=1`). NEVER run silently for minutes — print what is starting, accepted steps with energy decrease, and when finished. The user will not wait for scripts with no output.

## Documentation & Navigation

- Before writing: search existing implementations (skill:`doc-read-navigate`)
- After implementing: update README.md + topical audits (skill:`doc-task-summary`)
- Dedicated doc work: OKF format, extract/inline workflows (skill:`doc-audit`)
- `doc/topical_audit.md` — cross-implementation maps per scientific topic
- `doc/Caveats.md` — recurring scientific/numerical traps (all-e Δρ/NA multipoles, grid conventions)
- `doc/AGENTS/skills/` — all skills; `doc/AGENTS/protocols/` — domain protocols
- `README.md` per folder — local index; `CODEMAP.md` — repo structure
- Visualization: separate compute from plotting; `plt.show()` only in CLI/main
- **AFM E/Fz/df plots (SSOT):** never ad-hoc `imshow`/`plot` for energy, force, or df maps/z-profiles — use `spammm.SPM.AFM_utils` (`imshow_afm`, `plot_afm_height_panel`, `plot_afm_z_profiles`, `save_afm_images`, `plot_grid_Fz`). **FDBM vs Kriging / tip×sample z-diagnostics:** `plot_fdbm_vs_kriging_zlayout`, `plot_fdbm_methods_zcompare_4panel`, `fdbm_probe_sites_nch` (E−E(6), V−V(8); top=sites overlapped, bottom=per-site channels). See skill:`afm-plotting`. Generic 2D scalars → skill:`centralized-plotting` (`plotUtils.plot_2d_scalar`).
