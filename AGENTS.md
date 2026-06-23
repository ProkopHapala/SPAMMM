We develop rigorous scientific software where debuggability, physical consistency, numerical correctness and stability, as well as performance and simplicity are paramount. Follow these principles:

## Core Principles

- **KISS** (Keep It Simple), Simplest solution that works. One-liner > ten-liner.

- **AHA** (Avoid Hasty Abstractions), avoid boilerplate
- **YAGNI** : **Surgical Edits** — Touch only what's needed. No unrelated cleanup. Comment out, don't delete. Ask if ambiguous.
- **DRY** : Inventory existing code before writing new. Generalize rather than duplicate. 
- **SoC** (Separation of Concerns), separate module for Compute, plotting, Backend, CLI, GUI. Thin test scripts call general workhorse function from shared modules.
- **SSOT** : Authoritative single source of truth must be defined to avoid ambiguity and confusion
- **TDD** : Define verification before coding. Parity checks vs reference/analytical/physical invariants. Run tests after every change. See `numerical-parity/SKILL.md`.
- **Fail Fast** : No silent fallbacks (try-catch). Crashes with stack traces > masked bugs. Look for root cause, not symptoms.
- **Performance** : preallocate, minimize python orchestration; to C/C++/OpenCL kernels. Data-oriented-desing: Flat arrays, cache-aware, usel local memory in OpenCL. See `port-to-opencl/SKILL.md`.
- Compact code, unlimited line lengh (function call must be one line).  Short names for math symbols (`E_tot`, `T_ij`).



