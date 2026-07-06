# tests/surfaces/

Surface GridFF construction and contact-surface diagnostics.

## Files

| Script | Class | Purpose |
|--------|-------|---------|
| `ocl_GridFF_new.py` | 2 | Build GridFF from atomic system, sample at arbitrary points, fit B-spline basis, plot 1D cuts. Handles Morse (Pauli/London) and Coulomb potentials with Ewald summation. Saves PLQ grids to `data/` for reuse. |

**Related (repo root):** `tests/testplot_contact_surface.py` — GPU separable B-spline×poly + PIC vs brute Morse on single PTCDA; artifacts in `debug/testplot_contact_surface/`.
