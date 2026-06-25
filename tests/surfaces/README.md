# tests/surfaces/

Surface GridFF construction and sampling utilities using OpenCL.

## Files

| Script | Class | Purpose |
|--------|-------|---------|
| `ocl_GridFF_new.py` | 2 | Build GridFF from atomic system, sample at arbitrary points, fit B-spline basis, plot 1D cuts. Handles Morse (Pauli/London) and Coulomb potentials with Ewald summation. Saves PLQ grids to `data/` for reuse. |
