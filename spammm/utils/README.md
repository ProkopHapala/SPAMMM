# utils/

Shared OpenCL infrastructure and test utilities. All GPU modules inherit from OpenCLBase.

- **OpenCLBase.py** — Base class for all PyOpenCL computations: device selection, kernel loading/caching, buffer management, host-device transfer, kernel launch
- **clUtils.py** — OpenCL device selection by vendor (NVIDIA, AMD, Intel), GridShape/GridCL helpers for 3D grid dimensions
- **Lingebra_ocl.py** — Batched eigendecomposition of small symmetric matrices via parallel Jacobi rotations in local memory
- **test_utils.py** — Shared test helpers: reference potential computation, 1D plotting, RMS error calculation for parity checks
- **ocl_init_old.py** — Deprecated OpenCL init module (superseded by OpenCLBase)
