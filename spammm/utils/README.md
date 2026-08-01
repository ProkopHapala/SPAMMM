# utils/

Shared OpenCL infrastructure and test utilities. All GPU modules inherit from OpenCLBase.

- **OpenCLBase.py** — Base class for all PyOpenCL computations: device selection, kernel loading/caching, buffer management, host-device transfer, retained NDRange binding, optional eventless in-order enqueue
- **clUtils.py** — OpenCL device selection by vendor (NVIDIA, AMD, Intel), GridShape/GridCL helpers for 3D grid dimensions
- **Lingebra_ocl.py** — Batched eigendecomposition of small symmetric matrices via parallel Jacobi rotations in local memory
- **test_utils.py** — Shared test helpers: reference potential computation, 1D plotting, RMS error calculation for parity checks
- **ocl_init_old.py** — Deprecated OpenCL init module (superseded by OpenCLBase)

## Device selection (NVIDIA first)

`OpenCLBase` calls `select_device(preferred_vendor='nvidia')` by default: scan platforms, pick the first device with `"nvidia"` in name/vendor. PoCL/CPU is only a fallback when no NVIDIA device is visible to **that process**.

Policy note: [`doc/AGENTS/notes/opencl-nvidia-device.md`](../../doc/AGENTS/notes/opencl-nvidia-device.md). Agent rule: `.cursor/rules/opencl-nvidia-gpu.mdc`.

Quick check: `python -c "import pyopencl as cl; print([(p.name,[d.name for d in p.get_devices()]) for p in cl.get_platforms()])"` — must show NVIDIA for GPU work.
