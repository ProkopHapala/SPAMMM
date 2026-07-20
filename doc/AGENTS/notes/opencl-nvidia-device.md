# OpenCL device selection (NVIDIA policy)

**SSOT code:** `spammm/utils/OpenCLBase.py` → `select_device()`  
**Agent rule:** `.cursor/rules/opencl-nvidia-gpu.mdc`

## Short answer

SPAMMM prefers the **NVIDIA GPU**. Selection is automatic via `preferred_vendor='nvidia'` in `OpenCLBase`. PoCL (CPU) is only a fallback when no NVIDIA device is visible to the process.

## How it works

```
pyopencl.get_platforms()
        │
        ▼
scan devices; keep those with "nvidia" in name or vendor
        │
   found? ──yes──► Context([that device])   ← RTX 3090 here
        │
        no
        ▼
fallback: device index 0  (often PoCL CPU — avoid for real benches)
```

On a normal desktop session here:

| Platform index | Name | Device |
|----------------|------|--------|
| 0 | NVIDIA CUDA | GeForce RTX 3090 |
| 1 | Portable Computing Language | Ryzen CPU (PoCL) |

## Why agents sometimes hit PoCL

Cursor’s **command sandbox** can hide the NVIDIA OpenCL ICD. The GPU is still on the machine; the sandboxed Python process simply does not see it. Then `select_device` finds no NVIDIA and falls back to PoCL.

**Fix:** run OpenCL commands with unrestricted permissions (`required_permissions: ["all"]`). Then NVIDIA appears and is selected.

This is not “harness shows PoCL but we magically use NVIDIA.” Same machine, two process environments:

- sandboxed process → PoCL only → wrong
- unrestricted process → NVIDIA visible → correct

## Quick check

```bash
python -c "import pyopencl as cl; print([(p.name,[d.name for d in p.get_devices()]) for p in cl.get_platforms()])"
```

Must list NVIDIA before trusting GPU timings.
