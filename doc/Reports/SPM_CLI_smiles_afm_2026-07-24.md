---
type: Report
title: SMILES → planar opt → prolonged AFM (CLI)
tags: [SPM, AFM, SMILES, CLI]
timestamp: 2026-07-24
---

# Report: `smiles-afm` pipeline (2026-07-24)

**Status:** investigating — USER reviewed amp-aligned strips as good; science/opt quality not formally signed off.  
**CLI:** `python run_spm.py smiles-afm` · guide: [`user_guide/SPM_CLI.md`](../../user_guide/SPM_CLI.md) · task: [`SPM_CLI_Headless.md`](../Tasks/SPM_CLI_Headless.md)

## What it does

```
SMILES_EXAMPLES / --smiles
        ↓  smiles_to_system (pure OpenSMILES)
   optimize_vacuum (UFF|SPFF|LFF|DFTB)
        ↓  make_planar_xy → all z equal; orientPCA long→x
   prolonged FDBM AFM (CO tip) + atom-dot overlay
```

Gallery: `debug/spm_smiles_afm/<name>/` — `{name}_opt.xyz`, **`compare_per_column.png`**, **`stage_prolonged.png`**.

## Presentation defaults (USER OK for cosmetics)

| Knob | Default | Meaning |
|------|---------|---------|
| df window | 3.7–4.7 Å @ dz=0.1 | `--h-min/--h-max/--h-step` (aliases `--zmin/--zmax/--dz`) |
| Fz panels | 2.7–3.7 Å | amp-align: Fz @ **h − amp** (amp=1.0) |
| Plots | `compare,stage` | `--plots all` for tip/df/Fz/per_image |
| Geometry | planar + PCA | `--no-planar` / `--no-orient` to skip |

Planarity is forced (`zspan=0` on saved xyz). Non-flat-looking AFM features are tip/field physics, not atom heights.

## Open (do not mark done)

- UFF often stops at high `fmax` on some aromatics — images usable; opt quality TBD
- ASCII / `.mol`/`.mol2` shared CLI flags; GUI SMILES box
- Fukui `panel-fukui` still on older height/contrast conventions (see Fukui notes §2)
