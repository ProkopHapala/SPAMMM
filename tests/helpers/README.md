# tests/helpers/

Shared utility modules imported by Class 1 (pytest) and Class 2 (standalone) scripts.

## Files

| Module | Purpose |
|--------|---------|
| `parity.py` | `rmse`, `max_err`, `correlation`, `dir_cosine`, `plot_curves`, `overlay_plot`, `assert_parity` |
| `geometry.py` | `bond_lengths`, `bond_angle`, `planarity`, `distort`, `find_bonds`, `save_xyz_frames`, `plot_geometry` |
| `scan.py` | `z_scan`, `x_scan`, `compare_scans`, `assert_scan` — 1D scan runners with parity comparison |
| `folded_rigid.py` | Rigid body setup on folded basis: `setup_rigid_folded`, `relax_folded`, `relaxed_scan`, manipulation plots, reference data system (`save_reference`, `compare_to_reference`), `replicate_substrate` |
