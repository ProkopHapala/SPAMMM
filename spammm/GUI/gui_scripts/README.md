# Control scripts (`gui_scripts/`)

One-command Python scripts launched by the GUI after startup via
`./run_gui.sh --script spammm/GUI/gui_scripts/NAME.py [-- extra args]`.
They programmatically click the same buttons a user would click, so they are
not headless; use `rc_scan_offline.py` if you need the same pipeline without Qt.

## File index

- **folded_rigid_setup.py** — Load a molecule, fit or load a folded substrate
  potential, and start the `FoldedRigid` extension in run or manip (drag) mode.
  - Arguments: `--mol`, `--substrate`, `--fit`, `--fit-fit`, `--run`, `--step`,
    `--manip`, `--n`, `--dt`, `--x`, `--y`, `--z`, `--k`
  - Example: `./run_gui.sh --script spammm/GUI/gui_scripts/folded_rigid_setup.py -- --mol data/xyz/H2O.xyz --fit data/fits/h2o_nacl.npz --run`
- **rc_scan_review.py** — 2Quinolone symmetric pm-NEB relaxed review; `--preview` loads cached npz
- **rc_scan_offline.py** — Same reaction-coordinate path without Qt (`PYTHONPATH=. python3 …`)

## Adding a new script

1. Implement `run(window, argv=None)` returning the result of the setup.
2. Use helpers from `spammm.GUI.gui_script_utils` (`load_molecule`, `set_spin_value`, `set_edit_mode`, etc.).
3. Register any script-specific defaults in `SPAMMM_GUI` only if the extension
   does not already provide them.

## Related

- `spammm/GUI/gui_script_utils.py` — shared helper utilities
- `spammm/GUI/SPAMMM_GUI.py` — main window, `--script` / `-s` runner
