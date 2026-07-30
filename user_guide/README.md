# SPAMMM user guide

End-user documentation (how to run simulations **without** digging through `tests/` or opening the full molecular editor).

| Doc | Topic |
|-----|--------|
| [SPM_CLI.md](SPM_CLI.md) | Headless AFM/STM CLI — **`run_spm.py`** (FDBM, Morse, Kriging, `opt`, `smiles-afm`, STM orbitals/current/panel) |
| [RigidAssembly_GUI.md](RigidAssembly_GUI.md) | **Rigid Assembly** GUI extension — drag, MC/GA optimization, PME charge-ring STM (Build from file or from editor-drawn fragments) |
| [`../demos/PairFF_manual.md`](../demos/PairFF_manual.md) | **PairFF** rigid-body H-bond docking (Vispy, FIRE, multi-body, optional `--faf` NaCl map) |

**Repo entry points:** root [`run_spm.py`](../run_spm.py) · GUI [`./run_gui.sh`](../run_gui.sh) · PairFF [`demos/demo_pairff.py`](../demos/demo_pairff.py) · task/gaps [`doc/Tasks/SPM_CLI_Headless.md`](../doc/Tasks/SPM_CLI_Headless.md) · agent ToDo [`doc/ToDo/ToDo.agents.md`](../doc/ToDo/ToDo.agents.md).

Developer / agent material (architecture, skills, campaign reports) stays under `doc/` and `doc/AGENTS/`.
