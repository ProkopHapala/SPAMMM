"""__init__.py — Scanning Probe Microscopy (SPM) simulation package.

Core AFM simulator (**AFM.py**) with PyOpenCL GPU kernels for tip-sample interactions.
High-level orchestration and FDBM workflow in **AFM_utils.py**. Staged pipeline
with disk caching in **ModularPipeline.py**. DFT z-scan → GridFF in **KrigingGridFF.py**
(Kriging/RBF). Manipulation path optimization via evolutionary algorithms in
**ManipulationPathOpt.py**. Trajectory generators for scan patterns in **ScanUtils.py**.
"""
