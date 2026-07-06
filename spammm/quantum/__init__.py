"""__init__.py — Quantum chemistry integration for electron density computation.

Provides **DFTB_utils.py** for DFTB+ subprocess/C-API integration and
**pySCF_utils.py** for pySCF RHF/DFT calculations. The **DFTB/** subpackage
contains the ctypes wrapper (**DFTBcore**), basis parser (**DFTBplusParser**),
GPU density projection (**Grid_dftb**), and basis optimizer (**basis_optimizer**).
Densities feed into the FDBM AFM method in the SPM package.
"""
