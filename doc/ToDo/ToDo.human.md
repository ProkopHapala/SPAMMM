
1) stable cassette rots for coarse grid ptcda (https://graphics.cs.utah.edu/research/projects/stable-cosserat-rods/)
2) implement PPAFM kiring interactions 
3) Monte Carlo layer Pauli master equations and charge rings 
4) prolonged radial basis with DFTB+ (both STM and AFM)
5) Dyson orbital with DFTB+?
6) make a little pyOpenCl weave or jit or codon driver for forcefield optimization?


# GUI

- [ ] 3D view
- [ ] Molecuar Browser


# Problems Found

- [ ] Relaxation is still slow, why ? (SPFF/UFF)
- [ ] Need to make sure I can open GUI in specific folder (save/load in that folder)
- [ ] AFM panel in SPAMM_GUI does not show any wiggets in sub-panes Parameters, Visualization, STM/Orbitals




# QUICK

- [x] export kekule structures
- [ ] after load of molecule hydrogens are not connected
- [x] QEq
- [x] Molecule fragments/groups for easy edit and manipulation.  Automatic search for bridges
- [x] decouple `ascii_art_heterocycle.py` from `KekulePure.py`


# longer

- [ ] reoranize the editor menu - a but overcomplicated and slow, key-sortcuts
- [ ] 3D view