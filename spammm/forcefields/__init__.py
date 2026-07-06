"""__init__.py — Force field implementations for molecular simulations.

Contains UFF (**UFF_cl**, **UFFbuilder**) and SPFFsp3 (**SPFF_cl**, **SPFFbuilder**)
OpenCL force fields, **QEq** charge equilibration, **RigidBodyDynamics** for
6-DOF rigid body GPU simulation, **RigidBodyAFM** for AFM scanning with rigid
molecules, **Assembly** for GPU-accelerated molecular placement on surfaces,
and **FFController** as the pure-logic orchestrator bridging AtomicSystem to
forcefield build and GPU relaxation.
"""
