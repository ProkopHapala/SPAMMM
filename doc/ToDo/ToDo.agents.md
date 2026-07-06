# Agent ToDo Index

High-level pointers for agent-driven work. Detailed design lives in linked docs — do not duplicate here.

---

## AFM Contact Surface (quasi-2D representation)

Two related problems, **separate designs**, shared spatial infrastructure:

| Priority | Problem | Design doc | Summary |
|----------|---------|------------|---------|
| **Phase 1 — now** | Memory-efficient **static** AFM (rigid sample, classical FF, ND path) | [doc/Topics/AFM/ContactSurface_Static.md](../Topics/AFM/ContactSurface_Static.md) | **Prototype:** `spammm/surfaces/ContactSurface.py` + `kernels/contact_surface.cl` (separable B-spline×poly + PIC). Remaining: AFMulator integration, pytest L0 parity. |
| **Phase 2 — later** | **Elastic** AFM (flexible sample, stiffness/compliance) | [doc/Topics/AFM/ContactSurface_Elastic.md](../Topics/AFM/ContactSurface_Elastic.md) | Extend Phase 1 with h(x,y), K_z, ∇h (Winkler rubber surface), coarse-mesh indentation solves (Jacobi/PBD). Builds on static discretization. |

**External context:** FireCore brainstorming — `FireCore/doc/Topics/AFM/IndentationForce2D.chat.md`

**Existing code to inventory before implementing:**

- `spammm/surfaces/ContactSurface.py` — **implemented** GPU prototype (start here for extensions)
- `kernels/contact_surface.cl` — brute, separable CG, PIC kernels
- `tests/testplot_contact_surface.py` — visual diagnostic
- `spammm/SPM/AFM.py` — `AFMulator`, 3D grid path (integration target)
- `kernels/AFM.cl` — `interpFE`, `relaxStrokesTilted`
- `kernels/surface.cl` — FAF tensor kernels (`getSurfFolded_tensor_exp/poly`)
- `spammm/forcefields/SPFF_cl.py` — `fit_folded_surface_basis`
- `spammm/surfaces/GridFF.py`, `Ewald2D.py` — reference accuracy
- `tests/testplot_folded_surface_scan.py`, `tests/test_surface.py` — parity patterns

**Validation:** `doc/TEST_DESIGN.md` (L0 assert, L1 `.out`, L2 `.png`)

---

## Other agent-relevant docs

| Topic | Doc |
|-------|-----|
| AFM/STM overview | `doc/afm_stm_simulation.md` |
| Surface / GridFF / FAF | `doc/surface_interactions.md` |
| Folded basis + rigid body gap | `doc/Tasks/RigidBodyDynamicsWithFoldedBasisSubstrate.md` |
| Test conventions | `doc/TEST_DESIGN.md` |
| Repo map | `CODEMAP.md` |
