# demos/

Interactive and headless demos for SPAMMM physics that are meant to be **run by users**, not pytest regression.

Start here for PairFF (rigid-body H-bond / sigma-hole docking):

| File | What |
|------|------|
| **[PairFF_manual.md](PairFF_manual.md)** | User manual — concepts, controls, CLI, mixed molecules, FIRE vs MD |
| **[demo_pairff.py](demo_pairff.py)** | Runnable entry: classic 1+1, identical N-body, or mixed XYZs + Vispy |

```bash
# Interactive (FIRE ON by default) — click any molecule to make it mobile
python3 demos/demo_pairff.py --bodies 4 --active 0
python3 demos/demo_pairff.py --mols PTCDA.xyz HCOOH.xyz formamide.xyz --spacing 12

# Headless FIRE relax
python3 demos/demo_pairff.py --bodies 4 --active 2 --no-vis --steps 300
```

Developer notes / kernel design: [`doc/Tasks/PairFF_MultiBody_Kernel.md`](../doc/Tasks/PairFF_MultiBody_Kernel.md) · topical audit [`doc/TopicalAudit/PairFF_RigidBody.md`](../doc/TopicalAudit/PairFF_RigidBody.md).
