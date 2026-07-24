# demos/

Interactive and headless demos for SPAMMM physics that are meant to be **run by users**, not pytest regression.

Start here for PairFF (rigid-body H-bond / σ-hole docking, optional FAF substrate):

| File | What |
|------|------|
| **[PairFF_manual.md](PairFF_manual.md)** | User manual — concepts, controls, CLI, mixed molecules, FIRE vs MD, `--faf` map |
| **[demo_pairff.py](demo_pairff.py)** | Entry: classic 1+1, N-body allmol, mixed XYZs, optional NaCl FAF + Vispy |

```bash
# Interactive (FIRE ON) — click any molecule to make it mobile
python3 demos/demo_pairff.py --bodies 4 --active 0
python3 demos/demo_pairff.py --mols PTCDA.xyz HCOOH.xyz formamide.xyz --spacing 12

# On NaCl: dynamics PairFF+FAF; background map = PairFF(env) + FAF(probe) @ CoM z
python3 demos/demo_pairff.py --bodies 4 --faf

# Headless FIRE
python3 demos/demo_pairff.py --bodies 4 --active 2 --no-vis --steps 300
python3 demos/demo_pairff.py --bodies 4 --faf --no-vis --steps 80
```

Fit cache (created on first `--faf` if missing): `data/fits/hcooh_nacl.npz`.

Developer notes: [`doc/Tasks/PairFF_MultiBody_Kernel.md`](../doc/Tasks/PairFF_MultiBody_Kernel.md) · [`doc/Tasks/PairFF_FAF_Substrate.md`](../doc/Tasks/PairFF_FAF_Substrate.md) · audit [`doc/TopicalAudit/PairFF_RigidBody.md`](../doc/TopicalAudit/PairFF_RigidBody.md).
