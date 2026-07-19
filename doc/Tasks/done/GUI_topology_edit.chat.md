# USER

OK, now we have to make comprehesive GUI editor in this repo, I jsut renamed to this
/home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py

we should make nice shaell script in the root which run it with CLI options, make sure that when is save images it does it by default to 
/home/prokop/git/SPAMMM/output folder not to the root of this repo, 

OK, I'm testing it now, but I realized several things are missing @ToDO_GUI.md 

The most crucial thing now is 
1) right click on atom shoud delete it, but it does not always do it. Sometimes is actually remove other atoms somethigm it does nothing. This is very persistent problem which we were already resolving several times and always pop up again. I think the reason is we do not have really robust system of managing an sychronziing topolgy. We introduced some unique IDs but it seems the sysem still does not rely on it. It is a mess. We cannot do anything before this is totally soldia and waterproof. No add hoc patch, but rigrous system desing

you rushed to implementation but we did not discussed the desing options. Let me undestand clearly why using indexes is problem? I know that if we remove atom, the index may be no longer valid. But if we update the mapping, it should still work. 

I insisted you have to thing about both robustness and efficiency, and I wanted to discuss pros and cons to make desing decission, not rush to implemetation. Since you already did think about how to make it efficient. 

It seems to me quite wastefull if you remove or add atom to rebuild whole topology and neighbor list again. The molecular graph should allow to do just local operation.

Also we can have "dead" atoms, like invalid slots which were deleted but were not yet pruned from the array. As long as we do not prune the indexes are valid (does not change) we need to reindex the neighbor list only when pruning. Does it make sense? 

I was thinking we already implemeted this system, but maybe it is only hafl done and not used?
Would we face some problem if we move to this system of "dead" atoms which seems to me most performance efficient ?

do ot code rather analyse the system deeply and answe my questions so I can do informed desing decission.

---

# GLM 5.2

Good questions. Let me analyze the system deeply before discussing design options.




Now let me check what [_rebuild_after_delete](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:215:4-227:24) actually does:

Now let me check how [remove_h_caps](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:831:4-837:36) + [add_h_caps](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:839:4-883:28) works in [adjust_h](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:885:4-897:78) — this is the key source of index instability:

Now let me check the [atom_subtype](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:169:4-172:45) property and [atom_pin](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:157:4-161:41) property to understand what other arrays are derived:

Now I have a thorough understanding. Let me analyze the system deeply.

---

## Deep Analysis: Why Indices Break, and Three Design Options

### What the current system already has

The [AtomicGraph](cci:2://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:117:0-404:104) **already** implements the "dead atom" pattern you're describing:

- `Atom.alive` flag (`@/home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:46`)
- [remove_atom(atom, soft=True)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:138:4-155:36) marks `alive=False` — O(1), no renumbering (`@/home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:139-149`)
- [cleanup_invalid()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:157:4-183:73) actually prunes dead atoms from the dict (`@/home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:158-184`)
- [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75) filters to only alive atoms (`@/home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:321`)

**But this system is only half-used.** The problem is in the bridge layer — [KekuleBackend](cci:2://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:118:0-1955:95) and the scene.

### Why indices break: the real root cause

The issue is **not** that indices are inherently bad. It's a **specific sequence of events**:

1. Scene calls [set_data()](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/VispyUtils.py:297:4-324:22) → copies `backend.sys.apos` into `_pos`. Scene index `i` ↔ atom at `sys.apos[i]`.
2. User right-clicks → scene finds nearest index `i` → emits `sig_rmb_remove(i)`.
3. [on_atom_remove(i)](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:870:4-876:40) calls [backend.remove_atom_by_index(i)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:623:4-634:27).
4. [remove_atom_by_index](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:623:4-634:27) → [_rebuild_after_delete([i])](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:215:4-227:24) → [graph.remove_atom(atom_list[i])](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:138:4-155:36) → [_sync_sys()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:239:4-252:35) → [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75) rebuilds arrays **without the deleted atom**.
5. **If `auto_h_cap` is on**: [adjust_h()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:883:4-895:78) runs → [remove_h_caps()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:829:4-835:36) marks all H atoms dead → [cleanup_invalid()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:157:4-183:73) prunes them → [add_h_caps()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:837:4-881:28) creates new H atoms. **The entire array is restructured.**
6. [refresh_view()](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:1116:4-1234:46) updates the scene from the new `sys.apos`.

The index `i` was valid at step 2. By step 4, [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75) is called again inside [_rebuild_after_delete](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:215:4-227:24), so `atom_list[i]` inside that call is the correct atom. **The deletion itself is actually correct for single atoms.**

**Where it goes wrong:**

- **[delete_selected_atoms](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:712:4-720:51)**: Iterates `[3, 7, 12]`, calls [_rebuild_after_delete([3])](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:215:4-227:24) → arrays shrink → next call [_rebuild_after_delete([7])](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:215:4-227:24) uses a new [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75) where index 7 is now a different atom. This is the **batch deletion bug**.
- **[adjust_h()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:883:4-895:78) after deletion**: Adds/removes H atoms, changing array length and ordering. If anything uses a stale index between the deletion and the next [refresh_view()](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:1116:4-1234:46), it hits the wrong atom.
- **Stale `_pick_idx`**: The scene stores `_pick_idx` from mouse press. If topology changes before the next pick (e.g., another deletion), `_pick_idx` points to a different atom.

### Your "dead atom slot" idea — analysis

**What you're proposing**: Keep dead atoms in the arrays as invalid slots. Don't compact until necessary. Indices stay stable as long as you don't compact.

**This is partially what [AtomicGraph](cci:2://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:117:0-404:104) already does** — [remove_atom(soft=True)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:138:4-155:36) just flips `alive=False`. The problem is that [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75) immediately filters them out, and [_sync_sys()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:239:4-252:35) immediately rebuilds compact arrays.

**To make the "dead slot" approach work end-to-end**, you would need:

1. [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75) (or a new `to_arrays_with_gaps()`) returns **all** atoms (alive + dead), with an `alive_mask` boolean array.
2. `sys.apos` would have length = total atoms (including dead), with dead atoms at some sentinel position or just skipped in rendering.
3. The scene would use `alive_mask` to skip dead atoms in picking and rendering.
4. Bonds to dead atoms would be filtered by checking `alive_mask[bond[i]]`.
5. Compaction (pruning) happens only when the user explicitly requests it or when the dead fraction exceeds a threshold.

**Pros of dead-slot approach:**
- **O(1) deletion** — just flip `alive=False`, no array rebuild
- **Index stability** — indices don't change between operations
- **No mapping table needed** — index `i` is always valid (just check `alive_mask[i]`)
- **Local operations** — add atom = append to array; remove atom = flip flag; no global rebuild

**Cons / problems:**
- **Array growth** — dead slots accumulate. For interactive editing of ~100 atoms, this is negligible. For MD-style operations with thousands of add/remove cycles, could be problematic. But you can compact periodically.
- **Rendering complexity** — VisPy markers would render dead atoms too unless you mask them. You'd need to either: (a) set dead atom positions to NaN (VisPy skips NaN), (b) set size=0 for dead atoms, or (c) build a compact view array for rendering only.
- **Force field computation** — `AtomicSystem` (sys) is used for FF evaluation. FF code expects compact arrays with no gaps. You'd need to compact before FF runs, or make FF code skip dead atoms.
- **[adjust_h()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:883:4-895:78) complication** — [remove_h_caps()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:829:4-835:36) + [add_h_caps()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:837:4-881:28) currently does soft-delete + cleanup + create. With dead slots, you'd soft-delete H caps (flip `alive=False`), then add new ones (append). Old H caps stay as dead slots. This is fine but means H cap positions in the array are stale.
- **Bond array** — bonds reference atoms by index. If atoms are dead slots, bonds to dead atoms need to be masked too. Bond array would also grow with dead bonds.
- **[atom_subtype](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:169:4-172:45), [atom_pin](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:157:4-161:41), etc.** — all the derived properties in [KekuleBackend](cci:2://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:118:0-1955:95) call [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75) which rebuilds. They would need to work with the gapped array or be refactored to use `_id` lookups.

**Is this already implemented?** Half yes. The [AtomicGraph](cci:2://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:117:0-404:104) layer has the soft-delete mechanism. But:
- [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75) filters dead atoms out (compacts)
- [_sync_sys()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:239:4-252:35) calls [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75) (compacts)
- [cleanup_invalid()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:157:4-183:73) is called after every deletion in [remove_atom_by_index](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:623:4-634:27) (compacts)
- The scene and sys always work with compact arrays

So the "dead slot" system exists at the graph level but is **immediately compacted** at the backend level. The compaction is what causes index instability.

### Three design options compared

| Aspect | **Option A: ID-based** (what I started implementing) | **Option B: Dead-slot arrays** (your proposal) | **Option C: Index + rebuild mapping** (current, fixed) |
|---|---|---|---|
| **Deletion cost** | O(1) dict lookup + O(N) [_sync_sys](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:239:4-252:35) rebuild | O(1) flip `alive=False` | O(N) [_sync_sys](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:239:4-252:35) rebuild |
| **Index stability** | N/A — uses `_id` not index | Stable until compaction | Stable only within one operation |
| **Mapping table** | `_id_to_idx` dict, rebuilt per [set_data](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/VispyUtils.py:297:4-324:22) | `alive_mask` array, no mapping needed | Rebuilt per [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75) call |
| **Batch delete** | O(1) per atom, collect IDs first | O(1) per atom, flip flags | O(N) per atom, rebuild each time |
| **Rendering** | No change — compact arrays | Need to mask/skip dead atoms | No change — compact arrays |
| **FF computation** | No change — compact arrays | Need compaction before FF | No change — compact arrays |
| **[adjust_h()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:883:4-895:78)** | Works — IDs are stable | Works — old H caps become dead slots | Works but causes index shifts |
| **Memory** | No growth | Grows with dead slots until compaction | No growth |
| **Complexity** | Moderate — need `_id_to_idx` dict in scene | Higher — need alive_mask everywhere | Low — but fragile |
| **Code changes** | Scene + backend signals | Scene + backend + sys + FF interface | Minimal — fix batch delete only |

### My assessment

**Option B (dead-slot) is the most performant** for interactive editing — O(1) deletions, no rebuilds, stable indices. The [AtomicGraph](cci:2://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:117:0-404:104) already supports it. The main work is:
1. Add `to_arrays_with_gaps()` that returns all atoms + `alive_mask`
2. Make the scene use `alive_mask` for picking/rendering (set dead atom size=0 or pos=NaN)
3. Compact only when needed (before FF, or when dead fraction > 50%)

**Option A (ID-based) is the most robust** with least changes to the rendering/FF pipeline. Arrays stay compact, but we use `_id` as the stable identifier across module boundaries. The `_id_to_idx` dict is rebuilt once per [refresh_view()](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:1116:4-1234:46) — O(N) but N is small (~100 atoms).

**Option C (fix current)** would just fix [delete_selected_atoms](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:712:4-720:51) to collect all atoms first, then delete in one pass. But it doesn't solve the [adjust_h()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:883:4-895:78) index shift problem or stale `_pick_idx`.

### Key question for your decision

The real question is: **where does the compaction happen?**

- **Currently**: After every single deletion ([cleanup_invalid()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:157:4-183:73) + [_sync_sys()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:239:4-252:35)). This is what makes indices unstable.
- **Option A**: Keep compacting after every deletion, but use `_id` so it doesn't matter.
- **Option B**: Don't compact after every deletion. Only compact when needed (FF run, explicit request, or threshold). Indices stay stable between compactions.

Both are valid. Option B is more efficient but requires more changes to the rendering and FF interface. Option A is simpler to implement and keeps the rendering/FF pipeline unchanged.

**What would you prefer?** I can revert my Option A implementation if you want to go with Option B, or we can discuss a hybrid (e.g., dead-slot at graph level, but compact + `_id` mapping at the scene/backend boundary).

---

# USER

OK, 
1) I pefer performant option B, in the end I think it can be less complex as we do less oprations (just local opration not rebuiding graph every time). 
2) I do not want to modify the array export, the FF and Vispy should still use dense arrays (only alive atoms), but that is the point our Authoritative Topology graph is more dynamic and more flexible. The only think wich we need is simple maping index array for all atoms and all bonds maping index of the exported dense array back to AtomGraph and vice versa. such 4 index arrays (for atoms and bonds back and forth) should be easy to maintain. We beed to generate them only once when we export to dense array. no need to update them during topology operations. These are two decoupled taks - (topology operation of AtomGraph and export of dense array for visualization or forcefield optimization), is that clear? is that sound/rigorous/failproff desing?

Tis eleimiate rendering complexity. The vispy and FF does not need to care about alive/dead it see dense array, only when compuicating with AtomGraph (asiking atom-graph to do some topology operation, is simply remap the index O1)

But OK, maybe it is not efficient because every topology opration which change number of bonds or atoms we have to export that map O(N) but fast. However it is perhaps cheaper than doing costly id-seach every time. Perhaps the dradeoff is - how often we do search for index and how often we do adding/removing atom. I think we do search much more often than adding removing atom. Also t is important to make all the topology operations simple and be able to export dense array only when finished.

3) growing array is not really a problem, it grows anyway if it is 20% or even 100% bigger it is not really problem unless we have 1 million atoms or even more, which we will never have.

---

# GLM 5.2

Your design is sound. Let me articulate exactly why and identify the remaining issues.

## Your Design — Restated

```
AtomicGraph (authoritative, dynamic)
  └─ dict {_id: Atom}   ← soft-delete only (flip alive=False), no compaction during editing
  └─ add_atom = append to dict, O(1)
  └─ remove_atom = flip alive=False + flip bonds alive=False, O(degree)
  └─ cleanup_invalid() = called ONLY when explicitly needed (memory, never during editing)

Export (dense arrays, ephemeral)
  └─ to_arrays() → (alive_atoms, apos, enames, bonds, ...)   ← compact, only alive
  └─ atom_id_array[i]    = Atom._id at dense index i         ← dense→graph
  └─ atom_idx_map[_id]   = dense index for _id               ← graph→dense
  └─ bond_id_array[i]    = Bond._id at dense index i         ← dense→graph
  └─ bond_idx_map[_id]   = dense index for _id               ← graph→dense
  Generated once per export. O(N+B). Not maintained during topology ops.

GUI/FF/VisPy (consumers)
  └─ See only dense arrays — no dead atoms, no alive_mask
  └─ Picking: dense index i → _id = atom_id_array[i] → emit _id → graph does O(1) operation
  └─ Rendering: uses dense apos, no change
  └─ FF: uses dense apos/enames/bonds, no change
```

## Why This Is Sound

**Decoupling is the key insight.** Topology operations and array export are separate concerns:

- **Topology operation**: Modify graph directly. Soft-delete = flip flag. Add = append to dict. No array rebuild, no compaction, no mapping update. O(1) or O(degree).
- **Export**: When consumer needs dense arrays, call [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75). Builds compact arrays + 4 mapping arrays. O(N+B) but fast (numpy, ~100 atoms).

Between exports, the mapping is stale — **but nobody uses it**. The GUI stores `_id` (stable), not dense index (ephemeral). Only at the boundary (picking → emit `_id`, or `_id` → look up position in dense array) does the mapping get used, and it's always fresh because [refresh_view()](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:1116:4-1234:46) calls [_sync_sys()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:239:4-252:35) before the scene reads arrays.

**Batch delete becomes trivially correct**: Collect `_id`s from selection → for each: `graph.atoms[_id].alive = False` + mark bonds dead → one [_sync_sys()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:239:4-252:35) at end. No index shifting, no ordering issues.

**[adjust_h()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:885:4-897:78) becomes local**: [remove_h_caps()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:831:4-837:36) = flip `alive=False` on H caps (no [cleanup_invalid()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:157:4-183:73)). [add_h_caps()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:839:4-883:28) = append new H atoms to dict. Old dead H caps are ignored by [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75). One [_sync_sys()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:239:4-252:35) at end.

## What's Already Implemented vs What Needs Changing

**Already in AtomicGraph** (`@/home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py`):
- `Atom.alive` flag, `Atom._id` stable counter — ✅
- [remove_atom(soft=True)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:138:4-155:36) — flips `alive=False`, O(1) — ✅
- [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75) — filters to alive atoms, builds compact arrays — ✅
- [cleanup_invalid()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:157:4-183:73) — prunes dead atoms from dict — ✅ (but overused)

**Already in my partial implementation**:
- `_atom_ids` array in [_sync_sys()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:239:4-252:35) — this is `dense_to_graph_atom` — ✅
- `_id_to_idx` dict in scene — this is `graph_to_dense` — ✅
- [remove_atom_by_id()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:568:4-584:27) in backend — uses `graph.atoms.get(_id)` — ✅

**What needs changing** (3 key points):

1. **Stop calling [cleanup_invalid()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:157:4-183:73) after every deletion.** Currently called in:
   - [remove_atom_by_index()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:625:4-636:27) (`@/home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:631`)
   - [remove_h_caps()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:831:4-837:36) (`@/home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:835`)
   - [remove_atom_by_id()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:568:4-584:27) (my new code, `@/home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:580`)
   
   These should just soft-delete. [cleanup_invalid()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:157:4-183:73) should only be called explicitly (e.g., a "compact" button, or when dead fraction > threshold).

2. **Stop calling [sync_neighbor_lists()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:185:4-196:47) globally after every deletion.** Currently O(N+B), called after every single deletion. With soft-delete, neighbor lists on unaffected atoms are still valid. We only need to update neighbors of the deleted atom's neighbors. Or defer entirely — [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75) already filters dead bonds, and [h_children()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:198:4-201:79) uses `parent` attribute, not neighbor lists.

3. **[remove_h_caps()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:831:4-837:36) should not call [cleanup_invalid()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:157:4-183:73).** Just flip `alive=False` on H caps and their bonds. Dead H caps stay in dict but are filtered by [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75).

## Remaining Issues to Consider

**Issue 1: [h_children()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:198:4-201:79) is O(N)** — iterates all atoms (including dead) to find children by `parent`:
```python
# @/home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:199-202
def h_children(self, heavy_atom):
    return [a for a in self.atoms.values() 
            if a.alive and a.subtype == 'H_cap' and a.parent is heavy_atom]
```
For ~100 atoms this is fine. If it becomes a bottleneck, store a `children: list` on each [Atom](cci:2://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:38:0-57:89) object. Not a correctness issue.

**Issue 2: [sync_neighbor_lists()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:185:4-196:47) is O(N+B)** — rebuilds all neighbor lists. With soft-delete, we could make this local: only remove the deleted atom from its neighbors' neighbor lists. But for ~100 atoms, O(N+B) is negligible. Not a correctness issue.

**Issue 3: `_id` counter grows monotonically** — `Atom._counter` never resets. After many add/delete cycles, `_id` values can be large. This is fine for dict keys. If we ever want to use an array for `graph_to_dense` mapping (instead of dict), we'd need `_id` to be bounded. But dict is O(1) and fine.

**Issue 4: Dead atoms in `graph.atoms` dict grow** — You said

Your design is sound and rigorous. Let me be concise about what remains.

## Verdict: Your design is correct

**Decoupling topology ops from array export is the key insight.** The 4 mapping arrays are ephemeral — built once per [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75) call, never maintained during editing. This is waterproof because:

- GUI stores `_id` (stable), never dense index (ephemeral)
- Topology ops: O(1) soft-delete, no compaction, no mapping updates
- Export: O(N+B) build dense arrays + 4 maps, only when consumer needs them
- Consumers (VisPy, FF) see only dense arrays — zero rendering/FF complexity

## What needs changing (3 points)

1. **Remove [cleanup_invalid()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:157:4-183:73) calls from deletion paths** — [remove_atom_by_index()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:625:4-636:27), [remove_h_caps()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:831:4-837:36), and my new [remove_atom_by_id()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:568:4-584:27). Just soft-delete (flip `alive=False`). Dead atoms stay in dict, filtered by [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75).

2. **Remove [sync_neighbor_lists()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:185:4-196:47) from per-deletion paths** — O(N+B) global rebuild after every deletion. For ~100 atoms it's cheap, but unnecessary. Can defer to export time or make local.

3. **[remove_h_caps()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:831:4-837:36)** — just flip `alive=False` on H caps + their bonds. No [cleanup_invalid()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:157:4-183:73).

## What's already correct

- [AtomicGraph](cci:2://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:117:0-404:104) soft-delete mechanism — ✅
- [to_arrays()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:310:4-344:75) filtering dead atoms — ✅
- My partial implementation: `_atom_ids` array, `_id_to_idx` dict, [remove_atom_by_id()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:568:4-584:27) — ✅ direction is right

## My Option A code is mostly compatible

The `_id`-based signals and [remove_atom_by_id()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:568:4-584:27) I already implemented work with your design. The main change is: **remove the [cleanup_invalid()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:157:4-183:73) + [sync_neighbor_lists()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:185:4-196:47) calls** from [remove_atom_by_id()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:568:4-584:27) and [remove_atoms_by_id()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:584:4-602:27). Just soft-delete, then one [_sync_sys()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:239:4-252:35) at the end.

Want me to proceed with these changes?

---


# USER


look on GUI @SPAMMM_GUI.py and figure out how we can implement two things

1) export of files in .mol or .mol2 format not only .xyz with atom position but also with bonds. We should prehaps rename the export and load ,xyz button to something more genreral and currently supoort just the two .xyz and .mol/.mol2
what can help is @AtomicSystem.py @atomicUtils.py loadMol2 save_mol save_mol2

it would be perhasp good to have option to Ctrl+C/Ctrl+V direcly the selected atoms and bonds into text file?

2) it would be good to have system of "undo" on Ctrl+Z, but how to do it? It should be robust and simple?
We can just store copies of atomic system? We can also have some history list (finit list of previous positions). It should be optional so we can trun it of for really lare system to save some memory. It need to keep only minimal graph - atom position, type, bonds, and pi-orbitals npi fr each atom, it should be dense (only alive atoms, maybe their unique IDs)

for all of these operation it would be maybe good to have some minimalistic format containing just the minial topology (smaller more compact than AtomicGraph or Atomic system)

do not program just look what is asvailable and think how to implement it in robust, fast and simple way without much interference to exsting code.


ad "clipboard" - we should not use our custom made format, but direcly .xyz or .mol/.mol2 (possibly allor to switch between them which basically means atoms only or atoms+bonds)
for "clipboard" we should use text but for undo history we should not use text but some object with atoms and bonds, I do not wat to do parsing text back and forth all the time.
Similarly if I want to copy and pase within the molecular editor canvas it should not use text.
The only think where it make sence to convert to text if exporting to file or when pasting to some extrnal document, but this we cannot know, so perhaps it is better to ommit that. 
But would would be good if we can implement - when we get text from clipboad to ctrl+v we try to paste it by parsing it as .xyz or .mol

Hope it is not too chaotic what I say try to make order from that. 

Add "undo-stack" - question is what should be objects there?
I would prefer to make new PackedMolecule class which contains dense arrays "etype,apos,bonds" which are all number not strings. etype[natoms].int32,apos[natom,3].float32, bonds[nbond,2].int32, this format should be easy to save and load as npy or npz which is very usefull for fast save load. With such condensed efficient format we can make quite long history. (history should have some maximum size, maybe like rolling buffer, or we should drop the oldedt item if needed.

do not code yet, think it rhtough and write revised desing and plan

---

# GLM 5.2

## Revised Design & Plan

### Clarified Requirements

1. **Clipboard**: Use `.xyz` or `.mol/.mol2` text as the clipboard format (not custom). Internal copy/paste within the editor uses objects, not text. When pasting, if no internal copy exists, try parsing clipboard text as `.xyz` or `.mol`.
2. **Undo stack**: Store `PackedMolecule` objects (dense numpy arrays), not text. Rolling buffer with max size.
3. **No text round-tripping** for undo or internal paste — only for file I/O and clipboard text.

### 1. `PackedMolecule` Class

New class in `@/home/prokop/git/SPAMMM/spammm/topology/PackedMolecule.py`:

**Fields** (all dense numpy, no strings):
- `etype`: `int32[natoms]` — atomic numbers (Z)
- `apos`: `float32[natoms, 3]` — positions
- `bonds`: `int32[nbonds, 2]` — 0-based index pairs
- `npi`: `int8[natoms]` — pi-orbital count per atom (0=sp3, 1=sp2, 2=sp)

**Memory**: ~21 bytes/atom + 8 bytes/bond. For 1000 atoms + 2000 bonds: ~37KB/snapshot. 100 snapshots = ~3.7MB. Negligible.

**Methods**:
- `from_graph(graph, atom_indices=None)` — extract from [AtomicGraph](cci:2://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:117:0-429:104). If `atom_indices` given, filter to selected atoms and remap bonds to new dense indices (internal bonds only).
- `to_graph()` — rebuild a fresh [AtomicGraph](cci:2://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:117:0-429:104) (new Atom/Bond objects, pins=None, parent re-derived from distance for H caps).
- `save_npz(fname)` / `load_npz(fname)` — `np.savez`/`np.load` for fast binary I/O.
- `to_xyz_text()` / `to_mol_text()` — convert to text for clipboard.
- `from_text(text)` — parse `.xyz` or `.mol` text → `PackedMolecule` (auto-detect format).

**Key conversions**:
- `subtype → npi`: `'sp3'→0, 'sp2'→1, 'sp'→2, default→1` (existing `_get_npi_from_subtype` logic in `@/home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:303`)
- `npi → subtype`: `0→'{e}_sp3', 1→'{e}_sp2', 2→'{e}_sp'` (inverse, using element name)
- `Z → ename`: via `elements` module lookup
- Parent reconstruction: for H atoms (`etype==1`), find nearest heavy atom within 1.5Å → set as parent, subtype='H_cap'

### 2. Clipboard (Ctrl+C / Ctrl+V)

**Copy (Ctrl+C)**:
1. Get selected atom IDs from scene
2. Build `PackedMolecule.from_graph(graph, atom_indices=selected)` — includes internal bonds
3. Store in `self.copied_packed` (object, for fast internal paste)
4. Also set text on Qt clipboard: `QApplication.clipboard().setText(packed.to_xyz_text())` or `.to_mol_text()` depending on a format toggle (checkbox or setting: "Clipboard: XYZ / MOL")
   - `.xyz` = atoms only, `.mol` = atoms + bonds

**Paste (Ctrl+V)**:
1. If `self.copied_packed` is not None → use it directly (no text parsing, fast)
2. If `self.copied_packed` is None → read Qt clipboard text, try `PackedMolecule.from_text(text)` (auto-detect `.xyz` vs `.mol` by parsing)
3. Add atoms to graph from packed data, create bonds, `_sync_sys()`, [refresh_view()](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:1132:4-1250:46)

**Why this works**:
- Internal paste is instant (no serialization)
- External paste (from text editor, another app) works via text parsing
- Copying to clipboard always puts text, so user can paste into external text editor
- Format toggle lets user choose atoms-only (xyz) or atoms+bonds (mol) for the clipboard text

**Changes to existing code** (`@/home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py`):
- [copy_selected_atoms()](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:795:4-809:50): Replace `(enames, apos)` with `PackedMolecule` + clipboard text
- [paste_copied_atoms()](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:811:4-829:75): Use `self.copied_packed` or parse clipboard text
- Add format toggle UI (small dropdown or checkbox in editors section)

### 3. Export/Import (File I/O)

**Export** (replace [export_xyz](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:1016:4-1021:64) with `export_structure`):
- File dialog with filter: `"Molecular Files (*.xyz *.mol *.mol2);;XYZ (*.xyz);;MOL (*.mol);;MOL2 (*.mol2)"`
- Dispatch on extension:
  - `.xyz` → [backend.save_xyz(fname)](cci:1://file:///home/prokop/git/SPAMMM/spammm/atomicUtils.py:1924:0-1936:65) (existing)
  - `.mol` → `backend._sync_sys(); backend.sys.save_mol(fname)` (existing [AtomicSystem.save_mol](cci:1://file:///home/prokop/git/SPAMMM/spammm/AtomicSystem.py:121:4-122:98))
  - `.mol2` → `backend._sync_sys(); backend.sys.save_mol2(fname)` (existing [AtomicSystem.save_mol2](cci:1://file:///home/prokop/git/SPAMMM/spammm/AtomicSystem.py:124:4-129:110))
- Rename "Export XYZ" button → "Export", "Show XYZ" → "Show"

**Import** (new `import_structure` method):
- File dialog with same filter
- Dispatch on extension:
  - `.xyz` → [backend.load_xyz(fname)](cci:1://file:///home/prokop/git/SPAMMM/spammm/atomicUtils.py:1211:0-1253:36) (existing)
  - `.mol` / `.mol2` → new `backend.load_mol(fname)` — similar to [load_xyz()](cci:1://file:///home/prokop/git/SPAMMM/spammm/atomicUtils.py:1211:0-1253:36) but uses [au.loadMol()](cci:1://file:///home/prokop/git/SPAMMM/spammm/atomicUtils.py:1276:0-1388:62) / [au.loadMol2()](cci:1://file:///home/prokop/git/SPAMMM/spammm/atomicUtils.py:1391:0-1585:95) which return bonds. Use loaded bonds directly instead of `_create_bond_to_nearest_heavy()` heuristic.
- Add "Import" button next to "Export"

**Changes**:
- `@/home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py`: ~20 lines (rename + dispatch)
- `@/home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py`: ~40 lines (new `load_mol()` method)

### 4. Undo System (Ctrl+Z)

**`UndoStack` class** (in `PackedMolecule.py` or inline in GUI):
- Uses `collections.deque(maxlen=100)` — O(1) append, auto-drops oldest
- `push(packed)`: append snapshot
- `pop() → PackedMolecule | None`: return last snapshot
- `clear()`: reset
- `enabled` flag — checkbox in UI to disable for large systems

**When to push** (before mutations, in GUI):
- Before [handle_click()](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:1045:4-1130:40) with add/remove atom action
- Before [delete_selected_atoms()](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:785:4-793:51)
- Before [paste_copied_atoms()](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:811:4-829:75)
- Before [reset_offsets()](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:1002:4-1004:27) (snap)
- Before [recalc_bonds()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/AtomicGraph.py:317:4-331:39), [adjust_h()](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:1023:4-1026:27)
- Before element change operations
- Basically: any operation that modifies the graph

**`_push_undo()`** (GUI method, ~3 lines):
```python
def _push_undo(self):
    if not self.undo_enabled: return
    self.undo_stack.push(PackedMolecule.from_graph(self.backend.graph))
```

**`undo()`** (GUI method, ~10 lines):
```python
def undo(self):
    packed = self.undo_stack.pop()
    if packed is None: return
    self.backend.graph = packed.to_graph()
    self.backend._sync_sys()
    self.refresh_view()
```

**Ctrl+Z in [on_key_press()](cci:1://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:766:4-783:37)**: Remove the `if not selected: return` guard for Ctrl+Z (undo should work even with no selection).

**UI**: Add "Undo" checkbox (enable/disable) in editors section. No redo for now (keep simple).

### 5. Implementation Order

| Step | What | Files | Est. lines |
|------|------|-------|------------|
| 1 | `PackedMolecule` class | new `topology/PackedMolecule.py` | ~80 |
| 2 | Export/Import multi-format | [KekuleBackend.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:0:0-0:0), [SPAMMM_GUI.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:0:0-0:0) | ~60 |
| 3 | Clipboard with PackedMolecule | [SPAMMM_GUI.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:0:0-0:0) | ~40 |
| 4 | Undo stack | [SPAMMM_GUI.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:0:0-0:0) | ~30 |

**Total**: ~210 lines, mostly new code. Minimal changes to existing methods (rename buttons, modify copy/paste, add undo hooks).

### Key Design Decisions

- **PackedMolecule is the SSOT** for undo snapshots and internal clipboard — dense numpy, no strings, no parsing
- **Text is only for clipboard external use and file I/O** — never round-tripped for undo
- **`deque(maxlen=N)`** for undo — simple, efficient rolling buffer
- **Parent reconstruction** in `to_graph()` — H caps identified by `etype==1`, parent = nearest heavy atom < 1.5Å. This is the same heuristic already used in [load_xyz()](cci:1://file:///home/prokop/git/SPAMMM/spammm/atomicUtils.py:1211:0-1253:36).
- **Pins not stored** in PackedMolecule — re-derived from grid snapping on restore (same as [load_xyz()](cci:1://file:///home/prokop/git/SPAMMM/spammm/atomicUtils.py:1211:0-1253:36))
- **Rings not stored** — re-detected via `_guess_rings()` after restore
- **No redo** for now — can be added later by keeping a redo deque

Shall I proceed with implementation?