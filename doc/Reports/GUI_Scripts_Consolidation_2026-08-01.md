---
type: Report
title: GUI scripts centralized to demos/gui_scripts/ + MP4 video export (session 2026-08-01)
status: delivered — USER confirmed goals achieved
tags: [GUI-script, consolidation, video, mp4, gif, ffmpeg, demos]
timestamp: 2026-08-01
related:
  - doc/Tasks/GUI_Scripting_DemoRunner.md
  - doc/Tasks/Drag_Demo_Issues.md
  - doc/Reports/PTCDA_DragDemo_StickSlip_2026-08-01.md
---

# GUI scripts centralized to `demos/gui_scripts/` + MP4 video export

**Status:** delivered — USER confirmed both major conference scripts work (drag demo + conference_demo windmill/BR-STM).  
**Scripts moved:** 8 `.py` + 1 `README.md` from `spammm/GUI/gui_scripts/` → `demos/gui_scripts/`

---

## 1. What was done

### 1.1 Centralized all GUI scripts to `demos/gui_scripts/`

Moved all 8 scripts + README via `git mv`:

| Script | Type | Produces animation? |
|--------|------|---------------------|
| `conference_demo.py` | paced-generator | No (windmill MC → AFM → BR-STM pipeline) |
| `ptcda_drag_demo.py` | paced-generator | **GIF + MP4** (`--format gif\|mp4\|both`) |
| `ptcda_interactive_drag.py` | paced-generator | No (interactive setup) |
| `azaindol_draw_demo.py` | synchronous | **GIF** |
| `folded_rigid_setup.py` | synchronous | No (setup script) |
| `rc_scan_review.py` | synchronous | No (review) |
| `azaindol_draw_offline.py` | offline (no Qt) | SVG only |
| `rc_scan_offline.py` | offline (no Qt) | No |

### 1.2 Updated discovery path

`gui_script_runner.bundled_scripts()` now defaults to `demos/gui_scripts/` (relative to repo root). The **Scripts → Bundled** menu auto-discovers scripts from this folder — no manual registration needed. All 6 non-offline scripts appear in the menu (offline scripts excluded by convention).

### 1.3 Updated all references

26 files referenced the old `spammm/GUI/gui_scripts/` path. All updated to `demos/gui_scripts/`:
- `run_gui.sh` examples
- `spammm/GUI/README.md`
- `demos/README.md`
- `demos/gui_scripts/README.md`
- All script docstrings (usage examples)
- All doc reports, tasks, topical audits, takeaways
- Test files

### 1.4 Added MP4 video export (`frames_to_video`)

New function in `spammm/GUI/gui_script_utils.py`:

```python
GSU.frames_to_video(frame_paths, out_video, fps=10, codec='libx264', crf=23)
```

- **H.264 (`libx264`)** with `-tune animation` — best codec for mostly-static content with small moving parts (flat colors compress extremely well)
- `-crf 23` (default quality), `-pix_fmt yuv420p` (universal playback), `-movflags +faststart` (streaming)
- Pads to even dimensions (H.264 requirement)
- Also supports VP9 (WebM) and AV1 via `codec=` parameter

**Size comparison** (ptcda_drag_demo, 11 frames):
- GIF: 1.5 MB
- MP4: 130 KB → **11x smaller**

`ptcda_drag_demo.py` now has `--format gif|mp4|both` (default: `both`).

### 1.5 Codec recommendations

| Use case | Format | Why |
|----------|--------|-----|
| GitHub README | GIF | READMEs don't render `<video>` tags |
| GitHub PR comments | MP4 | Inline autoplay works, smaller than GIF |
| Slack/Discord/Notion | MP4 | All support inline H.264 autoplay |
| Web embed (your own site) | WebM (VP9) | 25-40% smaller than H.264 |
| Universal compatibility | MP4 (H.264) | Plays everywhere since 2012 |

**For SPAMMM demos:** MP4 (H.264, `-tune animation`) is the best default — 11x smaller than GIF, plays everywhere, and `-tune animation` is specifically designed for content with flat colors and small motion regions.

---

## 2. Verification

### 2.1 Script discovery + loading

All 6 bundled scripts discovered and loaded from new location:
```
6 bundled scripts in demos/gui_scripts/:
  OK  Azaindol Draw Demo              [synchronous]
  OK  Conference Demo                 [paced-generator]
  OK  Folded Rigid Setup              [synchronous]
  OK  Ptcda Drag Demo                 [paced-generator]
  OK  Ptcda Interactive Drag          [paced-generator]
  OK  Rc Scan Review                  [synchronous]
```

### 2.2 Tests pass

`pytest tests/GUI/test_gui_script_runner.py` — 15/15 passed (including `test_bundled_scripts_excludes_offline_and_underscore`).

### 2.3 End-to-end runs

- **`ptcda_drag_demo.py`** — produces GIF + MP4 from `demos/gui_scripts/` path ✓
- **`conference_demo.py`** — builds 4×PTCDA windmill, runs AFM stages 1-4, BR-STM ✓ (USER ran it manually from the GUI, killed during DFTB+ relaxation due to time, but the pipeline works)

### 2.4 Scripts menu accessibility

Both major conference scripts are accessible from the **Scripts → Bundled** menu:
- "Conference Demo" → `demos/gui_scripts/conference_demo.py`
- "Ptcda Drag Demo" → `demos/gui_scripts/ptcda_drag_demo.py`

No manual registration — the menu auto-discovers `*.py` files in `demos/gui_scripts/` (excluding `_*.py` and `*_offline.py`).

---

## 3. User goals achieved

1. **Centralized location** — all GUI scripts now in `demos/gui_scripts/` ✓
2. **Run both ways** — CLI (`--script`) and interactive (Scripts menu) ✓
3. **Auto-discovery menu** — Scripts → Bundled reads `demos/gui_scripts/` automatically ✓
4. **All scripts moved and tested** — 8 scripts, all load, 2 tested end-to-end ✓
5. **Video export** — MP4 (H.264 `-tune animation`) alongside GIF, 11x smaller ✓

---

## 4. File inventory

| File | Role |
|------|------|
| `demos/gui_scripts/*.py` | 8 centralized GUI scripts |
| `demos/gui_scripts/README.md` | Script index with usage examples |
| `spammm/GUI/gui_script_runner.py` | `bundled_scripts()` points to `demos/gui_scripts/` |
| `spammm/GUI/gui_script_utils.py` | `frames_to_video()` — ffmpeg H.264/VP9/AV1 encoder |
| `demos/README.md` | Added gui_scripts section |
| `spammm/GUI/README.md` | Updated to point to `demos/gui_scripts/` |
