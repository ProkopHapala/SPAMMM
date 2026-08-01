"""L0 tests for ShortcutRegistry — generic mechanism (no specific actions).

Verifies:
  - Registry is empty at import (no hardcoded actions)
  - register() + conflict detection (fail-loud)
  - Unicode encoding (⌃ ⇧ ⌥ ⏎ ␣ Del ↑)
  - register_button() auto-formats label
  - dispatch() calls callback, respects context_fn, skips doc-only
  - help_markdown / help_html / export_cheatsheet_section auto-generate
  - reset() clears state

Run: pytest tests/GUI/test_shortcut_registry.py -m "not slow"
"""
import pytest
from spammm.GUI.ShortcutRegistry import (
    ShortcutRegistry, ShortcutSpec, ShortcutConflictError, encode_keystroke,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry before and after each test."""
    ShortcutRegistry.reset()
    yield
    ShortcutRegistry.reset()


class _FakeEvent:
    """Minimal vispy-like key event."""
    def __init__(self, key, modifiers=()):
        self.key = key
        self.modifiers = modifiers
        self.handled = False


class _FakeWindow:
    """Minimal window for context_fn / callback testing."""
    def __init__(self, edit_mode='Unified'):
        self.edit_mode = edit_mode
        self.called = None
    def do_thing(self):
        self.called = 'do_thing'


# ── Encoding ──────────────────────────────────────────────────────────────────

def test_encode_ctrl():
    assert encode_keystroke('Z', ('Control',)) == '\u2303Z'  # ⌃Z

def test_encode_shift_ctrl():
    assert encode_keystroke('K', ('Control', 'Shift')) == '\u2303\u21e7K'  # ⌃⇧K

def test_encode_enter():
    assert encode_keystroke(['Enter', 'Return']) == '\u23ce'  # ⏎

def test_encode_space():
    assert encode_keystroke('Space') == '\u2423'  # ␣

def test_encode_delete():
    assert encode_keystroke('Delete') == 'Del'

def test_encode_arrows():
    assert encode_keystroke('ArrowUp') == '\u2191'  # ↑

def test_encode_numpad():
    # First non-KP_ key is preferred for display
    assert encode_keystroke(['+', 'KP_ADD', '=']) == '+'

def test_encode_ascii_fallback():
    assert encode_keystroke('Z', ('Control',), unicode=False) == '^Z'


# ── Registry: empty at import ─────────────────────────────────────────────────

def test_registry_empty_at_import():
    assert ShortcutRegistry.count() == 0
    assert ShortcutRegistry.all() == []


# ── Register + conflict ───────────────────────────────────────────────────────

def test_register_basic():
    spec = ShortcutRegistry.register('Z', ('Control',), "Undo", "Global")
    assert isinstance(spec, ShortcutSpec)
    assert spec.keys == ['Z']
    assert spec.modifiers == ('Control',)
    assert spec.description == "Undo"
    assert ShortcutRegistry.count() == 1

def test_register_multiple_keys():
    spec = ShortcutRegistry.register(['Enter', 'Return'], description="Toggle 2D/3D", group="Global")
    assert spec.keys == ['Enter', 'Return']
    assert spec.modifiers == ()

def test_register_conflict_same_key():
    ShortcutRegistry.register('Z', ('Control',), "Undo", "Global")
    with pytest.raises(ShortcutConflictError, match="already registered"):
        ShortcutRegistry.register('Z', ('Control',), "Redo", "Global")

def test_register_conflict_key_variant():
    # Registering ['Enter', 'Return'] then trying 'Enter' again should conflict
    ShortcutRegistry.register(['Enter', 'Return'], description="Toggle", group="Global")
    with pytest.raises(ShortcutConflictError):
        ShortcutRegistry.register('Enter', description="Something else", group="Global")

def test_register_no_conflict_different_modifiers():
    ShortcutRegistry.register('Z', ('Control',), "Undo", "Global")
    # Shift+Z is a different shortcut — no conflict
    spec = ShortcutRegistry.register('Z', ('Shift',), "Capital Z", "Global")
    assert spec.description == "Capital Z"

def test_register_no_conflict_different_keys():
    ShortcutRegistry.register('Z', ('Control',), "Undo", "Global")
    ShortcutRegistry.register('X', ('Control',), "Cut", "Global")
    assert ShortcutRegistry.count() == 2


# ── register_button ───────────────────────────────────────────────────────────

def test_register_button_formats_label():
    from PyQt5 import QtWidgets
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    btn = QtWidgets.QPushButton("Undo")
    spec = ShortcutRegistry.register_button(btn, "Undo", ('Z', ('Control',), "Undo", "Global"))
    assert btn.text() == "Undo [\u2303Z]"  # Undo [⌃Z]
    assert spec.widget is btn


# ── Dispatch ──────────────────────────────────────────────────────────────────

def test_dispatch_calls_callback():
    w = _FakeWindow()
    ShortcutRegistry.register('Z', ('Control',), "Undo", "Global",
                              callback=lambda win: setattr(win, 'called', 'undo'))
    ev = _FakeEvent('Z', ('Control',))
    assert ShortcutRegistry.dispatch(ev, w)
    assert w.called == 'undo'
    assert ev.handled is True

def test_dispatch_no_match_returns_false():
    w = _FakeWindow()
    ShortcutRegistry.register('Z', ('Control',), "Undo", "Global", callback=lambda win: None)
    ev = _FakeEvent('X', ('Control',))
    assert not ShortcutRegistry.dispatch(ev, w)
    assert not ev.handled

def test_dispatch_skips_doc_only():
    w = _FakeWindow()
    ShortcutRegistry.register('Z', ('Control',), "Undo", "Global", callback=None)
    ev = _FakeEvent('Z', ('Control',))
    assert not ShortcutRegistry.dispatch(ev, w)  # doc-only → not handled

def test_dispatch_respects_context_fn():
    w = _FakeWindow(edit_mode='Atom')
    called = []
    ShortcutRegistry.register('Delete', description="Delete sel", group="Select",
                              context_fn=lambda win: win.edit_mode == 'Select',
                              callback=lambda win: called.append('del'))
    # In Atom mode → context returns False → not dispatched
    ev = _FakeEvent('Delete')
    assert not ShortcutRegistry.dispatch(ev, w)
    assert called == []
    # In Select mode → dispatched
    w.edit_mode = 'Select'
    ev2 = _FakeEvent('Delete')
    assert ShortcutRegistry.dispatch(ev2, w)
    assert called == ['del']


# ── Help generation ───────────────────────────────────────────────────────────

def test_help_markdown_contains_groups():
    ShortcutRegistry.register('Z', ('Control',), "Undo", "Global")
    ShortcutRegistry.register('Delete', description="Delete sel", group="Select")
    md = ShortcutRegistry.help_markdown()
    assert "### Global" in md
    assert "### Select" in md
    assert "Undo" in md
    assert "Delete sel" in md

def test_help_html_contains_groups():
    ShortcutRegistry.register('Z', ('Control',), "Undo", "Global")
    html = ShortcutRegistry.help_html()
    assert "<b>Global</b>" in html
    assert "Undo" in html

def test_export_cheatsheet_section():
    ShortcutRegistry.register('Z', ('Control',), "Undo", "Global")
    ShortcutRegistry.register(['Enter', 'Return'], description="Toggle 2D/3D", group="Global")
    table = ShortcutRegistry.export_cheatsheet_section()
    assert "| Key | Action |" in table
    assert "Undo" in table
    assert "Toggle 2D/3D" in table


# ── Reset ──────────────────────────────────────────────────────────────────────

def test_reset_clears_all():
    ShortcutRegistry.register('Z', ('Control',), "Undo", "Global")
    assert ShortcutRegistry.count() == 1
    ShortcutRegistry.reset()
    assert ShortcutRegistry.count() == 0
    assert ShortcutRegistry.all() == []
