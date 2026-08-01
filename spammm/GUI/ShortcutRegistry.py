"""
ShortcutRegistry.py — Generic keyboard shortcut mechanism for SPAMMM GUI.

Purpose: Provide a centralized registry MECHANISM (not a hardcoded action list)
that extensions use to register their own shortcuts. The registry detects
conflicts, auto-syncs button labels with Unicode keystrokes, and auto-generates
help/cheatsheet content.

Key design principle: The registry knows NOTHING about specific actions.
Each extension and the main GUI register their own shortcuts at build_ui() /
initUI() time. The registry provides:
  - Conflict detection (fail-loud on duplicate key+modifiers)
  - Unicode modifier encoding (⌃ ⇧ ⌥)
  - Button label auto-sync ("Undo [⌃Z]")
  - Optional dispatch (for extensions that want registry-driven key handling)
  - Help/cheatsheet auto-generation from whatever was registered

Mouse actions are NOT registered here (hardwired in mode handlers — see
doc/Tasks/GUI_HelpPanel_ShortcutRegistry.md §7 decision 4).
"""


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------
class ShortcutConflictError(RuntimeError):
    """Raised when two shortcuts claim the same (key, modifiers) combination."""
    pass


# ---------------------------------------------------------------------------
# Modifier encoding (Unicode)
# ---------------------------------------------------------------------------
_MOD_SYMBOLS = {
    'Control': '\u2303',  # ⌃
    'Shift':   '\u21e7',  # ⇧
    'Alt':     '\u2325',  # ⌥
}
_MOD_ASCII = {
    'Control': '^',
    'Shift':   'S+',
    'Alt':     'A+',
}

_KEY_DISPLAY = {
    'Enter':   '\u23ce',  # ⏎
    'Return':  '\u23ce',  # ⏎
    'Space':   '\u2423',  # ␣
    ' ':       '\u2423',  # ␣
    'Delete':  'Del',
    'KP_ADD':  '+',
    'KP_SUBTRACT': '-',
    'ArrowUp':    '\u2191',  # ↑
    'ArrowDown':  '\u2193',  # ↓
    'ArrowLeft':  '\u2190',  # ←
    'ArrowRight': '\u2192',  # →
    'Up':    '\u2191',
    'Down':  '\u2193',
    'Left':  '\u2190',
    'Right': '\u2192',
}

USE_UNICODE = True


def _mod_str(modifiers, unicode=True):
    table = _MOD_SYMBOLS if unicode else _MOD_ASCII
    return ''.join(table[m] for m in modifiers if m in table)


def _key_str(key, unicode=True):
    if unicode and key in _KEY_DISPLAY:
        return _KEY_DISPLAY[key]
    if key.startswith('KP_'):
        return key[3:]
    return key


def encode_keystroke(keys, modifiers=(), unicode=True):
    """Encode a shortcut as a compact string, e.g. '⌃Z', '⏎', '⇧↑'.

    Args:
        keys: single key string or list of equivalent keys
        modifiers: tuple of 'Control'/'Shift'/'Alt'
    """
    if isinstance(keys, (list, tuple)):
        display_key = keys[0]
        for k in keys:
            if not k.startswith('KP_'):
                display_key = k
                break
    else:
        display_key = keys
    return _mod_str(modifiers, unicode) + _key_str(display_key, unicode)


# ---------------------------------------------------------------------------
# ShortcutSpec
# ---------------------------------------------------------------------------
class ShortcutSpec:
    """One shortcut declaration (created by register())."""
    def __init__(self, keys, modifiers=(), description="", group="Global",
                 context_fn=None, callback=None, widget=None):
        self.keys = [keys] if isinstance(keys, str) else list(keys)
        self.modifiers = tuple(modifiers)
        self.description = description
        self.group = group
        self.context_fn = context_fn   # (window) -> bool; None = always
        self.callback = callback       # (window) -> None; None = doc-only
        self.widget = widget           # the button this is bound to (for label sync)

    def matches(self, key, modifiers):
        if key not in self.keys:
            return False
        mods = set(modifiers) if isinstance(modifiers, (tuple, list)) else set()
        return mods == set(self.modifiers)

    def encode(self, unicode=True):
        return encode_keystroke(self.keys, self.modifiers, unicode)


# ---------------------------------------------------------------------------
# ShortcutRegistry — singleton via classmethods
# ---------------------------------------------------------------------------
class ShortcutRegistry:
    """Generic shortcut registry. Use classmethods — no instantiation needed.

    Empty at import time. Fills up as extensions call register() / register_button()
    during UI construction.
    """
    _shortcuts = []   # list of ShortcutSpec (registration order)
    _by_key = {}      # (key, frozenset(mods)) -> ShortcutSpec  (conflict index)

    @classmethod
    def register(cls, keys, modifiers=(), description="", group="Global",
                 context_fn=None, callback=None, widget=None):
        """Register a shortcut. Raises ShortcutConflictError on duplicate key+modifiers.

        Args:
            keys: vispy key name or list of equivalents, e.g. 'Z', ['Enter', 'Return']
            modifiers: tuple of 'Control'/'Shift'/'Alt'
            description: human-readable, for help/cheatsheet
            group: "Global", "Camera", or extension-defined group name
            context_fn: (window) -> bool — None = always active
            callback: (window) -> None — None = documentation-only (no dispatch)
            widget: the button this shortcut is bound to (for label sync)
        Returns:
            ShortcutSpec
        """
        key_list = [keys] if isinstance(keys, str) else list(keys)
        mods = tuple(modifiers)
        mod_set = frozenset(mods)
        # Conflict check: every key variant is checked
        for k in key_list:
            idx = (k, mod_set)
            if idx in cls._by_key:
                existing = cls._by_key[idx]
                raise ShortcutConflictError(
                    f"Shortcut {encode_keystroke(k, mods)} already registered: "
                    f"'{existing.description}' (group={existing.group}). "
                    f"Cannot register '{description}'.")
        spec = ShortcutSpec(key_list, mods, description, group, context_fn, callback, widget)
        cls._shortcuts.append(spec)
        for k in key_list:
            cls._by_key[(k, mod_set)] = spec
        return spec

    @classmethod
    def register_button(cls, btn, text, shortcut_tuple):
        """Register a shortcut for a button and auto-format its label.

        Args:
            btn: the QPushButton (or QCheckBox) widget
            text: the base label text (without shortcut suffix)
            shortcut_tuple: (keys, modifiers, description, group) or
                            (keys, modifiers, description, group, context_fn, callback)
        Returns:
            ShortcutSpec
        """
        n = len(shortcut_tuple)
        keys, modifiers, description, group = shortcut_tuple[:4]
        context_fn = shortcut_tuple[4] if n > 4 else None
        callback = shortcut_tuple[5] if n > 5 else None
        spec = cls.register(keys, modifiers, description, group,
                            context_fn=context_fn, callback=callback, widget=btn)
        btn.setText(cls.format_label(text, spec))
        return spec

    @classmethod
    def get(cls, name):
        """Lookup by description (not unique — returns first match). For testing."""
        for spec in cls._shortcuts:
            if spec.description == name:
                return spec
        return None

    @classmethod
    def all(cls):
        """All registered specs in registration order."""
        return list(cls._shortcuts)

    @classmethod
    def by_group(cls):
        """Dict: group_name -> list of ShortcutSpec (registration order)."""
        groups = {}
        for spec in cls._shortcuts:
            groups.setdefault(spec.group, []).append(spec)
        return groups

    @classmethod
    def dispatch(cls, event, window):
        """Try to dispatch a vispy key_press event. Returns True if handled.

        Skips documentation-only entries (callback=None). Checks context_fn.
        """
        key = getattr(event, 'key', None)
        if key is None:
            return False
        mods = event.modifiers if isinstance(event.modifiers, (tuple, list)) else ()
        mod_set = frozenset(mods)
        spec = cls._by_key.get((key, mod_set))
        if spec is None or spec.callback is None:
            return False
        if spec.context_fn is not None and not spec.context_fn(window):
            return False
        spec.callback(window)
        event.handled = True
        return True

    @classmethod
    def format_label(cls, text, spec):
        """Format a button label: 'ButtonName [⌃Z]'."""
        encoded = spec.encode(USE_UNICODE)
        if not encoded:
            return text
        return f"{text} [{encoded}]"

    @classmethod
    def help_markdown(cls, unicode=True):
        """Generate grouped markdown tables of all shortcuts."""
        lines = []
        for group, specs in cls.by_group().items():
            lines.append(f"### {group}")
            lines.append("")
            lines.append("| Key | Action |")
            lines.append("|-----|--------|")
            for spec in specs:
                key_disp = spec.encode(unicode)
                if len(spec.keys) > 1:
                    variants = []
                    for k in spec.keys:
                        e = encode_keystroke(k, spec.modifiers, unicode)
                        if e not in variants:
                            variants.append(e)
                    key_disp = ' / '.join(variants)
                lines.append(f"| `{key_disp}` | {spec.description} |")
            lines.append("")
        return '\n'.join(lines)

    @classmethod
    def help_html(cls, unicode=True):
        """Generate HTML table for in-GUI help panel."""
        lines = ['<table style="font-size:9pt;">']
        for group, specs in cls.by_group().items():
            lines.append(f'<tr><td colspan="2"><b>{group}</b></td></tr>')
            for spec in specs:
                key_disp = spec.encode(unicode)
                if len(spec.keys) > 1:
                    variants = []
                    for k in spec.keys:
                        e = encode_keystroke(k, spec.modifiers, unicode)
                        if e not in variants:
                            variants.append(e)
                    key_disp = ' / '.join(variants)
                lines.append(f'<tr><td><code>{key_disp}</code></td><td>{spec.description}</td></tr>')
        lines.append('</table>')
        return '\n'.join(lines)

    @classmethod
    def export_cheatsheet_section(cls, unicode=True):
        """Generate the markdown table for <!-- AUTOGEN:keyboard --> section."""
        lines = ["| Key | Action |", "|-----|--------|"]
        for spec in cls.all():
            key_disp = spec.encode(unicode)
            if len(spec.keys) > 1:
                variants = []
                for k in spec.keys:
                    e = encode_keystroke(k, spec.modifiers, unicode)
                    if e not in variants:
                        variants.append(e)
                key_disp = ' / '.join(variants)
            lines.append(f"| `{key_disp}` | {spec.description} |")
        return '\n'.join(lines)

    @classmethod
    def reset(cls):
        """Clear all registrations (for testing)."""
        cls._shortcuts.clear()
        cls._by_key.clear()

    @classmethod
    def count(cls):
        return len(cls._shortcuts)


# ---------------------------------------------------------------------------
# CLI: --export (regenerate cheatsheet keyboard section)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    if '--export' in sys.argv:
        # Extensions must be imported first so they register their shortcuts.
        # This CLI is meant to be run after the GUI modules are importable.
        print(ShortcutRegistry.export_cheatsheet_section())
    else:
        print("Usage: python -m spammm.GUI.ShortcutRegistry --export")
        print("  Outputs the keyboard table for GUI_CHEATSHEET.md (between AUTOGEN markers)")
        print("  Note: import GUI modules first so shortcuts are registered.")
