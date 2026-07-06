"""Load and run GUI control scripts (widget-parity setup after window.show())."""
import importlib.util
import os
import sys


def run_gui_script(window, script_path, script_argv=None):
    """Execute *script_path*; module must define ``run(window, argv=None)``."""
    script_path = os.path.abspath(script_path)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"GUI script not found: {script_path}")
    script_argv = list(script_argv or [])
    name = f"spammm_gui_script_{os.path.splitext(os.path.basename(script_path))[0]}"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load GUI script: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    run_fn = getattr(mod, 'run', None)
    if run_fn is None:
        raise RuntimeError(f"GUI script must define run(window, argv=None): {script_path}")
    return run_fn(window, script_argv)
