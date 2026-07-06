#!/usr/bin/env python3
"""Launch RC scan GUI review via run_gui.sh (preferred entry point).

Equivalent to:
  ./run_gui.sh --script spammm/GUI/gui_scripts/rc_scan_review.py [-- script args]
"""
import os
import subprocess
import sys

REPO = os.path.join(os.path.dirname(__file__), '..', '..')
RUN_GUI = os.path.join(REPO, 'run_gui.sh')
SCRIPT = os.path.join(REPO, 'spammm/GUI/gui_scripts/rc_scan_review.py')

if __name__ == '__main__':
    cmd = [RUN_GUI, '--script', SCRIPT, '--'] + sys.argv[1:]
    os.execv(RUN_GUI, cmd)
