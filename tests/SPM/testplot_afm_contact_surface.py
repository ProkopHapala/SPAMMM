#!/usr/bin/env python3
"""Deprecated: use tests/testplot_contact_surface.py (Phase1 fit + Phase2 PP parity)."""
import runpy
import os
runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testplot_contact_surface.py'), run_name='__main__')
