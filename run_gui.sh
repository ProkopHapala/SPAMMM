#!/bin/bash
# run_gui.sh — Launch the SPAMMM GUI molecular editor
#
# Usage:
#   ./run_gui.sh                    # Launch with defaults (output → <repo>/output)
#   ./run_gui.sh -o /tmp/myimgs     # Custom output directory
#   ./run_gui.sh -v 3               # Max verbosity
#   ./run_gui.sh -f /path/to/Fdata  # Custom Fdata path
#   ./run_gui.sh --help             # Show all CLI options
#
# All saved images (screenshots, plots) default to <repo>/output/

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output"

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Run the GUI, passing through all CLI arguments
# Default --output-dir to <repo>/output if not explicitly set
python3 "${REPO_ROOT}/spammm/GUI/SPAMMM_GUI.py" --output-dir "${OUTPUT_DIR}" "$@"
