#!/bin/bash
# Install build123d for STEP export support
# This installs into a virtual environment (NOT system Python or KiCad's Python)
# STEP export runs from the command line due to library conflicts with KiCad

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/step_venv"
CLI_SCRIPT="$SCRIPT_DIR/plugins/com_github_aightech_flexviz/step_export_cli.py"

echo "=== Flex Viewer STEP Export Setup ==="
echo ""
echo "This will create a virtual environment and install build123d."
echo "STEP export runs from the command line, not inside KiCad."
echo ""

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3."
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate venv
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install build123d
echo "Installing build123d (this may take a few minutes)..."
pip install build123d

# Verify installation
echo ""
echo "Verifying installation..."
if python -c "from build123d import Box; print('SUCCESS: build123d is installed')"; then
    echo ""
    echo "=== Setup Complete ==="
    echo ""
    echo "To export STEP files:"
    echo ""
    echo "  source $VENV_DIR/bin/activate"
    echo "  python $CLI_SCRIPT your_board.kicad_pcb output.step"
    echo ""
    echo "Options:"
    echo "  --direct              True CAD geometry (recommended)"
    echo "  --subdivisions N      Bend subdivisions (default: 8)"
    echo "  --marker-layer LAYER  Fold marker layer (default: User.1)"
    echo "  --stiffener-thickness T  Stiffener thickness in mm"
    echo "  --no-stiffeners       Disable stiffener export"
    echo ""
    echo "Example:"
    echo "  python $CLI_SCRIPT board.kicad_pcb board.step --direct"
    echo ""
else
    echo ""
    echo "ERROR: Installation verification failed."
    echo "Please check for errors above."
    exit 1
fi
