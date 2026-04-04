#!/bin/bash

# =============================================================
#  setup.sh — Python Virtual Environment Setup (Mac + VSCode)
# =============================================================
#
#  FIRST TIME SETUP:
#  ─────────────────
#  1. Open a terminal in the root of your project (Terminal >
#     New Terminal in VSCode, or cd into the project folder).
#
#  2. Make this script executable (only needed once):
#       chmod +x setup.sh
#
#  3. Run the script:
#       ./setup.sh
#
#  ACTIVATING THE VENV LATER (after first-time setup):
#  ────────────────────────────────────────────────────
#  Each time you open a new terminal session, activate with:
#       source .venv/bin/activate
#
#  To deactivate the venv:
#       deactivate
#
#  VSCODE PYTHON INTERPRETER:
#  ──────────────────────────
#  After running this script, point VSCode at the venv:
#  1. Open the Command Palette (Cmd + Shift + P)
#  2. Type: "Python: Select Interpreter"
#  3. Choose the option that shows: ./.venv/bin/python
#     (If it doesn't appear, click "Enter interpreter path..."
#      and type: .venv/bin/python)
#
# =============================================================

set -e  # Exit immediately if any command fails

VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"

echo ""
echo "🐍 Python Virtual Environment Setup"
echo "====================================="

# ── 1. Check Python is available ──────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "❌  python3 not found. Install it from https://www.python.org or via Homebrew:"
    echo "    brew install python"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✅  Found $PYTHON_VERSION"

# ── 2. Create the virtual environment (if it doesn't exist) ───
if [ -d "$VENV_DIR" ]; then
    echo "ℹ️   Virtual environment '$VENV_DIR' already exists — skipping creation."
else
    echo "📦  Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    echo "✅  Virtual environment created."
fi

# ── 3. Activate the virtual environment ───────────────────────
echo "⚡  Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "✅  Virtual environment activated."

# ── 4. Upgrade pip ────────────────────────────────────────────
echo "⬆️   Upgrading pip..."
pip install --upgrade pip --quiet
echo "✅  pip is up to date."

# ── 5. Install dependencies from requirements.txt ─────────────
if [ -f "$REQUIREMENTS" ]; then
    echo "📋  Installing dependencies from $REQUIREMENTS..."
    pip install -r "$REQUIREMENTS"
    echo "✅  All dependencies installed."
else
    echo "⚠️   No $REQUIREMENTS found in the current directory — skipping install."
    echo "    Add one and re-run this script, or install packages manually with:"
    echo "    pip install <package-name>"
fi

# ── Done ───────────────────────────────────────────────────────
echo ""
echo "🎉  Setup complete!"
echo ""
echo "Your virtual environment is active for this terminal session."
echo "To activate it again in a new terminal, run:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "To select this interpreter in VSCode:"
echo "  Cmd + Shift + P → 'Python: Select Interpreter' → .venv/bin/python"
echo ""
