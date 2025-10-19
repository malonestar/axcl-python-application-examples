#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# --- Define Variables ---
VENV_NAME="axcl_venv"
ALIAS_NAME="axclenv"
REQS_FILE="requirements.txt"
BASHRC_FILE=~/.bashrc

echo "Looking for Python 3..."
if command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
elif command -v python &>/dev/null; then
    PYTHON_EXEC="python"
else
    echo "Error: Python 3 is not installed." >&2
    exit 1
fi
echo "Using $PYTHON_EXEC"

# --- Create Virtual Environment ---
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment '$VENV_NAME' with --system-site-packages..."
    $PYTHON_EXEC -m venv --system-site-packages "$VENV_NAME"
else
    echo "Virtual environment '$VENV_NAME' already exists. Skipping creation."
fi

# --- Activate and Install ---
echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

if [ -f "$REQS_FILE" ]; then
    echo "Installing requirements from $REQS_FILE..."
    pip install -r "$REQS_FILE"
else
    echo "Warning: $REQS_FILE not found. Skipping package installation."
fi

# --- Add Alias to .bashrc ---
echo "Checking for '$ALIAS_NAME' alias in $BASHRC_FILE..."

ACTIVATE_SCRIPT_PATH=$(readlink -f "$VENV_NAME/bin/activate")

ALIAS_STRING="alias $ALIAS_NAME='source \"$ACTIVATE_SCRIPT_PATH\"'"

ALIAS_STRING_FOR_SED=$(echo "$ALIAS_STRING" | sed -e 's/\\/\\\\/g' -e 's/#/\\#/g')

if ! grep -q "alias $ALIAS_NAME=" "$BASHRC_FILE"; then
    echo "Alias not found. Adding..."
    echo "" >> "$BASHRC_FILE"
    echo "# Alias for $VENV_NAME environment" >> "$BASHRC_FILE"
    echo "$ALIAS_STRING" >> "$BASHRC_FILE"
    echo "Alias '$ALIAS_NAME' added to $BASHRC_FILE."
else
    echo "Alias '$ALIAS_NAME' already exists. Overwriting..."
    sed -i "s#alias $ALIAS_NAME=.*#$ALIAS_STRING_FOR_SED#" "$BASHRC_FILE"
    echo "Alias '$ALIAS_NAME' has been corrected in $BASHRC_FILE."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "IMPORTANT: To make the '$ALIAS_NAME' alias work, you must reload your shell:"
echo "source ~/.bashrc"
echo "Or close this terminal window."
echo ""
echo "After that, just type '$ALIAS_NAME' to activate your venv from anywhere."