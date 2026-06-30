#!/bin/bash
# Double-click this FIRST. It installs everything the project needs.
# Runs once, takes about 15 minutes. You can walk away while it works.
cd "$(dirname "$0")"

echo "======================================================"
echo "  YouTube Intelligence Engine - First-time Setup"
echo "  This takes ~15 minutes. It's safe to leave it running."
echo "======================================================"
echo ""

# 1. Install Homebrew (a tool installer) if it isn't already there.
if ! command -v brew >/dev/null 2>&1; then
  echo ">> Installing Homebrew. It may ask for your Mac password - just type it."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
# Make sure Homebrew is usable (works on both Apple-chip and Intel Macs).
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null)" || eval "$(/usr/local/bin/brew shellenv 2>/dev/null)"

# 2. Install Python 3.12.
echo ""
echo ">> Installing Python 3.12..."
brew install python@3.12

# 3. Build the project's private workspace and install all its libraries.
echo ""
echo ">> Installing the project (this is the long part - grab a coffee)..."
PY="$(brew --prefix)/bin/python3.12"
"$PY" -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m spacy download en_core_web_sm

# 4. Make the Run file double-clickable (in case the download stripped that).
chmod +x "2 - RUN DASHBOARD.command" 2>/dev/null

echo ""
echo "======================================================"
echo "  Setup complete!"
echo "  Now double-click:  2 - RUN DASHBOARD.command"
echo "======================================================"
