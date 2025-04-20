#!/bin/bash

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  uv venv .venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies using uv in the activated environment
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Setup complete. Virtual environment '.venv' is ready and dependencies are installed."
