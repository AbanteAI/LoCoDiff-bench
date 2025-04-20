#!/bin/bash

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first."
    echo "You can install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv || { echo "Failed to create virtual environment"; exit 1; }
else
    echo "Virtual environment already exists at .venv"
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Install dependencies using uv in the activated environment
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt || { echo "Failed to install dependencies"; exit 1; }

echo "Setup complete. Virtual environment '.venv' is ready and dependencies are installed."
