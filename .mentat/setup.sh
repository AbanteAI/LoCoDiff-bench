#!/bin/bash

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  uv venv .venv
fi

# Install dependencies using uv in the virtual environment
echo "Installing dependencies..."
uv pip install -r requirements.txt --venv .venv
