#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies to ensure new packages are available for type checking
echo "Installing dependencies..."
uv pip install --system -r requirements.txt

# Run formatter
echo "Running formatter..."
ruff format .

# Run linter with fix flag
echo "Running linter..."
ruff check --fix .

# Run type checking
echo "Running type checker..."
pyright utils.py

# Note: Skipping pytest as there are currently no tests in the repository
# If tests are added in the future, uncomment the following lines:
# echo "Running tests..."
# pytest
