#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Run formatter
echo "Running formatter..."
ruff format .

# Run linter with fix flag
echo "Running linter..."
ruff check --fix .

# Run type checking
echo "Running type checker..."
# Ensure pyright is installed in the venv and run it using the venv's executable path
uv pip install pyright &> /dev/null # Install quietly if already present
.venv/bin/pyright .

# Note: Skipping pytest as there are currently no tests in the repository
# If tests are added in the future, uncomment the following lines:
# echo "Running tests..."
# pytest
