#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Verify ruff version
echo "Ruff version in precommit:"
ruff --version

# Run formatter
echo "Running formatter..."
ruff format .

# Run linter with fix flag
echo "Running linter..."
ruff check --fix .

# Run type checking
echo "Running type checker..."
# Ensure all dependencies, including type checking tools, are installed
echo "Ensuring dependencies are installed..."
uv pip install -r requirements.txt
# Now run pyright using the venv's executable path
.venv/bin/pyright .

# Note: Skipping pytest as there are currently no tests in the repository
# If tests are added in the future, uncomment the following lines:
# echo "Running tests..."
# pytest
