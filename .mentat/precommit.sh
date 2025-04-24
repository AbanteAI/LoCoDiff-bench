#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please run .mentat/setup.sh first."
    exit 1
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Debug prints
echo "Running git status..."
git status

echo "Running git log -n 2..."
git log -n 2

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
# We'll use the executable from the virtual environment directly
# No need to reinstall dependencies as setup.sh should have done this
# Only check relevant source directories, exclude cached repos etc.
.venv/bin/pyright benchmark_pipeline results_explorer

# Note: Skipping pytest as there are currently no tests in the repository
# If tests are added in the future, uncomment the following lines:
# echo "Running tests..."
# pytest

echo "Precommit checks completed successfully!"
