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
pyright utils.py

# Run tests
echo "Running tests..."
pytest
