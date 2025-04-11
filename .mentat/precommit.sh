#!/bin/bash

# Run formatter
ruff format .

# Run linter with fix flag
ruff check --fix .

# Run type checking
pyright utils.py

# Run tests
pytest
