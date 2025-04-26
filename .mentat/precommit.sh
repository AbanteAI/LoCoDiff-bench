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
.venv/bin/pyright benchmark_pipeline

# JavaScript linting setup
echo "Checking for JavaScript in HTML files..."
if [ -d "docs" ] && [ -f "docs/index.html" ]; then
    echo "Found docs/index.html, checking for JavaScript..."
    
    # Check if Node.js is installed
    if command -v node &> /dev/null; then
        # Create temporary directory for extracted JS
        mkdir -p .tmp_js_lint
        
        # Extract JS from HTML files
        for file in docs/*.html; do
            if [ -f "$file" ]; then
                echo "Extracting JS from $file"
                # Extract script tags content
                grep -A 1000 "<script>" "$file" | grep -v "<script>" | grep -B 1000 "</script>" | grep -v "</script>" > ".tmp_js_lint/$(basename "$file").js"
            fi
        done
        
        # Basic JS syntax check with node
        for js_file in .tmp_js_lint/*.js; do
            if [ -s "$js_file" ]; then
                echo "Checking JavaScript syntax in $js_file"
                # Run node --check to verify syntax
                node --check "$js_file" || {
                    echo "ERROR: JavaScript syntax error found in extracted code from $js_file."
                    cat "$js_file"
                    exit 1
                }
                
                # Simple grep check for common undefined variables
                echo "Checking for undefined functions..."
                for func in wilson_score_interval initializeChart updateChart; do
                    if grep -q "\b$func\b" "$js_file"; then
                        if ! grep -q "function $func" "$js_file"; then
                            echo "WARNING: Function '$func' is used but might not be defined in the same file."
                            echo "Make sure it's defined before use or included from another file."
                        fi
                    fi
                done
            fi
        done
        
        # Clean up
        rm -rf .tmp_js_lint
        echo "JavaScript checks completed."
    else
        echo "Node.js not found. Skipping JavaScript syntax checks."
    fi
fi

# Note: Skipping pytest as there are currently no tests in the repository
# If tests are added in the future, uncomment the following lines:
# echo "Running tests..."
# pytest

echo "Precommit checks completed successfully!"
