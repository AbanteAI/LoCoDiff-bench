# JavaScript Linting Proposal for Precommit Script

## Current Issue
The precommit script currently only checks Python code, but we also have JavaScript code embedded in HTML templates. This led to a runtime reference error that wasn't caught during development.

## Proposed Solution
Add JavaScript linting to the precommit script to catch common JavaScript errors before they reach production.

### Implementation Options

#### Option 1: Light ESLint Integration
This option focuses on quick installation and basic checks:

```bash
# Add to .mentat/precommit.sh

# JavaScript linting setup
echo "Setting up JavaScript linting..."
if ! command -v eslint &> /dev/null; then
    echo "ESLint not found, installing..."
    npm install -g eslint
fi

# Extract JavaScript from HTML files in docs directory
echo "Extracting and linting JavaScript from HTML files..."
if [ -d "docs" ]; then
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
    
    # Run ESLint on extracted JS
    eslint --no-eslintrc --env browser --global fetch,Chart --rule 'no-undef: 2' .tmp_js_lint/*.js || {
        echo "WARNING: JavaScript linting found issues. These should be fixed before committing."
        # Optional: exit 1 to block commit if you want strict enforcement
    }
    
    # Clean up
    rm -rf .tmp_js_lint
fi
```

#### Option 2: More Comprehensive Setup
This option creates a proper ESLint configuration for the project:

1. Create a `.eslintrc.js` file in the project root:
```javascript
module.exports = {
  env: {
    browser: true,
    es2021: true,
  },
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
  },
  rules: {
    'no-undef': 'error',
    'no-unused-vars': 'warn'
  },
  globals: {
    Chart: 'readonly',
    fetch: 'readonly',
  }
};
```

2. Add to `.mentat/precommit.sh`:
```bash
# JavaScript linting
echo "Running JavaScript linting..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Warning: Node.js not found, skipping JavaScript linting"
else
    # Install ESLint if not already installed
    if ! command -v eslint &> /dev/null; then
        echo "ESLint not found, installing locally..."
        npm install --save-dev eslint
    fi
    
    # Extract JavaScript from HTML files in docs directory
    echo "Extracting and linting JavaScript from HTML files..."
    if [ -d "docs" ]; then
        # Create temporary directory for extracted JS
        mkdir -p .tmp_js_lint
        
        # Extract JS from HTML files
        for file in docs/*.html; do
            if [ -f "$file" ]; then
                echo "Extracting JS from $file"
                grep -A 1000 "<script>" "$file" | grep -v "<script>" | grep -B 1000 "</script>" | grep -v "</script>" > ".tmp_js_lint/$(basename "$file").js"
            fi
        done
        
        # Run ESLint on extracted JS
        npx eslint .tmp_js_lint/*.js || {
            echo "WARNING: JavaScript linting found issues. These should be fixed before committing."
            # exit 1 # Uncomment to block commit if linting fails
        }
        
        # Clean up
        rm -rf .tmp_js_lint
    fi
fi
```

## Recommendation
Since this is primarily a Python project with just a small amount of JavaScript in the generated HTML, **Option 1** is recommended for its simplicity and light footprint. It focuses specifically on catching undefined variables, which was the source of our recent bug.

If JavaScript usage grows in the project, **Option 2** would provide more comprehensive linting and could be extended with additional rules.
