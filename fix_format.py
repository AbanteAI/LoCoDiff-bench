
# Read current file content
filepath = "results_explorer/app.py"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Detect and print line endings
lines = content.splitlines()
print(f"File has {len(lines)} lines")

# Check for problematic lines with trailing whitespace
for i, line in enumerate(lines, 1):
    if line.rstrip() != line:
        print(f"Line {i} has trailing whitespace: '{line}'")

# Check for blank lines with whitespace
for i, line in enumerate(lines, 1):
    if line.strip() == "" and line != "":
        print(f"Line {i} has blank line with whitespace: '{line}'")

# Ensure the file ends with exactly one newline
if not content.endswith("\n"):
    print("File doesn't end with newline")
elif content.endswith("\n\n"):
    print("File ends with multiple newlines")

# Clean the file content - strip trailing whitespace and ensure consistent line endings
clean_lines = [line.rstrip() for line in lines]
clean_content = "\n".join(clean_lines) + "\n"  # Ensure exactly one newline at end

# Write back the clean content
with open(filepath, "w", encoding="utf-8") as f:
    f.write(clean_content)

print("File has been cleaned and rewritten")
