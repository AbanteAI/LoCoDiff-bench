import re

# Read the original file
with open("benchmark_pipeline/3_generate_pages.py", "r") as f:
    content = f.read()

# Read our new JS content
with open("new_chart_javascript.js", "r") as f:
    js_content = f.read()

# Find the beginning of the function
pattern = r'def create_chart_javascript\(\) -> str:.*?return """'
match = re.search(pattern, content, re.DOTALL)

if match:
    start_pos = match.end()

    # Find where to end - look for a triple quote followed by possible whitespace and a function definition
    end_pattern = r'"""[\s\n]*def '
    end_match = re.search(end_pattern, content[start_pos:])

    if end_match:
        end_pos = start_pos + end_match.start()

        # Build the new content
        new_content = content[:start_pos] + "\n"
        new_content += (
            '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n<script>\n'
        )
        new_content += js_content
        new_content += "\n</script>\n"
        new_content += content[end_pos:]

        # Write it back
        with open("benchmark_pipeline/3_generate_pages.py", "w") as f:
            f.write(new_content)
        print("Successfully replaced chart JavaScript")
    else:
        print("Could not find the end of the chart JavaScript function")
else:
    print("Could not find the chart JavaScript function")
