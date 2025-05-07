#!/usr/bin/env python3
"""
Generates an example git history for the LoCoDiff benchmark explanation.

This script:
1. Creates a temporary git repository
2. Makes a series of commits to demonstrate branching and merging
3. Outputs the git log with diffs in the same format used for benchmark prompts
"""

import os
import shutil
import subprocess
import tempfile


def run_command(cmd, cwd=None):
    """Run a shell command and return its output"""
    result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout


def setup_repo():
    """Set up a temporary git repository"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Initialize git repo
        run_command(["git", "init"], cwd=temp_dir)

        # Configure git user for commits
        run_command(
            ["git", "config", "user.email", "example@example.com"], cwd=temp_dir
        )
        run_command(["git", "config", "user.name", "Example User"], cwd=temp_dir)

        # Initial commit (A)
        with open(os.path.join(temp_dir, "simple_math.py"), "w") as f:
            f.write("""# simple_math.py
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

# Example usage
if __name__ == "__main__":
    print(calculate_sum([1, 2, 3, 4, 5]))  # Should output: 15
""")

        run_command(["git", "add", "simple_math.py"], cwd=temp_dir)
        run_command(
            ["git", "commit", "-m", "Initial implementation of sum calculator"],
            cwd=temp_dir,
        )

        # Create branch 1 (optimization)
        run_command(["git", "checkout", "-b", "feature/optimize"], cwd=temp_dir)

        # Make changes for branch 1 (Commit B)
        with open(os.path.join(temp_dir, "simple_math.py"), "w") as f:
            f.write("""# simple_math.py
def calculate_sum(numbers):
    return sum(numbers)  # More efficient implementation

# Example usage
if __name__ == "__main__":
    print(calculate_sum([1, 2, 3, 4, 5]))  # Should output: 15
""")

        run_command(["git", "add", "simple_math.py"], cwd=temp_dir)
        run_command(
            ["git", "commit", "-m", "Optimize sum calculation using built-in function"],
            cwd=temp_dir,
        )

        # Go back to main branch to create branch 2
        run_command(["git", "checkout", "master"], cwd=temp_dir)

        # Create branch 2 (new feature)
        run_command(["git", "checkout", "-b", "feature/average"], cwd=temp_dir)

        # Make changes for branch 2 (Commit C)
        with open(os.path.join(temp_dir, "simple_math.py"), "w") as f:
            f.write("""# simple_math.py
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

def calculate_average(numbers):
    if not numbers:
        return 0
    return calculate_sum(numbers) / len(numbers)

# Example usage
if __name__ == "__main__":
    print(calculate_sum([1, 2, 3, 4, 5]))  # Should output: 15
    print(calculate_average([1, 2, 3, 4, 5]))  # Should output: 3.0
""")

        run_command(["git", "add", "simple_math.py"], cwd=temp_dir)
        run_command(
            ["git", "commit", "-m", "Add average calculation function"], cwd=temp_dir
        )

        # Merge branch 2 into branch 1
        run_command(["git", "checkout", "feature/optimize"], cwd=temp_dir)

        try:
            # This will fail due to merge conflict
            run_command(["git", "merge", "feature/average"], cwd=temp_dir)
        except subprocess.CalledProcessError:
            # Resolve the conflict manually
            with open(os.path.join(temp_dir, "simple_math.py"), "w") as f:
                f.write("""# simple_math.py
def calculate_sum(numbers):
    return sum(numbers)  # Kept the optimized version

def calculate_average(numbers):
    if not numbers:
        return 0
    return calculate_sum(numbers) / len(numbers)

# Example usage
if __name__ == "__main__":
    print(calculate_sum([1, 2, 3, 4, 5]))  # Should output: 15
    print(calculate_average([1, 2, 3, 4, 5]))  # Should output: 3.0
""")

            run_command(["git", "add", "simple_math.py"], cwd=temp_dir)
            run_command(
                ["git", "commit", "-m", "Merge feature/average into feature/optimize"],
                cwd=temp_dir,
            )

        # Get the git log with patches (same format as benchmark prompts)
        git_log = run_command(
            [
                "git",
                "log",
                "-p",
                "--cc",
                "--topo-order",
                "--reverse",
                "--",
                "simple_math.py",
            ],
            cwd=temp_dir,
        )

        # Get final file state for verification
        with open(os.path.join(temp_dir, "simple_math.py"), "r") as f:
            final_state = f.read()

        return git_log, final_state

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


def save_output(git_log, final_state):
    """Save the git log and final file state to files in the benchmark directory"""
    # Create output directory if it doesn't exist
    os.makedirs("locodiff-example", exist_ok=True)

    # Save git log to a file
    with open("locodiff-example/example_prompt.txt", "w") as f:
        f.write(git_log)

    # Save final state to a file
    with open("locodiff-example/example_expected.txt", "w") as f:
        f.write(final_state)

    print("Example prompt saved to 'locodiff-example/example_prompt.txt'")
    print("Expected output saved to 'locodiff-example/example_expected.txt'")


def main():
    print("Generating example git history...")
    git_log, final_state = setup_repo()
    save_output(git_log, final_state)
    print("Done!")


if __name__ == "__main__":
    main()
