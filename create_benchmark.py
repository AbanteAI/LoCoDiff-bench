#!/usr/bin/env python3
import argparse
from utils import (
    clone_repo_to_cache,
    generate_prompts_and_expected,
    # Removed get_model_response_openrouter, sys, difflib
)


def print_stats_table(stats_list):
    """Prints a formatted table of the collected statistics."""
    if not stats_list:
        print("No statistics generated.")
        return

    # Sort by prompt tokens (descending) before printing
    stats_list.sort(key=lambda x: x["prompt_tokens"], reverse=True)

    # Determine column widths dynamically
    max_len_filename = max(len(s["filename"]) for s in stats_list) if stats_list else 10
    col_widths = {
        "filename": max(max_len_filename, 8),  # Min width 8 for "Filename"
        "prompt_tokens": 13,  # "Prompt Tokens"
        "expected_tokens": 15,  # "Expected Tokens"
        "num_commits": 8,  # "Commits"
        "lines_added": 7,  # "Added"
        "lines_deleted": 9,  # "Deleted"
        "final_lines": 11,  # "Final Lines"
    }

    # Header
    header = (
        f"{'Filename':<{col_widths['filename']}} | "
        f"{'Prompt Tokens':>{col_widths['prompt_tokens']}} | "
        f"{'Expected Tokens':>{col_widths['expected_tokens']}} | "
        f"{'Commits':>{col_widths['num_commits']}} | "
        f"{'Added':>{col_widths['lines_added']}} | "
        f"{'Deleted':>{col_widths['lines_deleted']}} | "
        f"{'Final Lines':>{col_widths['final_lines']}}"
    )
    print(header)
    print("-" * len(header))

    # Rows
    for stats in stats_list:
        row = (
            f"{stats['filename']:<{col_widths['filename']}} | "
            f"{stats['prompt_tokens']:>{col_widths['prompt_tokens']}} | "
            f"{stats['expected_tokens']:>{col_widths['expected_tokens']}} | "
            f"{stats['num_commits']:>{col_widths['num_commits']}} | "
            f"{stats['lines_added']:>{col_widths['lines_added']}} | "
            f"{stats['lines_deleted']:>{col_widths['lines_deleted']}} | "
            f"{stats['final_lines']:>{col_widths['final_lines']}}"
        )
        print(row)


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Generate benchmark prompts and expected outputs from a GitHub repository's history."
    )
    # Output directory is hardcoded
    output_dir = "generated_prompts"

    # Arguments for generating prompts (now required)
    parser.add_argument(
        "--repo",
        "-r",
        required=True,
        help="GitHub repository to clone for prompt generation (format: 'org/repo' or full URL)",
    )
    parser.add_argument(
        "--extensions",
        "-e",
        nargs="+",
        default=[".py"],  # Keep default but make it required
        required=True,
        help="File extensions to process for prompt generation (include the dot), e.g. .py .txt",
    )

    # Parse arguments
    args = parser.parse_args()

    # --- Generation Logic ---
    try:
        repo_path = clone_repo_to_cache(args.repo)
        print(f"Repository ready at: {repo_path}")

        # Generate prompts and expected outputs
        print(f"Generating prompts and expected outputs in '{output_dir}/'...")
        stats_list = generate_prompts_and_expected(
            repo_path, args.extensions, output_dir
        )

        # Print statistics table
        print("\n--- Statistics ---")
        print_stats_table(stats_list)
        print("\nBenchmark creation complete.")

    except ValueError as e:
        print(f"Error during benchmark creation: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred during benchmark creation: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
