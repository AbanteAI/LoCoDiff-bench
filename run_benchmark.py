#!/usr/bin/env python3
import argparse
from utils import (
    clone_repo_to_cache,
    generate_prompts_and_expected,
    count_tokens,
    analyze_file_history,
)


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run benchmarks on code repositories")
    parser.add_argument(
        "--repo",
        "-r",
        required=True,
        help="GitHub repository to clone (format: 'org/repo' or full URL)",
    )
    parser.add_argument(
        "--extensions",
        "-e",
        nargs="+",
        default=[".py"],
        help="File extensions to process (include the dot), e.g. .py .txt",
    )

    # Parse arguments
    args = parser.parse_args()

    # Clone the repository
    try:
        repo_path = clone_repo_to_cache(args.repo)
        print(f"Repository ready at: {repo_path}")

        # Generate prompts and expected outputs and collect file info
        files = generate_prompts_and_expected(repo_path, args.extensions)
        print("Generated prompts and expected outputs in 'generated_prompts/'.\n")

        # Prepare and print summary table
        print("## Benchmark Summary\n")
        header = [
            "File",
            "Prompt Tokens",
            "Expected Tokens",
            "Commits",
            "Lines Added",
            "Lines Removed",
            "Final Lines",
        ]
        # Markdown table header
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("---" for _ in header) + "|")

        for info in files:
            rel = info["rel_path"]
            # Load prompt and expected text
            with open(info["prompt_path"], "r", encoding="utf-8") as pf:
                prompt_text = pf.read()
            with open(info["expected_path"], "r", encoding="utf-8") as ef:
                expected_text = ef.read()

            # Compute token counts
            p_tokens = count_tokens(prompt_text)
            e_tokens = count_tokens(expected_text)

            # Compute history stats
            stats = analyze_file_history(repo_path, rel)

            # Print row
            row = [
                rel,
                str(p_tokens),
                str(e_tokens),
                str(stats["commits"]),
                str(stats["lines_added"]),
                str(stats["lines_removed"]),
                str(stats["final_lines"]),
            ]
            print("| " + " | ".join(row) + " |")

        print("\nReady for benchmarking.")

    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
