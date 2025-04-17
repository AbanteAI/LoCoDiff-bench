#!/usr/bin/env python3
import argparse
from typing import List

from utils import clone_repo_to_cache, generate_prompts_for_repo


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
        help="File extensions to include when generating prompts (default: .py)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help=(
            "Directory in which to place generated prompt/expected files. "
            "Defaults to 'generated-data/<repo-name>'."
        ),
    )

    # Parse arguments
    args = parser.parse_args()

    # Clone the repository
    try:
        repo_path = clone_repo_to_cache(args.repo)
        print(f"Repository ready at: {repo_path}")

        # ------------------------------------------------------------------ #
        # Generate LoCoDiff prompts & expected outputs                        #
        # ------------------------------------------------------------------ #
        written: List[str] = generate_prompts_for_repo(
            repo_path=repo_path,
            repo_name=args.repo
            if "/" in args.repo
            else args.repo.split("github.com/")[-1],
            extensions=args.extensions,
            output_dir=args.output_dir,
        )
        print(f"Generated {len(written)} prompt/expected-output pairs.")
        for f in written[:5]:  # only show a few so logs aren't huge
            print(f"  - {f}")

        print("Ready for benchmarking.")
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
