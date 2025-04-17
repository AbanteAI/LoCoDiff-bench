#!/usr/bin/env python3
import argparse
from utils import clone_repo_to_cache, generate_prompts_and_expected


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

        # Generate prompts and expected outputs for specified file types
        generate_prompts_and_expected(repo_path, args.extensions)
        print("Generated prompts and expected outputs in 'generated_prompts/'.")

        # In the future, additional benchmark functionality will be added here
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
