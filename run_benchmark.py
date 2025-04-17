#!/usr/bin/env python3
import argparse
from utils import clone_repo_to_cache, generate_prompt_and_expected_outputs


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
        "--ext",
        "-e",
        nargs="+",
        required=True,
        help="File extensions to process (e.g., .py .js .cpp)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Clone the repository
    try:
        repo_path = clone_repo_to_cache(args.repo)
        print(f"Repository ready at: {repo_path}")

        # Generate prompts and expected outputs
        output_dir = "benchmark-prompts"
        print(f"Generating prompt and expected output files for extensions: {args.ext}")
        generate_prompt_and_expected_outputs(
            repo_path=repo_path,
            extensions=args.ext,
            output_dir=output_dir,
        )
        print(f"All prompt and expected output files written to: {output_dir}")

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
