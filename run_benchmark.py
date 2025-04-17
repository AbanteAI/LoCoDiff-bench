#!/usr/bin/env python3
import argparse
from utils import clone_repo_to_cache, generate_locodiff_prompts


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Run LoCoDiff benchmarks on code repositories"
    )
    parser.add_argument(
        "--repo",
        "-r",
        required=True,
        help="GitHub repository to clone (format: 'org/repo' or full URL)",
    )
    parser.add_argument(
        "--extensions",
        "-e",
        required=True,
        help="Comma-separated list of file extensions to process (e.g., '.py,.md')",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="generated-prompts",
        help="Directory to store generated prompts and expected outputs",
    )

    # Parse arguments
    args = parser.parse_args()

    # Process extensions
    extensions = tuple(ext.strip() for ext in args.extensions.split(","))
    if not all(ext.startswith(".") for ext in extensions):
        print("Error: All extensions must start with a '.'")
        return 1

    # Clone the repository
    try:
        repo_path = clone_repo_to_cache(args.repo)
        print(f"Repository ready at: {repo_path}")

        # Generate prompts and expected outputs
        print(f"Generating prompts for extensions: {', '.join(extensions)}")
        generate_locodiff_prompts(repo_path, extensions, args.output_dir)
        print(f"Generated files stored in: {args.output_dir}")

    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
