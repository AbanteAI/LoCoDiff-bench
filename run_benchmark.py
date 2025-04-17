#!/usr/bin/env python3
import argparse
from utils import clone_repo_to_cache, create_benchmark_files


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
        help="File extensions to process (e.g., .py .js .ts). Default: .py",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip creating benchmark files (just clone the repository)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Clone the repository
    try:
        repo_path = clone_repo_to_cache(args.repo)
        print(f"Repository ready at: {repo_path}")

        # Skip benchmark if requested
        if args.skip_benchmark:
            print("Skipping benchmark file generation as requested.")
            return 0

        # Process file extensions
        extensions = [ext.lstrip(".") for ext in args.extensions]
        print(f"Creating benchmark files for extensions: {', '.join(extensions)}")

        # Create benchmark files
        prompt_files, expected_files = create_benchmark_files(repo_path, extensions)

        # Report results
        total_files = len(prompt_files)
        if total_files > 0:
            print(f"Successfully created {total_files} benchmark file pairs")
            print("Files are located in the 'benchmark-files' directory")
        else:
            print("No benchmark files were created")

    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
