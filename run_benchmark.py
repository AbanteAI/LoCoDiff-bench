#!/usr/bin/env python3
import argparse
from utils import clone_repo_to_cache, get_file_modification_history


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
        "--file",
        "-f",
        help="Find all commits that modified this file (following first parent path)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Clone the repository
    try:
        repo_path = clone_repo_to_cache(args.repo)
        print(f"Repository ready at: {repo_path}")

        # If file is specified, find its modification history
        if args.file:
            try:
                commits = get_file_modification_history(repo_path, args.file)
                if commits:
                    print(f"\nCommits that modified '{args.file}' (newest to oldest):")
                    for i, commit in enumerate(commits, 1):
                        print(f"{i}. {commit}")
                else:
                    print(f"\nNo commits found that modified '{args.file}'")
            except ValueError as e:
                print(f"Error: {e}")
                return 1
        else:
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
