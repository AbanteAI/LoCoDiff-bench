#!/usr/bin/env python3
import argparse
import os
from utils import clone_repo_to_cache


def clone_repo(repo_name, shallow=False):
    """
    Clone a GitHub repository into the cached-repos directory.
    Wrapper around clone_repo_to_cache that adds shallow clone support.

    Args:
        repo_name: A GitHub repository name in 'org/repo' format or full URL
        shallow: If True, performs a shallow clone (--depth 1)

    Returns:
        The path to the cloned repository
    """
    # Set environment variable for shallow clone if requested
    if shallow:
        # Keep track of the original environment value, if any
        orig_value = os.environ.get("GIT_CLONE_DEPTH")
        try:
            # Set depth=1 for shallow clone
            os.environ["GIT_CLONE_DEPTH"] = "1"
            return clone_repo_to_cache(repo_name)
        finally:
            # Restore original environment
            if orig_value is not None:
                os.environ["GIT_CLONE_DEPTH"] = orig_value
            else:
                os.environ.pop("GIT_CLONE_DEPTH", None)
    else:
        # Normal (full) clone
        return clone_repo_to_cache(repo_name)


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
        "--shallow",
        action="store_true",
        help="Perform a shallow clone (--depth 1) for faster cloning",
    )

    # Parse arguments
    args = parser.parse_args()

    # Clone the repository
    try:
        repo_path = clone_repo(args.repo, args.shallow)
        print(f"Repository ready at: {repo_path}")

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
    exit(main())
