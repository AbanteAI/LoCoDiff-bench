#!/usr/bin/env python3
import argparse
import sys
from utils import clone_repo_to_cache


def main():
    parser = argparse.ArgumentParser(
        description="Clone multiple GitHub repositories into the 'cached-repos' directory."
    )
    parser.add_argument(
        "--repos",
        "-r",
        nargs="+",
        required=True,
        help="List of GitHub repositories to clone (format: 'org/repo' or full URL).",
    )

    args = parser.parse_args()

    print("--- Starting Repository Cloning ---")
    success_count = 0
    fail_count = 0

    for repo_name in args.repos:
        print(f"\nCloning {repo_name}...")
        try:
            repo_path = clone_repo_to_cache(repo_name)
            print(f"Successfully cloned or found {repo_name} at {repo_path}")
            success_count += 1
        except ValueError as e:
            print(f"Error processing repository name {repo_name}: {e}")
            fail_count += 1
        except Exception as e:
            print(f"Failed to clone {repo_name}: {e}")
            fail_count += 1

    print("\n--- Cloning Summary ---")
    print(f"Successfully cloned/found: {success_count}")
    print(f"Failed: {fail_count}")

    if fail_count > 0:
        return 1  # Indicate failure
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
