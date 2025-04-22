#!/usr/bin/env python3
"""
Clones specified GitHub repositories into a local cache directory.

Purpose:
  This script takes a list of GitHub repository identifiers (either 'org/repo'
  or full HTTPS URLs) and clones them into a structured directory named
  'cached-repos'. It checks if a repository already exists in the cache
  and skips cloning if it does, printing a message instead.

Arguments:
  --repos, -r (required): One or more GitHub repository identifiers separated
                          by spaces.
                          Example: python 1_clone_repos.py -r org1/repoA https://github.com/org2/repoB

Inputs:
  - Command-line arguments specifying the repositories to clone.
  - Does not depend on outputs from previous scripts in the pipeline.

Outputs:
  - Creates the 'cached-repos/' directory in the current working directory
    if it doesn't already exist.
  - Clones each specified repository into a subdirectory structure within
    'cached-repos/'. The structure is 'cached-repos/<organization>/<repository_name>/'.
    Example: Cloning 'AbanteAI/mentat' results in 'cached-repos/AbanteAI/mentat/'.

File Modifications:
  - Creates the 'cached-repos/' directory.
  - Creates subdirectories within 'cached-repos/' for each organization and repository.
  - Populates these directories with the cloned repository files via 'git clone'.
  - Does *not* modify any files outside the 'cached-repos/' directory.
  - Does *not* modify existing cloned repositories if they are already present
    (e.g., it does not pull updates).
"""

import argparse
import sys
import os
import subprocess
from urllib.parse import urlparse


def standardize_repo_name(repo_name):
    """
    Convert various GitHub repository reference formats to a standard 'org/repo' format.

    Args:
        repo_name: A GitHub repository name, either as a full URL (https://github.com/org/repo)
                   or in the shorter format (org/repo).

    Returns:
        A standardized repository name in 'org/repo' format.

    Raises:
        ValueError: If the repository name cannot be parsed into a valid
                    org/repo format.
    """
    if repo_name.startswith("http"):
        # It's a full URL, extract the path
        parsed_url = urlparse(repo_name)
        path_parts = [p for p in parsed_url.path.split("/") if p]
        if len(path_parts) >= 2:
            org, repo = path_parts[:2]
            return f"{org}/{repo}"
        else:
            raise ValueError(f"Invalid GitHub URL: {repo_name}")
    else:
        # Assume it's in org/repo format
        if repo_name.count("/") != 1:
            raise ValueError(
                f"Repository name should be in 'org/repo' format: {repo_name}"
            )
        return repo_name


def clone_repo_to_cache(repo_name):
    """
    Clone a GitHub repository into the cached-repos directory.

    Args:
        repo_name: A GitHub repository name, either as a full URL (https://github.com/org/repo)
                   or in the shorter format (org/repo).

    Returns:
        The path to the cloned repository (cached-repos/org/repo).
    """
    # Create cached-repos directory if it doesn't exist
    cache_dir = "cached-repos"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created cache directory: {cache_dir}")

    # Standardize the repository name
    std_repo_name = standardize_repo_name(repo_name)

    # Generate target directory name
    org, repo = std_repo_name.split("/")
    target_dir = os.path.join(cache_dir, org, repo)

    # Check if repo already exists
    if os.path.exists(target_dir):
        print(f"Repository already exists at {target_dir}")
        # Optionally, you could pull the latest changes here
        return target_dir

    # Create parent directory if needed
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    # Clone the repository
    github_url = f"https://github.com/{std_repo_name}.git"
    try:
        # Don't capture output so clone progress is visible to the user
        subprocess.run(
            ["git", "clone", github_url, target_dir],
            check=True,
        )
        print(f"Successfully cloned {std_repo_name} to {target_dir}")
        return target_dir
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        raise


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
