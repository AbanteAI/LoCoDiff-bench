#!/usr/bin/env python3
import argparse
import os
import subprocess
from utils import standardize_repo_name


def is_public_repo(repo_name):
    """
    Check if a GitHub repository is public by making an unauthenticated API request.

    Args:
        repo_name: Repository name in 'org/repo' format

    Returns:
        True if the repository is public, False otherwise
    """
    import urllib.request
    import urllib.error
    import json

    std_repo_name = standardize_repo_name(repo_name)
    org, repo = std_repo_name.split("/")

    url = f"https://api.github.com/repos/{org}/{repo}"

    try:
        # Create a request with a custom user agent
        req = urllib.request.Request(url, headers={"User-Agent": "LoCoDiff-bench"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            # If we got a response without auth, it's public
            return True
    except urllib.error.HTTPError as e:
        # 404 means it doesn't exist or is private
        if e.code == 404:
            return False
        # 403 can mean API rate limiting
        elif e.code == 403:
            # Assume it might be public but we're rate limited
            print("Warning: GitHub API rate limit may have been reached.")
            return True
        else:
            print(f"HTTP error checking repo visibility: {e.code}")
            return False
    except Exception as e:
        print(f"Error checking repo visibility: {e}")
        # Assume it might be public if we can't check
        return True


def clone_repo(repo_name, shallow=False):
    """
    Clone a GitHub repository into the cached-repos directory.
    Uses anonymous cloning for public repositories.

    Args:
        repo_name: A GitHub repository name in 'org/repo' format or full URL
        shallow: If True, performs a shallow clone (--depth 1)

    Returns:
        The path to the cloned repository
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
        return target_dir

    # Create parent directory if needed
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    # Check if the repository is public
    is_public = is_public_repo(std_repo_name)
    if is_public:
        print(f"Detected that {std_repo_name} is a public repository.")
    else:
        print(
            f"Warning: {std_repo_name} appears to be a private repository or doesn't exist."
        )
        print(
            "You will need to have appropriate credentials configured in your git config."
        )

    # For public repositories, we can try to clone without credentials
    github_url = f"https://github.com/{std_repo_name}"

    try:
        # Basic clone command
        cmd = ["git", "clone"]

        # Add --depth flag if shallow cloning is requested
        if shallow:
            cmd.extend(["--depth", "1"])

        # Add repository URL and target directory
        cmd.extend([github_url, target_dir])

        print(f"Cloning repository {std_repo_name}...")

        # Set GIT_TERMINAL_PROMPT=0 to prevent git from prompting for credentials
        # This ensures it fails instead of hanging if credentials are required
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"

        # Execute the clone command
        result = subprocess.run(
            cmd,
            env=env,
            check=False,
            timeout=60,  # Longer timeout for larger repos
        )

        if result.returncode == 0:
            print(f"Successfully cloned {std_repo_name} to {target_dir}")
            return target_dir
        else:
            if is_public:
                print(
                    "Failed to clone public repository. This might be due to network issues or rate limiting."
                )
            else:
                print(
                    "Failed to clone repository. If this is a private repository, make sure you have:"
                )
                print("1. Proper authentication configured in your git config")
                print("2. Access to the repository")

            # Clean up any partial clone
            if os.path.exists(target_dir):
                import shutil

                shutil.rmtree(target_dir)

            raise RuntimeError(f"Failed to clone repository {std_repo_name}")

    except subprocess.TimeoutExpired:
        print(
            f"Timeout while cloning {std_repo_name}. The repository might be too large or there might be network issues."
        )

        # Clean up partial clone
        if os.path.exists(target_dir):
            import shutil

            shutil.rmtree(target_dir)

        raise RuntimeError(f"Timeout while cloning repository {std_repo_name}")

    except Exception as e:
        print(f"Error cloning repository: {e}")

        # Clean up partial clone
        if os.path.exists(target_dir):
            import shutil

            shutil.rmtree(target_dir)

        raise


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
