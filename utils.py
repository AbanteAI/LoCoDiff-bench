import os
import subprocess
from urllib.parse import urlparse
from typing import List


def standardize_repo_name(repo_name):
    """
    Convert various GitHub repository reference formats to a standard 'org/repo' format.

    Args:
        repo_name: A GitHub repository name, either as a full URL (https://github.com/org/repo)
                   or in the shorter format (org/repo).

    Returns:
        A standardized repository name in 'org/repo' format.

    Raises:
        ValueError: If the repository name cannot be parsed into a valid org/repo format.
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


def find_files_with_extensions(root_dir: str, extensions: List[str]) -> List[str]:
    """
    Recursively find all files in root_dir with the given extensions.

    Args:
        root_dir: Directory to search.
        extensions: List of file extensions (e.g., ['.py', '.js']).

    Returns:
        List of file paths (relative to root_dir) matching the extensions.
    """
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            for ext in extensions:
                if filename.endswith(ext):
                    full_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(full_path, root_dir)
                    matches.append(rel_path)
                    break
    return matches


def generate_prompt_and_expected_outputs(
    repo_path: str,
    extensions: List[str],
    output_dir: str,
    repo_name: str = None,
):
    """
    For each file in repo_path with the given extensions, create a prompt and expected output file.

    Args:
        repo_path: Path to the cloned repo.
        extensions: List of file extensions to process.
        output_dir: Directory to write prompt/expected output files.
        repo_name: Optional, used for naming output files. If None, uses the repo directory name.
    """
    if repo_name is None:
        repo_name = os.path.basename(repo_path.rstrip("/"))

    os.makedirs(output_dir, exist_ok=True)
    files = find_files_with_extensions(repo_path, extensions)
    if not files:
        print(f"No files with extensions {extensions} found in {repo_path}")
        return

    for rel_file in files:
        abs_file = os.path.join(repo_path, rel_file)
        # Prepare output filenames
        safe_rel_file = rel_file.replace(os.sep, "_")
        prompt_filename = f"{repo_name}_{safe_rel_file}_prompt.txt"
        expected_filename = f"{repo_name}_{safe_rel_file}_expectedoutput.txt"
        prompt_path = os.path.join(output_dir, prompt_filename)
        expected_path = os.path.join(output_dir, expected_filename)

        # Run git log command
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    repo_path,
                    "log",
                    "-p",
                    "--cc",
                    "--follow",
                    "--topo-order",
                    "--reverse",
                    "--",
                    rel_file,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            git_log_output = result.stdout
        except subprocess.CalledProcessError as e:
            git_log_output = f"Error running git log: {e}"

        # Write prompt file
        prompt_content = (
            "You are being tested. Your goal is to reconstruct the current state of a file, given the history of changes made to that file. For your response, simply output the exact final state of the file, wrapped in triple backticks (```):\n\n"
            f"> git log -p --cc --follow --topo-order --reverse -- {rel_file}\n\n"
            f"{git_log_output}"
        )
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt_content)

        # Write expected output file
        try:
            with open(abs_file, "r", encoding="utf-8") as src:
                file_contents = src.read()
        except Exception as e:
            file_contents = f"Error reading file: {e}"

        expected_content = f"```\n{file_contents}\n```"
        with open(expected_path, "w", encoding="utf-8") as f:
            f.write(expected_content)

        print(f"Wrote prompt and expected output for {rel_file}")
