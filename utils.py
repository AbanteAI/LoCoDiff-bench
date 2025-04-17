import os
import re
import subprocess
from pathlib import Path
from typing import Sequence
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


def _sanitize_filepath(filepath: str) -> str:
    """Sanitize a filepath to be used in a filename."""
    # Remove leading './' if present
    if filepath.startswith("./"):
        filepath = filepath[2:]
    # Replace path separators with underscores
    sanitized = filepath.replace(os.path.sep, "_")
    # Remove or replace other potentially problematic characters (optional, adjust as needed)
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", sanitized)
    return sanitized


def generate_locodiff_prompts(
    repo_path: str, extensions: Sequence[str], output_dir: str
):
    """
    Generate LoCoDiff prompts and expected outputs for files in a repository.

    Args:
        repo_path: The local path to the cloned repository.
        extensions: A sequence of file extensions to process (e.g., ['.py', '.md']).
        output_dir: The directory to store the generated files.
    """
    repo_path_obj = Path(repo_path)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    # Extract org and repo name from the repo_path (assuming structure 'cached-repos/org/repo')
    try:
        org = repo_path_obj.parts[-2]
        repo = repo_path_obj.parts[-1]
        repo_prefix = f"{org}_{repo}"
    except IndexError:
        print(
            f"Warning: Could not determine org/repo from path '{repo_path}'. Using 'unknown_repo' as prefix."
        )
        repo_prefix = "unknown_repo"

    prompt_template = """You are being tested. Your goal is to reconstruct the current state of a file, given the history of changes made to that file. For your response, simply output the exact final state of the file, wrapped in triple backticks (```):

> git log -p --cc --follow --topo-order --reverse -- {filepath}

{git_log_output}"""

    files_processed = 0
    for current_dir, _, files in os.walk(repo_path):
        for filename in files:
            if any(filename.endswith(ext) for ext in extensions):
                file_path_abs = Path(current_dir) / filename
                # Get relative path *within* the cloned repo
                try:
                    file_path_rel = file_path_abs.relative_to(repo_path_obj)
                except ValueError:
                    print(f"Warning: Could not get relative path for {file_path_abs}")
                    continue  # Skip files not directly under repo_path_obj

                file_path_rel_str = str(file_path_rel)
                print(f"Processing: {file_path_rel_str}")

                # 1. Run git log command
                git_log_command = [
                    "git",
                    "log",
                    "-p",
                    "--cc",
                    "--follow",
                    "--topo-order",
                    "--reverse",
                    "--",
                    file_path_rel_str,  # Use relative path for git command
                ]
                try:
                    # Run git command from within the repo directory
                    result = subprocess.run(
                        git_log_command,
                        cwd=repo_path,  # Crucial: run git in the repo's context
                        capture_output=True,
                        text=True,
                        check=True,
                        encoding="utf-8",  # Specify encoding
                        errors="replace",  # Handle potential decoding errors
                    )
                    git_log_output = result.stdout
                except subprocess.CalledProcessError as e:
                    print(f"Error running git log for {file_path_rel_str}: {e}")
                    print(f"Stderr: {e.stderr}")
                    continue  # Skip this file if git log fails
                except UnicodeDecodeError as e:
                    print(f"Error decoding git log output for {file_path_rel_str}: {e}")
                    continue  # Skip this file if output can't be decoded

                # 2. Read current file content
                try:
                    with open(
                        file_path_abs, "r", encoding="utf-8", errors="replace"
                    ) as f:
                        file_content = f.read()
                except Exception as e:
                    print(f"Error reading file {file_path_abs}: {e}")
                    continue  # Skip this file if reading fails

                # 3. Sanitize filename and create output paths
                sanitized_name = _sanitize_filepath(file_path_rel_str)
                prompt_filename = f"{repo_prefix}_{sanitized_name}_prompt.txt"
                output_filename = f"{repo_prefix}_{sanitized_name}_expectedoutput.txt"

                prompt_file_path = output_dir_obj / prompt_filename
                output_file_path = output_dir_obj / output_filename

                # 4. Write prompt file
                prompt_content = prompt_template.format(
                    filepath=file_path_rel_str, git_log_output=git_log_output
                )
                try:
                    with open(prompt_file_path, "w", encoding="utf-8") as f:
                        f.write(prompt_content)
                except Exception as e:
                    print(f"Error writing prompt file {prompt_file_path}: {e}")
                    continue  # Skip if writing fails

                # 5. Write expected output file
                try:
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        f.write(file_content)
                except Exception as e:
                    print(f"Error writing output file {output_file_path}: {e}")
                    # Optionally remove the prompt file if output fails
                    try:
                        prompt_file_path.unlink()
                    except OSError:
                        pass  # Ignore error if prompt file couldn't be removed
                    continue  # Skip if writing fails

                files_processed += 1

    print(f"Finished processing. Generated prompts for {files_processed} files.")
