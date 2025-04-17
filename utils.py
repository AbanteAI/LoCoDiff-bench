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


def generate_prompts_and_expected(
    repo_path, extensions, output_dir="generated_prompts"
):
    """
    For every file in the repository at repo_path with one of the specified extensions,
    create two files in output_dir:
      - {repo_name}_{relative_path_with_underscores}_prompt.txt containing
        a reconstruction prompt with git history.
      - {repo_name}_{relative_path_with_underscores}_expectedoutput.txt containing
        the file's final content.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    repo_name = os.path.basename(os.path.normpath(repo_path))

    for root, _, files in os.walk(repo_path):
        for filename in files:
            if any(filename.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, repo_path)
                safe_rel = rel_path.replace(os.sep, "_")
                prompt_fname = f"{repo_name}_{safe_rel}_prompt.txt"
                expected_fname = f"{repo_name}_{safe_rel}_expectedoutput.txt"
                prompt_path = os.path.join(output_dir, prompt_fname)
                expected_path = os.path.join(output_dir, expected_fname)

                try:
                    result = subprocess.run(
                        [
                            "git",
                            "log",
                            "-p",
                            "--cc",
                            "--follow",
                            "--topo-order",
                            "--reverse",
                            "--",
                            rel_path,
                        ],
                        cwd=repo_path,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    git_history = result.stdout
                except subprocess.CalledProcessError as e:
                    git_history = f"Error retrieving git history: {e}"

                prompt_content = (
                    "You are being tested. Your goal is to reconstruct the current state of a file, "
                    "given the history of changes made to that file. For your response, simply output "
                    "the exact final state of the file, wrapped in triple backticks (```):\n\n"
                    f"> git log -p --cc --follow --topo-order --reverse -- {rel_path}\n\n"
                    f"{git_history}"
                )
                with open(prompt_path, "w", encoding="utf-8") as pf:
                    pf.write(prompt_content)

                try:
                    with open(full_path, "r", encoding="utf-8") as original:
                        final_content = original.read()
                except Exception as e:
                    final_content = f"Error reading file content: {e}"

                with open(expected_path, "w", encoding="utf-8") as ef:
                    ef.write(final_content)
