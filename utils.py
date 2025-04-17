import os
import re
import subprocess
import tiktoken
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


# Global tiktoken encoder instance
_ENCODER = None


def get_encoder():
    """Initializes and returns the tiktoken encoder."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text):
    """
    Counts the number of tokens in a given text using the cl100k_base encoder.

    Args:
        text: The string to count tokens for.

    Returns:
        The number of tokens in the text.
    """
    encoder = get_encoder()
    return len(encoder.encode(text))


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

    Also calculates statistics about the generated prompts and files.

    Args:
        repo_path: Path to the cloned repository.
        extensions: List of file extensions to process.
        output_dir: Directory to save generated files.

    Returns:
        A list of dictionaries, where each dictionary contains statistics
        for one processed file:
        {
            'filename': str,
            'prompt_tokens': int,
            'expected_tokens': int,
            'num_commits': int,
            'lines_added': int,
            'lines_deleted': int,
            'final_lines': int
        }
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    repo_name = os.path.basename(os.path.normpath(repo_path))
    stats_list = []

    for root, _, files in os.walk(repo_path):
        # Skip .git directory
        if ".git" in root.split(os.sep):
            continue

        for filename in files:
            if any(filename.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, repo_path)
                safe_rel = rel_path.replace(os.sep, "_")
                prompt_fname = f"{repo_name}_{safe_rel}_prompt.txt"
                expected_fname = f"{repo_name}_{safe_rel}_expectedoutput.txt"
                prompt_path = os.path.join(output_dir, prompt_fname)
                expected_path = os.path.join(output_dir, expected_fname)

                # 1. Get git history with diffs for the prompt
                try:
                    history_result = subprocess.run(
                        [
                            "git",
                            "log",
                            "-p",
                            "--cc",  # Show combined diff for merge commits
                            "--topo-order",
                            "--reverse",
                            "--",
                            rel_path,
                        ],
                        cwd=repo_path,
                        check=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="ignore",  # Ignore decoding errors
                    )
                    git_history = history_result.stdout
                except subprocess.CalledProcessError as e:
                    print(f"Warning: Error getting git history for {rel_path}: {e}")
                    git_history = f"Error retrieving git history: {e}\n"
                except FileNotFoundError:
                    print(
                        "Warning: git command not found. Skipping history generation."
                    )
                    git_history = "git command not found.\n"

                # 2. Construct prompt content
                prompt_content = (
                    "You are being tested. Your goal is to reconstruct the current state of a file, "
                    "given the history of changes made to that file. For your response, simply output "
                    "the exact final state of the file, wrapped in triple backticks (```):\n\n"
                    f"> git log -p --cc --topo-order --reverse -- {rel_path}\n\n"
                    f"{git_history}"
                )
                with open(prompt_path, "w", encoding="utf-8") as pf:
                    pf.write(prompt_content)

                # 3. Read final content
                try:
                    with open(
                        full_path, "r", encoding="utf-8", errors="ignore"
                    ) as original:
                        final_content = original.read()
                except Exception as e:
                    print(f"Warning: Error reading file {full_path}: {e}")
                    final_content = f"Error reading file: {e}"

                with open(expected_path, "w", encoding="utf-8") as ef:
                    ef.write(final_content)

                # 4. Calculate statistics
                prompt_tokens = count_tokens(prompt_content)
                expected_tokens = count_tokens(final_content)
                final_lines = len(final_content.splitlines())

                # Count commits
                num_commits = len(re.findall(r"^commit ", git_history, re.MULTILINE))

                # Get lines added/deleted using git log --numstat
                lines_added = 0
                lines_deleted = 0
                try:
                    numstat_result = subprocess.run(
                        [
                            "git",
                            "log",
                            "--format=format:",  # Only show numstat
                            "--numstat",
                            "--",
                            rel_path,
                        ],
                        cwd=repo_path,
                        check=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="ignore",
                    )
                    for line in numstat_result.stdout.splitlines():
                        if not line.strip():
                            continue
                        parts = line.split("\t")
                        if len(parts) == 3:
                            # Handle binary files marked with '-'
                            added = parts[0]
                            deleted = parts[1]
                            if added != "-":
                                lines_added += int(added)
                            if deleted != "-":
                                lines_deleted += int(deleted)
                except subprocess.CalledProcessError as e:
                    print(f"Warning: Error getting numstat for {rel_path}: {e}")
                except FileNotFoundError:
                    print(
                        "Warning: git command not found. Skipping numstat calculation."
                    )

                # 5. Store stats
                file_stats = {
                    "filename": rel_path,
                    "prompt_tokens": prompt_tokens,
                    "expected_tokens": expected_tokens,
                    "num_commits": num_commits,
                    "lines_added": lines_added,
                    "lines_deleted": lines_deleted,
                    "final_lines": final_lines,
                }
                stats_list.append(file_stats)

    return stats_list
