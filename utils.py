import os
import subprocess
import glob
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


def find_files_with_extensions(repo_path, extensions):
    """
    Find all files in a repository with the specified extensions.

    Args:
        repo_path: Path to the repository.
        extensions: List of file extensions to match (e.g., ['.py', '.js']).

    Returns:
        List of file paths relative to the repo_path.
    """
    all_matching_files = []

    for ext in extensions:
        # Ensure extension starts with dot
        if not ext.startswith("."):
            ext = f".{ext}"

        # Find all files with this extension in the repo
        pattern = os.path.join(repo_path, f"**/*{ext}")
        matching_files = glob.glob(pattern, recursive=True)

        # Convert to paths relative to repo_path
        rel_paths = [os.path.relpath(f, repo_path) for f in matching_files]
        all_matching_files.extend(rel_paths)

    return all_matching_files


def generate_prompts(repo_path, files, output_dir):
    """
    Generate prompt files containing git history for each file.

    Args:
        repo_path: Path to the repository.
        files: List of file paths relative to repo_path.
        output_dir: Directory to save the generated prompts.

    Returns:
        List of paths to the generated prompt files.
    """
    # Extract repo name for use in filename
    repo_name = os.path.basename(os.path.normpath(repo_path))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    prompt_files = []
    for file_path in files:
        # Skip files that might be in .git directory
        if ".git/" in file_path:
            continue

        # Create a sanitized filename for the prompt
        sanitized_path = file_path.replace("/", "_").replace("\\", "_")
        prompt_filename = f"{repo_name}_{sanitized_path}_prompt.txt"
        prompt_filepath = os.path.join(output_dir, prompt_filename)

        # Run git log command to get file history
        try:
            git_log_cmd = [
                "git",
                "log",
                "-p",
                "--cc",
                "--follow",
                "--topo-order",
                "--reverse",
                "--",
                file_path,
            ]
            git_log_output = subprocess.run(
                git_log_cmd, cwd=repo_path, text=True, capture_output=True, check=True
            ).stdout

            # Create the prompt file
            with open(prompt_filepath, "w", encoding="utf-8") as f:
                f.write(f"""You are being tested. Your goal is to reconstruct the current state of a file, given the history of changes made to that file. For your response, simply output the exact final state of the file, wrapped in triple backticks (```):

> git log -p --cc --follow --topo-order --reverse -- {file_path}

{git_log_output}
""")

            prompt_files.append(prompt_filepath)
            print(f"Created prompt file: {prompt_filepath}")

        except subprocess.CalledProcessError as e:
            print(f"Error generating prompt for {file_path}: {e}")
            continue

    return prompt_files


def generate_expected_outputs(repo_path, files, output_dir):
    """
    Generate expected output files containing the current content of each file.

    Args:
        repo_path: Path to the repository.
        files: List of file paths relative to repo_path.
        output_dir: Directory to save the expected outputs.

    Returns:
        List of paths to the generated expected output files.
    """
    # Extract repo name for use in filename
    repo_name = os.path.basename(os.path.normpath(repo_path))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_files = []
    for file_path in files:
        # Skip files that might be in .git directory
        if ".git/" in file_path:
            continue

        # Create a sanitized filename for the expected output
        sanitized_path = file_path.replace("/", "_").replace("\\", "_")
        output_filename = f"{repo_name}_{sanitized_path}_expectedoutput.txt"
        output_filepath = os.path.join(output_dir, output_filename)

        try:
            # Read the current content of the file
            with open(
                os.path.join(repo_path, file_path), "r", encoding="utf-8"
            ) as src_file:
                file_content = src_file.read()

            # Write the content to the expected output file
            with open(output_filepath, "w", encoding="utf-8") as dest_file:
                dest_file.write(file_content)

            output_files.append(output_filepath)
            print(f"Created expected output file: {output_filepath}")

        except Exception as e:
            print(f"Error generating expected output for {file_path}: {e}")
            continue

    return output_files


def create_benchmark_files(repo_path, extensions):
    """
    Create benchmark files (prompts and expected outputs) for all files
    with the specified extensions in the repository.

    Args:
        repo_path: Path to the repository.
        extensions: List of file extensions to process.

    Returns:
        Tuple containing (list of prompt files, list of expected output files).
    """
    # Find all matching files
    matching_files = find_files_with_extensions(repo_path, extensions)
    print(
        f"Found {len(matching_files)} files matching extensions: {', '.join(extensions)}"
    )

    if not matching_files:
        print("No matching files found. Check the extensions provided.")
        return [], []

    # Create output directory
    output_dir = "benchmark-files"
    os.makedirs(output_dir, exist_ok=True)

    # Generate prompt files
    prompt_files = generate_prompts(repo_path, matching_files, output_dir)

    # Generate expected output files
    expected_output_files = generate_expected_outputs(
        repo_path, matching_files, output_dir
    )

    return prompt_files, expected_output_files
