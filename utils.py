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
            raise ValueError(f"Repository name should be in 'org/repo' format: {repo_name}")
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
        subprocess.run(
            ["git", "clone", github_url, target_dir],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Successfully cloned {std_repo_name} to {target_dir}")
        return target_dir
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e.stderr}")
        raise
