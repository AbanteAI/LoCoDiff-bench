import os
import subprocess
from pathlib import Path
from typing import Iterable, List
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
        repo_name: A GitHub repository name, either as a full URL
                   (https://github.com/org/repo) or in the shorter format (org/repo).

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


# --------------------------------------------------------------------------- #
# New helper functionality for generating LoCoDiff-style prompts              #
# --------------------------------------------------------------------------- #
OUTPUT_BASE_DIR = "generated-data"


def _ensure_output_dir(repo_name: str) -> Path:
    """
    Ensure an output directory exists for a given repository.

    The structure will look like:
        generated-data/
            org_repo/  # underscores instead of slash
                <files>

    Args:
        repo_name: Repository name in ``org/repo`` format.

    Returns
    -------
    Path
        Path object pointing to the directory.
    """
    safe_repo_name = repo_name.replace("/", "_")
    out_dir = Path(OUTPUT_BASE_DIR) / safe_repo_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _is_target_file(path: Path, exts: Iterable[str]) -> bool:
    """
    Determine whether *path* ends with one of the supplied extensions.

    Args:
        path: The Path to test.
        exts: Iterable of extensions such as [".py", ".txt"].

    Returns
    -------
    bool
    """
    return any(str(path).endswith(ext) for ext in exts)


def list_files_with_extensions(
    repo_path: str | os.PathLike, extensions: Iterable[str]
) -> List[Path]:
    """
    Recursively list every file in *repo_path* that ends with one of *extensions*.

    Parameters
    ----------
    repo_path
        Path to a cloned repository.
    extensions
        Iterable of extensions to include (e.g., ``[".py", ".md"]``).

    Returns
    -------
    list[pathlib.Path]
        All matching files, absolute paths.
    """
    repo_path = Path(repo_path)
    exts = list(extensions)
    matches: List[Path] = []
    for file_path in repo_path.rglob("*"):
        if file_path.is_file() and _is_target_file(file_path, exts):
            matches.append(file_path)
    return matches


PROMPT_TEMPLATE = """You are being tested. Your goal is to reconstruct the current state of a file, given the history of changes made to that file. For your response, simply output the exact final state of the file, wrapped in triple backticks (```):

> git log -p --cc --follow --topo-order --reverse -- {filepath}

{git_history}
"""


def _git_log_history(repo_path: Path, relative_path: str) -> str:
    """
    Call git log -p ... and return its text output for *relative_path*.

    Parameters
    ----------
    repo_path
        Path object pointing to repo root.
    relative_path
        File path relative to *repo_path*.

    Returns
    -------
    str
        Git log output (stdout).
    """
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_path),
            "log",
            "-p",
            "--cc",
            "--follow",
            "--topo-order",
            "--reverse",
            "--",
            relative_path,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:  # pragma: no cover
        # If git has no history for the file (e.g., a brand new repo) we can still
        # continue by returning whatever output we did get plus stderr for context.
        return result.stdout + "\n" + result.stderr
    return result.stdout


def _safe_filename(repo_name: str, relative_path: str) -> str:
    """
    Build a safe filename for prompt/expected outputs.

    Slashes are replaced with underscores.

    Returns
    -------
    str
    """
    return f"{repo_name.replace('/', '_')}_{relative_path.replace(os.sep, '_')}"


def write_prompt_and_expected(
    repo_name: str,
    repo_path: str | os.PathLike,
    file_path: Path,
    output_dir: Path | None = None,
) -> None:
    """
    For *file_path*, write a prompt file and expected output file.

    Parameters
    ----------
    repo_name
        'org/repo' string.
    repo_path
        Root of cloned repo.
    file_path
        Absolute path of the file for which we generate inputs.
    output_dir
        Where to place output files.  If None, a directory inside
        ``generated-data`` will be created/used automatically.
    """
    repo_path = Path(repo_path)
    relative_path = str(file_path.relative_to(repo_path))

    if output_dir is None:
        output_dir = _ensure_output_dir(repo_name)
    else:
        output_dir = Path(output_dir)

    # Prompt -----------------------------------------------------------------
    git_history = _git_log_history(repo_path, relative_path)
    prompt_text = PROMPT_TEMPLATE.format(
        filepath=relative_path, git_history=git_history
    )

    safe_prefix = _safe_filename(repo_name, relative_path)
    prompt_file = output_dir / f"{safe_prefix}_prompt.txt"
    prompt_file.write_text(prompt_text, encoding="utf-8")

    # Expected output --------------------------------------------------------
    expected_text = file_path.read_text(encoding="utf-8")
    expected_file = output_dir / f"{safe_prefix}_expectedoutput.txt"
    expected_file.write_text(expected_text, encoding="utf-8")


def generate_prompts_for_repo(
    repo_path: str | os.PathLike,
    repo_name: str,
    extensions: Iterable[str],
    output_dir: str | os.PathLike | None = None,
) -> List[Path]:
    """
    Generate all prompt and expected output pairs for *repo_path*.

    Parameters
    ----------
    repo_path
        Location of cloned repo.
    repo_name
        'org/repo' string.
    extensions
        Iterable of file extensions to consider.
    output_dir
        Optional directory in which to write files.  If None, defaults to
        a sub‑directory inside ``generated-data``.

    Returns
    -------
    list[pathlib.Path]
        List of written prompt file paths (one per input file).
    """
    repo_path = Path(repo_path)
    files = list_files_with_extensions(repo_path, extensions)
    written_prompts: List[Path] = []
    for path in files:
        write_prompt_and_expected(repo_name, repo_path, path, output_dir=output_dir)
        safe_prefix = _safe_filename(repo_name, str(path.relative_to(repo_path)))
        out_dir = (
            Path(output_dir)
            if output_dir is not None
            else _ensure_output_dir(repo_name)
        )
        written_prompts.append(out_dir / f"{safe_prefix}_prompt.txt")
    return written_prompts
