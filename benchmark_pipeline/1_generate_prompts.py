#!/usr/bin/env python3
"""
Generates benchmark prompts and expected outputs from specified repositories.

Purpose:
  This script clones specified repositories (if not already cached) and then
  iterates through files (matching given extensions) within those repositories.
  For each eligible file, it generates two corresponding files in the
  'generated_prompts/' directory:
    1. A prompt file (*_prompt.txt): Contains instructions and the full git log
       history (with diffs) for the original file. The model's task is to
       reconstruct the file's final state based on this history.
    2. An expected output file (*_expectedoutput.txt): Contains the exact final
       content of the original file. This serves as the ground truth for scoring.

  The script applies filters based on file modification date and the token counts
  of both the prompt and the expected output. It then samples a specified number
  of prompts from the filtered set, aiming for a distribution across the allowed
  token range, and saves metadata about the final benchmark set.

Arguments:
  --repos, -r (required): List of GitHub repositories to process (format: 'org/repo'
                          or full URL). These will be cloned if not already in cache.
  --extensions, -e (required): List of file extensions to process (e.g., '.py' '.js').
  --benchmark-run-dir (required): Path to the directory where benchmark run data
                                  (prompts, results) will be stored.
  --min-prompt-tokens (optional): Minimum number of tokens (in thousands, e.g., 0)
                                  allowed in the generated prompt file (default: 0).
  --max-prompt-tokens (optional): Maximum number of tokens (in thousands, e.g., 50)
                                  allowed in the generated prompt file (default: 50).
  --add-prompts (optional): The target number of *additional* prompts to generate
                            and add to the existing set (default: 0). If 0, only
                            reports existing count and exits.
  --modified-within-months (optional): Only process files last modified within the
                                       specified number of months (default: 6).
                                       Set <= 0 to disable.
  --max-expected-tokens (optional): Skip files whose final content (expected output)
                                    exceeds this token count (default: 12000).
                                    Set <= 0 to disable.

Inputs:
  - Command-line arguments specifying repositories, extensions, directories, and
    filtering/sampling parameters.

Outputs:
  - Creates the `<benchmark_run_dir>/prompts/` directory if it doesn't exist.
  - Creates the `<benchmark_run_dir>/prompts_temp/` directory for temporary files.
  - Populates `<benchmark_run_dir>/prompts/` with pairs of files for each processed source file:
    - `repo_path_prompt.txt`: Contains the reconstruction prompt (filename format uses repo name and sanitized relative path).
    - `repo_path_expectedoutput.txt`: Contains the ground truth file content (filename format uses repo name and sanitized relative path).
  - Creates `<benchmark_run_dir>/prompts/metadata.json`: Contains metadata about the generation
    parameters and the final list of benchmark cases.
  - Creates or updates `<benchmark_run_dir>/benchmark_history.log`: Records timestamps and command-line
    arguments used for each script run.
  - Prints statistics about the generation process, filtering, and final sampling results to the console.

File Modifications:
  - Creates the `<benchmark_run_dir>/prompts/` directory if it doesn't exist.
  - Creates the `<benchmark_run_dir>/prompts_temp/` directory if it doesn't exist.
  - Creates `*_prompt.txt` files within `<benchmark_run_dir>/prompts_temp/` initially.
  - Creates `*_expectedoutput.txt` files within `<benchmark_run_dir>/prompts_temp/` initially.
  - Copies selected `*_prompt.txt` and `*_expectedoutput.txt` files from `<benchmark_run_dir>/prompts_temp/` to `<benchmark_run_dir>/prompts/`.
  - Creates or overwrites `metadata.json` within `<benchmark_run_dir>/prompts/`.
  - Creates or updates `benchmark_history.log` in the benchmark run directory.
  - Deletes the `<benchmark_run_dir>/prompts_temp/` directory after completion.
  - Creates the `cached-repos/` directory if it doesn't exist.
  - Clones repositories into the `cached-repos/` directory if they don't already exist there.
"""

import argparse
import os
import sys
import shutil
from glob import glob
import json
import re
import subprocess
import time
from datetime import datetime, timezone
import tiktoken
import random
import yaml
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Set, Optional
from urllib.parse import urlparse


# --- Benchmark Utilities ---

# Use path-based import to ensure it works regardless of how the script is invoked

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import log_script_run


# --- Repository Cloning Functions ---


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


def clone_repo_to_cache(repo_name, cache_dir):
    """
    Clone a GitHub repository into the cached-repos directory.

    Args:
        repo_name: A GitHub repository name, either as a full URL (https://github.com/org/repo)
                   or in the shorter format (org/repo).
        cache_dir: Directory to store cloned repositories.

    Returns:
        The path to the cloned repository (cache_dir/org/repo).
    """
    # Create cache_dir directory if it doesn't exist
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


# --- Helper Functions ---


def get_repo_head_commit_hash(repo_path: str) -> str:
    """
    Get the HEAD commit hash of a repository.

    Args:
        repo_path: Path to the cloned repository.

    Returns:
        The full commit hash as a string.

    Raises:
        subprocess.CalledProcessError: If git command fails.
        FileNotFoundError: If git command is not found.
    """
    hash_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return hash_result.stdout.strip()


@dataclass(frozen=True)
class Config:
    """Configuration settings for the prompt generation script."""

    benchmark_run_dir: str  # New required directory
    prompts_dir: str  # Derived: benchmark_run_dir / "prompts"
    temp_dir: str  # Derived: benchmark_run_dir / "prompts_temp"
    min_prompt_tokens: int
    max_prompt_tokens: int
    add_prompts: int  # Renamed from num_prompts
    modified_within_months: int
    max_expected_tokens: int
    encoder: tiktoken.Encoding  # Encoder instance is now required


def count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    """
    Counts the number of tokens in a given text using the provided encoder.

    Args:
        text: The string to count tokens for.
        encoder: The tiktoken encoder instance to use.

    Returns:
        The number of tokens in the text.
    """
    return len(encoder.encode(text))


def run_git_command(args: List[str], repo_path: str, description: str) -> str:
    """
    Runs a git command and raises exception on error.

    Args:
        args: List of arguments for the git command (e.g., ["log", "-1", "--format=%ct"]).
        repo_path: Path to the repository.
        description: Description of the action for error messages.

    Returns:
        The stdout of the command.

    Raises:
        subprocess.CalledProcessError: If the git command fails.
        FileNotFoundError: If the git command is not found.
    """
    result = subprocess.run(
        ["git"] + args,
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return result.stdout


def is_file_recently_modified(
    rel_path: str, repo_path: str, threshold_timestamp: float | None
) -> bool:
    """
    Checks if a file was modified after the threshold timestamp.

    Args:
        rel_path: Relative path of the file within the repository.
        repo_path: Path to the repository.
        threshold_timestamp: The minimum modification timestamp (Unix epoch).

    Returns:
        True if the file is recent enough or if the check is disabled.

    Raises:
        ValueError: If there are issues getting or parsing the commit timestamp.
    """
    if threshold_timestamp is None:
        return True  # Date filter disabled

    try:
        stdout = run_git_command(
            ["log", "-1", "--format=%ct", "--", rel_path], repo_path, "commit time"
        )

        if not stdout.strip():
            raise ValueError(f"No commit timestamp returned for {rel_path}")

        last_commit_timestamp = int(stdout.strip())
        return last_commit_timestamp >= threshold_timestamp

    except subprocess.CalledProcessError as e:
        raise ValueError(f"Git command failed for {rel_path}: {e}")
    except ValueError as e:
        raise ValueError(f"Failed to parse commit timestamp for {rel_path}: {e}")


def get_git_history(rel_path: str, repo_path: str) -> str:
    """Gets the full git log history with patches for a file."""
    try:
        return run_git_command(
            ["log", "-p", "--cc", "--topo-order", "--reverse", "--", rel_path],
            repo_path,
            "git history",
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get git history for {rel_path}: {e}")


def get_git_numstat(rel_path: str, repo_path: str) -> Tuple[int, int]:
    """Gets the total lines added and deleted for a file from git history."""
    lines_added = 0
    lines_deleted = 0

    try:
        stdout = run_git_command(
            ["log", "--format=format:", "--numstat", "--", rel_path],
            repo_path,
            "numstat",
        )

        for line in stdout.splitlines():
            if not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) == 3:
                added_str, deleted_str, _ = parts
                if added_str != "-":
                    lines_added += int(added_str)
                if deleted_str != "-":
                    lines_deleted += int(deleted_str)

        return lines_added, lines_deleted

    except (subprocess.CalledProcessError, ValueError) as e:
        # If we can't get numstat, return 0/0 rather than crashing
        print(f"\nWarning: Failed to get numstat for {rel_path}: {e}")
        return 0, 0


def write_output_files(
    prompt_path: str,
    expected_path: str,
    prompt_content: str,
    final_content: str,
):
    """Writes the prompt and expected output content to their respective files."""
    os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
    os.makedirs(os.path.dirname(expected_path), exist_ok=True)

    with open(prompt_path, "w", encoding="utf-8") as pf:
        pf.write(prompt_content)

    with open(expected_path, "w", encoding="utf-8") as ef:
        ef.write(final_content)


def build_prompt_content(rel_path: str, git_history: str) -> str:
    """Constructs the prompt content using the git history."""
    return f"""\
# Instructions

You are being benchmarked. You will see the output of a git log command, and from that must infer the current state of a file. Think carefully, as you must output the exact state of the file to earn full marks.

**Important:** Your goal is to reproduce the file's content *exactly* as it exists at the final commit, even if the code appears broken, buggy, or contains obvious errors. Do **not** try to "fix" the code. Attempting to correct issues will result in a poor score, as this benchmark evaluates your ability to reproduce the precise state of the file based on its history.

# Required Response Format

Wrap the content of the file in triple backticks (```). Any text outside the final closing backticks will be ignored. End your response after outputting the closing backticks.

# Example Response

```python
#!/usr/bin/env python
print('Hello, world!')
```

# File History

> git log -p --cc --topo-order --reverse -- {rel_path}

{git_history}
"""


# --- Language Configuration Loading ---

DEFAULT_BENCHMARK_CONFIG_PATH = "benchmark_pipeline/benchmark_config.yaml"


def load_language_config(
    filepath: str = DEFAULT_BENCHMARK_CONFIG_PATH,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Loads language configuration from a YAML file.

    Args:
        filepath: Path to the YAML configuration file.

    Returns:
        A dictionary mapping language names to their configuration
        (e.g., {"python": {"extensions": [".py"]}}).

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the file cannot be parsed.
        ValueError: If the config format is invalid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Benchmark configuration file not found: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing benchmark config file {filepath}: {e}")

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid format in {filepath}: Top level must be a dictionary."
        )

    # Extract languages section
    language_config = config["languages"]

    # Basic validation
    for lang, settings in language_config.items():
        if not isinstance(settings, dict) or "extensions" not in settings:
            raise ValueError(
                f"Invalid format for language '{lang}' in {filepath}: Must be a dict with 'extensions' key."
            )
        if not isinstance(settings["extensions"], list) or not all(
            isinstance(ext, str) and ext.startswith(".")
            for ext in settings["extensions"]
        ):
            raise ValueError(
                f"Invalid 'extensions' list for language '{lang}' in {filepath}: Must be a list of strings starting with '.'."
            )

    print(f"Loaded language configuration from: {filepath}")
    return language_config


def get_all_extensions_from_config(
    language_config: Dict[str, Dict[str, List[str]]],
) -> Set[str]:
    """Extracts a set of all unique extensions from the language config."""
    all_extensions = set()
    for lang_settings in language_config.values():
        all_extensions.update(lang_settings.get("extensions", []))
    return all_extensions


def build_extension_to_language_map(
    language_config: Dict[str, Dict[str, List[str]]],
) -> Dict[str, str]:
    """Creates a reverse map from extension to language."""
    ext_to_lang = {}
    for lang, settings in language_config.items():
        for ext in settings.get("extensions", []):
            if ext in ext_to_lang:
                print(
                    f"Warning: Extension '{ext}' is mapped to multiple languages ('{ext_to_lang[ext]}' and '{lang}'). Using '{lang}'."
                )
            ext_to_lang[ext] = lang
    return ext_to_lang


# --- Statistics for Existing Prompts ---


@dataclass
class PromptInfo:
    filepath: str
    language: str
    token_count: int


def get_detailed_existing_prompt_info(
    prompts_dir: str,
    ext_to_lang_map: Dict[str, str],
    encoder: tiktoken.Encoding,
) -> List[PromptInfo]:
    """
    Scans the prompts directory, reads each prompt, counts tokens, and infers language.

    Args:
        prompts_dir: Directory containing the '*_prompt.txt' files.
        ext_to_lang_map: Dictionary mapping file extensions (e.g., '.py') to language names.
        encoder: The tiktoken encoder instance.

    Returns:
        A list of PromptInfo objects containing details for each found prompt.
    """
    prompt_infos = []
    prompt_files = glob(os.path.join(prompts_dir, "*_prompt.txt"))

    print(f"Analyzing {len(prompt_files)} existing prompt files in {prompts_dir}...")
    if not prompt_files:
        return []  # Return early if no files found

    for filepath in tqdm(prompt_files, desc="Analyzing prompts", mininterval=3):
        filename = os.path.basename(filepath)
        # Attempt to extract the original extension
        # Format: org_repo_path_with_underscores_ext_prompt.txt
        base_name = filename.replace("_prompt.txt", "")
        language = "unknown"
        # found_ext = None # Unused variable

        # Iterate through known extensions to find the longest match at the end
        possible_exts = sorted(ext_to_lang_map.keys(), key=len, reverse=True)
        for ext in possible_exts:
            # Check for sanitized extension (e.g., _py)
            sanitized_ext = ext.replace(".", "_")
            if base_name.endswith(sanitized_ext):
                # Check if the part before the extension looks like a path separator replacement
                potential_path_part = base_name[: -len(sanitized_ext)]
                if (
                    potential_path_part.endswith("_") or not potential_path_part
                ):  # Handle case where filename is just the extension
                    # found_ext = ext # Unused variable
                    language = ext_to_lang_map[ext]
                    break
            # Check for original extension (e.g., .py) just in case (less likely)
            elif base_name.endswith(ext):
                # Check if the part before the extension looks like a path separator replacement
                potential_path_part = base_name[: -len(ext)]
                if potential_path_part.endswith("_") or not potential_path_part:
                    # found_ext = ext # Unused variable
                    language = ext_to_lang_map[ext]
                    break

        if language == "unknown":
            # Fallback: Check if any known extension is present *anywhere* after the last likely path separator '_'
            # This is less precise but might catch cases missed by the endswith logic
            last_underscore_idx = base_name.rfind("_")
            if last_underscore_idx != -1:
                potential_filename_part = base_name[last_underscore_idx + 1 :]
                for ext in possible_exts:
                    sanitized_ext = ext.replace(".", "_")
                    if (
                        sanitized_ext in potential_filename_part
                        or ext in potential_filename_part
                    ):
                        language = ext_to_lang_map[ext]
                        # Take the first match found in this fallback
                        break

            if language == "unknown":  # Still unknown after fallback
                print(f"\nWarning: Could not determine language for {filename}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            token_count = count_tokens(content, encoder)
            prompt_infos.append(PromptInfo(filepath, language, token_count))
        except Exception as e:
            print(f"\nError processing file {filepath}: {e}")

    return prompt_infos


@dataclass
class LanguageStats:
    count: int = 0
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    quartile_counts: List[int] = lambda: [0, 0, 0, 0]  # type: ignore


def calculate_prompt_statistics(
    prompt_infos: List[PromptInfo],
) -> Tuple[Dict[str, LanguageStats], List[float]]:
    """
    Calculates token statistics globally and per language, including quartile distribution.

    Args:
        prompt_infos: A list of PromptInfo objects.

    Returns:
        A tuple containing:
        - A dictionary mapping language names (including "All") to LanguageStats objects.
        - A list of 5 floats representing the quartile boundaries (min, q1, q2, q3, max).
    """
    if not prompt_infos:
        return {}, [0.0, 0.0, 0.0, 0.0, 0.0]

    all_token_counts = [info.token_count for info in prompt_infos]
    global_min = min(all_token_counts)
    global_max = max(all_token_counts)

    # Define quartile boundaries
    if global_min == global_max:
        # Handle edge case where all prompts have the same length
        boundaries = [float(global_min)] * 5
        quartile_ranges = [(global_min, global_min)] * 4
    else:
        q1 = global_min + (global_max - global_min) / 4.0
        q2 = global_min + 2.0 * (global_max - global_min) / 4.0
        q3 = global_min + 3.0 * (global_max - global_min) / 4.0
        boundaries = [float(global_min), q1, q2, q3, float(global_max)]
        # Define ranges carefully: [b0, b1], (b1, b2], (b2, b3], (b3, b4]
        quartile_ranges = [
            (boundaries[0], boundaries[1]),
            (boundaries[1], boundaries[2]),
            (boundaries[2], boundaries[3]),
            (boundaries[3], boundaries[4]),
        ]

    stats: Dict[str, LanguageStats] = {"All": LanguageStats()}
    stats["All"].quartile_counts = [0, 0, 0, 0]  # Ensure list is initialized

    for info in prompt_infos:
        # Update language stats
        lang = info.language
        if lang not in stats:
            stats[lang] = LanguageStats()
            stats[lang].quartile_counts = [0, 0, 0, 0]  # Ensure list is initialized

        stats[lang].count += 1
        # Handle None case for min/max update
        current_min = stats[lang].min_tokens
        stats[lang].min_tokens = (
            info.token_count
            if current_min is None
            else min(current_min, info.token_count)
        )
        current_max = stats[lang].max_tokens
        stats[lang].max_tokens = (
            info.token_count
            if current_max is None
            else max(current_max, info.token_count)
        )

        # Update "All" stats
        stats["All"].count += 1
        # Handle None case for min/max update
        all_current_min = stats["All"].min_tokens
        stats["All"].min_tokens = (
            info.token_count
            if all_current_min is None
            else min(all_current_min, info.token_count)
        )
        all_current_max = stats["All"].max_tokens
        stats["All"].max_tokens = (
            info.token_count
            if all_current_max is None
            else max(all_current_max, info.token_count)
        )

        # Assign to quartile
        assigned = False
        token_val = info.token_count
        # Special case: if token_val is exactly the min, it goes in Q1
        if token_val == boundaries[0]:
            stats[lang].quartile_counts[0] += 1
            stats["All"].quartile_counts[0] += 1
            assigned = True
        else:
            for i, (q_min, q_max) in enumerate(quartile_ranges):
                # Check if token falls into (q_min, q_max]
                # For the last quartile, include the max value: (q3, q4]
                # is_last_quartile = i == 3 # Unused variable
                in_range = token_val > q_min and token_val <= q_max

                if in_range:
                    stats[lang].quartile_counts[i] += 1
                    stats["All"].quartile_counts[i] += 1
                    assigned = True
                    break

        # This should theoretically not happen if ranges cover min to max and edge cases handled
        if not assigned:
            # This case might happen if token_val == global_max and global_max was the upper bound of Q3 due to float precision.
            # Assign such cases to the last quartile.
            if token_val == global_max:
                stats[lang].quartile_counts[3] += 1
                stats["All"].quartile_counts[3] += 1
            else:
                print(
                    f"\nWarning: Prompt {info.filepath} with {info.token_count} tokens did not fall into any quartile range based on boundaries {boundaries}."
                )

    return stats, boundaries


def print_detailed_prompt_stats(
    stats: Dict[str, LanguageStats], boundaries: List[float]
):
    """Prints the calculated prompt statistics in a formatted way."""
    print("\n--- Existing Prompt Statistics ---")

    # Define quartile labels based on boundaries
    # Helper to format boundaries: show in thousands (k)
    def format_boundary(x):
        if x < 1000:
            # For values less than 1000, show as is (integer)
            return f"{x:.0f}"
        else:
            # For values 1000 or more, show in k
            return f"{round(x / 1000):.0f}k"

    q_labels = [
        f"Q1 [{format_boundary(boundaries[0])} - {format_boundary(boundaries[1])}]",
        f"Q2 ({format_boundary(boundaries[1])} - {format_boundary(boundaries[2])}]",
        f"Q3 ({format_boundary(boundaries[2])} - {format_boundary(boundaries[3])}]",
        f"Q4 ({format_boundary(boundaries[3])} - {format_boundary(boundaries[4])}]",
    ]
    # Handle edge case for labels where min == max
    if boundaries[0] == boundaries[4]:
        q_labels = [f"Q1-4 [{format_boundary(boundaries[0])}]"] * 4

    # Sort languages, keeping "All" first, then alphabetically, "unknown" last
    sorted_langs = sorted([lang for lang in stats if lang not in ["All", "unknown"]])
    langs_to_print = ["All"] + sorted_langs
    if "unknown" in stats and stats["unknown"].count > 0:
        langs_to_print.append("unknown")

    for lang in langs_to_print:
        if lang not in stats:
            continue  # Skip if somehow a lang is in the list but not stats
        lang_stat = stats[lang]
        title = lang if lang != "All" else "All Languages"
        print(f"\n{title}")
        print("-" * len(title))  # Separator matches title length

        if lang_stat.count == 0:
            print("  Number of prompts: 0")
            continue

        print(f"  Number of prompts: {lang_stat.count}")
        min_t = lang_stat.min_tokens if lang_stat.min_tokens is not None else "N/A"
        max_t = lang_stat.max_tokens if lang_stat.max_tokens is not None else "N/A"
        print(f"  Shortest / Longest Prompts: {min_t} tokens / {max_t} tokens")
        print("  Prompt count by Quartile:")

        # Format quartile counts - adjust label width dynamically
        max_label_len = 0
        if boundaries[0] != boundaries[4]:
            max_label_len = max(len(label) for label in q_labels)
        else:
            max_label_len = len(q_labels[0])  # Only one label in edge case

        for i, count in enumerate(lang_stat.quartile_counts):
            # Handle edge case display (only print first quartile)
            if boundaries[0] == boundaries[4] and i > 0:
                continue
            label = q_labels[i]
            print(f"    {label:<{max_label_len}} : {count}")

    print("\n" + "-" * 30 + "\n")  # Final separator


# --- Core Generation Logic ---


def generate_prompts_and_expected(
    repo_path: str,
    cfg: Config,
    existing_prefixes: set[str],
    target_extensions: Set[str],  # Added: Set of extensions to process
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """
    Generates prompts and expected outputs for eligible files in a repository.

    Iterates through files matching specified extensions, applies date and token
    filters, fetches git history, calculates stats, and writes output files to
    the temporary directory (`cfg.temp_dir`).

    Args:
        repo_path: Path to the cloned repository.
        cfg: Configuration object.
        existing_prefixes: Set of benchmark case prefixes that already exist.

    Returns:
        Tuple: (stats_list, date_filtered_count, expected_token_filtered_count, already_exists_count)
            - stats_list: List of dictionaries with statistics for each generated case.
            - date_filtered_count: Number of files skipped due to modification date.
            - expected_token_filtered_count: Number of files skipped due to expected token limit.
            - already_exists_count: Number of files skipped because they already exist.
    """
    # Ensure temporary directory exists
    if not os.path.exists(cfg.temp_dir):
        os.makedirs(cfg.temp_dir, exist_ok=True)

    repo_name = os.path.basename(os.path.normpath(repo_path))
    org_name = os.path.basename(os.path.dirname(repo_path))
    full_repo_name = f"{org_name}/{repo_name}"
    stats_list = []
    files_to_process = []
    date_filtered_count = 0
    expected_token_filtered_count = 0
    already_exists_count = 0

    # --- Calculate Date Threshold ---
    threshold_timestamp = None
    if cfg.modified_within_months > 0:
        avg_seconds_per_month = 30.44 * 24 * 60 * 60
        threshold_timestamp = time.time() - (
            cfg.modified_within_months * avg_seconds_per_month
        )
        print(
            f"Date filter enabled: Processing files modified since {datetime.fromtimestamp(threshold_timestamp, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    # --- Get Repo Info ---
    print(f"Getting head commit hash for {full_repo_name}...")
    head_commit_hash = get_repo_head_commit_hash(repo_path)
    print(f"Repository at commit: {head_commit_hash}")

    # --- Collect Files ---
    print("Collecting files matching extensions...")
    for root, _, files in os.walk(repo_path):
        if ".git" in root.split(os.sep):
            continue
        for filename in files:
            # Use the target_extensions set derived from languages.yaml
            if any(filename.endswith(ext) for ext in target_extensions):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, repo_path)
                files_to_process.append((full_path, rel_path))
    print(f"Found {len(files_to_process)} files to potentially process.")

    # --- Process Files ---
    for full_path, rel_path in tqdm(
        files_to_process, desc=f"Generating prompts for {full_repo_name}", mininterval=3
    ):
        try:  # Wrap processing for a single file to catch tokenization errors
            # 1. Date Filter Check
            try:
                if not is_file_recently_modified(
                    rel_path, repo_path, threshold_timestamp
                ):
                    date_filtered_count += 1
                    continue
            except ValueError as e:
                # Handle errors getting commit time (e.g., file not in git)
                tqdm.write(
                    f"Skipping {rel_path}: Error checking modification date: {e}"
                )
                date_filtered_count += 1
                continue

            # 2. Read Final Content & Expected Token Filter Check
            try:
                with open(
                    full_path, "r", encoding="utf-8", errors="ignore"
                ) as original:
                    final_content = original.read()
            except Exception as e:
                tqdm.write(f"Skipping {rel_path}: Error reading file: {e}")
                continue

            # Calculate expected tokens (potential ValueError from count_tokens)
            expected_tokens = count_tokens(final_content, cfg.encoder)
            if (
                cfg.max_expected_tokens > 0
                and expected_tokens > cfg.max_expected_tokens
            ):
                expected_token_filtered_count += 1
                continue

            # 3. Prepare Filenames and Paths (write to temporary directory)
            safe_rel = rel_path.replace(os.sep, "_")
            repo_file_prefix = f"{repo_name}_{safe_rel}"
            prompt_fname = f"{repo_file_prefix}_prompt.txt"
            expected_fname = f"{repo_file_prefix}_expectedoutput.txt"

            # Check if this file already exists in the benchmark set (based on existing prefixes loaded from prompts_dir)
            if repo_file_prefix in existing_prefixes:
                already_exists_count += 1
                continue

            # Write to temp directory (e.g., <benchmark_run_dir>/prompts_temp/)
            prompt_path = os.path.join(cfg.temp_dir, prompt_fname)
            expected_path = os.path.join(cfg.temp_dir, expected_fname)

            # 4. Get Git History
            try:
                git_history = get_git_history(rel_path, repo_path)
            except RuntimeError as e:
                # Handle errors getting git history (e.g., file deleted/renamed weirdly)
                tqdm.write(f"Skipping {rel_path}: Error getting git history: {e}")
                continue

            # 5. Construct and Write Prompt File
            prompt_content = build_prompt_content(rel_path, git_history)
            try:
                write_output_files(
                    prompt_path, expected_path, prompt_content, final_content
                )
            except Exception as e:
                tqdm.write(f"Skipping {rel_path}: Error writing output files: {e}")
                continue

            # 6. Calculate Statistics
            # Calculate prompt tokens (potential ValueError from count_tokens)
            prompt_tokens = count_tokens(prompt_content, cfg.encoder)
            final_lines = len(final_content.splitlines())
            num_commits = len(re.findall(r"^commit ", git_history, re.MULTILINE))
            lines_added, lines_deleted = get_git_numstat(rel_path, repo_path)

            # 7. Store Stats
            file_stats = {
                "filename": rel_path,
                "prompt_tokens": prompt_tokens,
                "expected_tokens": expected_tokens,
                "num_commits": num_commits,
                "lines_added": lines_added,
                "lines_deleted": lines_deleted,
                "final_lines": final_lines,
                "repo_name": full_repo_name,
                "repo_commit_hash": head_commit_hash,
                "prompt_filename": prompt_fname,
                "expected_filename": expected_fname,
                "benchmark_case_prefix": repo_file_prefix,
            }
            stats_list.append(file_stats)

        except ValueError as e:
            # Catch tokenization errors specifically
            if "disallowed special token" in str(e):
                tqdm.write(f"Skipping {rel_path}: Contains disallowed special tokens.")
                # Optionally, increment a dedicated counter for this skip reason
                continue  # Skip to the next file
            else:
                # Handle other potential ValueErrors during processing
                tqdm.write(
                    f"Skipping {rel_path}: Encountered unexpected ValueError: {e}"
                )
                continue  # Skip file on other ValueErrors too for robustness

    return (
        stats_list,
        date_filtered_count,
        expected_token_filtered_count,
        already_exists_count,
    )


# --- Statistics, Filtering, Sampling, and Deletion Functions ---


def print_stats_summary(stats_list: List[Dict[str, Any]], title: str):
    """Prints a summary of statistics for a list of prompts."""
    if not stats_list:
        print(f"\n--- {title} ---")
        print("No prompts in this set.")
        return

    count = len(stats_list)

    print(f"\n--- {title} (Count: {count}) ---")


def filter_prompts_by_token_range(
    stats_list: List[Dict[str, Any]], cfg: Config
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Filters the list of generated prompt statistics based on prompt token limits.

    Args:
        stats_list: The initial list of statistics for all generated prompts.
        cfg: The configuration object containing min_prompt_tokens and max_prompt_tokens.

    Returns:
        A tuple containing:
        - A list of statistics dictionaries for prompts that pass the filter.
        - The count of prompts that were filtered out.
    """
    min_tokens = cfg.min_prompt_tokens
    max_tokens = cfg.max_prompt_tokens

    print(f"\n--- Filtering Prompts (Range: [{min_tokens}, {max_tokens}] Tokens) ---")
    kept_stats = []
    filtered_count = 0

    for stats in stats_list:
        tokens = stats["prompt_tokens"]
        if tokens < min_tokens or tokens > max_tokens:
            filtered_count += 1
        else:
            kept_stats.append(stats)

    print(
        f"Filtered out {filtered_count} prompts outside the token range [{min_tokens}, {max_tokens}]."
    )
    return kept_stats, filtered_count


def sample_prompts(
    stats_list: List[Dict[str, Any]], cfg: Config
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Samples prompts from the filtered list to meet the num_prompts target.
    Uses targeted sampling to aim for distribution across the token range.

    Args:
        stats_list: List of statistics dictionaries (already filtered by token range).
        cfg: The configuration object containing add_prompts, min_prompt_tokens, max_prompt_tokens.

    Returns:
        A tuple containing:
        - A list containing the final sampled prompt statistics.
        - The total number of prompts sampled out (discarded).
    """
    num_to_add = cfg.add_prompts  # Use add_prompts from config
    min_tk = cfg.min_prompt_tokens
    max_tk = cfg.max_prompt_tokens

    print(
        f"\n--- Sampling Prompts to Add (Target: {num_to_add}, Range: [{min_tk}, {max_tk}]) ---"
    )

    if not stats_list:
        print("No new candidate prompts available to sample from.")
        return [], 0

    if len(stats_list) <= num_to_add:
        print(
            f"Number of available new prompts ({len(stats_list)}) is less than or equal to target ({num_to_add}). Keeping all available."
        )
        # Return the available stats, 0 sampled out
        return stats_list, 0

    # --- Targeted Sampling Logic ---
    items_to_keep = []
    available_items = list(stats_list)  # Create a mutable copy
    # Sample exactly num_to_add items
    num_to_sample = num_to_add

    print(
        f"Sampling {num_to_sample} prompts from {len(available_items)} new candidates..."
    )

    for i in tqdm(range(num_to_sample), desc="Sampling prompts to add", mininterval=3):
        if not available_items:
            print("\nWarning: Ran out of available items during sampling.")
            break

        # 1. Choose a random target token count within the overall range.
        target_token_count = random.uniform(min_tk, max_tk)

        # 2. Find the available prompt that is closest to this target token count.
        closest_item = min(
            available_items,
            key=lambda item: abs(item["prompt_tokens"] - target_token_count),
        )

        # 3. Select this closest prompt and remove it from the available pool.
        items_to_keep.append(closest_item)
        available_items.remove(closest_item)
    # --- End Targeted Sampling ---

    items_sampled_out_count = len(stats_list) - len(items_to_keep)
    print(
        f"\nSampled down from {len(stats_list)} to {len(items_to_keep)} (removed {items_sampled_out_count}, targeted sampling)."
    )

    return items_to_keep, items_sampled_out_count


def copy_selected_files(
    kept_prefixes: set[str],
    temp_dir: str,
    prompts_dir: str,  # Changed from output_dir
):
    """
    Copies selected prompt and expected output files from temporary directory to the final prompts directory.

    Args:
        kept_prefixes: A set of benchmark case prefixes to copy.
        temp_dir: The source directory containing temporary files (e.g., <run_dir>/prompts_temp/).
        prompts_dir: The destination directory to copy selected files to (e.g., <run_dir>/prompts/).
    """
    # Ensure prompts directory exists
    os.makedirs(prompts_dir, exist_ok=True)

    copied_files_count = 0

    print(
        f"\n--- Copying {len(kept_prefixes)} Selected Benchmark Files to Final Location ({prompts_dir}) ---"
    )

    if not kept_prefixes:
        print("No files to copy.")
        return

    for prefix in tqdm(kept_prefixes, desc="Copying files", mininterval=3):
        prompt_filename = f"{prefix}_prompt.txt"
        expected_filename = f"{prefix}_expectedoutput.txt"

        source_prompt_path = os.path.join(temp_dir, prompt_filename)
        source_expected_path = os.path.join(temp_dir, expected_filename)

        dest_prompt_path = os.path.join(
            prompts_dir, prompt_filename
        )  # Changed from output_dir
        dest_expected_path = os.path.join(
            prompts_dir, expected_filename
        )  # Changed from output_dir

        # Skip if source files don't exist
        if not os.path.exists(source_prompt_path) or not os.path.exists(
            source_expected_path
        ):
            # This indicates an inconsistency between kept prefixes and generated files
            raise FileNotFoundError(
                f"Error: Source files missing for prefix {prefix}. Expected prompt: {source_prompt_path}, Expected output: {source_expected_path}"
            )
            # error_count += 1 # Unreachable
            # continue

        # Copy the files
        try:
            shutil.copy2(source_prompt_path, dest_prompt_path)
            shutil.copy2(source_expected_path, dest_expected_path)
            copied_files_count += 1
        except OSError as e:
            # Raise the error instead of just printing and counting
            raise OSError(f"Error copying files for prefix {prefix}: {e}") from e
            # error_count += 1 # Unreachable

    print(f"Successfully copied {copied_files_count} pairs of prompt/expected files.")
    # if error_count > 0: # Error count logic removed as errors are now raised
    #     print(f"Encountered errors with {error_count} benchmark sets.")


def load_existing_metadata_and_prefixes(
    prompts_dir: str,  # Changed from output_dir
) -> Tuple[List[Dict[str, Any]], set[str]]:
    """
    Loads existing metadata and identifies all existing benchmark case prefixes from the prompts directory.

    Args:
        prompts_dir: The directory where metadata and final prompt files are stored (e.g., <run_dir>/prompts/).

    Returns:
        A tuple containing:
        - A list of previous generation run dictionaries loaded from metadata.json (or an empty list).
        - A set of all unique benchmark_case_prefix strings found in the metadata and by scanning the prompts_dir.
    """
    metadata_path = os.path.join(
        prompts_dir, "metadata.json"
    )  # Changed from output_dir
    existing_metadata_runs: List[Dict[str, Any]] = []
    existing_prefixes: set[str] = set()

    # 1. Load from metadata.json
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as mf:
                # Expect metadata to be a list of run dictionaries
                loaded_data = json.load(mf)
                if isinstance(loaded_data, list):
                    existing_metadata_runs = loaded_data
                    for run_index, run_data in enumerate(existing_metadata_runs):
                        # Check if the item is a dictionary and contains the expected key for the current format
                        if (
                            isinstance(run_data, dict)
                            and "benchmark_cases_added" in run_data
                        ):
                            cases_list = run_data["benchmark_cases_added"]
                            if isinstance(cases_list, list):
                                for case in cases_list:
                                    if (
                                        isinstance(case, dict)
                                        and "benchmark_case_prefix" in case
                                    ):
                                        existing_prefixes.add(
                                            case["benchmark_case_prefix"]
                                        )
                            else:
                                # The expected key exists, but its value is not a list
                                print(
                                    f"Warning: 'benchmark_cases_added' in run item {run_index} of {metadata_path} is not a list."
                                )
                        else:
                            # This item in the list doesn't look like the expected run dictionary structure
                            print(
                                f"Warning: Skipping item at index {run_index} in {metadata_path} as it does not match expected structure (missing 'benchmark_cases_added' key or not a dictionary): {type(run_data)}"
                            )
                else:
                    # The top-level structure is not a list, which is the only supported format now
                    print(
                        f"Warning: Existing {metadata_path} is not a list. Only list format is supported. Starting fresh."
                    )
                    existing_metadata_runs = []  # Reset as the format is unsupported

        except (json.JSONDecodeError, IOError) as e:
            print(
                f"Warning: Error reading or parsing existing {metadata_path}: {e}. Starting fresh."
            )
            existing_metadata_runs = []  # Reset on error

    # 2. Scan prompts directory for existing prompt files (keeps working as before)
    try:
        # Use glob directly due to 'from glob import glob'
        prompt_files = glob(
            os.path.join(prompts_dir, "*_prompt.txt")
        )  # Changed from output_dir
        for f_path in prompt_files:
            basename = os.path.basename(f_path)
            prefix = basename[:-11]  # Remove '_prompt.txt'
            existing_prefixes.add(prefix)
    except OSError as e:
        print(
            f"Warning: Error scanning prompts directory {prompts_dir}: {e}"
        )  # Changed from output_dir

    return existing_metadata_runs, existing_prefixes


def save_benchmark_metadata(
    existing_runs: List[Dict[str, Any]],
    newly_added_stats: List[Dict[str, Any]],
    cfg: Config,
    language_config: Dict[str, Any],  # Added language config
):
    """
    Appends the current generation run's metadata to the list of existing runs
    and saves the updated list to metadata.json.

    Args:
        existing_runs: The list of run dictionaries loaded from the existing metadata.
        newly_added_stats: List of statistics for the prompts added in *this* run.
        cfg: The configuration object for the current run.
    """
    # Create generation_params dict for the current run, excluding derived/internal paths
    current_run_params = asdict(cfg)
    current_run_params.pop("encoder", None)  # Exclude non-serializable encoder
    current_run_params.pop("prompts_dir", None)  # Exclude derived path
    current_run_params.pop("temp_dir", None)  # Exclude derived path

    # Create the list of benchmark cases for the current run
    current_run_cases = []
    for stats in newly_added_stats:
        current_run_cases.append(
            {
                "benchmark_case_prefix": stats["benchmark_case_prefix"],
                "original_filename": stats["filename"],
                "repo_name": stats["repo_name"],
                "repo_commit_hash": stats["repo_commit_hash"],
                "prompt_tokens": stats["prompt_tokens"],
                "expected_tokens": stats["expected_tokens"],
                "num_commits": stats["num_commits"],
            }
        )

    # Create the dictionary for the current run
    current_run_metadata = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        # "language_config_path": DEFAULT_LANGUAGE_CONFIG_PATH,  # Removed per review
        "language_config_content": language_config,  # Store the actual config used
        "generation_parameters": current_run_params,
        "benchmark_cases_added": current_run_cases,  # Store only newly added cases
    }

    # Append the current run's metadata to the list of existing runs
    updated_metadata_runs = existing_runs + [current_run_metadata]

    # Save the updated list back to metadata.json in the prompts directory
    metadata_path = os.path.join(
        cfg.prompts_dir, "metadata.json"
    )  # Changed from output_dir
    try:
        # Ensure parent directory exists (prompts_dir should already exist from copy step)
        os.makedirs(cfg.prompts_dir, exist_ok=True)  # Changed from output_dir
        with open(metadata_path, "w", encoding="utf-8") as mf:
            json.dump(updated_metadata_runs, mf, indent=4)
        print(
            f"\nAppended current run metadata to {metadata_path} ({len(newly_added_stats)} cases added)."
        )
    except (IOError, TypeError) as e:
        raise RuntimeError(
            f"Error: Failed to save updated metadata to {metadata_path}: {e}"
        ) from e


# --- Main Script Logic ---


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark prompts and expected outputs from specified repositories."
    )
    parser.add_argument(
        "--repos",
        "-r",
        nargs="+",
        required=True,
        help="List of GitHub repositories to process (format: 'org/repo' or full URL).",
    )
    parser.add_argument(
        "--benchmark-run-dir",
        required=True,
        help="Directory where benchmark run data (prompts, results) will be stored. Will be created if it doesn't exist.",
    )
    # Add arguments for filtering/sampling parameters
    parser.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=0,
        help="Minimum number of tokens (in thousands) allowed in the prompt file (default: 0).",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=50,  # Default is 50k tokens
        help="Maximum number of tokens (in thousands) allowed in the prompt file (default: 50).",
    )
    parser.add_argument(
        "--add-prompts",
        type=int,
        default=0,  # Default to 0, meaning only report existing count
        help="Target number of *additional* prompts to generate and add to the existing set (default: 0).",
    )
    parser.add_argument(
        "--modified-within-months",
        type=int,
        default=6,
        metavar="N",
        help="Only process files modified in the last N months (default: 6). Set to 0 or negative to disable.",
    )
    parser.add_argument(
        "--max-expected-tokens",
        type=int,
        default=12000,
        metavar="T",
        help="Maximum number of tokens allowed in the expected output file (default: 12000). Set to 0 or negative to disable.",
    )

    args = parser.parse_args()

    print("--- Starting Prompt Generation ---")

    # --- Validate arguments ---
    if args.min_prompt_tokens < 0:
        print("Error: --min-prompt-tokens cannot be negative.")
        return 1
    # Note: Comparison is done on k-token values here before conversion
    if args.max_prompt_tokens <= args.min_prompt_tokens:
        print("Error: --max-prompt-tokens must be greater than --min-prompt-tokens.")
        return 1
    # Allow add_prompts to be 0 (for reporting mode)
    if args.add_prompts < 0:
        print("Error: --add-prompts cannot be negative.")
        return 1
    # --- End argument validation ---

    # --- Load Language Config ---
    try:
        language_config = load_language_config()  # Uses default path
        all_target_extensions = get_all_extensions_from_config(language_config)
        if not all_target_extensions:
            print("Error: No extensions found in language configuration. Exiting.")
            return 1  # Use return inside main()
        print(
            f"Targeting extensions from {DEFAULT_BENCHMARK_CONFIG_PATH}: {', '.join(sorted(list(all_target_extensions)))}"
        )
    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        print(f"Error loading language configuration: {e}")
        return 1  # Use return inside main()

    # Build reverse map for language lookup later
    ext_to_lang_map = build_extension_to_language_map(language_config)

    # --- Create Config object ---
    # Define derived paths
    benchmark_run_dir = args.benchmark_run_dir
    prompts_dir = os.path.join(benchmark_run_dir, "prompts")
    temp_dir = os.path.join(benchmark_run_dir, "prompts_temp")

    cfg = Config(
        benchmark_run_dir=benchmark_run_dir,
        prompts_dir=prompts_dir,
        temp_dir=temp_dir,
        # Convert k-tokens from args to absolute tokens for internal use
        min_prompt_tokens=args.min_prompt_tokens * 1000,
        max_prompt_tokens=args.max_prompt_tokens * 1000,
        add_prompts=args.add_prompts,
        modified_within_months=args.modified_within_months,
        max_expected_tokens=args.max_expected_tokens,
        encoder=tiktoken.get_encoding("cl100k_base"),  # Initialize encoder here
    )
    print(
        f"Configuration loaded. Benchmark run dir: {cfg.benchmark_run_dir}, Add prompts: {cfg.add_prompts}"
    )
    # --- End Config creation ---

    # --- Log script execution ---
    log_script_run(benchmark_run_dir, "1_generate_prompts.py", args)

    # --- Handle add_prompts == 0 Case (Analyze and Exit) ---
    if cfg.add_prompts == 0:
        print("\nadd_prompts is 0. Analyzing existing prompts...")
        if not os.path.exists(cfg.prompts_dir):
            print(f"Prompts directory '{cfg.prompts_dir}' does not exist.")
            print("Exiting.")
            return 0  # Successful exit, nothing to do

        # Perform detailed analysis
        prompt_infos = get_detailed_existing_prompt_info(
            cfg.prompts_dir, ext_to_lang_map, cfg.encoder
        )

        if not prompt_infos:
            print(f"No prompt files found in '{cfg.prompts_dir}' to analyze.")
            print("Exiting.")
            return 0  # Successful exit, nothing to analyze

        # Calculate statistics
        stats, boundaries = calculate_prompt_statistics(prompt_infos)

        # Print detailed statistics
        print_detailed_prompt_stats(stats, boundaries)

        print("Exiting without generating new prompts.")
        return 0  # Successful exit after analysis

    # --- Load existing data (Only if generating new prompts) ---
    # Ensure prompts directory exists before trying to load from it
    os.makedirs(cfg.prompts_dir, exist_ok=True)
    existing_metadata_runs, existing_prefixes = load_existing_metadata_and_prefixes(
        cfg.prompts_dir  # Load from the final prompts directory
    )
    print(
        f"\nFound {len(existing_prefixes)} existing benchmark prompts in {cfg.prompts_dir}."
    )
    # --- End Load existing data ---

    # --- Reporting Mode Check ---
    if cfg.add_prompts == 0:
        print(
            "\n--add-prompts is 0. Counting existing prompts per configured language..."
        )
        if not os.path.exists(cfg.prompts_dir):
            print(
                f"Prompts directory '{cfg.prompts_dir}' does not exist. No prompts found."
            )
            return 0  # Use return inside main()

        existing_prompt_files = glob(os.path.join(cfg.prompts_dir, "*_prompt.txt"))
        print(f"Found {len(existing_prompt_files)} total existing prompt files.")

        lang_counts = {lang: 0 for lang in language_config}
        unknown_ext_count = 0

        # Create a reverse map: extension -> language_name
        ext_to_lang = {}
        for lang, settings in language_config.items():
            for ext in settings.get("extensions", []):
                # Handle potential conflicts (e.g., if .js is in multiple groups) - last one wins here
                ext_to_lang[ext] = lang

        prompt_suffix = "_prompt.txt"
        for prompt_file in existing_prompt_files:
            basename = os.path.basename(prompt_file)
            if not basename.endswith(prompt_suffix):
                continue  # Should not happen with glob pattern

            # Attempt to extract original extension from filename based on the generation format:
            # Example: aider_aider_cli.py_prompt.txt
            base_no_suffix = basename[: -len(prompt_suffix)]
            inferred_ext = ""

            # Extract extension: everything after the last dot in the base name
            if "." in base_no_suffix:
                potential_ext = (
                    "." + base_no_suffix.rsplit(".", 1)[1]
                )  # ".py", ".js", ...
                # Check if this potential extension is defined in our config
                if potential_ext in ext_to_lang:
                    inferred_ext = potential_ext
            else:
                potential_ext = ""  # no dot found → leave blank

            lang_found = ext_to_lang.get(inferred_ext)
            if lang_found:
                lang_counts[lang_found] += 1
            else:
                # This case means the inferred extension wasn't in our config map
                # Or we couldn't infer an extension reliably from the filename parts
                unknown_ext_count += 1

        print("\nExisting prompts per language (based on filename inference):")
        for lang, count in lang_counts.items():
            print(f"- {lang}: {count}")
        if unknown_ext_count > 0:
            print(f"- Unknown/Unmatched: {unknown_ext_count}")
        print("\nExiting as --add-prompts is 0.")
        return 0  # Use return inside main()
    # --- End Reporting Mode Check ---

    # --- Clone repos if needed and build repo paths list ---
    print(f"Processing {len(args.repos)} repositories...")
    repo_paths = []
    success_count = 0
    fail_count = 0

    for repo_name in args.repos:
        print(f"\nProcessing repository: {repo_name}")
        try:
            repo_path = clone_repo_to_cache(repo_name, "cached-repos")
            repo_paths.append(repo_path)
            success_count += 1
        except ValueError as e:
            print(f"Error processing repository name {repo_name}: {e}")
            fail_count += 1
        except Exception as e:
            print(f"Failed to process {repo_name}: {e}")
            fail_count += 1

    if fail_count > 0:
        print(f"\nWarning: Failed to process {fail_count} repositories.")

    if not repo_paths:
        print("Error: No valid repository directories found to process.")
        return 1

    print(
        f"Successfully prepared {len(repo_paths)} repositories for prompt generation."
    )

    all_candidate_stats = []  # Changed name from all_stats
    generation_errors = 0
    total_date_filtered_count = 0  # Initialize counter for date filtering
    total_expected_token_filtered_count = (
        0  # Initialize counter for expected token filtering
    )
    total_already_exists_count = 0  # Initialize counter for already existing files

    for repo_path in repo_paths:
        repo_name = os.path.basename(os.path.normpath(repo_path))
        org_name = os.path.basename(os.path.dirname(repo_path))
        print(f"\nProcessing repository: {org_name}/{repo_name} ({repo_path})")
        try:
            # Pass the config object and existing prefixes
            (
                stats_list,
                date_filtered_count,
                expected_token_filtered_count,
                already_exists_count,
            ) = generate_prompts_and_expected(
                repo_path,
                cfg,
                existing_prefixes,
                all_target_extensions,  # Pass the set of extensions derived from language config
            )
            all_candidate_stats.extend(stats_list)  # Add to candidates
            total_date_filtered_count += date_filtered_count  # Accumulate date count
            total_expected_token_filtered_count += (
                expected_token_filtered_count  # Accumulate token count
            )
            total_already_exists_count += (
                already_exists_count  # Accumulate already exists count
            )
            print(
                f"Generated {len(stats_list)} prompts for {org_name}/{repo_name} (skipped {date_filtered_count} by date, "
                f"{expected_token_filtered_count} by expected tokens, {already_exists_count} already exist)."
            )
        except Exception as e:
            print(f"Error generating prompts for {org_name}/{repo_name}: {e}")
            generation_errors += 1

    if generation_errors > 0:
        print(
            f"\nWarning: Encountered errors during prompt generation for {generation_errors} repositories."
        )

    if not all_candidate_stats:
        print("\nError: No potential prompt candidates were generated successfully.")
        return 1

    # Report total files filtered by date
    if cfg.modified_within_months > 0:
        print(
            f"\nFiltered out a total of {total_date_filtered_count} files across all repositories due to modification date constraint (older than {cfg.modified_within_months} months)."
        )
    # Report total files filtered by expected token count
    if cfg.max_expected_tokens > 0:
        print(
            f"\nFiltered out a total of {total_expected_token_filtered_count} files across all repositories due to expected output token constraint (more than {cfg.max_expected_tokens} tokens)."
        )
    # Report total files skipped because they already exist
    print(
        f"\nSkipped processing a total of {total_already_exists_count} files across all repositories because they already exist in the benchmark set."
    )

    # --- Post-Generation Processing ---

    # Print summary statistics for all potential candidates
    print_stats_summary(
        all_candidate_stats, "Initial Candidate Statistics (All Repos, Pre-Filtering)"
    )

    # 1. Filter candidates by token range
    filtered_candidate_stats, _ = filter_prompts_by_token_range(
        all_candidate_stats, cfg
    )
    print_stats_summary(
        filtered_candidate_stats, "Statistics After Token Range Filtering"
    )

    # 2. Filter out existing prompts
    new_candidate_stats = [
        stats
        for stats in filtered_candidate_stats
        if stats["benchmark_case_prefix"] not in existing_prefixes
    ]
    print(
        f"\nFiltered out {len(filtered_candidate_stats) - len(new_candidate_stats)} candidates that already exist."
    )
    print_stats_summary(
        new_candidate_stats, "Statistics of New Candidates (Non-Existing)"
    )

    if not new_candidate_stats:
        print("\nNo new, valid candidates found to add. Exiting.")
        # Clean up temp dir even if no files are added
        if os.path.exists(cfg.temp_dir):
            try:
                shutil.rmtree(cfg.temp_dir)
                print(f"Successfully removed temporary directory: {cfg.temp_dir}")
            except OSError as e:
                print(
                    f"Warning: Failed to remove temporary directory {cfg.temp_dir}: {e}"
                )
        return 0

    # 3. Sample prompts to add from the new candidates
    prompts_to_add_stats, _ = sample_prompts(new_candidate_stats, cfg)
    print_stats_summary(
        prompts_to_add_stats,
        f"Statistics of {len(prompts_to_add_stats)} Prompts Selected to Add",
    )

    if not prompts_to_add_stats:
        print("\nNo prompts were selected during sampling. Exiting.")
        # Clean up temp dir even if no files are added
        if os.path.exists(cfg.temp_dir):
            try:
                shutil.rmtree(cfg.temp_dir)
                print(f"Successfully removed temporary directory: {cfg.temp_dir}")
            except OSError as e:
                print(
                    f"Warning: Failed to remove temporary directory {cfg.temp_dir}: {e}"
                )
        return 0

    # 4. Determine prefixes of the prompts to add
    prefixes_to_add = set()
    for stats in prompts_to_add_stats:
        prefixes_to_add.add(stats["benchmark_case_prefix"])

    # 5. Copy the selected new files from temp directory to the final prompts directory
    copy_selected_files(
        prefixes_to_add, cfg.temp_dir, cfg.prompts_dir
    )  # Changed output_dir to prompts_dir

    # 6. Save the updated metadata (appending the new run) to the prompts directory
    save_benchmark_metadata(
        existing_metadata_runs,
        prompts_to_add_stats,
        cfg,
        language_config,  # Pass language config here
    )

    # 7. Clean up temporary directory
    print(f"\nCleaning up temporary directory: {cfg.temp_dir}")
    if os.path.exists(cfg.temp_dir):
        try:
            shutil.rmtree(cfg.temp_dir)
            print(f"Successfully removed temporary directory: {cfg.temp_dir}")
        except OSError as e:
            print(f"Warning: Failed to remove temporary directory {cfg.temp_dir}: {e}")
            print("You may need to manually delete this directory.")

    print("\nBenchmark prompt generation and processing complete.")
    return 0


if __name__ == "__main__":
    # Example Usage:
    # Generate 10 new prompts for python/tsx files from mentat repo:
    #   ./benchmark_pipeline/1_generate_prompts.py -r AbanteAI/mentat -e .py .tsx --benchmark-run-dir ./benchmark_runs/mentat_dev --add-prompts 10
    # Analyze existing prompts in a specific run directory:
    #   ./benchmark_pipeline/1_generate_prompts.py --benchmark-run-dir ./benchmark_runs/mentat_dev --add-prompts 0
    exit_code = main()
    sys.exit(exit_code)
