#!/usr/bin/env python3
"""
Generates benchmark prompts and expected outputs from cloned repositories.

Purpose:
  This script iterates through specified files (matching given extensions) within
  repositories located in the 'cached-repos/' directory. For each eligible file,
  it generates two corresponding files in the 'generated_prompts/' directory:
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
  --extensions, -e (required): List of file extensions to process (e.g., '.py' '.js').
  --cache-dir (optional): Path to the directory containing cached repositories
                          (default: 'cached-repos').
  --output-dir (optional): Directory to save generated prompt/expected files
                           (default: 'generated_prompts').
  --min-prompt-tokens (optional): Minimum number of tokens allowed in the generated
                                  prompt file (default: 0).
  --max-prompt-tokens (optional): Maximum number of tokens allowed in the generated
                                  prompt file (default: 50000).
  --num-prompts (optional): The target number of prompts to generate in the final
                            benchmark set after sampling (default: 100).
  --modified-within-months (optional): Only process files last modified within the
                                       specified number of months (default: 3).
                                       Set <= 0 to disable.
  --max-expected-tokens (optional): Skip files whose final content (expected output)
                                    exceeds this token count (default: 12000).
                                    Set <= 0 to disable.

Inputs:
  - Repositories cloned by `1_clone_repos.py` located in the `cache-dir`
    (default: 'cached-repos/'). Expects standard git repository structure within.
  - Command-line arguments specifying extensions, directories, and filtering/sampling parameters.

Outputs:
  - Creates the `output-dir` (default: 'generated_prompts/') if it doesn't exist.
  - Populates `output-dir` with pairs of files for each processed source file:
    - `repo_path_prompt.txt`: Contains the reconstruction prompt (filename format uses repo name and sanitized relative path).
    - `repo_path_expectedoutput.txt`: Contains the ground truth file content (filename format uses repo name and sanitized relative path).
  - Creates `output-dir/metadata.json`: Contains metadata about the generation
    parameters and the final list of benchmark cases.
  - Prints statistics about the generation process, filtering, and final sampling results to the console.

File Modifications:
  - Creates the `output-dir` if it doesn't exist.
  - Creates `*_prompt.txt` files within `output-dir`.
  - Creates `*_expectedoutput.txt` files within `output-dir`.
  - Creates or overwrites `metadata.json` within `output-dir`.
  - Deletes prompt/expected file pairs from `output-dir` that are filtered out due
    to token limits (prompt token outside min/max range, or expected token > max_expected_tokens)
    or removed during the sampling process (`num-prompts`).
  - Does *not* modify files within the `cache-dir`.
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
from statistics import mean, median, stdev
import random
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any


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

    extensions: List[str]
    cache_dir: str
    output_dir: str
    temp_dir: str  # Added temporary directory
    min_prompt_tokens: int
    max_prompt_tokens: int
    num_prompts: int
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


# --- Core Generation Logic ---


def generate_prompts_and_expected(
    repo_path: str, cfg: Config
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Generates prompts and expected outputs for eligible files in a repository.

    Iterates through files matching specified extensions, applies date and token
    filters, fetches git history, calculates stats, and writes output files to
    the temporary directory.

    Args:
        repo_path: Path to the cloned repository.
        cfg: Configuration object.

    Returns:
        Tuple: (stats_list, date_filtered_count, expected_token_filtered_count)
            - stats_list: List of dictionaries with statistics for each generated case.
            - date_filtered_count: Number of files skipped due to modification date.
            - expected_token_filtered_count: Number of files skipped due to expected token limit.
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
            if any(filename.endswith(ext) for ext in cfg.extensions):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, repo_path)
                files_to_process.append((full_path, rel_path))
    print(f"Found {len(files_to_process)} files to potentially process.")

    # --- Process Files ---
    for full_path, rel_path in tqdm(
        files_to_process, desc=f"Generating prompts for {full_repo_name}"
    ):
        # 1. Date Filter Check
        try:
            if not is_file_recently_modified(rel_path, repo_path, threshold_timestamp):
                date_filtered_count += 1
                continue
        except ValueError as e:
            print(f"\nSkipping {rel_path}: {e}")
            date_filtered_count += 1
            continue

        # 2. Read Final Content & Expected Token Filter Check
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as original:
                final_content = original.read()
        except Exception as e:
            print(f"\nSkipping {rel_path}: Error reading file: {e}")
            continue

        expected_tokens = count_tokens(final_content, cfg.encoder)
        if cfg.max_expected_tokens > 0 and expected_tokens > cfg.max_expected_tokens:
            expected_token_filtered_count += 1
            continue

        # 3. Prepare Filenames and Paths (write to temporary directory)
        safe_rel = rel_path.replace(os.sep, "_")
        repo_file_prefix = f"{repo_name}_{safe_rel}"
        prompt_fname = f"{repo_file_prefix}_prompt.txt"
        expected_fname = f"{repo_file_prefix}_expectedoutput.txt"

        # Write to temp directory instead of output directory
        prompt_path = os.path.join(cfg.temp_dir, prompt_fname)
        expected_path = os.path.join(cfg.temp_dir, expected_fname)

        # 4. Get Git History
        try:
            git_history = get_git_history(rel_path, repo_path)
        except RuntimeError as e:
            print(f"\n{e}")
            continue

        # 5. Construct and Write Prompt File
        prompt_content = build_prompt_content(rel_path, git_history)
        try:
            write_output_files(
                prompt_path, expected_path, prompt_content, final_content
            )
        except Exception as e:
            print(f"\nError writing output files for {rel_path}: {e}")
            continue

        # 6. Calculate Statistics
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

    return stats_list, date_filtered_count, expected_token_filtered_count


# --- Statistics, Filtering, Sampling, and Deletion Functions ---


def print_stats_summary(stats_list: List[Dict[str, Any]], title: str):
    """Prints a summary of statistics for a list of prompts."""
    if not stats_list:
        print(f"\n--- {title} ---")
        print("No prompts in this set.")
        return

    count = len(stats_list)
    prompt_tokens = [s["prompt_tokens"] for s in stats_list]
    expected_tokens = [s["expected_tokens"] for s in stats_list]
    num_commits = [s["num_commits"] for s in stats_list]
    lines_added = [s["lines_added"] for s in stats_list]
    lines_deleted = [s["lines_deleted"] for s in stats_list]
    final_lines = [s["final_lines"] for s in stats_list]

    print(f"\n--- {title} (Count: {count}) ---")
    if count > 0:
        print(
            f"  Prompt Tokens:  Min={min(prompt_tokens)}, Max={max(prompt_tokens)}, Avg={mean(prompt_tokens):.0f}, Median={median(prompt_tokens):.0f}, StdDev={stdev(prompt_tokens) if count > 1 else 0:.0f}"
        )
        print(
            f"  Expected Tokens: Min={min(expected_tokens)}, Max={max(expected_tokens)}, Avg={mean(expected_tokens):.0f}, Median={median(expected_tokens):.0f}, StdDev={stdev(expected_tokens) if count > 1 else 0:.0f}"
        )
        print(
            f"  Num Commits:    Min={min(num_commits)}, Max={max(num_commits)}, Avg={mean(num_commits):.1f}, Median={median(num_commits):.1f}"
        )
        print(
            f"  Lines Added:    Min={min(lines_added)}, Max={max(lines_added)}, Avg={mean(lines_added):.0f}, Median={median(lines_added):.0f}"
        )
        print(
            f"  Lines Deleted:  Min={min(lines_deleted)}, Max={max(lines_deleted)}, Avg={mean(lines_deleted):.0f}, Median={median(lines_deleted):.0f}"
        )
        print(
            f"  Final Lines:    Min={min(final_lines)}, Max={max(final_lines)}, Avg={mean(final_lines):.0f}, Median={median(final_lines):.0f}"
        )


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
        cfg: The configuration object containing num_prompts, min_prompt_tokens, max_prompt_tokens.

    Returns:
        A tuple containing:
        - A list containing the final sampled prompt statistics.
        - The total number of prompts sampled out (discarded).
    """
    target_num_prompts = cfg.num_prompts
    min_tk = cfg.min_prompt_tokens
    max_tk = cfg.max_prompt_tokens

    print(
        f"\n--- Sampling Prompts (Target: {target_num_prompts}, Range: [{min_tk}, {max_tk}]) ---"
    )

    if not stats_list:
        print("No prompts available to sample from.")
        return [], 0

    if len(stats_list) <= target_num_prompts:
        print(
            f"Number of available prompts ({len(stats_list)}) is less than or equal to target ({target_num_prompts}). Keeping all."
        )
        return stats_list, 0

    # --- Targeted Sampling Logic ---
    items_to_keep = []
    available_items = list(stats_list)  # Create a mutable copy
    num_to_sample = min(target_num_prompts, len(available_items))

    print(f"Sampling {num_to_sample} prompts from {len(available_items)} available...")

    for i in tqdm(range(num_to_sample), desc="Sampling prompts"):
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
    output_dir: str,
):
    """
    Copies selected prompt and expected output files from temporary directory to output directory.

    Args:
        kept_prefixes: A set of benchmark case prefixes to copy.
        temp_dir: The source directory containing temporary files.
        output_dir: The destination directory to copy selected files to.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    copied_files_count = 0

    print(
        f"\n--- Copying {len(kept_prefixes)} Selected Benchmark Files to Final Location ---"
    )

    if not kept_prefixes:
        print("No files to copy.")
        return

    for prefix in tqdm(kept_prefixes, desc="Copying files"):
        prompt_filename = f"{prefix}_prompt.txt"
        expected_filename = f"{prefix}_expectedoutput.txt"

        source_prompt_path = os.path.join(temp_dir, prompt_filename)
        source_expected_path = os.path.join(temp_dir, expected_filename)

        dest_prompt_path = os.path.join(output_dir, prompt_filename)
        dest_expected_path = os.path.join(output_dir, expected_filename)

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


def save_benchmark_metadata(final_stats_list: List[Dict[str, Any]], cfg: Config):
    """
    Saves the final benchmark structure and generation parameters to metadata.json.

    Args:
        final_stats_list: List of statistics for the final sampled prompts.
        cfg: The configuration object containing output_dir and other parameters.
    """
    # Create generation_params dict from Config, excluding non-serializable fields
    generation_params = asdict(cfg)
    # Remove fields that are not JSON serializable or derived
    generation_params.pop("encoder", None)
    # Remove bucket related fields if they somehow lingered
    generation_params.pop("buckets_str", None)
    generation_params.pop("bucket_boundaries", None)
    generation_params.pop("max_per_bucket", None)

    metadata = {
        "generation_parameters": generation_params,
        "benchmark_cases": [],  # Flat list of benchmark cases
    }

    # Collect all benchmark cases from the final list
    for stats in final_stats_list:
        # Store only the necessary info for analysis
        metadata["benchmark_cases"].append(
            {
                "benchmark_case_prefix": stats["benchmark_case_prefix"],
                "original_filename": stats["filename"],
                "repo_name": stats["repo_name"],
                "repo_commit_hash": stats["repo_commit_hash"],
                "prompt_tokens": stats["prompt_tokens"],
                # Optionally add other stats if needed for analysis later
                "expected_tokens": stats["expected_tokens"],
                "num_commits": stats["num_commits"],
            }
        )

    metadata_path = os.path.join(cfg.output_dir, "metadata.json")
    try:
        with open(metadata_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=4)
        print(f"\nSaved final benchmark structure metadata to {metadata_path}")
    except (
        IOError,
        TypeError,
    ) as e:  # Catch specific errors related to writing/serialization
        # Raise error instead of printing warning
        raise RuntimeError(
            f"Error: Failed to save metadata to {metadata_path}: {e}"
        ) from e


# --- Main Script Logic ---


def find_repo_dirs(cache_dir="cached-repos"):
    """Finds repository directories within the cache directory."""
    # Assumes structure cache_dir/org/repo
    pattern = os.path.join(cache_dir, "*", "*")
    repo_dirs = [
        d
        for d in glob(pattern)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, ".git"))
    ]
    return repo_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark prompts and expected outputs from previously cloned repositories."
    )
    parser.add_argument(
        "--extensions",
        "-e",
        type=str,
        required=True,
        help="Comma-separated list of file extensions to process (include the dot), e.g., .py,.txt,.js",
    )
    parser.add_argument(
        "--cache-dir",
        default="cached-repos",
        help="Directory containing the cached repositories (default: 'cached-repos').",
    )
    parser.add_argument(
        "--output-dir",
        default="generated_prompts",
        help="Directory to save generated prompt/expected files (default: 'generated_prompts').",
    )
    parser.add_argument(
        "--temp-dir",
        default="generated_prompts_temp",
        help="Directory for storing temporarily generated files (default: 'generated_prompts_temp').",
    )
    # Add arguments for filtering/sampling parameters
    parser.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=0,
        help="Minimum number of tokens allowed in the prompt file (default: 0).",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=50000,
        help="Maximum number of tokens allowed in the prompt file (default: 50000).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Target number of prompts to generate in the final set after sampling (default: 100).",
    )
    parser.add_argument(
        "--modified-within-months",
        type=int,
        default=3,
        metavar="N",
        help="Only process files modified in the last N months (default: 3). Set to 0 or negative to disable.",
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
    if args.max_prompt_tokens <= args.min_prompt_tokens:
        print("Error: --max-prompt-tokens must be greater than --min-prompt-tokens.")
        return 1
    if args.num_prompts <= 0:
        print("Error: --num-prompts must be positive.")
        return 1
    # --- End argument validation ---

    # --- Create Config object ---
    # Parse extensions from comma-separated string to list
    extension_list = [ext.strip() for ext in args.extensions.split(",")]

    cfg = Config(
        extensions=extension_list,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,  # Add temp_dir
        min_prompt_tokens=args.min_prompt_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        num_prompts=args.num_prompts,
        modified_within_months=args.modified_within_months,
        max_expected_tokens=args.max_expected_tokens,
        encoder=tiktoken.get_encoding("cl100k_base"),  # Initialize encoder here
    )
    print(f"Configuration loaded: {cfg}")
    # --- End Config creation ---

    repo_paths = find_repo_dirs(cfg.cache_dir)

    if not repo_paths:
        print(f"Error: No valid repository directories found in {cfg.cache_dir}")
        print("Please run the clone_repos.py script first.")
        return 1

    print(f"Found {len(repo_paths)} repositories to process.")

    all_stats = []
    generation_errors = 0
    total_date_filtered_count = 0  # Initialize counter for date filtering
    total_expected_token_filtered_count = (
        0  # Initialize counter for expected token filtering
    )

    for repo_path in repo_paths:
        repo_name = os.path.basename(os.path.normpath(repo_path))
        org_name = os.path.basename(os.path.dirname(repo_path))
        print(f"\nProcessing repository: {org_name}/{repo_name} ({repo_path})")
        try:
            # Pass the config object
            (
                stats_list,
                date_filtered_count,
                expected_token_filtered_count,
            ) = generate_prompts_and_expected(repo_path, cfg)  # Pass cfg object
            all_stats.extend(stats_list)
            total_date_filtered_count += date_filtered_count  # Accumulate date count
            total_expected_token_filtered_count += (
                expected_token_filtered_count  # Accumulate token count
            )
            print(
                f"Generated {len(stats_list)} prompts for {org_name}/{repo_name} (skipped {date_filtered_count} by date, {expected_token_filtered_count} by expected tokens)."
            )
        except Exception as e:
            print(f"Error generating prompts for {org_name}/{repo_name}: {e}")
            generation_errors += 1

    if generation_errors > 0:
        print(
            f"\nWarning: Encountered errors during prompt generation for {generation_errors} repositories."
        )

    if not all_stats:
        print("\nError: No prompts were generated successfully.")
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

    # --- Post-Generation Processing ---

    # Print summary statistics for all generated prompts (before filtering/sampling)
    print_stats_summary(
        all_stats, "Initial Generation Statistics (All Repos, Pre-Filtering/Sampling)"
    )

    # 1. Filter prompts by token range
    filtered_stats, _ = filter_prompts_by_token_range(all_stats, cfg)
    print_stats_summary(filtered_stats, "Statistics After Token Range Filtering")

    # 2. Sample prompts from the filtered list
    final_sampled_stats, _ = sample_prompts(filtered_stats, cfg)
    print_stats_summary(final_sampled_stats, "Final Sampled Benchmark Set Statistics")

    # 3. Determine which files were kept after sampling and filtering
    kept_prefixes = set()
    for stats in final_sampled_stats:
        kept_prefixes.add(stats["benchmark_case_prefix"])

    # 4. Copy selected files from temp directory to output directory
    copy_selected_files(kept_prefixes, cfg.temp_dir, cfg.output_dir)

    # 5. Save the final benchmark structure metadata to the output directory
    save_benchmark_metadata(final_sampled_stats, cfg)

    # 6. Clean up temporary directory
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
    exit_code = main()
    sys.exit(exit_code)
