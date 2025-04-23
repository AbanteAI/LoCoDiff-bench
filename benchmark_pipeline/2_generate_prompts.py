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

  The script applies filters based on file modification date and the token count
  of the expected output. It then buckets the generated prompts based on their
  token count, samples prompts within each bucket to meet a specified maximum,
  and saves metadata about the final benchmark set.

Arguments:
  --extensions, -e (required): List of file extensions to process (e.g., '.py' '.js').
  --cache-dir (optional): Path to the directory containing cached repositories
                          (default: 'cached-repos').
  --output-dir (optional): Directory to save generated prompt/expected files
                           (default: 'generated_prompts').
  --buckets (optional): Comma-separated list of bucket boundaries in thousands
                        of prompt tokens (e.g., "0,20,40,80,100"). Defines the
                        token ranges for grouping prompts. (default: "0,20,40,60,80,100").
                        The highest value also acts as a maximum token filter.
  --max-per-bucket (optional): Maximum number of prompts to keep per bucket after
                               sampling (default: 10).
  --modified-within-months (optional): Only process files last modified within the
                                       specified number of months (default: 3).
                                       Set <= 0 to disable.
  --max-expected-tokens (optional): Skip files whose final content (expected output)
                                    exceeds this token count (default: 12000).
                                    Set <= 0 to disable.

Inputs:
  - Repositories cloned by `1_clone_repos.py` located in the `cache-dir`
    (default: 'cached-repos/'). Expects standard git repository structure within.
  - Command-line arguments specifying extensions, directories, and filtering/bucketing parameters.

Outputs:
  - Creates the `output-dir` (default: 'generated_prompts/') if it doesn't exist.
  - Populates `output-dir` with pairs of files for each processed source file:
    - `repo_path_prompt.txt`: Contains the reconstruction prompt (filename format uses repo name and sanitized relative path).
    - `repo_path_expectedoutput.txt`: Contains the ground truth file content (filename format uses repo name and sanitized relative path).
  - Creates `output-dir/metadata.json`: Contains metadata about the generation
    parameters and the final set of benchmark cases organized by bucket.
  - Prints statistics about the generation process, filtering, and final bucket distribution to the console.

File Modifications:
  - Creates the `output-dir` if it doesn't exist.
  - Creates `*_prompt.txt` files within `output-dir`.
  - Creates `*_expectedoutput.txt` files within `output-dir`.
  - Creates or overwrites `metadata.json` within `output-dir`.
  - Deletes prompt/expected file pairs from `output-dir` that are filtered out due
    to token limits (prompt token > max bucket boundary, or expected token > max_expected_tokens)
    or removed during the sampling process (`max-per-bucket`).
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
from collections import defaultdict
from statistics import mean
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
    buckets_str: str  # Original comma-separated string for metadata
    bucket_boundaries: List[int]  # Processed list of token boundaries
    max_per_bucket: int
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


# --- Statistics, Filtering, Bucketing, Sampling, and Deletion Functions ---


def print_stats_table(stats_list: List[Dict[str, Any]]):
    """Prints a formatted table of the collected statistics."""
    if not stats_list:
        print("No statistics generated.")
        return

    # Sort by prompt tokens (descending) before printing
    stats_list.sort(key=lambda x: x["prompt_tokens"], reverse=True)

    # Determine column widths dynamically
    max_len_filename = max(len(s["filename"]) for s in stats_list) if stats_list else 10
    col_widths = {
        "filename": max(max_len_filename, 8),  # Min width 8 for "Filename"
        "prompt_tokens": 13,  # "Prompt Tokens"
        "expected_tokens": 15,  # "Expected Tokens"
        "num_commits": 8,  # "Commits"
        "lines_added": 7,  # "Added"
        "lines_deleted": 9,  # "Deleted"
        "final_lines": 11,  # "Final Lines"
    }

    # Header
    header = (
        f"{'Filename':<{col_widths['filename']}} | "
        f"{'Prompt Tokens':>{col_widths['prompt_tokens']}} | "
        f"{'Expected Tokens':>{col_widths['expected_tokens']}} | "
        f"{'Commits':>{col_widths['num_commits']}} | "
        f"{'Added':>{col_widths['lines_added']}} | "
        f"{'Deleted':>{col_widths['lines_deleted']}} | "
        f"{'Final Lines':>{col_widths['final_lines']}}"
    )
    print(header)
    print("-" * len(header))

    # Rows
    for stats in stats_list:
        row = (
            f"{stats['filename']:<{col_widths['filename']}} | "
            f"{stats['prompt_tokens']:>{col_widths['prompt_tokens']}} | "
            f"{stats['expected_tokens']:>{col_widths['expected_tokens']}} | "
            f"{stats['num_commits']:>{col_widths['num_commits']}} | "
            f"{stats['lines_added']:>{col_widths['lines_added']}} | "
            f"{stats['lines_deleted']:>{col_widths['lines_deleted']}} | "
            f"{stats['final_lines']:>{col_widths['final_lines']}}"
        )
        print(row)


def filter_prompts(
    stats_list: List[Dict[str, Any]], cfg: Config
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Filters the list of generated prompt statistics based on token limits.

    Args:
        stats_list: The initial list of statistics for all generated prompts.
        cfg: The configuration object containing bucket_boundaries.

    Returns:
        A tuple containing:
        - A list of statistics dictionaries for prompts that pass the filter.
        - The count of prompts that were filtered out.
    """
    bucket_boundaries = cfg.bucket_boundaries
    if not bucket_boundaries or len(bucket_boundaries) < 2:
        raise ValueError("Config bucket_boundaries must contain at least two elements.")
    max_tokens = bucket_boundaries[-1]  # Highest boundary is the max prompt tokens

    print(f"\n--- Filtering Prompts (Max Prompt Tokens: {max_tokens}) ---")
    kept_stats = []
    filtered_count = 0

    for stats in stats_list:
        # Filter prompts strictly greater than the max boundary,
        # or exactly zero if the first boundary is > 0 (meaning 0 is excluded).
        if stats["prompt_tokens"] > max_tokens or (
            bucket_boundaries[0] > 0 and stats["prompt_tokens"] == 0
        ):
            filtered_count += 1
        else:
            # Keep prompts within the overall range [min_boundary, max_boundary]
            # Note: The lower bound check happens implicitly during bucketing.
            kept_stats.append(stats)

    print(
        f"Filtered out {filtered_count} prompts exceeding token limit ({max_tokens}) or below minimum boundary ({bucket_boundaries[0]})."
    )
    return kept_stats, filtered_count


def assign_prompts_to_buckets(
    stats_list: List[Dict[str, Any]], cfg: Config
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """
    Assigns filtered prompt statistics to buckets based on prompt token count.

    Args:
        stats_list: List of statistics dictionaries (already filtered).
        cfg: The configuration object containing bucket_boundaries.

    Returns:
        A dictionary where keys are bucket range tuples (min_token, max_token)
        and values are lists of stats dictionaries belonging to that bucket.
    """
    bucket_boundaries = cfg.bucket_boundaries
    max_tokens = bucket_boundaries[-1]

    print("\n--- Assigning Prompts to Buckets ---")
    buckets = defaultdict(list)
    # Define bucket ranges and initialize dictionary
    bucket_ranges = []
    for i in range(len(bucket_boundaries) - 1):
        min_tk = bucket_boundaries[i]
        max_tk = bucket_boundaries[i + 1]
        bucket_key = (min_tk, max_tk)
        buckets[bucket_key] = []  # Initialize empty list for each defined bucket
        bucket_ranges.append(bucket_key)

    # unassigned_count = 0 # Removed as it became unused after changing warning to error
    for stats in stats_list:
        tokens = stats["prompt_tokens"]
        assigned = False
        for min_tk, max_tk in bucket_ranges:
            # Check if token count falls within the bucket range [min_tk, max_tk).
            # Special case: Include 0 tokens if min_tk is 0.
            # Special case: Include the upper boundary max_tk if it's the very last boundary.
            is_last_bucket = max_tk == max_tokens
            in_range = (tokens >= min_tk and tokens < max_tk) or (
                min_tk == 0 and tokens == 0
            )
            # Include the max_tokens value itself in the last bucket
            if is_last_bucket and tokens == max_tk:
                in_range = True

            if in_range:
                buckets[(min_tk, max_tk)].append(stats)
                assigned = True
                break

        if not assigned:
            # This should ideally not happen if filtering and bucketing logic is correct
            # This should ideally not happen if filtering and bucketing logic is correct
            raise ValueError(
                f"Error: Prompt with {tokens} tokens did not fit into any bucket defined by {bucket_boundaries}. Stats: {stats}"
            )
            # unassigned_count += 1 # Unreachable after raise

    # if unassigned_count > 0: # Unreachable after raise
    #     print(f"Warning: {unassigned_count} prompts could not be assigned to a bucket.")

    # Print summary of bucket assignments
    for bucket_key, items in buckets.items():
        print(f"  Bucket {bucket_key}: Assigned {len(items)} prompts.")

    return buckets


def sample_prompts_from_buckets(
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]], cfg: Config
) -> Tuple[Dict[Tuple[int, int], List[Dict[str, Any]]], int]:
    """
    Samples prompts within each bucket to meet the max_per_bucket limit.

    Args:
        buckets: Dictionary mapping bucket ranges to lists of prompt stats.
        cfg: The configuration object containing max_per_bucket.

    Returns:
        A tuple containing:
        - A dictionary containing the final sampled buckets.
        - The total number of prompts sampled out (discarded).
    """
    print(f"\n--- Sampling Buckets (Max per Bucket: {cfg.max_per_bucket}) ---")
    final_sampled_buckets = {}
    total_sampled_out = 0

    # Iterate through the buckets, ensuring consistent order for logging
    sorted_bucket_keys = sorted(buckets.keys(), key=lambda x: x[0])

    for bucket_key in sorted_bucket_keys:
        items = buckets[bucket_key]
        if len(items) > cfg.max_per_bucket:
            # --- Targeted Sampling Logic ---
            # This logic aims to select prompts that are somewhat evenly distributed
            # within the token range of the bucket, rather than purely random selection.
            min_tk, max_tk = bucket_key
            items_to_keep = []
            available_items = list(items)  # Create a mutable copy

            for _ in range(cfg.max_per_bucket):
                if not available_items:
                    break  # Should not happen if len(items) > max_per_bucket initially

                # 1. Choose a random target token count within the bucket's range.
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

            # Items remaining in available_items are the ones sampled out.
            items_sampled_out_count = len(available_items)
            total_sampled_out += items_sampled_out_count
            print(
                f"  Bucket {bucket_key}: Sampled down from {len(items)} to {cfg.max_per_bucket} (removed {items_sampled_out_count}, targeted sampling)."
            )

            # Store the kept items for this bucket
            final_sampled_buckets[bucket_key] = items_to_keep
        else:
            # Keep all items if count is within limit
            final_sampled_buckets[bucket_key] = items
            if items:  # Only print if bucket wasn't empty
                print(
                    f"  Bucket {bucket_key}: Kept all {len(items)} items (within limit)."
                )

    print(f"\nTotal prompts removed during sampling: {total_sampled_out}")
    return final_sampled_buckets, total_sampled_out


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
    # error_count = 0 # Removed as it became unused after changing warning to error

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


def print_bucket_stats_table(buckets):
    """Prints a formatted table of the bucket statistics."""
    print("\n--- Final Benchmark Set Statistics (Averages per Bucket) ---")
    if not buckets:
        print("No buckets to display statistics for.")
        return

    # Define columns and widths (reduced for better terminal fit)
    col_widths = {
        "bucket_range": 22,  # e.g., "0 - 20000 tokens"
        "count": 6,
        "avg_prompt_tokens": 12,  # Reduced width
        "avg_expected_tokens": 13,  # Reduced width
        "avg_num_commits": 10,  # Reduced width
        "avg_lines_added": 10,  # Reduced width
        "avg_lines_deleted": 11,  # Reduced width
        "avg_final_lines": 11,  # Reduced width
    }

    # Header
    header = (
        f"{'Bucket Range':<{col_widths['bucket_range']}} | "
        f"{'Count':>{col_widths['count']}} | "
        f"{'Avg Prompt':>{col_widths['avg_prompt_tokens']}} | "  # Shortened title
        f"{'Avg Expected':>{col_widths['avg_expected_tokens']}} | "  # Shortened title
        f"{'Avg Commits':>{col_widths['avg_num_commits']}} | "
        f"{'Avg Added':>{col_widths['avg_lines_added']}} | "
        f"{'Avg Deleted':>{col_widths['avg_lines_deleted']}} | "
        f"{'Avg Final':>{col_widths['avg_final_lines']}}"  # Shortened title
    )
    print(header)
    print("-" * len(header))

    # Sort buckets by the lower bound of the token range for consistent order
    # Filter out empty buckets before sorting and printing
    sorted_bucket_keys = sorted(
        [key for key, items in buckets.items() if items], key=lambda x: x[0]
    )

    total_count = 0
    # Rows
    for bucket_key in sorted_bucket_keys:
        items = buckets[bucket_key]
        count = len(items)
        total_count += count
        range_str = f"{bucket_key[0]} - {bucket_key[1]} tokens"

        # Calculate averages and round them to the nearest integer
        avg_prompt_tokens = round(mean(item["prompt_tokens"] for item in items))
        avg_expected_tokens = round(mean(item["expected_tokens"] for item in items))
        avg_num_commits = round(mean(item["num_commits"] for item in items))
        avg_lines_added = round(mean(item["lines_added"] for item in items))
        avg_lines_deleted = round(mean(item["lines_deleted"] for item in items))
        avg_final_lines = round(mean(item["final_lines"] for item in items))

        # Format rounded averages as integers (:d)
        row = (
            f"{range_str:<{col_widths['bucket_range']}} | "
            f"{count:>{col_widths['count']}} | "
            f"{avg_prompt_tokens:>{col_widths['avg_prompt_tokens']}d} | "
            f"{avg_expected_tokens:>{col_widths['avg_expected_tokens']}d} | "
            f"{avg_num_commits:>{col_widths['avg_num_commits']}d} | "
            f"{avg_lines_added:>{col_widths['avg_lines_added']}d} | "
            f"{avg_lines_deleted:>{col_widths['avg_lines_deleted']}d} | "
            f"{avg_final_lines:>{col_widths['avg_final_lines']}d}"
        )
        print(row)

    print("-" * len(header))
    print(f"Total prompts in final set: {total_count}")


def save_benchmark_metadata(
    final_buckets: Dict[Tuple[int, int], List[Dict[str, Any]]], cfg: Config
):
    """
    Saves the final benchmark structure and generation parameters to metadata.json.

    Args:
        final_buckets: Dictionary mapping bucket ranges to lists of kept prompt stats.
        cfg: The configuration object containing output_dir and other parameters.
    """
    # Create generation_params dict from Config, excluding non-serializable fields
    generation_params = asdict(cfg)
    # Remove fields that are not JSON serializable or derived
    generation_params.pop("encoder", None)
    generation_params.pop("bucket_boundaries", None)  # Remove the list, keep the string

    metadata = {
        "generation_parameters": generation_params,
        "benchmark_buckets": {},
    }

    # Convert tuple keys to string representation for JSON compatibility
    for bucket_key, stats_list in final_buckets.items():
        # Only include buckets that have prompts in them
        if stats_list:
            # Convert tuple key (min_token, max_token) to string "min_token-max_token"
            bucket_key_str = f"{bucket_key[0]}-{bucket_key[1]}"
            # Store only the necessary info for analysis: benchmark_case_prefix and original filename
            metadata["benchmark_buckets"][bucket_key_str] = [
                {
                    "benchmark_case_prefix": stats["benchmark_case_prefix"],
                    "original_filename": stats["filename"],
                    "repo_name": stats["repo_name"],
                    "repo_commit_hash": stats["repo_commit_hash"],
                    "prompt_tokens": stats["prompt_tokens"],
                }
                for stats in stats_list
            ]

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
        nargs="+",
        required=True,
        help="File extensions to process (include the dot), e.g., .py .txt",
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
    # Add arguments for filtering/bucketing/sampling parameters
    parser.add_argument(
        "--buckets",
        type=str,
        default="0,20,40,60,80,100",  # Default buckets up to 100k
        help='Comma-separated list of bucket boundaries in thousands of tokens (e.g., "0,10,20,40,80"). Defines buckets [0k-10k), [10k-20k), [20k-40k), [40k-80k).',
    )
    parser.add_argument(
        "--max-per-bucket",
        type=int,
        default=10,
        help="Maximum number of prompts per bucket after sampling (default: 10).",
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

    # --- Parse and validate bucket boundaries ---
    try:
        bucket_boundaries_k = [int(b.strip()) for b in args.buckets.split(",")]
        if len(bucket_boundaries_k) < 2:
            raise ValueError(
                "Must specify at least two bucket boundaries (e.g., '0,100')."
            )
        if not all(
            bucket_boundaries_k[i] < bucket_boundaries_k[i + 1]
            for i in range(len(bucket_boundaries_k) - 1)
        ):
            raise ValueError("Bucket boundaries must be strictly increasing.")
        if bucket_boundaries_k[0] < 0:
            raise ValueError("Bucket boundaries cannot be negative.")
        # Convert k-tokens to actual token counts
        bucket_boundaries = [b * 1000 for b in bucket_boundaries_k]
        print(f"Using bucket boundaries (tokens): {bucket_boundaries}")
    except ValueError as e:
        print(f"Error parsing --buckets argument '{args.buckets}': {e}")
        return 1
    # --- End bucket parsing ---

    # --- Create Config object ---
    cfg = Config(
        extensions=args.extensions,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,  # Add temp_dir
        buckets_str=args.buckets,  # Store original string
        bucket_boundaries=bucket_boundaries,  # Store processed list
        max_per_bucket=args.max_per_bucket,
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

    # Print initial statistics table for all generated prompts (before filtering/sampling)
    print("\n--- Initial Generation Statistics (All Repos, Pre-Filtering/Sampling) ---")
    print_stats_table(all_stats)

    # 1. Filter prompts by token limits
    filtered_stats, _ = filter_prompts(all_stats, cfg)

    # 2. Assign filtered prompts to buckets
    buckets = assign_prompts_to_buckets(filtered_stats, cfg)

    # 3. Sample prompts from buckets
    final_buckets, _ = sample_prompts_from_buckets(buckets, cfg)

    # Print statistics for the final selected buckets
    print_bucket_stats_table(final_buckets)

    # 4. Determine which files were kept after sampling and filtering
    kept_prefixes = set()
    for bucket_key, stats_list in final_buckets.items():
        for stats in stats_list:
            kept_prefixes.add(stats["benchmark_case_prefix"])

    # 5. Copy selected files from temp directory to output directory
    copy_selected_files(kept_prefixes, cfg.temp_dir, cfg.output_dir)

    # 6. Save the final benchmark structure metadata to the output directory
    save_benchmark_metadata(final_buckets, cfg)

    print("\nBenchmark prompt generation and processing complete.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
