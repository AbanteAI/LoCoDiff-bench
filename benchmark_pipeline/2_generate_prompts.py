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
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict, Any


# --- Helper Functions ---


def get_repo_head_commit_hash(repo_path):
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
    # Get commit hash - will raise an exception if it fails
    hash_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    commit_hash = hash_result.stdout.strip()
    return commit_hash


@dataclass(frozen=True)
class Config:
    """Configuration settings for the prompt generation script."""

    extensions: List[str]
    cache_dir: str
    output_dir: str
    buckets_str: str  # Original comma-separated string for metadata
    bucket_boundaries: List[int]  # Processed list of token boundaries
    max_per_bucket: int
    modified_within_months: int
    max_expected_tokens: int
    encoder: tiktoken.Encoding # Encoder instance is now required


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


# --- Core Generation Logic ---


def generate_prompts_and_expected(
    repo_path: str, cfg: Config
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    For every file in the repository at repo_path matching extensions in cfg,
    and optionally modified within cfg.modified_within_months, create two files in cfg.output_dir:
      - {repo_name}_{relative_path_with_underscores}_prompt.txt containing
        a reconstruction prompt with git history.
      - {repo_name}_{relative_path_with_underscores}_expectedoutput.txt containing
        the file's final content.

    Also calculates statistics about the generated prompts and files.

    Args:
        repo_path: Path to the cloned repository.
        cfg: The configuration object containing settings like extensions, output_dir,
             modified_within_months, and max_expected_tokens.

    Returns:
        A tuple containing:
        - A list of dictionaries, where each dictionary contains statistics
          for one processed file:
          {
            'filename': str,
            'prompt_tokens': int,
            'expected_tokens': int,
            'num_commits': int,
            'lines_added': int,
            'lines_deleted': int,
            'final_lines': int,
            'repo_name': str,
            'repo_commit_hash': str,
            'prompt_filename': str,
            'expected_filename': str,
            'benchmark_case_prefix': str
          }
        - An integer representing the count of files skipped due to the date filter.
        - An integer representing the count of files skipped due to the expected token limit.
    """
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)

    repo_name = os.path.basename(os.path.normpath(repo_path))
    org_name = os.path.basename(os.path.dirname(repo_path))
    full_repo_name = f"{org_name}/{repo_name}"
    stats_list = []
    files_to_process = []
    date_filtered_count = 0  # Counter for files skipped by date filter
    expected_token_filtered_count = (
        0  # Counter for files skipped by expected token limit
    )

    # Calculate date threshold if filter is enabled
    threshold_timestamp = None
    if cfg.modified_within_months > 0:
        # Approximate seconds per month (average)
        avg_seconds_per_month = 30.44 * 24 * 60 * 60
        current_timestamp = time.time()
        threshold_timestamp = current_timestamp - (
            cfg.modified_within_months * avg_seconds_per_month
        )
        print(
            f"Date filter enabled: Processing files modified since {datetime.fromtimestamp(threshold_timestamp, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    # Get repository head commit hash
    print(f"Getting head commit hash for {full_repo_name}...")
    head_commit_hash = get_repo_head_commit_hash(repo_path)
    print(f"Repository at commit: {head_commit_hash}")

    # First, collect all files matching the extensions
    for root, _, files in os.walk(repo_path):
        # Skip .git directory
        if ".git" in root.split(os.sep):
            continue
        for filename in files:
            if any(filename.endswith(ext) for ext in cfg.extensions):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, repo_path)
                files_to_process.append((full_path, rel_path))

    # Now, process the files with a progress bar
    for full_path, rel_path in tqdm(files_to_process, desc="Generating prompts"):
        # --- Date Filter Check ---
        if threshold_timestamp is not None:
            last_commit_timestamp_str = ""  # Initialize to ensure it's bound
            try:
                # Get the commit timestamp (Unix timestamp) of the last commit affecting the file
                commit_time_result = subprocess.run(
                    ["git", "log", "-1", "--format=%ct", "--", rel_path],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )
                last_commit_timestamp_str = commit_time_result.stdout.strip()

                # Check if we got a valid timestamp
                if not last_commit_timestamp_str:
                    print(
                        f"\nWarning: Could not get commit timestamp for {rel_path}. Skipping date check."
                    )
                else:
                    last_commit_timestamp = int(last_commit_timestamp_str)
                    # Compare with the threshold
                    if last_commit_timestamp < threshold_timestamp:
                        date_filtered_count += 1
                        continue  # Skip this file
            except subprocess.CalledProcessError as e:
                print(
                    f"\nWarning: Error getting commit time for {rel_path}: {e}. Skipping file."
                )
                date_filtered_count += 1  # Count as filtered if we can't get time
                continue
            except ValueError as e:
                print(
                    f"\nWarning: Could not parse commit timestamp for {rel_path}: '{last_commit_timestamp_str}'. Skipping file. Error: {e}"
                )
                date_filtered_count += 1
                continue
            except FileNotFoundError:
                print(
                    "\nWarning: git command not found. Cannot perform date filtering."
                )
                threshold_timestamp = None  # Disable filter if git is missing

        # --- Proceed with generation if not filtered by date ---

        # --- Read final content first to check expected token count ---
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as original:
                final_content = original.read()
        except Exception as e:
            print(f"\nWarning: Error reading file {full_path}: {e}. Skipping.")
            continue  # Skip if we can't even read the file

        # --- Expected Token Filter Check ---
        expected_tokens = count_tokens(
            final_content, cfg.encoder
        )  # Pass encoder from config
        if cfg.max_expected_tokens > 0 and expected_tokens > cfg.max_expected_tokens:
            expected_token_filtered_count += 1
            continue  # Skip this file

        # --- Proceed with prompt generation if not filtered by expected tokens ---
        safe_rel = rel_path.replace(os.sep, "_")
        prompt_fname = f"{repo_name}_{safe_rel}_prompt.txt"
        expected_fname = f"{repo_name}_{safe_rel}_expectedoutput.txt"
        prompt_path = os.path.join(cfg.output_dir, prompt_fname)
        expected_path = os.path.join(cfg.output_dir, expected_fname)

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
            print(f"\nWarning: Error getting git history for {rel_path}: {e}")
            git_history = f"Error retrieving git history: {e}\n"
        except FileNotFoundError:
            print("\nWarning: git command not found. Skipping history generation.")
            git_history = "git command not found.\n"

        # 2. Construct prompt content using Markdown structure
        prompt_content = f"""\
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
        with open(prompt_path, "w", encoding="utf-8") as pf:
            pf.write(prompt_content)

        # 3. Write final content (already read and token count checked)
        with open(expected_path, "w", encoding="utf-8") as ef:
            ef.write(final_content)

        # 4. Calculate statistics (expected_tokens already calculated)
        prompt_tokens = count_tokens(
            prompt_content, cfg.encoder
        )  # Pass encoder from config
        # expected_tokens = count_tokens(final_content, cfg.encoder) # Already done above
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
            print(f"\nWarning: Error getting numstat for {rel_path}: {e}")
        except FileNotFoundError:
            print("\nWarning: git command not found. Skipping numstat calculation.")

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
        # Include generated filenames and repo info in the stats for metadata/analysis
        file_stats["repo_name"] = full_repo_name
        file_stats["repo_commit_hash"] = head_commit_hash
        file_stats["prompt_filename"] = prompt_fname
        file_stats["expected_filename"] = expected_fname
        # Generate a unique prefix for this benchmark case
        # Format: {repo_name}_{relative_path_with_underscores}
        # Example: celery_celery_app___init__.py
        # This MUST match the prefix derived from filenames by run_benchmark.py
        benchmark_case_prefix = f"{repo_name}_{safe_rel}"
        file_stats["benchmark_case_prefix"] = benchmark_case_prefix

        stats_list.append(file_stats)

    # Return the list of stats and the counts of files filtered by date and expected tokens
    return stats_list, date_filtered_count, expected_token_filtered_count


# --- Statistics and Filtering Functions ---


def print_stats_table(stats_list):
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


def filter_bucket_sample_stats(
    stats_list: List[Dict[str, Any]], cfg: Config
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """
    Filters stats based on the highest bucket boundary, assigns them to buckets defined by the boundaries,
    samples buckets, and deletes discarded files.

    Args:
        stats_list: List of dictionaries containing statistics for each file.
        cfg: The configuration object containing output_dir, bucket_boundaries, and max_per_bucket.

    Returns:
        A dictionary where keys are bucket range tuples (min_token, max_token)
        and values are lists of stats dictionaries belonging to that bucket after filtering and sampling.
    """
    bucket_boundaries = cfg.bucket_boundaries  # Use boundaries from config
    if not bucket_boundaries or len(bucket_boundaries) < 2:
        raise ValueError("Config bucket_boundaries must contain at least two elements.")
    if not all(
        bucket_boundaries[i] < bucket_boundaries[i + 1]
        for i in range(len(bucket_boundaries) - 1)
    ):
        raise ValueError("Config bucket_boundaries must be strictly increasing.")
    if bucket_boundaries[0] < 0:
        raise ValueError("Config bucket boundaries cannot be negative.")

    max_tokens = bucket_boundaries[
        -1
    ]  # The highest boundary defines the max tokens allowed

    print(
        f"\n--- Filtering, Bucketing, and Sampling (Max Tokens: {max_tokens}, Max per Bucket: {cfg.max_per_bucket}) ---"
    )
    filtered_stats = []
    deleted_count = 0

    # 1. Filter by max_tokens (highest boundary) and delete corresponding files
    print(f"Filtering prompts with more than {max_tokens} tokens...")
    for stats in stats_list:
        # Filter prompts strictly greater than the max boundary, or exactly zero if the first boundary is > 0
        if stats["prompt_tokens"] > max_tokens or (
            bucket_boundaries[0] > 0 and stats["prompt_tokens"] == 0
        ):
            prompt_file = os.path.join(cfg.output_dir, stats["prompt_filename"])
            expected_file = os.path.join(cfg.output_dir, stats["expected_filename"])
            try:
                os.remove(prompt_file)
            except FileNotFoundError:
                print(f"Warning: Prompt file not found for deletion: {prompt_file}")
            try:
                os.remove(expected_file)
            except FileNotFoundError:
                print(f"Warning: Expected file not found for deletion: {expected_file}")
            deleted_count += 1
        else:
            # Keep prompts within the overall range [min_boundary, max_boundary]
            # Note: We handle the lower bound during bucketing.
            filtered_stats.append(stats)

    print(
        f"Filtered out {deleted_count} prompts exceeding token limit or below minimum boundary."
    )

    # 2. Bucket the filtered stats
    print("Assigning remaining prompts to buckets...")
    buckets = defaultdict(list)
    # Define bucket ranges directly from the boundaries list
    bucket_ranges = []
    for i in range(len(bucket_boundaries) - 1):
        min_tk = bucket_boundaries[i]
        max_tk = bucket_boundaries[i + 1]
        bucket_key = (min_tk, max_tk)
        buckets[bucket_key] = []  # Initialize empty list
        bucket_ranges.append(bucket_key)

    for stats in filtered_stats:
        tokens = stats["prompt_tokens"]
        assigned = False
        for min_tk, max_tk in bucket_ranges:
            # Assign if tokens are within the bucket range [min_tk, max_tk)
            # Special case: if min_tk is 0, include 0 tokens.
            # Also include the upper boundary max_tk if it's the very last boundary
            is_last_bucket = max_tk == max_tokens
            in_range = (tokens >= min_tk and tokens < max_tk) or (
                min_tk == 0 and tokens == 0
            )
            # Include the max_tokens value in the last bucket
            if is_last_bucket and tokens == max_tk:
                in_range = True

            if in_range:
                buckets[(min_tk, max_tk)].append(stats)
                assigned = True
                break

        if not assigned:
            # This should ideally not happen if filtering and bucketing logic is correct
            print(
                f"Warning: Prompt with {tokens} tokens did not fit into any bucket defined by {bucket_boundaries}. Stats: {stats}"
            )

    # 3. Sample buckets exceeding max_per_bucket and delete corresponding files
    print(f"Sampling buckets to have at most {cfg.max_per_bucket} prompts each...")
    final_buckets = {}
    total_sampled_out = 0
    # Iterate through the buckets created in step 2
    for bucket_key, items in buckets.items():
        if len(items) > cfg.max_per_bucket:
            # Targeted sampling logic
            min_tk, max_tk = bucket_key
            items_to_keep = []
            available_items = list(items)  # Copy to modify

            for _ in range(cfg.max_per_bucket):
                if not available_items:
                    break
                # Select item closest to a random target within the bucket range
                target_token_count = random.uniform(min_tk, max_tk)
                closest_item = min(
                    available_items,
                    key=lambda item: abs(item["prompt_tokens"] - target_token_count),
                )
                items_to_keep.append(closest_item)
                available_items.remove(closest_item)

            # Items remaining in available_items are discarded
            items_to_discard = available_items
            print(
                f"  Bucket {bucket_key}: Sampled down from {len(items)} to {cfg.max_per_bucket} (targeted sampling)."
            )
            total_sampled_out += len(items_to_discard)

            # Delete files for discarded items
            for stats in items_to_discard:
                prompt_file = os.path.join(cfg.output_dir, stats["prompt_filename"])
                expected_file = os.path.join(cfg.output_dir, stats["expected_filename"])
                try:
                    os.remove(prompt_file)
                except FileNotFoundError:
                    print(
                        f"Warning: Prompt file not found for deletion during sampling: {prompt_file}"
                    )
                try:
                    os.remove(expected_file)
                except FileNotFoundError:
                    print(
                        f"Warning: Expected file not found for deletion during sampling: {expected_file}"
                    )

            # Store the kept items for this bucket
            final_buckets[bucket_key] = items_to_keep
        else:
            # Keep all items if count is within limit
            final_buckets[bucket_key] = items
            if items:  # Only print if bucket wasn't empty
                print(f"  Bucket {bucket_key}: Kept all {len(items)} items.")

    print(f"Removed {total_sampled_out} prompts during sampling.")
    print("Filtering, bucketing, and sampling complete.")
    # Return the final buckets containing the stats of the kept prompts
    return final_buckets


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
    # Create generation_params dict from Config, excluding derived fields if needed
    # Using asdict and then removing the derived list of boundaries
    generation_params = asdict(cfg)
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
    except Exception as e:
        print(f"\nWarning: Failed to save metadata to {metadata_path}: {e}")


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
        buckets_str=args.buckets,  # Store original string
        bucket_boundaries=bucket_boundaries,  # Store processed list
        max_per_bucket=args.max_per_bucket,
        modified_within_months=args.modified_within_months,
        max_expected_tokens=args.max_expected_tokens,
        encoder=tiktoken.get_encoding("cl100k_base") # Initialize encoder here
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

    # Print initial statistics table for all generated prompts (after all filtering)
    print("\n--- Initial Generation Statistics (All Repos, Post-Filtering) ---")
    print_stats_table(all_stats)

    # Filter, bucket, and sample the results using the config object
    final_buckets = filter_bucket_sample_stats(all_stats, cfg)  # Pass cfg object

    # Print statistics for the final buckets
    print_bucket_stats_table(final_buckets)

    # Save the final benchmark structure metadata using the config object
    save_benchmark_metadata(final_buckets, cfg)  # Pass cfg object

    print("\nBenchmark prompt generation and processing complete.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
