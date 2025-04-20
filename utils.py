import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
import tiktoken
import openai
import aiohttp  # For async requests
import asyncio  # For sleep
from dotenv import load_dotenv
import math
from collections import defaultdict
from statistics import mean
import random

from tqdm import tqdm
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
    repo_path,
    extensions,
    output_dir="generated_prompts",
    modified_within_months=3,
):
    """
    For every file in the repository at repo_path with one of the specified extensions,
    and optionally modified within the last `modified_within_months` months, create two files in output_dir:
      - {repo_name}_{relative_path_with_underscores}_prompt.txt containing
        a reconstruction prompt with git history.
      - {repo_name}_{relative_path_with_underscores}_expectedoutput.txt containing
        the file's final content.

    Also calculates statistics about the generated prompts and files.

    Args:
        repo_path: Path to the cloned repository.
        extensions: List of file extensions to process.
        output_dir: Directory to save generated files.
        modified_within_months: Integer, only process files modified in the last N months.
                                If <= 0, this filter is disabled.

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
            'final_lines': int
        }
        - An integer representing the count of files skipped due to the date filter.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    repo_name = os.path.basename(os.path.normpath(repo_path))
    org_name = os.path.basename(os.path.dirname(repo_path))
    full_repo_name = f"{org_name}/{repo_name}"
    stats_list = []
    files_to_process = []
    date_filtered_count = 0  # Counter for files skipped by date filter

    # Calculate date threshold if filter is enabled
    threshold_timestamp = None
    if modified_within_months > 0:
        # Approximate seconds per month (average)
        avg_seconds_per_month = 30.44 * 24 * 60 * 60
        current_timestamp = time.time()
        threshold_timestamp = current_timestamp - (
            modified_within_months * avg_seconds_per_month
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
            if any(filename.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, repo_path)
                files_to_process.append((full_path, rel_path))

    # Now, process the files with a progress bar
    for full_path, rel_path in tqdm(files_to_process, desc="Generating prompts"):
        # --- Date Filter Check ---
        if threshold_timestamp is not None:
            last_commit_timestamp_str = "" # Initialize to ensure it's bound
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

        # --- Proceed with generation if not filtered ---
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

        # 3. Read final content (Let it raise errors if file cannot be read)
        with open(full_path, "r", encoding="utf-8", errors="ignore") as original:
            final_content = original.read()

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

    # Return the list of stats and the count of files filtered by date
    return stats_list, date_filtered_count


def save_benchmark_metadata(output_dir, final_buckets, generation_params):
    """
    Saves the final benchmark structure and generation parameters to metadata.json.

    Args:
        output_dir: The directory where prompts are saved.
        final_buckets: Dictionary mapping bucket ranges to lists of kept prompt stats.
        generation_params: Dictionary of parameters used for generation/filtering.
    """
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

    metadata_path = os.path.join(output_dir, "metadata.json")
    try:
        with open(metadata_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=4)
        print(f"\nSaved final benchmark structure metadata to {metadata_path}")
    except Exception as e:
        print(f"\nWarning: Failed to save metadata to {metadata_path}: {e}")


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
    stats_list, output_dir, max_tokens=100000, bucket_size=20000, max_per_bucket=10
):
    """
    Filters stats, assigns them to buckets, samples buckets, and deletes discarded files.

    Args:
        stats_list: List of dictionaries containing statistics for each file.
        output_dir: Directory where prompt and expected files are stored.
        max_tokens: Maximum prompt tokens allowed.
        bucket_size: Size of each token bucket.
        max_per_bucket: Maximum number of items allowed per bucket after sampling.

    Returns:
        A dictionary where keys are bucket range tuples (min_token, max_token)
        and values are lists of stats dictionaries belonging to that bucket after filtering and sampling.
    """
    print(
        f"\n--- Filtering, Bucketing, and Sampling (Max Tokens: {max_tokens}, Max per Bucket: {max_per_bucket}) ---"
    )
    filtered_stats = []
    deleted_count = 0

    # 1. Filter by max_tokens and delete corresponding files
    print(f"Filtering prompts with more than {max_tokens} tokens...")
    for stats in stats_list:
        if stats["prompt_tokens"] <= max_tokens:
            filtered_stats.append(stats)
        else:
            prompt_file = os.path.join(output_dir, stats["prompt_filename"])
            expected_file = os.path.join(output_dir, stats["expected_filename"])
            try:
                os.remove(prompt_file)
                # print(f"Deleted {prompt_file}")
            except FileNotFoundError:
                print(f"Warning: Prompt file not found for deletion: {prompt_file}")
            try:
                os.remove(expected_file)
                # print(f"Deleted {expected_file}")
            except FileNotFoundError:
                print(f"Warning: Expected file not found for deletion: {expected_file}")
            deleted_count += 1
    print(f"Filtered out {deleted_count} prompts exceeding token limit.")

    # 2. Bucket the filtered stats
    print("Assigning remaining prompts to buckets...")
    num_buckets = math.ceil(max_tokens / bucket_size)
    buckets = defaultdict(list)
    bucket_ranges = {}  # Store range tuple for printing

    for i in range(num_buckets):
        min_tk = i * bucket_size + (
            1 if i > 0 else 0
        )  # Start from 1 for non-zero buckets
        max_tk = (i + 1) * bucket_size
        # Ensure the last bucket goes exactly up to max_tokens
        if max_tk > max_tokens:
            max_tk = max_tokens
        bucket_key = (min_tk, max_tk)
        buckets[bucket_key] = []  # Initialize empty list
        bucket_ranges[i] = bucket_key  # Map index to range for lookup

    for stats in filtered_stats:
        tokens = stats["prompt_tokens"]
        # Find the correct bucket index
        bucket_index = (tokens - 1) // bucket_size if tokens > 0 else 0
        # Ensure index is within bounds (handles edge case of exactly max_tokens)
        bucket_index = min(bucket_index, num_buckets - 1)

        bucket_key = bucket_ranges[bucket_index]
        buckets[bucket_key].append(stats)

    # 3. Sample buckets exceeding max_per_bucket and delete corresponding files
    print(f"Sampling buckets to have at most {max_per_bucket} prompts each...")
    final_buckets = {}
    total_sampled_out = 0
    for bucket_key, items in buckets.items():
        if len(items) > max_per_bucket:
            # Targeted sampling: Instead of pure random sampling, select items closest to
            # randomly chosen target token counts within the bucket range. This aims to
            # create a sample whose average token count is closer to the midpoint of the
            # bucket, counteracting potential skew towards lower token counts.
            min_tk, max_tk = bucket_key
            items_to_keep = []
            available_items = list(items)  # Copy to modify

            for _ in range(max_per_bucket):
                if (
                    not available_items
                ):  # Should not happen if len(items) > max_per_bucket
                    break
                target_token_count = random.uniform(min_tk, max_tk)
                # Find item in available_items closest to target_token_count
                closest_item = min(
                    available_items,
                    key=lambda item: abs(item["prompt_tokens"] - target_token_count),
                )
                items_to_keep.append(closest_item)
                available_items.remove(closest_item)  # Remove selected item

            # Items remaining in available_items are discarded
            items_to_discard = available_items
            print(
                f"  Bucket {bucket_key}: Sampled down from {len(items)} to {max_per_bucket} (targeted sampling)."
            )
            total_sampled_out += len(items_to_discard)

            for stats in items_to_discard:
                prompt_file = os.path.join(output_dir, stats["prompt_filename"])
                expected_file = os.path.join(output_dir, stats["expected_filename"])
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


# --- Model Interaction (Async) ---

# Global async client instance
_ASYNC_CLIENT = None


def _get_async_openai_client():
    """Initializes and returns the async OpenAI client for OpenRouter."""
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is None:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables. "
                "Ensure it's set in a .env file or exported."
            )
        _ASYNC_CLIENT = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _ASYNC_CLIENT


async def get_model_response_openrouter(
    prompt_content: str, model_name: str
) -> tuple[str | None, str | None, str | None]:
    """
    Sends a prompt to a specified model via OpenRouter asynchronously.

    Args:
        prompt_content: The full content of the prompt to send to the model.
        model_name: The identifier of the model on OpenRouter (e.g., 'openai/gpt-4o').

    Returns:
        A tuple containing:
        - The content of the model's response message (str) if successful, else None.
        - The generation ID (str) if available, else None.
        - An error message (str) if an API error occurred, else None.

    Raises:
        ValueError: If the OPENROUTER_API_KEY environment variable is not set (raised by _get_async_openai_client).
    """
    client = _get_async_openai_client()
    error_message = None

    try:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt_content,
                },
            ],
            # Optional: Add other parameters like temperature, max_tokens if needed
            # temperature=0.7,
            # max_tokens=2000,
        )

        response_content = ""
        generation_id = None

        # Check for API-level errors returned in the response body (e.g., credit limits)
        # Use getattr for safer access to potentially dynamic attributes
        error_payload = getattr(completion, "error", None)
        if error_payload:
            # Try to serialize the full error payload for more detail
            try:
                if isinstance(error_payload, dict):
                    error_details = json.dumps(error_payload)
                else:
                    error_details = str(error_payload)
                error_message = f"Provider error in response body: {error_details}"
            except Exception as serialize_err:
                # Fallback if serialization fails
                error_message = f"Provider error in response body (serialization failed: {serialize_err}): {str(error_payload)}"

            print(f"OpenRouter API reported an error: {error_message}")
            return None, None, error_message

        # Extract content if successful and no error in body
        if completion.choices and completion.choices[0].message:
            response_content = completion.choices[0].message.content or ""

        # Extract generation ID
        if hasattr(completion, "id") and isinstance(completion.id, str):
            generation_id = completion.id
        else:
            # Log the full response if ID extraction fails, might reveal structure changes
            print(
                f"Warning: Could not extract generation ID from OpenRouter response object: {completion}"
            )

        return response_content, generation_id, None  # Success

    except openai.APIError as e:
        # This catches errors where the API call itself failed (e.g., 4xx/5xx status codes)
        # Use getattr for status_code and body as they might not be statically typed
        status_code = getattr(e, "status_code", "Unknown")
        base_error_message = f"OpenRouter API Error: Status {status_code} - {e.message}"
        detailed_error_message = base_error_message  # Start with base message

        # Attempt to extract more detail from the body
        body = getattr(e, "body", None)
        if body:
            try:
                if isinstance(body, dict):
                    # Try to get nested message first
                    nested_message = body.get("error", {}).get("message")
                    if nested_message and nested_message != e.message:
                        detailed_error_message = (
                            f"{base_error_message} | Detail: {nested_message}"
                        )
                    # Include full body if it might be useful and isn't just repeating the message
                    body_str = json.dumps(body)
                    if body_str not in detailed_error_message:  # Avoid redundancy
                        detailed_error_message += f" | Body: {body_str}"
                else:
                    # If body is not a dict, include its string representation if informative
                    body_str = str(body)
                    if body_str and body_str not in detailed_error_message:
                        detailed_error_message += f" | Body: {body_str}"
            except Exception as serialize_err:
                detailed_error_message += (
                    f" (Failed to serialize body: {serialize_err})"
                )

        print(detailed_error_message)  # Print the most detailed message obtained
        return None, None, detailed_error_message  # Return the detailed message
    except Exception as e:
        error_message = f"Unexpected Error during API call: {type(e).__name__}: {e}"
        print(error_message)
        # Log traceback for unexpected errors
        # import traceback
        # traceback.print_exc()
        return None, None, error_message


async def get_generation_stats_openrouter(generation_id: str) -> dict | None:
    """
    Queries the OpenRouter Generation Stats API asynchronously for cost and token information.

    Args:
        generation_id: The ID of the generation to query (e.g., "gen-12345").

    Returns:
        A dictionary containing statistics like cost and token counts, or None if
        the query fails or the API key is missing.
        Example return format:
        {
            'cost_usd': float,
            'prompt_tokens': int,
            'completion_tokens': int,
            'total_tokens': int,
                    'native_prompt_tokens': int | None,
                    'native_completion_tokens': int | None,
                    'native_finish_reason': str | None
                }

    Raises:
        ValueError: If the OPENROUTER_API_KEY environment variable is not set.
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment variables for stats query."
        )

    stats_url = f"https://openrouter.ai/api/v1/generation?id={generation_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    max_retries = 3
    retry_delay_seconds = 1

    for attempt in range(max_retries):
        try:
            # Use aiohttp for the async request
            async with aiohttp.ClientSession() as session:
                async with session.get(stats_url, headers=headers) as response:
                    # Check for 404 specifically for retry
                    if response.status == 404:
                        print(
                            f"Attempt {attempt + 1}/{max_retries}: Stats not found (404) for {generation_id}. Retrying in {retry_delay_seconds}s..."
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay_seconds)
                            continue  # Go to next retry iteration
                        else:
                            print(
                                f"Max retries reached for {generation_id}. Giving up."
                            )
                            return None  # Failed after retries

                    # Raise HTTPError for other bad responses (4xx or 5xx, excluding 404 handled above)
                    response.raise_for_status()
                    response_data = await response.json()

            # If successful (status 200 and no exception raised), process data
            if "data" in response_data and response_data["data"] is not None:
                stats_data = response_data["data"]
                # Extract relevant fields, providing defaults or None if missing
                cost_usd = stats_data.get("total_cost", 0.0)
                prompt_tokens = stats_data.get("tokens_prompt", 0)
                completion_tokens = stats_data.get("tokens_completion", 0)
                native_prompt_tokens = stats_data.get(
                    "native_tokens_prompt"
                )  # Can be None
                native_completion_tokens = stats_data.get(
                    "native_tokens_completion"
                )  # Can be None
                native_finish_reason = stats_data.get(
                    "native_finish_reason"
                )  # Can be None

                return {
                    "cost_usd": float(cost_usd) if cost_usd is not None else 0.0,
                    "prompt_tokens": int(prompt_tokens)
                    if prompt_tokens is not None
                    else 0,
                    "completion_tokens": int(completion_tokens)
                    if completion_tokens is not None
                    else 0,
                    "total_tokens": (
                        int(prompt_tokens or 0) + int(completion_tokens or 0)
                    ),
                    "native_prompt_tokens": int(native_prompt_tokens)
                    if native_prompt_tokens is not None
                    else None,
                    "native_completion_tokens": int(native_completion_tokens)
                    if native_completion_tokens is not None
                    else None,
                    "native_finish_reason": str(native_finish_reason)
                    if native_finish_reason is not None
                    else None,
                }
            else:
                print(
                    f"Warning: 'data' field missing or null in OpenRouter stats response for ID {generation_id}."
                )
                print(f"Full response: {response_data}")
                return None  # Indicate stats could not be retrieved (even on success status)

        except aiohttp.ClientResponseError as e:
            # Catch non-404 HTTP errors after raise_for_status
            print(
                f"HTTP error querying OpenRouter generation stats API (Attempt {attempt + 1}/{max_retries}): {e.status} {e.message}"
            )
            # Don't retry on non-404 errors
            return None
        except aiohttp.ClientError as e:
            # Catch other aiohttp client errors (e.g., connection issues)
            print(
                f"Client error querying OpenRouter generation stats API (Attempt {attempt + 1}/{max_retries}): {e}"
            )
            # Don't retry on general client errors
            return None
        except json.JSONDecodeError as e:
            print(
                f"Error decoding JSON response from OpenRouter stats API (Attempt {attempt + 1}/{max_retries}): {e}"
            )
            # Don't retry on JSON errors
            return None
        except Exception as e:
            print(
                f"An unexpected error occurred during the async stats API call (Attempt {attempt + 1}/{max_retries}): {e}"
            )
            # Log traceback for unexpected errors
            # import traceback
            # traceback.print_exc()
            # Don't retry on unexpected errors
            return None

    # Should be unreachable if logic is correct, but as a fallback
    return None
