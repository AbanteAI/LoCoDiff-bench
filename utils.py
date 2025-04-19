import json
import os
import re
import subprocess
import tiktoken
import openai
import aiohttp  # For async requests
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
    org_name = os.path.basename(os.path.dirname(repo_path))
    full_repo_name = f"{org_name}/{repo_name}"
    stats_list = []
    files_to_process = []

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
        # Include generated filenames in the stats for metadata
        file_stats["prompt_filename"] = prompt_fname
        file_stats["expected_filename"] = expected_fname
        stats_list.append(file_stats)

    # After processing all files, save the metadata
    # Create a structured metadata with clear sections
    metadata = {
        # Repository information section
        "repository": {
            "name": full_repo_name,
            "head_commit_hash": head_commit_hash,
        },
        # Files section to store all file entries
        "files": {},
    }

    # Add file statistics to the files section
    for stats in stats_list:
        metadata["files"][stats["filename"]] = {
            "prompt_filename": stats["prompt_filename"],
            "expected_filename": stats["expected_filename"],
            "stats": {
                "prompt_tokens": stats["prompt_tokens"],
                "expected_tokens": stats["expected_tokens"],
                "num_commits": stats["num_commits"],
                "lines_added": stats["lines_added"],
                "lines_deleted": stats["lines_deleted"],
                "final_lines": stats["final_lines"],
            },
        }

    metadata_path = os.path.join(output_dir, "metadata.json")
    try:
        with open(metadata_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=4)
        print(f"\nSaved statistics metadata to {metadata_path}")
    except Exception as e:
        print(f"\nWarning: Failed to save metadata to {metadata_path}: {e}")

    return stats_list  # Still return the list for table printing


# --- Statistics and Filtering Functions (Moved from create_benchmark.py) ---


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
    return final_buckets


def print_bucket_stats_table(buckets):
    """Prints a formatted table of the bucket statistics."""
    print("\n--- Bucket Statistics (Averages) ---")
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
    sorted_bucket_keys = sorted(buckets.keys(), key=lambda x: x[0])

    # Rows
    for bucket_key in sorted_bucket_keys:
        items = buckets[bucket_key]
        count = len(items)
        range_str = f"{bucket_key[0]} - {bucket_key[1]} tokens"

        if count > 0:
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
        else:
            # Display empty buckets clearly
            row = (
                f"{range_str:<{col_widths['bucket_range']}} | "
                f"{count:>{col_widths['count']}} | "
                f"{'-':>{col_widths['avg_prompt_tokens']}} | "
                f"{'-':>{col_widths['avg_expected_tokens']}} | "
                f"{'-':>{col_widths['avg_num_commits']}} | "
                f"{'-':>{col_widths['avg_lines_added']}} | "
                f"{'-':>{col_widths['avg_lines_deleted']}} | "
                f"{'-':>{col_widths['avg_final_lines']}}"
            )
        print(row)


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
) -> tuple[str, str | None]:
    """
    Sends a prompt to a specified model via OpenRouter asynchronously and returns
    the response content and the generation ID.

    Args:
        prompt_content: The full content of the prompt to send to the model.
        model_name: The identifier of the model on OpenRouter (e.g., 'openai/gpt-4o').

    Returns:
        A tuple containing:
        - The content of the model's response message (str).
        - The generation ID (str) if available, otherwise None.

    Raises:
        ValueError: If the OPENROUTER_API_KEY environment variable is not set.
        openai.APIError: If there's an issue communicating with the OpenRouter API.
    """
    client = _get_async_openai_client()

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

        # Extract content
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

        return response_content, generation_id

    except openai.APIError as e:
        print(f"OpenRouter API error during async chat completion: {e}")
        raise
    except Exception as e:
        print(
            f"An unexpected error occurred during the async chat completion API call: {e}"
        )
        raise


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
            'native_completion_tokens': int | None
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

    try:
        # Use aiohttp for the async request
        async with aiohttp.ClientSession() as session:
            async with session.get(stats_url, headers=headers) as response:
                # Raise HTTPError for bad responses (4xx or 5xx)
                response.raise_for_status()
                response_data = await response.json()

        # Check if 'data' field exists and is not None
        if "data" in response_data and response_data["data"] is not None:
            stats_data = response_data["data"]
            # Extract relevant fields, providing defaults or None if missing
            cost_usd = stats_data.get("total_cost", 0.0)
            prompt_tokens = stats_data.get("tokens_prompt", 0)
            completion_tokens = stats_data.get("tokens_completion", 0)
            native_prompt_tokens = stats_data.get("native_tokens_prompt")  # Can be None
            native_completion_tokens = stats_data.get(
                "native_tokens_completion"
            )  # Can be None

            return {
                "cost_usd": float(cost_usd) if cost_usd is not None else 0.0,
                "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else 0,
                "completion_tokens": int(completion_tokens)
                if completion_tokens is not None
                else 0,
                "total_tokens": (int(prompt_tokens or 0) + int(completion_tokens or 0)),
                "native_prompt_tokens": int(native_prompt_tokens)
                if native_prompt_tokens is not None
                else None,
                "native_completion_tokens": int(native_completion_tokens)
                if native_completion_tokens is not None
                else None,
            }
        else:
            print(
                f"Warning: 'data' field missing or null in OpenRouter stats response for ID {generation_id}."
            )
            print(f"Full response: {response_data}")
            return None  # Indicate stats could not be retrieved

    except aiohttp.ClientResponseError as e:
        # Catch specific aiohttp HTTP errors
        print(
            f"HTTP error querying OpenRouter generation stats API: {e.status} {e.message}"
        )
        return None  # Indicate stats could not be retrieved
    except aiohttp.ClientError as e:
        # Catch other aiohttp client errors (e.g., connection issues)
        print(f"Client error querying OpenRouter generation stats API: {e}")
        return None  # Indicate stats could not be retrieved
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from OpenRouter stats API: {e}")
        # Consider logging response.text() if needed, but handle potential exceptions
        return None  # Indicate stats could not be retrieved
    except Exception as e:
        print(f"An unexpected error occurred during the async stats API call: {e}")
        # Log traceback for unexpected errors
        # import traceback
        # traceback.print_exc()
        return None  # Indicate stats could not be retrieved
