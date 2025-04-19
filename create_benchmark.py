#!/usr/bin/env python3
import argparse
import os
import random
import math  # For ceiling division if needed, or just general math ops
from collections import defaultdict
from statistics import mean

from utils import (
    clone_repo_to_cache,
    generate_prompts_and_expected,
    # Removed get_model_response_openrouter, sys, difflib
)


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
            avg_prompt_tokens = mean(item["prompt_tokens"] for item in items)
            avg_expected_tokens = mean(item["expected_tokens"] for item in items)
            avg_num_commits = mean(item["num_commits"] for item in items)
            avg_lines_added = mean(item["lines_added"] for item in items)
            avg_lines_deleted = mean(item["lines_deleted"] for item in items)
            avg_final_lines = mean(item["final_lines"] for item in items)

            row = (
                f"{range_str:<{col_widths['bucket_range']}} | "
                f"{count:>{col_widths['count']}} | "
                f"{avg_prompt_tokens:>{col_widths['avg_prompt_tokens']:.0f}} | "  # Use :.0f (integer format)
                f"{avg_expected_tokens:>{col_widths['avg_expected_tokens']:.0f}} | "
                f"{avg_num_commits:>{col_widths['avg_num_commits']:.0f}} | "
                f"{avg_lines_added:>{col_widths['avg_lines_added']:.0f}} | "
                f"{avg_lines_deleted:>{col_widths['avg_lines_deleted']:.0f}} | "
                f"{avg_final_lines:>{col_widths['avg_final_lines']:.0f}}"
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


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Generate benchmark prompts and expected outputs from a GitHub repository's history."
    )
    # Output directory is hardcoded
    output_dir = "generated_prompts"

    # Arguments for generating prompts (now required)
    parser.add_argument(
        "--repo",
        "-r",
        required=True,
        help="GitHub repository to clone for prompt generation (format: 'org/repo' or full URL)",
    )
    parser.add_argument(
        "--extensions",
        "-e",
        nargs="+",
        default=[".py"],  # Keep default but make it required
        required=True,
        help="File extensions to process for prompt generation (include the dot), e.g. .py .txt",
    )

    # Parse arguments
    args = parser.parse_args()

    # --- Generation Logic ---
    try:
        repo_path = clone_repo_to_cache(args.repo)
        print(f"Repository ready at: {repo_path}")

        # Generate prompts and expected outputs
        print(f"Generating prompts and expected outputs in '{output_dir}/'...")
        stats_list = generate_prompts_and_expected(
            repo_path, args.extensions, output_dir
        )

        # Print initial statistics table
        print("\n--- Initial Generation Statistics ---")
        print_stats_table(stats_list)

        # Filter, bucket, and sample the results
        # Uses defaults: max_tokens=100000, bucket_size=20000, max_per_bucket=10
        final_buckets = filter_bucket_sample_stats(stats_list, output_dir)

        # Print statistics for the final buckets
        print_bucket_stats_table(final_buckets)

        print("\nBenchmark creation and processing complete.")

    except ValueError as e:
        print(f"Error during benchmark creation or processing: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred during benchmark creation: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
