#!/usr/bin/env python3
import argparse
import os
import sys
from glob import glob

# Import necessary functions from utils
from utils import (
    generate_prompts_and_expected,
    print_stats_table,
    filter_bucket_sample_stats,
    print_bucket_stats_table,
    save_benchmark_metadata,  # Import the new function
)


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
        "--max-tokens",
        type=int,
        default=100000,
        help="Maximum prompt tokens allowed (default: 100000).",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=20000,
        help="Size of token buckets for sampling (default: 20000).",
    )
    parser.add_argument(
        "--max-per-bucket",
        type=int,
        default=10,
        help="Maximum number of prompts per bucket after sampling (default: 10).",
    )
    parser.add_argument(
        "--months-ago",
        type=int,
        default=3,
        help="Only process files modified in the last N months (default: 3). Set to 0 or negative to disable.",
    )

    args = parser.parse_args()

    print("--- Starting Prompt Generation ---")
    print(f"Processing extensions: {args.extensions}")
    print(f"Looking for repositories in: {args.cache_dir}")
    print(f"Outputting prompts to: {args.output_dir}")

    repo_paths = find_repo_dirs(args.cache_dir)

    if not repo_paths:
        print(f"Error: No valid repository directories found in {args.cache_dir}")
        print("Please run the clone_repos.py script first.")
        return 1

    print(f"Found {len(repo_paths)} repositories to process.")

    all_stats = []
    generation_errors = 0
    total_date_filtered_count = 0  # Initialize counter for date filtering

    for repo_path in repo_paths:
        repo_name = os.path.basename(os.path.normpath(repo_path))
        org_name = os.path.basename(os.path.dirname(repo_path))
        print(f"\nProcessing repository: {org_name}/{repo_name} ({repo_path})")
        try:
            # Pass months_ago argument and receive the date filtered count
            stats_list, date_filtered_count = generate_prompts_and_expected(
                repo_path, args.extensions, args.output_dir, args.months_ago
            )
            all_stats.extend(stats_list)
            total_date_filtered_count += date_filtered_count  # Accumulate count
            print(
                f"Generated {len(stats_list)} prompts for {org_name}/{repo_name} (skipped {date_filtered_count} due to date filter)."
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
    if args.months_ago > 0:
        print(
            f"\nFiltered out a total of {total_date_filtered_count} files across all repositories due to modification date constraint (older than {args.months_ago} months)."
        )

    # Print initial statistics table for all generated prompts (after date filtering)
    print("\n--- Initial Generation Statistics (All Repos, Post-Date-Filter) ---")
    print_stats_table(all_stats)

    # Filter, bucket, and sample the results using provided arguments
    final_buckets = filter_bucket_sample_stats(
        all_stats,
        args.output_dir,
        max_tokens=args.max_tokens,
        bucket_size=args.bucket_size,
        max_per_bucket=args.max_per_bucket,
    )

    # Print statistics for the final buckets
    print_bucket_stats_table(final_buckets)

    # Save the final benchmark structure metadata
    generation_params = {
        "extensions": args.extensions,
        "cache_dir": args.cache_dir,
        "output_dir": args.output_dir,
        "max_tokens": args.max_tokens,
        "bucket_size": args.bucket_size,
        "max_per_bucket": args.max_per_bucket,
    }
    save_benchmark_metadata(args.output_dir, final_buckets, generation_params)

    print("\nBenchmark prompt generation and processing complete.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
