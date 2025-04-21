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
    print(f"Processing extensions: {args.extensions}")
    print(f"Looking for repositories in: {args.cache_dir}")
    print(f"Outputting prompts to: {args.output_dir}")

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

    repo_paths = find_repo_dirs(args.cache_dir)

    if not repo_paths:
        print(f"Error: No valid repository directories found in {args.cache_dir}")
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
            # Pass modified_within_months and max_expected_tokens arguments
            # Receive both date and expected token filtered counts
            (
                stats_list,
                date_filtered_count,
                expected_token_filtered_count,
            ) = generate_prompts_and_expected(
                repo_path,
                args.extensions,
                args.output_dir,
                args.modified_within_months,
                args.max_expected_tokens,  # Pass the new argument
            )
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
    if args.modified_within_months > 0:
        print(
            f"\nFiltered out a total of {total_date_filtered_count} files across all repositories due to modification date constraint (older than {args.modified_within_months} months)."
        )
    # Report total files filtered by expected token count
    if args.max_expected_tokens > 0:
        print(
            f"\nFiltered out a total of {total_expected_token_filtered_count} files across all repositories due to expected output token constraint (more than {args.max_expected_tokens} tokens)."
        )

    # Print initial statistics table for all generated prompts (after all filtering)
    print("\n--- Initial Generation Statistics (All Repos, Post-Filtering) ---")
    print_stats_table(all_stats)

    # Filter, bucket, and sample the results using provided arguments
    final_buckets = filter_bucket_sample_stats(
        all_stats,
        args.output_dir,
        bucket_boundaries=bucket_boundaries,  # Pass the parsed list of token boundaries
        max_per_bucket=args.max_per_bucket,
    )

    # Print statistics for the final buckets
    print_bucket_stats_table(final_buckets)

    # Save the final benchmark structure metadata
    generation_params = {
        "extensions": args.extensions,
        "cache_dir": args.cache_dir,
        "output_dir": args.output_dir,
        "buckets": args.buckets,  # Store the original string argument
        "max_per_bucket": args.max_per_bucket,
        "modified_within_months": args.modified_within_months,
        "max_expected_tokens": args.max_expected_tokens,  # Store the new parameter
    }
    save_benchmark_metadata(args.output_dir, final_buckets, generation_params)

    print("\nBenchmark prompt generation and processing complete.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
