#!/usr/bin/env python3
"""
Temporary script to backfill native_tokens_reasoning metric in existing benchmark metadata files.

This script:
1. Finds all metadata.json files in the benchmark results directory
2. Extracts the generation IDs from those files
3. Queries the OpenRouter API to get native_tokens_reasoning for each generation
4. Updates the metadata files with the new information

Usage:
  python benchmark_pipeline/backfill_reasoning_tokens.py --benchmark-run-dir BENCHMARK_DIR

Arguments:
  --benchmark-run-dir: Path to the directory containing benchmark run data
                      (subdirectories: 'results/').
  --dry-run: Optional flag to simulate updates without modifying files

Requirements:
  - OPENROUTER_API_KEY environment variable must be set
"""

import argparse
import json
import os
import requests
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv


def get_generation_stats_openrouter(generation_id: str) -> Optional[Dict[str, Any]]:
    """
    Queries the OpenRouter Generation Stats API for a specific generation.
    This is a synchronous version of the function from 2_run_benchmark.py.

    Args:
        generation_id: The ID of the generation to query.

    Returns:
        A dictionary containing the native_tokens_reasoning value if available,
        or None if the query fails or the data is not available.
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment variables. "
            "Ensure it's set in a .env file or exported."
        )

    stats_url = f"https://openrouter.ai/api/v1/generation?id={generation_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    max_retries = 3
    retry_delay_seconds = 1

    for attempt in range(max_retries):
        try:
            response = requests.get(stats_url, headers=headers)

            # Handle 404s with retry logic
            if response.status_code == 404:
                print(
                    f"Attempt {attempt + 1}/{max_retries}: Stats not found (404) for {generation_id}. Retrying in {retry_delay_seconds}s..."
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_seconds)
                    continue  # Go to next retry iteration
                else:
                    print(f"Max retries reached for {generation_id}. Giving up.")
                    return None  # Failed after retries

            # Raise exception for other errors
            response.raise_for_status()
            response_data = response.json()

            # If successful, extract the needed data
            if "data" in response_data and response_data["data"] is not None:
                stats_data = response_data["data"]
                native_tokens_reasoning = stats_data.get("native_tokens_reasoning")

                if native_tokens_reasoning is not None:
                    return {"native_tokens_reasoning": int(native_tokens_reasoning)}
                else:
                    print(
                        f"Warning: native_tokens_reasoning not found for generation {generation_id}"
                    )
                    return {"native_tokens_reasoning": None}
            else:
                print(
                    f"Warning: 'data' field missing or null in OpenRouter stats response for ID {generation_id}"
                )
                print(f"Full response: {response_data}")
                return None

        except requests.exceptions.HTTPError as e:
            print(
                f"HTTP error querying OpenRouter generation stats API (Attempt {attempt + 1}/{max_retries}): {e}"
            )
            return None
        except requests.exceptions.RequestException as e:
            print(
                f"Request error querying OpenRouter generation stats API (Attempt {attempt + 1}/{max_retries}): {e}"
            )
            return None
        except json.JSONDecodeError as e:
            print(
                f"Error decoding JSON response from OpenRouter stats API (Attempt {attempt + 1}/{max_retries}): {e}"
            )
            return None
        except Exception as e:
            print(
                f"Unexpected error during API call (Attempt {attempt + 1}/{max_retries}): {e}"
            )
            return None

    return None


def find_metadata_files(results_dir: Path) -> List[Path]:
    """
    Recursively finds all metadata.json files in the results directory.

    Args:
        results_dir: Path to the results directory

    Returns:
        List of paths to metadata.json files
    """
    metadata_files = []

    # Verify the directory exists
    if not results_dir.exists() or not results_dir.is_dir():
        print(f"Error: Results directory not found: {results_dir}")
        return []

    # Walk through the directory structure to find all metadata.json files
    for root, dirs, files in os.walk(results_dir):
        if "metadata.json" in files:
            metadata_files.append(Path(root) / "metadata.json")

    return metadata_files


def process_metadata_file(
    metadata_file: Path, dry_run: bool = False
) -> Tuple[bool, bool, Optional[str]]:
    """
    Processes a single metadata.json file to update it with native_tokens_reasoning.

    Args:
        metadata_file: Path to the metadata.json file
        dry_run: If True, simulate the update without modifying the file

    Returns:
        Tuple of (file_read_success, update_made, generation_id)
    """
    try:
        # Read the metadata file
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Check if it already has native_tokens_reasoning
        if "native_tokens_reasoning" in metadata:
            print(f"Skipping {metadata_file} - already has native_tokens_reasoning")
            return True, False, None

        # Check if it has a generation_id
        generation_id = metadata.get("generation_id")
        if not generation_id:
            print(f"Skipping {metadata_file} - no generation_id found")
            return True, False, None

        # Query the OpenRouter API for the generation stats
        stats = get_generation_stats_openrouter(generation_id)
        if not stats:
            print(f"Could not get stats for {generation_id} in {metadata_file}")
            return True, False, generation_id

        # Update the metadata with the new information
        metadata["native_tokens_reasoning"] = stats.get("native_tokens_reasoning")

        if dry_run:
            print(
                f"DRY RUN - Would update {metadata_file} with native_tokens_reasoning: {stats.get('native_tokens_reasoning')}"
            )
            return True, True, generation_id

        # Write the updated metadata back to the file
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"Updated {metadata_file} with native_tokens_reasoning: {stats.get('native_tokens_reasoning')}"
        )
        return True, True, generation_id

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error processing {metadata_file}: {e}")
        return False, False, None
    except Exception as e:
        print(f"Unexpected error processing {metadata_file}: {e}")
        return False, False, None


def main():
    parser = argparse.ArgumentParser(
        description="Backfill native_tokens_reasoning metric in existing benchmark metadata files."
    )
    parser.add_argument(
        "--benchmark-run-dir",
        type=Path,
        required=True,
        help="Path to the directory containing benchmark run data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making actual changes to files.",
    )
    args = parser.parse_args()

    benchmark_run_dir = args.benchmark_run_dir
    results_dir = benchmark_run_dir / "results"

    print(f"Searching for metadata files in {results_dir}...")
    metadata_files = find_metadata_files(results_dir)
    print(f"Found {len(metadata_files)} metadata files.")

    # Process statistics
    stats = {
        "total_files": len(metadata_files),
        "processed_files": 0,
        "updated_files": 0,
        "error_files": 0,
        "skipped_files": 0,
        "api_calls": 0,
        "generation_ids": set(),
    }

    # Process each metadata file
    for i, metadata_file in enumerate(metadata_files):
        print(f"Processing file {i + 1}/{len(metadata_files)}: {metadata_file}")
        success, updated, generation_id = process_metadata_file(
            metadata_file, args.dry_run
        )

        stats["processed_files"] += 1
        if not success:
            stats["error_files"] += 1
        elif updated:
            stats["updated_files"] += 1
            stats["api_calls"] += 1
            if generation_id:
                stats["generation_ids"].add(generation_id)
        else:
            stats["skipped_files"] += 1

        # Add a small delay between API calls to avoid rate limiting
        if i < len(metadata_files) - 1:
            time.sleep(0.1)

    # Print summary
    print("\nProcess complete!")
    print(f"Total files: {stats['total_files']}")
    print(f"Processed files: {stats['processed_files']}")
    print(f"Updated files: {stats['updated_files']}")
    print(
        f"Skipped files: {stats['skipped_files']} (already had the field or no generation ID)"
    )
    print(f"Error files: {stats['error_files']}")
    print(f"API calls made: {stats['api_calls']}")
    print(f"Unique generation IDs processed: {len(stats['generation_ids'])}")

    if args.dry_run:
        print("\nThis was a DRY RUN. No files were actually modified.")

    return 0


if __name__ == "__main__":
    main()
