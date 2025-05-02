#!/usr/bin/env python3
"""
Analyzes LoCoDiff benchmark results to report extraction failures by model.

Purpose:
  This script analyzes the results of benchmark runs to identify and report
  extraction failures per model. Extraction failures occur when the script
  cannot extract code from the model's response (e.g., missing triple backticks).

  For each model found in the benchmark results, it reports:
  - The number of benchmark cases that failed due to extraction issues
  - How many extraction failures were due to empty model outputs
  - The total number of benchmark cases run for that model
  - The percentage of cases with extraction failures
  - A detailed list of all cases where extraction failed but output wasn't empty

Arguments:
  --benchmark-run-dir (required): Path to the directory containing the benchmark run data
                                  (must contain a 'results/' subdirectory).

Inputs:
  - Result directories and metadata.json files created by the '2_run_benchmark.py' script
    within the specified benchmark run directory.

Outputs:
  - Prints a summary table showing extraction failures by model.
  - Distinguishes between empty output failures and other extraction failures.
  - Lists models in descending order of extraction failure rate.
  - Includes counts and percentages for each model.
  - Lists all model/case pairs where extraction failed but output wasn't empty.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results to report extraction failures by model."
    )
    parser.add_argument(
        "--benchmark-run-dir",
        required=True,
        help="Path to the directory containing the benchmark run data",
    )
    return parser.parse_args()


def find_metadata_files(results_dir: Path) -> List[Path]:
    """
    Find all metadata.json files in the results directory.

    Args:
        results_dir: Path to the results directory.

    Returns:
        List of paths to metadata.json files.
    """
    metadata_files = []
    # Walk through the directory tree to find all metadata.json files
    for root, _, files in os.walk(results_dir):
        if "metadata.json" in files:
            metadata_files.append(Path(root) / "metadata.json")
    return metadata_files


def analyze_extraction_failures(
    metadata_files: List[Path],
) -> Tuple[Dict[str, Dict], Dict[str, List]]:
    """
    Analyze metadata files to count extraction failures by model,
    distinguishing between empty outputs and other extraction failures.
    Also collects detailed information about each failure.

    Args:
        metadata_files: List of paths to metadata.json files.

    Returns:
        A tuple containing:
        - Dictionary with model name as key and statistics as value
        - Dictionary with model name as key and a list of failure details
    """
    # Initialize dictionaries to store counts and details for each model
    model_stats = defaultdict(
        lambda: {"total": 0, "extraction_failures": 0, "empty_output_failures": 0}
    )

    # Store details about each failure for reporting
    failure_details = defaultdict(list)

    for metadata_path in metadata_files:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            model = metadata.get("model")
            if not model:
                continue

            benchmark_case = metadata.get("benchmark_case", "Unknown")

            # Increment total count for this model
            model_stats[model]["total"] += 1

            # Check if extraction failed
            error = metadata.get("error")
            if error and "backticks not found" in error:
                model_stats[model]["extraction_failures"] += 1

                # Assume it's not empty by default
                is_empty = False

                # Check if it was due to empty output
                raw_response_length = metadata.get("raw_response_length", 0)
                if raw_response_length == 0 or (
                    raw_response_length < 10
                    and "raw_response.txt" in str(metadata_path)
                ):
                    is_empty = True
                    model_stats[model]["empty_output_failures"] += 1

                    # Check the actual raw response file for more accurate detection
                    raw_response_path = metadata_path.parent / "raw_response.txt"
                    if raw_response_path.exists():
                        try:
                            content = raw_response_path.read_text(
                                encoding="utf-8"
                            ).strip()
                            is_empty = not content
                        except Exception:
                            # If we can't read the file, rely on the metadata
                            pass

                # Store details about this failure
                failure_details[model].append(
                    {
                        "benchmark_case": benchmark_case,
                        "is_empty": is_empty,
                        "results_dir": str(metadata_path.parent),
                    }
                )

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading metadata file {metadata_path}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {metadata_path}: {e}")

    return model_stats, failure_details


def calculate_percentages(model_stats: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
    """
    Calculate extraction failure percentages and sort models by failure rate.

    Args:
        model_stats: Dictionary with model name as key and statistics as value.

    Returns:
        List of (model, stats) tuples sorted by extraction failure percentage.
    """
    result = []

    for model, stats in model_stats.items():
        total = stats["total"]
        failures = stats["extraction_failures"]
        empty_failures = stats["empty_output_failures"]

        if total > 0:
            percentage = (failures / total) * 100
            empty_percentage = (
                (empty_failures / total) * 100 if empty_failures > 0 else 0
            )
        else:
            percentage = 0
            empty_percentage = 0

        result.append(
            (
                model,
                {
                    "total": total,
                    "extraction_failures": failures,
                    "empty_output_failures": empty_failures,
                    "percentage": percentage,
                    "empty_percentage": empty_percentage,
                },
            )
        )

    # Sort by percentage in descending order
    return sorted(result, key=lambda x: x[1]["percentage"], reverse=True)


def display_results(sorted_stats: List[Tuple[str, Dict]]):
    """
    Display the analysis results in a formatted table.

    Args:
        sorted_stats: List of (model, stats) tuples sorted by extraction failure percentage.
    """
    if not sorted_stats:
        print("No benchmark results found.")
        return

    # Print header
    print("\nExtraction Failures by Model")
    print("=" * 120)
    print(
        f"{'Model':<40} {'Total Extraction':<15} {'Empty Output':<15} {'Other Extraction':<15} {'Total Cases':<12} {'% Failed':<8} {'% Empty':<8}"
    )
    print("-" * 120)

    # Print data rows
    for model, stats in sorted_stats:
        total_failures = stats["extraction_failures"]
        empty_failures = stats["empty_output_failures"]
        other_failures = total_failures - empty_failures

        print(
            f"{model:<40} "
            f"{total_failures:<15} "
            f"{empty_failures:<15} "
            f"{other_failures:<15} "
            f"{stats['total']:<12} "
            f"{stats['percentage']:.2f}%{' ':<2} "
            f"{stats['empty_percentage']:.2f}%"
        )

    print("=" * 120)


def display_non_empty_failures(failure_details: Dict[str, List[Dict[str, Any]]]):
    """
    Display a list of all model/case pairs where extraction failed but output wasn't empty.

    Args:
        failure_details: Dictionary with model name as key and list of failure details as value.
    """
    # Count non-empty failures
    non_empty_count = 0
    for model_failures in failure_details.values():
        for failure in model_failures:
            if not failure["is_empty"]:
                non_empty_count += 1

    if non_empty_count == 0:
        print("\nNo non-empty extraction failures found.")
        return

    print(f"\nList of Non-Empty Extraction Failures ({non_empty_count} total)")
    print("=" * 100)
    print(f"{'Model':<40} {'Benchmark Case':<40} {'Results Directory'}")
    print("-" * 100)

    # Sort by model name for consistent output
    for model in sorted(failure_details.keys()):
        # Filter to only non-empty failures
        non_empty_failures = [f for f in failure_details[model] if not f["is_empty"]]

        if non_empty_failures:
            # Sort by benchmark case for consistent output
            for failure in sorted(
                non_empty_failures, key=lambda f: f["benchmark_case"]
            ):
                print(
                    f"{model:<40} "
                    f"{failure['benchmark_case']:<40} "
                    f"{failure['results_dir']}"
                )

    print("=" * 100)


def main():
    """Main function to analyze extraction failures."""
    args = parse_args()
    benchmark_dir = Path(args.benchmark_run_dir)
    results_dir = benchmark_dir / "results"

    if not results_dir.exists() or not results_dir.is_dir():
        print(f"Error: Results directory not found at {results_dir}")
        return 1

    print(f"Analyzing benchmark results in {results_dir}...")

    # Find all metadata files
    metadata_files = find_metadata_files(results_dir)
    print(f"Found {len(metadata_files)} result files to analyze.")

    if not metadata_files:
        print("No benchmark result files found.")
        return 1

    # Analyze the metadata files
    model_stats, failure_details = analyze_extraction_failures(metadata_files)

    # Calculate percentages and sort
    sorted_stats = calculate_percentages(model_stats)

    # Display summary results
    display_results(sorted_stats)

    # Display list of non-empty extraction failures
    display_non_empty_failures(failure_details)

    return 0


if __name__ == "__main__":
    exit(main())
