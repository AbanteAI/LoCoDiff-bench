#!/usr/bin/env python3
"""
Analyzes LoCoDiff benchmark results to report extraction failures by model.

Purpose:
  This script analyzes the results of benchmark runs to identify and report
  extraction failures per model. Extraction failures occur when the script
  cannot extract code from the model's response (e.g., missing triple backticks).

  For each model found in the benchmark results, it reports:
  - The number of benchmark cases that failed due to extraction issues
  - The total number of benchmark cases run for that model
  - The percentage of cases with extraction failures

Arguments:
  --benchmark-run-dir (required): Path to the directory containing the benchmark run data
                                  (must contain a 'results/' subdirectory).

Inputs:
  - Result directories and metadata.json files created by the '2_run_benchmark.py' script
    within the specified benchmark run directory.

Outputs:
  - Prints a summary table showing extraction failures by model.
  - Lists models in descending order of extraction failure rate.
  - Includes counts and percentages for each model.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


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


def analyze_extraction_failures(metadata_files: List[Path]) -> Dict[str, Dict]:
    """
    Analyze metadata files to count extraction failures by model.

    Args:
        metadata_files: List of paths to metadata.json files.

    Returns:
        Dictionary with model name as key and statistics as value.
    """
    # Initialize a dictionary to store counts for each model
    model_stats = defaultdict(lambda: {"total": 0, "extraction_failures": 0})

    for metadata_path in metadata_files:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            model = metadata.get("model")
            if not model:
                continue

            # Increment total count for this model
            model_stats[model]["total"] += 1

            # Check if extraction failed
            error = metadata.get("error")
            if error and "backticks not found" in error:
                model_stats[model]["extraction_failures"] += 1

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading metadata file {metadata_path}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {metadata_path}: {e}")

    return model_stats


def calculate_percentages(model_stats: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
    """
    Calculate extraction failure percentages and sort models by failure rate.

    Args:
        model_stats: Dictionary with model statistics.

    Returns:
        List of (model, stats) tuples sorted by extraction failure percentage.
    """
    result = []

    for model, stats in model_stats.items():
        total = stats["total"]
        failures = stats["extraction_failures"]

        if total > 0:
            percentage = (failures / total) * 100
        else:
            percentage = 0

        result.append(
            (
                model,
                {
                    "total": total,
                    "extraction_failures": failures,
                    "percentage": percentage,
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
    print("=" * 80)
    print(
        f"{'Model':<40} {'Extraction Failures':<20} {'Total Cases':<15} {'Percentage':<10}"
    )
    print("-" * 80)

    # Print data rows
    for model, stats in sorted_stats:
        print(
            f"{model:<40} "
            f"{stats['extraction_failures']:<20} "
            f"{stats['total']:<15} "
            f"{stats['percentage']:.2f}%"
        )

    print("=" * 80)


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
    model_stats = analyze_extraction_failures(metadata_files)

    # Calculate percentages and sort
    sorted_stats = calculate_percentages(model_stats)

    # Display results
    display_results(sorted_stats)

    return 0


if __name__ == "__main__":
    exit(main())
