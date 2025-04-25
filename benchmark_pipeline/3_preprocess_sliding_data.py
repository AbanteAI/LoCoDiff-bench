#!/usr/bin/env python3
"""
Preprocesses sliding window data for the benchmark explorer.

Purpose:
  This script analyzes benchmark results and generates a static JSON file
  containing sliding window data for visualization in the explorer web app.
  It extracts the same data that would be generated dynamically by the
  /api/sliding-plot-data endpoint in the app, but saves it to a static file.
  This allows the web app to be served as a static site without requiring
  runtime analysis of all the benchmark results.

  The script should be run after completing benchmark runs and before
  serving the explorer web app.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Add parent directory to path to import from results_explorer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from results_explorer for consistency
from results_explorer.app import (
    load_benchmark_metadata,
    scan_results_directory,
    analyze_results,
    PROMPTS_SUBDIR,
    RESULTS_SUBDIR,
)


def generate_sliding_plot_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats the sliding window analysis results into the structure expected by the chart.
    This function is adapted from the get_sliding_plot_data() endpoint in app.py.

    Args:
        analysis_results: The results from analyze_results()

    Returns:
        The formatted data structure for the sliding window chart
    """
    if (
        not analysis_results
        or not analysis_results.get("sliding_window")
        or not analysis_results["sliding_window"].get("models")
    ):
        return {"error": "No sliding window analysis results available"}

    sliding_data = analysis_results["sliding_window"]
    # Labels are the window centers in k tokens
    labels = [f"{k}k" for k in sliding_data.get("window_centers_k", [])]
    datasets = []

    # Sort models for consistent coloring
    models = sorted(list(sliding_data["models"].keys()))
    window_centers = [
        c * 1000 for c in sliding_data.get("window_centers_k", [])
    ]  # Get original centers

    for model_name in models:
        model_sliding_stats = sliding_data["models"][model_name]
        success_rates = []
        wilson_lower_bounds = []  # Initialize list for lower bounds
        wilson_upper_bounds = []  # Initialize list for upper bounds
        totals_in_window = []  # Initialize list for total counts
        successes_in_window = []  # Initialize list for success counts

        for center in window_centers:
            stats = model_sliding_stats.get(center, {})
            success_rates.append(stats.get("rate"))  # Append rate (can be None)
            wilson_lower_bounds.append(stats.get("wilson_lower"))  # Append lower bound
            wilson_upper_bounds.append(stats.get("wilson_upper"))  # Append upper bound
            totals_in_window.append(stats.get("total"))  # Append total count
            successes_in_window.append(stats.get("successful"))  # Append success count

        # Get overall cost for the label from the main analysis part
        total_cost = (
            analysis_results.get("models", {})
            .get(model_name, {})
            .get("total_cost_usd", 0.0)
        )

        datasets.append(
            {
                "label": f"{model_name} (${total_cost:.2f})",
                "data": success_rates,
                "wilson_lower": wilson_lower_bounds,  # Add lower bounds to dataset
                "wilson_upper": wilson_upper_bounds,  # Add upper bounds to dataset
                "totals": totals_in_window,  # Add total counts to dataset
                "successes": successes_in_window,  # Add success counts to dataset
                "borderWidth": 2,
                "tension": 0.1,
                "fill": False,
                "spanGaps": True,  # Connect lines even if there are null data points
            }
        )

    return {"labels": labels, "datasets": datasets}


def main():
    """
    Main function to preprocess sliding window data.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess sliding window data for the benchmark explorer."
    )
    parser.add_argument(
        "--benchmark-run-dir",
        required=True,
        help="Path to the directory containing the benchmark run data (subdirectories: 'prompts/', 'results/').",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path where the output JSON file will be saved. Defaults to 'results_explorer/static/sliding-plot-data.json'.",
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        static_dir = os.path.join(script_dir, "results_explorer", "static")
        os.makedirs(static_dir, exist_ok=True)
        args.output = os.path.join(static_dir, "sliding-plot-data.json")
    else:
        # Ensure the directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    benchmark_run_dir = args.benchmark_run_dir
    prompts_dir = os.path.join(benchmark_run_dir, PROMPTS_SUBDIR)
    results_dir = os.path.join(benchmark_run_dir, RESULTS_SUBDIR)

    print(f"Using benchmark run directory: {benchmark_run_dir}")
    print(f"Using prompts directory: {prompts_dir}")
    print(f"Using results directory: {results_dir}")

    # Load benchmark metadata
    print("Loading benchmark metadata...")
    benchmark_metadata = load_benchmark_metadata(prompts_dir)
    if not benchmark_metadata:
        print(
            f"Error: Benchmark metadata not found in {prompts_dir}. Cannot perform analysis."
        )
        sys.exit(1)

    # Scan results directory
    print("Scanning results directory...")
    models_found, results_data = scan_results_directory(results_dir)
    if not models_found:
        print("Warning: No models found in results directory. Analysis will be empty.")
        sliding_data = {"labels": [], "datasets": []}
    else:
        # Analyze results
        print("Analyzing results...")
        analysis_results = analyze_results(
            benchmark_metadata, models_found, results_data
        )

        # Generate sliding plot data
        print("Generating sliding plot data...")
        sliding_data = generate_sliding_plot_data(analysis_results)

    # Save to file
    print(f"Saving sliding plot data to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sliding_data, f, indent=2)

    print("Done! The sliding plot data has been preprocessed successfully.")


if __name__ == "__main__":
    main()
