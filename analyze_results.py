#!/usr/bin/env python3
import argparse
import os
import json
import glob
import re
from collections import defaultdict
import sys

# Attempt to import pandas and matplotlib, provide guidance if missing
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as e:
    print(
        f"Error importing libraries: {e}. Please ensure pandas and matplotlib are installed."
    )
    print("You may need to run: pip install pandas matplotlib")
    sys.exit(1)


def load_benchmark_metadata(benchmark_dir: str) -> dict | None:
    """Loads the benchmark structure metadata from metadata.json."""
    metadata_path = os.path.join(benchmark_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: Benchmark metadata file not found at {metadata_path}")
        print("Please run generate_prompts.py first.")
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading or parsing benchmark metadata file {metadata_path}: {e}")
        return None


def find_models_in_results(results_base_dir: str) -> list[str]:
    """Finds all unique model names present in the results directory structure."""
    models = set()
    # Pattern: results_base_dir / * (benchmark_case) / * (model_name) / * (timestamp)
    pattern = os.path.join(results_base_dir, "*", "*")
    potential_model_dirs = glob.glob(pattern)
    for path in potential_model_dirs:
        if os.path.isdir(path):
            # The model name is the basename of this directory
            model_name = os.path.basename(path)
            # Basic check: avoid adding timestamp dirs if structure is unexpected
            if not re.match(r"\d{8}_\d{6}", model_name):
                models.add(model_name)
    return sorted(list(models))


def find_latest_result_dir(
    benchmark_case_prefix: str, model_name: str, results_base_dir: str
) -> str | None:
    """Finds the path to the latest timestamped result directory for a given case and model."""
    pattern = os.path.join(
        results_base_dir, benchmark_case_prefix, model_name, "*"
    )  # Match any timestamp dir
    potential_dirs = glob.glob(pattern)
    latest_dir = None
    latest_timestamp = ""

    for result_dir in potential_dirs:
        if not os.path.isdir(result_dir):
            continue
        dir_name = os.path.basename(result_dir)
        # Check if it looks like a timestamp directory
        if re.match(r"\d{8}_\d{6}", dir_name):
            if dir_name > latest_timestamp:
                latest_timestamp = dir_name
                latest_dir = result_dir

    return latest_dir


def load_result_metadata(result_dir: str) -> dict | None:
    """Loads the metadata.json from a specific result directory."""
    metadata_path = os.path.join(result_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        # This might happen if a run failed very early
        # print(f"Warning: Result metadata not found in {result_dir}")
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Error reading result metadata from {metadata_path}: {e}")
        return None


def analyze_results(benchmark_metadata: dict, results_base_dir: str) -> dict:
    """
    Analyzes benchmark results by comparing benchmark structure with found results.

    Args:
        benchmark_metadata: Loaded metadata from generated_prompts/metadata.json.
        results_base_dir: Path to the benchmark_results directory.

    Returns:
        A dictionary containing aggregated analysis results per model and per bucket.
        Structure:
        {
            "models": {
                "model_name": {
                    "total_benchmarks": int, # Total benchmarks defined for this model
                    "runs_found": int,
                    "successful_runs": int,
                    "total_cost_usd": float,
                    "success_rate": float, # 0.0 to 1.0
                    "buckets": {
                        "bucket_key_str": {
                            "total_in_bucket": int,
                            "runs_found": int,
                            "successful_runs": int,
                            "success_rate": float
                        }, ...
                    }
                }, ...
            },
            "bucket_keys": list[str] # Sorted list of bucket keys like "0-20000"
        }
    """
    analysis = {"models": defaultdict(lambda: defaultdict(lambda: 0.0))}
    models_found_in_results = find_models_in_results(results_base_dir)
    all_bucket_keys = sorted(benchmark_metadata["benchmark_buckets"].keys())
    analysis["bucket_keys"] = all_bucket_keys

    print(f"Found models in results directory: {models_found_in_results}")
    print(f"Analyzing {len(all_bucket_keys)} buckets defined in benchmark metadata.")

    # Initialize structure for all models found and all defined buckets
    for model_name in models_found_in_results:
        model_stats = {
            "total_benchmarks": 0,
            "runs_found": 0,
            "successful_runs": 0,
            "total_cost_usd": 0.0,
            "success_rate": 0.0,
            "buckets": {
                key: {
                    "total_in_bucket": 0,
                    "runs_found": 0,
                    "successful_runs": 0,
                    "success_rate": 0.0,
                }
                for key in all_bucket_keys
            },
        }
        analysis["models"][model_name] = model_stats

    # Iterate through the defined benchmark structure
    for bucket_key, benchmark_cases in benchmark_metadata["benchmark_buckets"].items():
        for case_info in benchmark_cases:
            benchmark_case_prefix = case_info["benchmark_case_prefix"]

            # Update total counts for each model
            for model_name in models_found_in_results:
                analysis["models"][model_name]["total_benchmarks"] += 1
                analysis["models"][model_name]["buckets"][bucket_key][
                    "total_in_bucket"
                ] += 1

                # Find the latest result for this specific case and model
                latest_result_dir = find_latest_result_dir(
                    benchmark_case_prefix, model_name, results_base_dir
                )

                if latest_result_dir:
                    result_meta = load_result_metadata(latest_result_dir)
                    if result_meta:
                        # Increment found counts
                        analysis["models"][model_name]["runs_found"] += 1
                        analysis["models"][model_name]["buckets"][bucket_key][
                            "runs_found"
                        ] += 1

                        # Check success
                        if result_meta.get("success", False):
                            analysis["models"][model_name]["successful_runs"] += 1
                            analysis["models"][model_name]["buckets"][bucket_key][
                                "successful_runs"
                            ] += 1

                        # Accumulate cost
                        cost = result_meta.get("cost_usd", 0.0) or 0.0  # Handle None
                        analysis["models"][model_name]["total_cost_usd"] += float(cost)

    # Calculate success rates
    for model_name, model_stats in analysis["models"].items():
        if model_stats["runs_found"] > 0:
            model_stats["success_rate"] = (
                model_stats["successful_runs"] / model_stats["runs_found"]
            )
        for bucket_key, bucket_stats in model_stats["buckets"].items():
            if bucket_stats["runs_found"] > 0:
                bucket_stats["success_rate"] = (
                    bucket_stats["successful_runs"] / bucket_stats["runs_found"]
                )

    return analysis


def print_summary_tables(analysis_results: dict):
    """Prints formatted summary tables to the console."""
    print("\n--- Overall Model Performance Summary ---")
    models_data = []
    for model_name, stats in analysis_results["models"].items():
        models_data.append(
            {
                "Model": model_name,
                "Defined": stats["total_benchmarks"],
                "Runs Found": stats["runs_found"],
                "Successful": stats["successful_runs"],
                "Success Rate": f"{stats['success_rate']:.1%}",
                "Total Cost (USD)": f"${stats['total_cost_usd']:.4f}",
            }
        )

    if not models_data:
        print("No model results found to summarize.")
        return

    df_overall = pd.DataFrame(models_data)
    print(df_overall.to_string(index=False))

    print("\n--- Per-Bucket Success Rate (%) ---")
    bucket_keys = analysis_results["bucket_keys"]
    bucket_data = {"Bucket": [f"{k} tk" for k in bucket_keys]}  # Shorten bucket label

    for model_name, stats in analysis_results["models"].items():
        rates = []
        for key in bucket_keys:
            bucket_stats = stats["buckets"][key]
            # Display rate if runs were found, otherwise indicate N/A or 0 runs
            if bucket_stats["runs_found"] > 0:
                rates.append(f"{bucket_stats['success_rate']:.1%}")
            elif bucket_stats["total_in_bucket"] > 0:
                rates.append("0/0")  # Indicate 0 runs found for defined benchmarks
            else:
                rates.append(
                    "-"
                )  # Bucket was empty for this model (shouldn't happen with current logic)
        bucket_data[model_name] = rates

    df_buckets = pd.DataFrame(bucket_data)
    print(df_buckets.to_string(index=False))


def generate_plot(analysis_results: dict, output_filename: str):
    """Generates and saves a plot of per-bucket success rates."""
    print(f"\nGenerating plot and saving to {output_filename}...")
    models = list(analysis_results["models"].keys())
    bucket_keys = analysis_results["bucket_keys"]
    # Use bucket midpoints for plotting x-axis? Or just indices? Indices are simpler.
    # Let's use indices for simplicity, label ticks with bucket ranges.
    x_indices = range(len(bucket_keys))

    plt.figure(figsize=(12, 7))

    for model_name in models:
        success_rates = []
        for key in bucket_keys:
            # Get rate, default to NaN if no runs found so it creates gaps in lines
            rate = analysis_results["models"][model_name]["buckets"][key][
                "success_rate"
            ]
            runs_found = analysis_results["models"][model_name]["buckets"][key][
                "runs_found"
            ]
            success_rates.append(rate if runs_found > 0 else float("nan"))

        plt.plot(x_indices, success_rates, marker="o", linestyle="-", label=model_name)

    # Formatting the plot
    plt.title("Benchmark Success Rate per Prompt Token Bucket")
    plt.xlabel("Prompt Token Bucket")
    plt.ylabel("Success Rate")
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.ylim(0, 1.05)  # Extend slightly beyond 100%
    plt.xticks(x_indices, [f"{k} tk" for k in bucket_keys], rotation=45, ha="right")
    plt.legend(title="Models", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend

    # Save the plot
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"Plot saved successfully to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # Optionally display the plot
    # plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LoCoDiff benchmark results, print summaries, and generate plots."
    )
    default_benchmark_dir = "generated_prompts"
    default_results_dir = "benchmark_results"
    default_plot_file = "benchmark_success_rate.png"

    parser.add_argument(
        "--benchmark-dir",
        default=default_benchmark_dir,
        help=f"Directory containing the benchmark metadata.json (default: '{default_benchmark_dir}').",
    )
    parser.add_argument(
        "--results-dir",
        default=default_results_dir,
        help=f"Base directory containing the benchmark run results (default: '{default_results_dir}').",
    )
    parser.add_argument(
        "--plot-file",
        default=default_plot_file,
        help=f"Filename to save the generated plot (default: '{default_plot_file}').",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="If set, skip generating and saving the plot.",
    )

    args = parser.parse_args()

    print("--- Starting Benchmark Analysis ---")
    print(f"Benchmark Metadata Directory: {args.benchmark_dir}")
    print(f"Results Directory: {args.results_dir}")
    if not args.no_plot:
        print(f"Output Plot File: {args.plot_file}")
    else:
        print("Plot generation disabled.")

    # 1. Load Benchmark Metadata
    benchmark_metadata = load_benchmark_metadata(args.benchmark_dir)
    if benchmark_metadata is None:
        return 1  # Error loading metadata

    # 2. Analyze Results
    analysis = analyze_results(benchmark_metadata, args.results_dir)

    # 3. Print Summary Tables
    print_summary_tables(analysis)

    # 4. Generate Plot (if not disabled)
    if not args.no_plot:
        if not analysis["models"]:
            print("\nSkipping plot generation as no model results were found.")
        else:
            generate_plot(analysis, args.plot_file)

    print("\n--- Benchmark Analysis Complete ---")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
