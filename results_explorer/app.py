#!/usr/bin/env python3
import os
import json
import glob
import re
import sys
import math  # Added for Wilson score calculation
from flask import (
    Flask,
    render_template,
    abort,
    send_from_directory,
    current_app,
)
from markupsafe import escape
import webbrowser
import threading
from typing import Any, Dict, Optional, Tuple, List
from collections import defaultdict

# pandas is still useful for analysis but we no longer need matplotlib
try:
    import pandas as pd  # noqa: F401
except ImportError as e:
    print(f"Error importing pandas: {e}. Please ensure it's installed.")
    print("You may need to run: pip install pandas")
    # Not exiting as pandas is not strictly required for basic functionality


# --- Configuration ---
# Assuming the app is run from the root of the repository
BENCHMARK_DIR = "generated_prompts"
RESULTS_BASE_DIR = "benchmark_results"
STATIC_DIR = "results_explorer/static"  # Updated directory name

# --- Flask App Initialization ---
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["BENCHMARK_DIR"] = BENCHMARK_DIR
app.config["RESULTS_BASE_DIR"] = RESULTS_BASE_DIR
app.config["STATIC_DIR"] = STATIC_DIR
# Placeholder for analysis results calculated at startup
app.config["ANALYSIS_RESULTS"] = None


# --- Helper Functions (Moved from analyze_results.py and app.py) ---


def format_bucket_key(key_str: str) -> str:
    """Formats a bucket key string 'min-max' into 'min/1k-max/1k k'."""
    try:
        min_val, max_val = map(int, key_str.split("-"))
        min_k = min_val // 1000
        max_k = max_val // 1000
        return f"{min_k}-{max_k}k"
    except ValueError:
        return key_str


def load_json_file(filepath: str) -> Optional[Dict[str, Any]]:
    """Loads a JSON file, returning None on error."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Error reading or parsing JSON file {filepath}: {e}")
        return None


def load_benchmark_metadata(benchmark_dir):
    """Loads the benchmark structure metadata from metadata.json."""
    metadata_path = os.path.join(benchmark_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Warning: Benchmark metadata file not found at {metadata_path}")
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading or parsing benchmark metadata file {metadata_path}: {e}")
        return None


# Removed duplicate definition of load_benchmark_metadata here


def scan_results_directory(
    results_base_dir: str,
) -> Tuple[List[str], Dict[str, Dict[str, Optional[Dict[str, Any]]]]]:
    """
    Scans the results directory to find all models and their latest run metadata for each benchmark case.
    (Combined logic from find_models_in_results and loading latest metadata)
    """
    print(f"Scanning results directory: {results_base_dir}...")
    latest_runs: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(
        lambda: defaultdict(lambda: ("", ""))
    )
    models = set()

    pattern = os.path.join(results_base_dir, "*", "*", "*", "metadata.json")
    metadata_files = glob.glob(pattern)

    for metadata_path in metadata_files:
        try:
            parts = metadata_path.split(os.sep)
            if len(parts) >= 4:
                timestamp = parts[-2]
                model_name = parts[-3]
                benchmark_case_prefix = parts[-4]

                if not re.match(r"\d{8}_\d{6}", timestamp):
                    continue

                models.add(model_name)
                current_latest_ts, _ = latest_runs[model_name][benchmark_case_prefix]
                if timestamp > current_latest_ts:
                    latest_runs[model_name][benchmark_case_prefix] = (
                        timestamp,
                        metadata_path,
                    )
        except IndexError:
            pass  # Ignore paths that don't match expected structure

    results_data: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = defaultdict(dict)
    for model_name, cases in latest_runs.items():
        for case_prefix, (_, metadata_path) in cases.items():
            results_data[model_name][case_prefix] = load_json_file(metadata_path)

    sorted_models = sorted(list(models))
    print(f"Scan complete. Found {len(sorted_models)} models.")
    return sorted_models, results_data


def analyze_results(
    benchmark_metadata: dict,
    models_found: List[str],
    results_data: Dict[str, Dict[str, Optional[Dict[str, Any]]]],
) -> dict:
    """
    Analyzes benchmark results using pre-scanned data.
    (Directly from analyze_results.py)
    """
    analysis: dict[str, Any] = {"models": {}}
    all_bucket_keys = sorted(
        benchmark_metadata.get("benchmark_buckets", {}).keys(),
        key=lambda k: int(k.split("-")[0]),
    )
    analysis["bucket_keys"] = all_bucket_keys
    analysis["formatted_bucket_keys"] = [
        format_bucket_key(k) for k in all_bucket_keys
    ]  # Add formatted keys

    print(f"Analyzing results for models: {models_found}")
    print(f"Using {len(all_bucket_keys)} buckets defined in benchmark metadata.")

    for model_name in models_found:
        analysis["models"][model_name] = {
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

    defined_buckets = benchmark_metadata.get("benchmark_buckets", {})
    if not defined_buckets:
        print("Warning: No benchmark buckets found in metadata.")
        return analysis

    for bucket_key, benchmark_cases in defined_buckets.items():
        for case_info in benchmark_cases:
            benchmark_case_prefix = case_info["benchmark_case_prefix"]

            for model_name in models_found:
                analysis["models"][model_name]["total_benchmarks"] += 1
                analysis["models"][model_name]["buckets"][bucket_key][
                    "total_in_bucket"
                ] += 1

                result_meta = results_data.get(model_name, {}).get(
                    benchmark_case_prefix
                )

                if result_meta is not None:
                    analysis["models"][model_name]["runs_found"] += 1
                    analysis["models"][model_name]["buckets"][bucket_key][
                        "runs_found"
                    ] += 1

                    if result_meta.get("success", False):
                        analysis["models"][model_name]["successful_runs"] += 1
                        analysis["models"][model_name]["buckets"][bucket_key][
                            "successful_runs"
                        ] += 1

                    cost = result_meta.get("cost_usd", 0.0) or 0.0
                    analysis["models"][model_name]["total_cost_usd"] += float(cost)

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

    # --- Sliding Window Analysis ---
    print("Starting sliding window analysis...")
    max_token_limit = 0
    try:
        if benchmark_metadata and "generation_parameters" in benchmark_metadata:
            gen_params = benchmark_metadata["generation_parameters"]
            if "buckets_str" in gen_params:
                buckets_k_str = gen_params["buckets_str"]
                # Parse the string "0,20,40,..." into a list of integers
                bucket_boundaries_k = [
                    int(b.strip()) for b in buckets_k_str.split(",") if b.strip()
                ]
                if bucket_boundaries_k:
                    # Get the max k-token value and convert to actual token count
                    max_token_limit = max(bucket_boundaries_k) * 1000
                else:
                    print("Warning: Parsed bucket boundaries string is empty.")
            else:
                print("Warning: 'buckets_str' not found in generation_parameters.")
        else:
            print("Warning: 'generation_parameters' not found in benchmark_metadata.")
    except (ValueError, TypeError, AttributeError) as e:
        print(f"Error parsing bucket boundaries for sliding window: {e}")
        max_token_limit = 0  # Ensure it defaults to 0 on error

    if max_token_limit < 5000:
        print(
            f"Warning: Max token limit ({max_token_limit}) is less than 5000. Skipping sliding window analysis."
        )
        analysis["sliding_window"] = None
    else:
        # Define x-axis points (centers of the sliding windows)
        window_centers = list(range(5000, max_token_limit + 1, 1000))
        window_radius = 5000
        analysis["sliding_window"] = {
            "window_centers_k": [c // 1000 for c in window_centers],
            "models": {},
        }

        # Initialize structure for each model
        for model_name in models_found:
            analysis["sliding_window"]["models"][model_name] = {
                center: {
                    "total": 0,
                    "successful": 0,
                    "rate": None,
                    "wilson_lower": None,
                    "wilson_upper": None,
                }
                for center in window_centers
            }

        # Iterate through all benchmark cases
        all_cases_info = []
        for _bucket_key, cases in defined_buckets.items():
            all_cases_info.extend(cases)

        print(
            f"Processing {len(all_cases_info)} total benchmark cases for sliding window..."
        )
        for case_info in all_cases_info:
            benchmark_case_prefix = case_info["benchmark_case_prefix"]
            prompt_tokens = case_info.get("prompt_tokens")

            if prompt_tokens is None:
                print(
                    f"Warning: Missing prompt_tokens for case {benchmark_case_prefix}. Skipping for sliding window."
                )
                continue

            for model_name in models_found:
                result_meta = results_data.get(model_name, {}).get(
                    benchmark_case_prefix
                )
                if result_meta is not None:
                    is_success = result_meta.get("success", False)

                    # Check which windows this case falls into
                    for center in window_centers:
                        if abs(prompt_tokens - center) <= window_radius:
                            window_stats = analysis["sliding_window"]["models"][
                                model_name
                            ][center]
                            window_stats["total"] += 1
                            if is_success:
                                window_stats["successful"] += 1

        # Calculate rates
        for model_name in models_found:
            for center in window_centers:
                window_stats = analysis["sliding_window"]["models"][model_name][center]
                if window_stats["total"] > 0:
                    window_stats["rate"] = (
                        window_stats["successful"] / window_stats["total"]
                    )
                    # Calculate Wilson interval
                    lower, upper = calculate_wilson_interval(
                        window_stats["successful"], window_stats["total"]
                    )
                    window_stats["wilson_lower"] = lower
                    window_stats["wilson_upper"] = upper
                else:
                    # Ensure bounds are None if total is 0
                    window_stats["wilson_lower"] = None
                    window_stats["wilson_upper"] = None

        print("Sliding window analysis complete.")
    # --- End Sliding Window Analysis ---

    return analysis


# --- Wilson Score Interval Calculation ---
def calculate_wilson_interval(
    successes: int, total: int, confidence: float = 0.95
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates the Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successful outcomes.
        total: Total number of trials.
        confidence: The desired confidence level (e.g., 0.95 for 95%).

    Returns:
        A tuple containing (lower_bound, upper_bound), or (None, None) if total is 0.
    """
    if total == 0:
        return None, None

    # Calculate z-score for the given confidence level (approximation for common levels)
    # For simplicity, we'll hardcode z for 95% confidence.
    # For other levels, use: from scipy.stats import norm; z = norm.ppf(1 - (1 - confidence) / 2)
    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.99:
        z = 2.576
    elif confidence == 0.90:
        z = 1.645
    else:
        # Fallback or raise error for unsupported confidence levels
        # Using 95% as default if confidence is not recognized
        print(
            f"Warning: Unsupported confidence level {confidence}. Using z=1.96 (95%)."
        )
        z = 1.96  # Default to 95%

    p_hat = successes / total
    z2 = z * z
    n = total

    center = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    margin = (z / (1 + z2 / n)) * math.sqrt(
        (p_hat * (1 - p_hat) / n) + (z2 / (4 * n * n))
    )

    lower = center - margin
    upper = center + margin

    # Ensure bounds are within [0, 1]
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    return lower, upper


# Plot generation function has been removed in favor of the client-side Chart.js visualization


def find_models_in_results(results_base_dir):
    """Finds all unique model names present in the results directory structure."""
    # This function is now effectively part of scan_results_directory,
    # but keep it for the model_results route fallback check.
    models = set()
    # Pattern: results_base_dir / * (benchmark_case) / * (model_name) / * (timestamp)
    pattern = os.path.join(results_base_dir, "*", "*")
    potential_model_dirs = glob.glob(pattern)
    for path in potential_model_dirs:
        if os.path.isdir(path):
            model_name = os.path.basename(path)
            # Basic check: avoid adding timestamp dirs if structure is unexpected
            if not re.match(r"\d{8}_\d{6}", model_name):
                # Further check: ensure parent is not also a model dir (handles nested structures if any)
                parent_dir_name = os.path.basename(os.path.dirname(path))
                if not re.match(
                    r"\d{8}_\d{6}", parent_dir_name
                ):  # Parent shouldn't be a timestamp
                    models.add(model_name)  # Add the model name
    # Replace underscores back to slashes for display if needed (assuming sanitize_filename replaced them)
    # This might be too aggressive, let's keep original names found in filesystem for now.
    # sanitized_models = {m.replace("_", "/") for m in models} # Example if needed
    return sorted(list(models))


def find_runs_for_model(model_name, results_base_dir):
    """Finds all run directories for a specific model."""
    runs = []
    # Pattern: results_base_dir / * (benchmark_case) / model_name / * (timestamp)
    pattern = os.path.join(results_base_dir, "*", model_name, "*")
    potential_run_dirs = glob.glob(pattern)
    for run_dir in potential_run_dirs:
        if os.path.isdir(run_dir) and re.match(
            r"\d{8}_\d{6}", os.path.basename(run_dir)
        ):
            benchmark_case_prefix = os.path.basename(
                os.path.dirname(os.path.dirname(run_dir))
            )
            timestamp = os.path.basename(run_dir)
            metadata_path = os.path.join(run_dir, "metadata.json")
            run_info = {
                "benchmark_case_prefix": benchmark_case_prefix,
                "model_name": model_name,  # Use original model name passed in
                "timestamp": timestamp,
                "run_dir": run_dir,
                "metadata": None,
                "success": False,  # Default
            }
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    run_info["metadata"] = metadata
                    run_info["success"] = metadata.get("success", False)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not load metadata for {run_dir}: {e}")
            runs.append(run_info)
    return runs


def get_run_details(
    benchmark_case_prefix, model_name, timestamp, benchmark_dir, results_base_dir
):
    """Loads all details for a specific run."""
    run_dir = os.path.join(
        results_base_dir, benchmark_case_prefix, model_name, timestamp
    )
    details = {
        "benchmark_case_prefix": benchmark_case_prefix,
        "model_name": model_name,
        "timestamp": timestamp,
        "run_dir": run_dir,
        "metadata": None,
        "prompt_exists": False,
        "prompt_rel_path": None,
        "expected_exists": False,
        "expected_rel_path": None,
        "raw_response_exists": False,
        "raw_response_rel_path": None,
        "extracted_output_exists": False,
        "extracted_output_rel_path": None,
        "diff_exists": False,
        "diff_rel_path": None,
        "error": None,
    }

    if not os.path.isdir(run_dir):
        details["error"] = "Run directory not found."
        return details

    # Load metadata
    metadata_path = os.path.join(run_dir, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                details["metadata"] = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            details["error"] = f"Error loading metadata.json: {e}"
            # Continue loading other files if possible

    # Load prompt and expected from benchmark_dir
    prompt_filename = f"{benchmark_case_prefix}_prompt.txt"
    expected_filename = f"{benchmark_case_prefix}_expectedoutput.txt"
    # Check existence and store relative paths for prompt and expected files
    prompt_rel_path = os.path.join(benchmark_dir, prompt_filename)
    expected_rel_path = os.path.join(benchmark_dir, expected_filename)

    if os.path.exists(prompt_rel_path):
        details["prompt_exists"] = True
        details["prompt_rel_path"] = prompt_rel_path
    else:
        details["error"] = (
            details.get("error", "") + f" Prompt file not found: {prompt_rel_path}"
        )

    if os.path.exists(expected_rel_path):
        details["expected_exists"] = True
        details["expected_rel_path"] = expected_rel_path
    else:
        details["error"] = (
            details.get("error", "")
            + f" Expected output file not found: {expected_rel_path}"
        )

    # Check existence and store relative paths for files in run_dir
    def check_run_file(filename, exists_key, path_key):
        rel_path = os.path.join(run_dir, filename)
        if os.path.exists(rel_path):
            details[exists_key] = True
            details[path_key] = rel_path
        # No error message here, as missing files might be expected (e.g., no diff on success)

    check_run_file("raw_response.txt", "raw_response_exists", "raw_response_rel_path")
    check_run_file(
        "extracted_output.txt", "extracted_output_exists", "extracted_output_rel_path"
    )
    check_run_file("output.diff", "diff_exists", "diff_rel_path")

    return details


# --- Routes ---


@app.route("/")
def index():
    """Index page: Shows overall summary, models, and dynamic chart."""
    benchmark_metadata = load_benchmark_metadata(current_app.config["BENCHMARK_DIR"])
    # Retrieve pre-calculated analysis results and models found
    analysis_results = current_app.config.get("ANALYSIS_RESULTS")
    models = (
        sorted(list(analysis_results["models"].keys()))
        if analysis_results and analysis_results.get("models")
        else []
    )

    total_cases = 0
    if benchmark_metadata and "benchmark_buckets" in benchmark_metadata:
        for _bucket_key, cases in benchmark_metadata["benchmark_buckets"].items():
            total_cases += len(cases)

    return render_template(
        "index.html",
        models=models,
        total_cases=total_cases,
        benchmark_metadata=benchmark_metadata,
        analysis_results=analysis_results,  # Pass full analysis results
    )


@app.route("/model/<path:model_name>")
def model_results(model_name):
    """Shows results for a specific model, grouped by bucket."""
    safe_model_name = escape(model_name)
    benchmark_metadata = load_benchmark_metadata(current_app.config["BENCHMARK_DIR"])
    all_runs = find_runs_for_model(model_name, current_app.config["RESULTS_BASE_DIR"])

    # Check if model exists based on analysis results (more reliable than scanning again)
    analysis_results = current_app.config.get("ANALYSIS_RESULTS")
    if not analysis_results or model_name not in analysis_results.get("models", {}):
        # Check filesystem as fallback if analysis hasn't run or failed
        if not any(
            m == model_name
            for m in find_models_in_results(current_app.config["RESULTS_BASE_DIR"])
        ):
            abort(404, description=f"Model '{safe_model_name}' not found in results.")

    runs_by_bucket = {}
    if benchmark_metadata and "benchmark_buckets" in benchmark_metadata:
        # Use bucket keys from analysis results if available, otherwise from metadata
        bucket_keys = (
            analysis_results.get("bucket_keys")
            if analysis_results
            else sorted(
                benchmark_metadata["benchmark_buckets"].keys(),
                key=lambda k: int(k.split("-")[0]),
            )
        )
        for bucket_key in bucket_keys:
            runs_by_bucket[bucket_key] = []

        case_to_bucket = {}
        for bucket_key, cases in benchmark_metadata["benchmark_buckets"].items():
            for case_info in cases:
                case_to_bucket[case_info["benchmark_case_prefix"]] = bucket_key

        for run in all_runs:
            bucket_key = case_to_bucket.get(run["benchmark_case_prefix"])
            if bucket_key and bucket_key in runs_by_bucket:
                runs_by_bucket[bucket_key].append(run)
            else:
                if "unknown" not in runs_by_bucket:
                    runs_by_bucket["unknown"] = []
                runs_by_bucket["unknown"].append(run)

        # Use the sorted keys for the final dict to maintain order
        sorted_runs_by_bucket = {
            k: runs_by_bucket[k] for k in bucket_keys if k in runs_by_bucket
        }
        if "unknown" in runs_by_bucket:
            sorted_runs_by_bucket["unknown"] = runs_by_bucket["unknown"]

    else:
        sorted_runs_by_bucket = {"all_runs": all_runs}

    return render_template(
        "model_results.html",
        model_name=safe_model_name,
        original_model_name=model_name,
        runs_by_bucket=sorted_runs_by_bucket,
        benchmark_metadata=benchmark_metadata,
        # Pass formatted bucket keys from analysis if available
        formatted_bucket_keys=analysis_results.get("formatted_bucket_keys")
        if analysis_results
        else None,
        bucket_keys=analysis_results.get("bucket_keys") if analysis_results else None,
    )


@app.route("/case/<benchmark_case_prefix>/<path:model_name>/<timestamp>")
def case_details(benchmark_case_prefix, model_name, timestamp):
    """Shows details for a specific benchmark run."""
    # Escape components for display, but use original for fetching data
    safe_case_prefix = escape(benchmark_case_prefix)
    safe_model_name = escape(model_name)
    safe_timestamp = escape(timestamp)

    details = get_run_details(
        benchmark_case_prefix,
        model_name,  # Use original model name for path construction
        timestamp,
        app.config["BENCHMARK_DIR"],
        app.config["RESULTS_BASE_DIR"],
    )

    if details.get("error") and "Run directory not found" in details["error"]:
        abort(
            404,
            description=f"Details not found for case '{safe_case_prefix}', model '{safe_model_name}', timestamp '{safe_timestamp}'. Error: {details.get('error')}",
        )
    elif details.get("error"):
        # Show page but display error prominently
        pass

    return render_template(
        "case_details.html",
        details=details,
        # Pass safe versions for display in template if needed
        safe_case_prefix=safe_case_prefix,
        safe_model_name=safe_model_name,
        safe_timestamp=safe_timestamp,
    )


# Optional: Add a route to serve files directly if needed, e.g., download prompt
# @app.route('/files/<path:filepath>')
# def serve_file(filepath):
#     # Be VERY careful with security here if implementing
#     # Ensure path traversal is prevented
#     base_dir = os.path.abspath(".") # Or specific allowed directories
#     safe_path = os.path.abspath(os.path.join(base_dir, filepath))
#     if not safe_path.startswith(base_dir):
#          abort(403)
#     # Add checks for allowed directories (e.g., only generated_prompts, benchmark_results)
#     allowed_dirs = [os.path.abspath(app.config['BENCHMARK_DIR']), os.path.abspath(app.config['RESULTS_BASE_DIR'])]
#     is_allowed = any(safe_path.startswith(allowed_dir) for allowed_dir in allowed_dirs)
#     if not is_allowed:
#         abort(403)

#     try:
#         return send_from_directory(os.path.dirname(safe_path), os.path.basename(safe_path))
#     except FileNotFoundError:
#         abort(404)


@app.route("/api/plot-data")
def get_plot_data():
    """Returns the plot data as JSON for the chart."""
    analysis_results = current_app.config.get("ANALYSIS_RESULTS")

    if not analysis_results or not analysis_results.get("models"):
        return {"error": "No analysis results available"}, 404

    # Prepare data for Chart.js format
    bucket_labels = analysis_results.get("formatted_bucket_keys", [])
    datasets = []

    # Sort models for consistent coloring
    models = sorted(list(analysis_results["models"].keys()))

    for model_name in models:
        model_stats = analysis_results["models"][model_name]
        success_rates = []

        for bucket_key in analysis_results["bucket_keys"]:
            bucket_stats = model_stats["buckets"][bucket_key]
            rate = (
                bucket_stats["success_rate"] if bucket_stats["runs_found"] > 0 else None
            )
            success_rates.append(rate)

        # Format the cost for the label
        total_cost = model_stats.get("total_cost_usd", 0.0)

        datasets.append(
            {
                "label": f"{model_name} (${total_cost:.2f})",
                "data": success_rates,
                "borderWidth": 2,
                "tension": 0.1,  # Slight curve for nicer appearance
                "fill": False,
            }
        )

    return {"labels": bucket_labels, "datasets": datasets}


@app.route("/api/sliding-plot-data")
def get_sliding_plot_data():
    """Returns the sliding window plot data as JSON for the chart."""
    analysis_results = current_app.config.get("ANALYSIS_RESULTS")

    if (
        not analysis_results
        or not analysis_results.get("sliding_window")
        or not analysis_results["sliding_window"].get("models")
    ):
        return {"error": "No sliding window analysis results available"}, 404

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


@app.route("/files/<path:filepath>")
def serve_file(filepath):
    """Serves files securely from allowed directories."""
    # Basic sanitization (Flask's path converter helps, but extra checks are good)
    if ".." in filepath or filepath.startswith("/"):
        abort(403, "Invalid file path.")

    # Define allowed base directories relative to app root
    allowed_dirs_rel = [app.config["BENCHMARK_DIR"], app.config["RESULTS_BASE_DIR"]]
    # Get the absolute path to the repository root directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    allowed_dirs_abs = [
        os.path.abspath(os.path.join(repo_root, d)) for d in allowed_dirs_rel
    ]

    # Construct the absolute path requested by the user, relative to repo root
    requested_path_abs = os.path.abspath(os.path.join(repo_root, filepath))

    # Security Check: Ensure the requested path is within one of the allowed directories
    is_allowed = False
    serving_directory = None
    filename = None
    for allowed_dir in allowed_dirs_abs:
        # Check if the requested path starts with the allowed directory path + separator
        # This ensures we don't match partial directory names
        # Use os.path.normcase for case-insensitive comparison on relevant systems
        if os.path.normcase(requested_path_abs).startswith(
            os.path.normcase(allowed_dir + os.sep)
        ):
            is_allowed = True
            serving_directory = allowed_dir
            # Calculate filename relative to the serving directory
            filename = os.path.relpath(requested_path_abs, allowed_dir)
            # Double-check filename doesn't try to escape upwards (should be prevented by startswith check)
            if ".." in filename or filename.startswith(os.sep):
                is_allowed = False  # Abort if relpath calculation seems suspicious
                break
            break  # Found the allowed directory

    if not is_allowed or serving_directory is None or filename is None:
        abort(
            403,
            "Access denied: File is outside allowed directories or path calculation failed.",
        )

    # Determine mimetype (simple check for text files)
    mimetype = (
        "text/plain"
        if filepath.endswith(
            (".txt", ".diff", ".py", ".js", ".html", ".css", ".md", ".log")
        )
        else None
    )

    try:
        # Use send_from_directory for safer serving
        # It requires the directory and the filename relative to that directory
        # print(f"Serving file: directory='{serving_directory}', filename='{filename}'") # Debugging
        return send_from_directory(
            serving_directory, filename, mimetype=mimetype, as_attachment=False
        )
    except FileNotFoundError:
        abort(404, "File not found.")
    except Exception as e:
        print(f"Error serving file {filepath}: {e}")
        abort(500, "Internal server error while serving file.")


def open_browser(host, port):
    """Opens the browser to the specified host and port."""
    # Use 127.0.0.1 for the browser URL even if hosting on 0.0.0.0
    url_host = "127.0.0.1" if host == "0.0.0.0" else host
    webbrowser.open_new_tab(f"http://{url_host}:{port}")


# --- Analysis and Plotting Execution (at startup) ---


def run_data_analysis():
    """Performs benchmark results analysis, storing results in app.config."""
    # Use app context to access config
    with app.app_context():
        print("\n--- Running Benchmark Data Analysis ---")
        benchmark_metadata = load_benchmark_metadata(
            current_app.config["BENCHMARK_DIR"]
        )
        if not benchmark_metadata:
            print(
                "Error: Benchmark metadata not found. Cannot perform analysis.",
                file=sys.stderr,
            )
            current_app.config["ANALYSIS_RESULTS"] = None  # Ensure it's None
            return

        models_found, results_data = scan_results_directory(
            current_app.config["RESULTS_BASE_DIR"]
        )
        if not models_found:
            print(
                "Warning: No models found in results directory. Analysis will be empty."
            )
            # Store empty structure
            current_app.config["ANALYSIS_RESULTS"] = {
                "models": {},
                "bucket_keys": [],
                "formatted_bucket_keys": [],
            }
            return

        analysis_results = analyze_results(
            benchmark_metadata, models_found, results_data
        )
        current_app.config["ANALYSIS_RESULTS"] = analysis_results  # Store results

        print("--- Finished Data Analysis ---\n")


if __name__ == "__main__":
    # Configuration for running directly
    HOST = "127.0.0.1"
    PORT = 5001
    DEBUG = True  # Flask debug mode

    # Run data analysis before starting the server
    # This happens only once when the main process starts (not in reloader subprocess)
    if not DEBUG or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        run_data_analysis()

    # Open browser tab shortly after starting the server
    if not DEBUG or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        # Delay slightly to ensure server is likely up
        threading.Timer(1.5, lambda: open_browser(HOST, PORT)).start()

    # Run the Flask app
    app.run(debug=DEBUG, host=HOST, port=PORT)
