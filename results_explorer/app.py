#!/usr/bin/env python3
import os
import json
import glob
import argparse
import re
import sys
import math
import typing # Added typing import
from flask import (
    Flask,
    render_template,
    abort,
    send_from_directory,
    current_app,
    request,  # Added request
    jsonify,  # Added jsonify
)
from markupsafe import escape
import webbrowser
import threading
from typing import Any, Dict, Optional, Tuple, List
from collections import defaultdict

try:
    import pandas as pd  # noqa: F401
except ImportError as e:
    print(f"Error importing pandas: {e}. Please ensure it's installed.")
    print("You may need to run: pip install pandas")
    # Not exiting as pandas is not strictly required for basic functionality


# --- Configuration (Set via command-line arg when run directly) ---
# Constants for subdirectory names
PROMPTS_SUBDIR = "prompts"
RESULTS_SUBDIR = "results"
STATIC_DIR_NAME = "static"  # Relative to this script's location

# --- Flask App Initialization ---
# Determine static folder path relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
static_folder_path = os.path.join(script_dir, STATIC_DIR_NAME)

app = Flask(__name__, template_folder="templates", static_folder=static_folder_path)
# BENCHMARK_RUN_DIR will be set in app.config when run directly via __main__
# Placeholder for analysis results calculated at startup
app.config["ANALYSIS_RESULTS"] = None
app.config["AVAILABLE_LANGUAGES"] = []  # Add placeholder for languages
app.config["BENCHMARK_METADATA"] = None  # Add placeholder for benchmark metadata


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


def load_benchmark_metadata(prompts_dir: str):
    """
    Loads the benchmark structure metadata from metadata.json located in the prompts subdirectory.

    Args:
        prompts_dir: The path to the 'prompts' subdirectory within the benchmark run directory.
    """
    metadata_path = os.path.join(prompts_dir, "metadata.json")
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
    results_dir: str,
) -> Tuple[List[str], Dict[str, Dict[str, Optional[Dict[str, Any]]]]]:
    """
    Scans the results subdirectory to find all models and their latest run metadata for each benchmark case.
    (Combined logic from find_models_in_results and loading latest metadata)

    Args:
        results_dir: The path to the 'results' subdirectory within the benchmark run directory.
    """
    print(f"Scanning results directory: {results_dir}...")
    latest_runs: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(
        lambda: defaultdict(lambda: ("", ""))
    )
    models = set()

    # Pattern: <run_dir>/results/[benchmark_case]/[model_name]/[timestamp]/metadata.json
    pattern = os.path.join(results_dir, "*", "*", "*", "metadata.json")
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
    selected_languages: Optional[
        List[str]
    ] = None,  # Added selected_languages parameter
) -> dict:
    """
    Analyzes benchmark results using pre-scanned data.

    Args:
        benchmark_metadata: The loaded benchmark metadata.
        models_found: A list of model names found in the results.
        results_data: A nested dictionary containing the latest result metadata for each model/case.
        selected_languages: An optional list of languages to filter the sliding window analysis by.
                            If None or empty, all languages are included.
    """
    analysis: dict[str, Any] = {"models": {}}
    print(f"Analyzing results for models: {models_found}")
    if selected_languages:
        print(f"Filtering sliding window analysis by languages: {selected_languages}")

    # --- Aggregate All Benchmark Cases from Metadata ---
    all_cases_info_dict: Dict[
        str, Dict[str, Any]
    ] = {}  # prefix -> {token_count, language, etc.}
    max_token_limit = 0  # Track max token limit across all runs for sliding window
    available_languages_from_metadata = set()  # Track languages found in metadata

    if isinstance(benchmark_metadata, list):  # New format: list of runs
        print(f"Processing {len(benchmark_metadata)} runs from metadata...")
        for run_index, run_data in enumerate(benchmark_metadata):
            cases_key = "benchmark_cases_added"  # New key
            if cases_key not in run_data and "benchmark_cases" in run_data:
                cases_key = "benchmark_cases"  # Legacy key
                # print(f"Note: Using legacy 'benchmark_cases' key for run {run_index}.")

            if (
                isinstance(run_data, dict)
                and cases_key in run_data
                and isinstance(run_data[cases_key], list)
            ):
                for case in run_data[cases_key]:
                    if isinstance(case, dict) and "benchmark_case_prefix" in case:
                        prefix = case["benchmark_case_prefix"]
                        # Store info, potentially overwriting older info if prefix repeats (unlikely but possible)
                        case_info = {
                            "prompt_tokens": case.get("prompt_tokens"),
                            # Add other relevant case info if needed later
                        }
                        # Extract language if available
                        lang = case.get("language")
                        if lang:
                            case_info["language"] = lang
                            available_languages_from_metadata.add(lang)
                        else:
                            # Attempt to infer language from prefix if not explicitly provided
                            # Example: 'python_some_file_...' -> 'python'
                            try:
                                inferred_lang = prefix.split("_")[0]
                                # Basic check if it looks like a language name (e.g., all lowercase letters)
                                if inferred_lang.isalpha() and inferred_lang.islower():
                                    case_info["language"] = inferred_lang
                                    available_languages_from_metadata.add(inferred_lang)
                                    # print(f"Inferred language '{inferred_lang}' for case {prefix}") # Optional debug log
                                else:
                                    # print(f"Warning: Could not infer language for case {prefix}") # Optional debug log
                                    pass
                            except IndexError:
                                # print(f"Warning: Could not infer language for case {prefix}") # Optional debug log
                                pass

                        all_cases_info_dict[prefix] = case_info
            else:
                print(
                    f"Warning: Skipping run {run_index} due to invalid structure or missing '{cases_key}'."
                )

            # Update max_token_limit based on this run's parameters
            try:
                gen_params = run_data.get("generation_parameters", {})
                run_max_tokens = gen_params.get(
                    "max_prompt_tokens", 0
                )  # Absolute tokens
                # Handle potential k-tokens in older metadata if necessary (adjust logic if needed)
                # Example: if 'max_prompt_tokens' < 1000 and 'max_prompt_tokens' > 0: run_max_tokens *= 1000
                max_token_limit = max(max_token_limit, run_max_tokens)
            except Exception as e:
                print(
                    f"Warning: Could not determine max token limit for run {run_index}: {e}"
                )

    elif (
        isinstance(benchmark_metadata, dict) and "benchmark_cases" in benchmark_metadata
    ):  # Handle legacy format
        print("Processing legacy metadata format (single run)...")
        if isinstance(benchmark_metadata["benchmark_cases"], list):
            for case in benchmark_metadata["benchmark_cases"]:
                if isinstance(case, dict) and "benchmark_case_prefix" in case:
                    prefix = case["benchmark_case_prefix"]
                    case_info = {"prompt_tokens": case.get("prompt_tokens")}
                    # Extract language if available (also for legacy)
                    lang = case.get("language")
                    if lang:
                        case_info["language"] = lang
                        available_languages_from_metadata.add(lang)
                    else:
                        # Attempt inference for legacy too
                        try:
                            inferred_lang = prefix.split("_")[0]
                            if inferred_lang.isalpha() and inferred_lang.islower():
                                case_info["language"] = inferred_lang
                                available_languages_from_metadata.add(inferred_lang)
                        except IndexError:
                            pass  # Ignore if inference fails

                    all_cases_info_dict[prefix] = case_info
        # Try to get max token limit from legacy params
        try:
            gen_params = benchmark_metadata.get("generation_parameters", {})
            # Check for bucket boundaries first
            if "buckets_str" in gen_params:
                buckets_k_str = gen_params["buckets_str"]
                bucket_boundaries_k = [
                    int(b.strip()) for b in buckets_k_str.split(",") if b.strip()
                ]
                if bucket_boundaries_k:
                    max_token_limit = max(bucket_boundaries_k) * 1000
            # Fallback to max_prompt_tokens if buckets_str not present
            elif "max_prompt_tokens" in gen_params:
                max_token_limit = gen_params[
                    "max_prompt_tokens"
                ]  # Assume absolute in legacy? Adjust if needed.
        except Exception as e:
            print(
                f"Warning: Could not determine max token limit from legacy metadata: {e}"
            )

    else:
        print(
            "Warning: Benchmark metadata is not in a recognized format (list of runs or legacy object). Analysis might be incomplete."
        )

    total_unique_cases = len(all_cases_info_dict)
    print(
        f"Found {total_unique_cases} total unique benchmark cases defined across all metadata runs."
    )
    # Update global list of languages found in actual benchmark data
    # Only update if this is the initial run (no language filter)
    if selected_languages is None:
        current_app.config["AVAILABLE_LANGUAGES"] = sorted(
            list(available_languages_from_metadata)
        )
        print(
            f"Available languages set to: {current_app.config['AVAILABLE_LANGUAGES']}"
        )  # Log found languages
    else:
        print("Skipping update of AVAILABLE_LANGUAGES during filtered run.")

    if total_unique_cases == 0:
        analysis["sliding_window"] = None
        return analysis  # Cannot proceed without defined cases

    # --- Initialize Model Stats ---
    for model_name in models_found:
        analysis["models"][model_name] = {
            "total_benchmarks": total_unique_cases,  # Total defined cases
            "runs_found": 0,  # Will count actual results found
            "successful_runs": 0,
            "total_cost_usd": 0.0,
            "success_rate": 0.0,
        }

    # --- Calculate Run Stats (Based on Latest Results Found) ---
    for model_name in models_found:
        model_results = results_data.get(model_name, {})
        analysis["models"][model_name]["runs_found"] = len(model_results)
        for case_prefix, result_meta in model_results.items():
            if result_meta is not None:
                if result_meta.get("success", False):
                    analysis["models"][model_name]["successful_runs"] += 1
                cost = result_meta.get("cost_usd", 0.0) or 0.0
                analysis["models"][model_name]["total_cost_usd"] += float(cost)

        # Calculate overall success rates based on actual runs found
        if analysis["models"][model_name]["runs_found"] > 0:
            analysis["models"][model_name]["success_rate"] = (
                analysis["models"][model_name]["successful_runs"]
                / analysis["models"][model_name]["runs_found"]
            )

    # --- Sliding Window Analysis ---
    print("Starting sliding window analysis...")
    # Use max_token_limit determined from metadata runs
    print(f"Using max token limit for sliding window: {max_token_limit}")

    # Define x-axis points (centers of the sliding windows)
    # Start centers at 0k, step by 1k
    window_centers = list(range(0, max_token_limit + 1, 1000))
    window_radius = 10000  # +/- 10k tokens window

    if not window_centers:
        print(
            "Warning: No window centers generated (max_token_limit might be too low). Skipping sliding window analysis."
        )
        analysis["sliding_window"] = None
    else:
        analysis["sliding_window"] = {
            "max_token_limit": max_token_limit,  # Store the calculated limit
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

        # Iterate through the aggregated benchmark cases
        print(
            f"Processing {total_unique_cases} unique benchmark cases for sliding window..."
        )
        for benchmark_case_prefix, case_info in all_cases_info_dict.items():
            prompt_tokens = case_info.get("prompt_tokens")
            case_language = case_info.get("language")  # Get language for filtering

            if prompt_tokens is None:
                print(
                    f"Warning: Missing prompt_tokens for case {benchmark_case_prefix}. Skipping for sliding window."
                )
                continue

            # Filter by language if selected_languages is provided and not empty
            # Also skip if the case doesn't have language info when filtering is active
            if selected_languages:
                if not case_language:
                    # print(f"Skipping case {benchmark_case_prefix} (missing language info) due to active language filter.") # Debug log
                    continue
                if case_language not in selected_languages:
                    # print(f"Skipping case {benchmark_case_prefix} (lang: {case_language}) due to language filter.") # Debug log
                    continue
                # If we reach here, the case language is in the selected list

            # Check results for each model for this specific case
            for model_name in models_found:
                # Get the latest result for this model/case pair
                result_meta = results_data.get(model_name, {}).get(
                    benchmark_case_prefix
                )

                # Only consider cases where a result actually exists for this model
                if result_meta is not None:
                    is_success = result_meta.get("success", False)

                    # Check which windows this case falls into based on its prompt_tokens
                    for center in window_centers:
                        if abs(prompt_tokens - center) <= window_radius:
                            window_stats = analysis["sliding_window"]["models"][
                                model_name
                            ][center]
                            # Increment total count for this window/model pair
                            window_stats["total"] += 1
                            if is_success:
                                # Increment success count if the result was successful
                                window_stats["successful"] += 1

        # Calculate rates and confidence intervals after processing all cases
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


def find_models_in_results(results_dir: str):
    """
    Finds all unique model names present in the results subdirectory structure.

    Args:
        results_dir: The path to the 'results' subdirectory within the benchmark run directory.
    """
    # This function is now effectively part of scan_results_directory,
    # but keep it for the model_results route fallback check.
    models = set()
    # Pattern: <run_dir>/results/[benchmark_case]/[model_name]
    pattern = os.path.join(results_dir, "*", "*")
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


def find_runs_for_model(
    model_name: str, results_dir: str
):  # Changed from results_base_dir
    """
    Finds all run directories for a specific model within the results subdirectory.

    Args:
        model_name: The name of the model.
        results_dir: The path to the 'results' subdirectory within the benchmark run directory.
    """
    runs = []
    # Pattern: <run_dir>/results/[benchmark_case]/[model_name]/[timestamp]
    pattern = os.path.join(results_dir, "*", model_name, "*")
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
    benchmark_case_prefix: str,
    model_name: str,
    timestamp: str,
    prompts_dir: str,
    results_dir: str,
    benchmark_run_dir: Optional[str] = None,
):
    """
    Loads all details for a specific run, accessing files from prompts and results subdirectories.

    Args:
        benchmark_case_prefix: The benchmark case identifier.
        model_name: The model name.
        timestamp: The run timestamp.
        prompts_dir: Path to the 'prompts' subdirectory.
        results_dir: Path to the 'results' subdirectory.
        benchmark_run_dir: Path to the benchmark run directory (parent of prompts and results).
    """
    # If benchmark_run_dir isn't provided, derive it from prompts_dir or results_dir
    if benchmark_run_dir is None:
        # Assume prompts_dir is benchmark_run_dir/prompts
        benchmark_run_dir = os.path.dirname(prompts_dir)
    # Path to the specific run directory within the results subdirectory
    run_dir = os.path.join(results_dir, benchmark_case_prefix, model_name, timestamp)
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

    # Load prompt and expected from prompts_dir
    prompt_filename = f"{benchmark_case_prefix}_prompt.txt"
    expected_filename = f"{benchmark_case_prefix}_expectedoutput.txt"
    # Check existence and store relative paths for prompt and expected files within prompts_dir
    prompt_abs_path = os.path.join(prompts_dir, prompt_filename)
    expected_abs_path = os.path.join(prompts_dir, expected_filename)

    if os.path.exists(prompt_abs_path):
        details["prompt_exists"] = True
        # Store path relative to the benchmark_run_dir, not absolute path
        prompt_rel_path = os.path.relpath(prompt_abs_path, benchmark_run_dir)
        details["prompt_rel_path"] = prompt_rel_path
    else:
        details["error"] = (
            details.get("error", "") + f" Prompt file not found: {prompt_abs_path}"
        )

    if os.path.exists(expected_abs_path):
        details["expected_exists"] = True
        # Store path relative to the benchmark_run_dir, not absolute path
        expected_rel_path = os.path.relpath(expected_abs_path, benchmark_run_dir)
        details["expected_rel_path"] = expected_rel_path
    else:
        details["error"] = (
            details.get("error", "")
            + f" Expected output file not found: {expected_abs_path}"
        )

    # Check existence and store relative paths for files in run_dir
    def check_run_file(filename, exists_key, path_key):
        abs_path = os.path.join(run_dir, filename)
        if os.path.exists(abs_path):
            details[exists_key] = True
            # Store path relative to the benchmark_run_dir, not absolute path
            rel_path = os.path.relpath(abs_path, benchmark_run_dir)
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
    benchmark_run_dir_maybe_none = current_app.config.get("BENCHMARK_RUN_DIR")
    if benchmark_run_dir_maybe_none is None:
        abort(500, "Benchmark run directory not configured.")
    benchmark_run_dir: str = (
        benchmark_run_dir_maybe_none  # Explicitly typed after check
    )
    prompts_dir = os.path.join(benchmark_run_dir, PROMPTS_SUBDIR)
    results_dir = os.path.join(benchmark_run_dir, RESULTS_SUBDIR)  # Needed for scanning

    # Load benchmark metadata first (and store it)
    # Use cached version if available
    if current_app.config.get("BENCHMARK_METADATA") is None:
        print("Loading benchmark metadata...")
        current_app.config["BENCHMARK_METADATA"] = load_benchmark_metadata(prompts_dir)

    benchmark_metadata = current_app.config["BENCHMARK_METADATA"]
    if benchmark_metadata is None:
        print("Warning: Could not load benchmark metadata. Analysis may be limited.")
        # Allow rendering even without metadata, but show warnings/limited info

    # Retrieve or run initial analysis results (unfiltered)
    # This populates AVAILABLE_LANGUAGES
    analysis_results = current_app.config.get("ANALYSIS_RESULTS")
    if analysis_results is None:
        print("Running initial analysis (cached results not found)...")
        models_found, results_data = scan_results_directory(results_dir)
        if benchmark_metadata and models_found:
            analysis_results = analyze_results(
                benchmark_metadata, models_found, results_data, selected_languages=None
            )
            current_app.config["ANALYSIS_RESULTS"] = analysis_results  # Cache results
        else:
            current_app.config["ANALYSIS_RESULTS"] = None
            current_app.config["AVAILABLE_LANGUAGES"] = []
    else:
        print("Using cached initial analysis results.")

    # Get models from analysis results (either fresh or cached)
    models = (
        sorted(list(analysis_results["models"].keys()))
        if analysis_results and analysis_results.get("models")
        else []
    )

    # Get available languages determined during analysis
    available_languages = current_app.config.get("AVAILABLE_LANGUAGES", [])

    # Calculate total cases from metadata if possible, considering available languages
    total_cases = None
    if benchmark_metadata:
        if isinstance(benchmark_metadata, list):  # New format
            if available_languages:
                total_cases = 0
                for run in benchmark_metadata:
                    if isinstance(run, dict):
                        cases_key = (
                            "benchmark_cases_added"
                            if "benchmark_cases_added" in run
                            else "benchmark_cases"
                        )
                        for case in run.get(cases_key, []):
                            if isinstance(case, dict):
                                lang = case.get("language")
                                if not lang:  # Try inferring if missing
                                    prefix = case.get("benchmark_case_prefix", "")
                                    try:
                                        inferred_lang = prefix.split("_")[0]
                                        if (
                                            inferred_lang.isalpha()
                                            and inferred_lang.islower()
                                        ):
                                            lang = inferred_lang
                                    except IndexError:
                                        pass
                                if (
                                    not available_languages
                                    or lang in available_languages
                                ):
                                    total_cases += 1
            else:  # Fallback if no languages found
                total_cases = sum(
                    len(run.get("benchmark_cases_added", []))
                    for run in benchmark_metadata
                    if isinstance(run, dict)
                )

        elif (
            isinstance(benchmark_metadata, dict)
            and "benchmark_cases" in benchmark_metadata
        ):  # Legacy
            if available_languages:
                total_cases = 0
                for case in benchmark_metadata.get("benchmark_cases", []):
                    if isinstance(case, dict):
                        lang = case.get("language")
                        if not lang:  # Try inferring if missing
                            prefix = case.get("benchmark_case_prefix", "")
                            try:
                                inferred_lang = prefix.split("_")[0]
                                if inferred_lang.isalpha() and inferred_lang.islower():
                                    lang = inferred_lang
                            except IndexError:
                                pass
                        if not available_languages or lang in available_languages:
                            total_cases += 1
            else:  # Fallback
                total_cases = len(benchmark_metadata.get("benchmark_cases", []))

    # Extract max_token_limit for the template, default to None if not available
    max_token_limit_value = None
    if (
        analysis_results
        and analysis_results.get("sliding_window")
        and "max_token_limit" in analysis_results["sliding_window"]
    ):
        max_token_limit_value = analysis_results["sliding_window"]["max_token_limit"]

    return render_template(
        "index.html",
        models=models,
        total_cases=total_cases,  # Pass potentially language-aware count
        benchmark_metadata=benchmark_metadata,
        analysis_results=analysis_results,  # Pass initial analysis results
        max_token_limit=max_token_limit_value,  # Pass the limit to the template
        available_languages=available_languages,  # Pass languages for checkboxes
    )


@app.route("/model/<path:model_name>")
def model_results(model_name):
    """Shows results for a specific model, grouped by bucket."""
    safe_model_name = escape(model_name)
    benchmark_run_dir_maybe_none = current_app.config.get("BENCHMARK_RUN_DIR")
    if benchmark_run_dir_maybe_none is None:
        abort(500, "Benchmark run directory not configured.")
    benchmark_run_dir: str = (
        benchmark_run_dir_maybe_none  # Explicitly typed after check
    )
    prompts_dir = os.path.join(benchmark_run_dir, PROMPTS_SUBDIR)
    results_dir = os.path.join(benchmark_run_dir, RESULTS_SUBDIR)

    benchmark_metadata = load_benchmark_metadata(prompts_dir)
    all_runs = find_runs_for_model(model_name, results_dir)

    # Check if model exists based on analysis results (more reliable than scanning again)
    analysis_results = current_app.config.get("ANALYSIS_RESULTS")
    if not analysis_results or model_name not in analysis_results.get("models", {}):
        # Check filesystem as fallback if analysis hasn't run or failed
        if not any(m == model_name for m in find_models_in_results(results_dir)):
            abort(404, description=f"Model '{safe_model_name}' not found in results.")

    # --- Aggregate prompt tokens from all metadata runs ---
    case_prompt_tokens = {}  # prefix -> token_count
    if isinstance(benchmark_metadata, list):  # New format
        for run_data in benchmark_metadata:
            cases_key = "benchmark_cases_added"
            if cases_key not in run_data and "benchmark_cases" in run_data:
                cases_key = "benchmark_cases"  # Legacy

            if (
                isinstance(run_data, dict)
                and cases_key in run_data
                and isinstance(run_data[cases_key], list)
            ):
                for case in run_data[cases_key]:
                    if isinstance(case, dict) and "benchmark_case_prefix" in case:
                        prefix = case["benchmark_case_prefix"]
                        # Store token count, potentially overwriting (should be consistent)
                        case_prompt_tokens[prefix] = case.get("prompt_tokens")

    elif (
        isinstance(benchmark_metadata, dict) and "benchmark_cases" in benchmark_metadata
    ):  # Legacy format
        if isinstance(benchmark_metadata["benchmark_cases"], list):
            for case in benchmark_metadata["benchmark_cases"]:
                if isinstance(case, dict) and "benchmark_case_prefix" in case:
                    prefix = case["benchmark_case_prefix"]
                    case_prompt_tokens[prefix] = case.get("prompt_tokens")
    # --- End aggregation ---

    # Assign prompt tokens to runs based on the aggregated map
    missing_token_count = 0
    for run in all_runs:
        prefix = run["benchmark_case_prefix"]
        if prefix in case_prompt_tokens and case_prompt_tokens[prefix] is not None:
            run["prompt_tokens"] = case_prompt_tokens[prefix]
        else:
            # Fallback: Try getting from the run's own metadata if lookup failed
            if run.get("metadata") and run["metadata"].get("prompt_tokens") is not None:
                run["prompt_tokens"] = run["metadata"]["prompt_tokens"]
                if prefix not in case_prompt_tokens:  # Log if it wasn't in the main map
                    print(
                        f"Warning: Prompt tokens for case '{prefix}' found in run metadata but not in aggregated benchmark metadata."
                    )
            else:
                run["prompt_tokens"] = 0  # Default if not found anywhere
                missing_token_count += 1
                print(
                    f"Warning: Could not determine prompt tokens for case '{prefix}'. Defaulting to 0."
                )

    if missing_token_count > 0:
        print(
            f"Warning: Failed to find prompt tokens for {missing_token_count} run(s)."
        )

    # Sort all runs by prompt token count (ascending order)
    sorted_runs = sorted(all_runs, key=lambda run: run["prompt_tokens"])

    return render_template(
        "model_results.html",
        model_name=safe_model_name,
        original_model_name=model_name,
        runs=sorted_runs,
        benchmark_metadata=benchmark_metadata,
    )


@app.route("/case/<benchmark_case_prefix>/<path:model_name>/<timestamp>")
def case_details(benchmark_case_prefix, model_name, timestamp):
    """Shows details for a specific benchmark run."""
    # Escape components for display, but use original for fetching data
    safe_case_prefix = escape(benchmark_case_prefix)
    safe_model_name = escape(model_name)
    safe_timestamp = escape(timestamp)

    benchmark_run_dir_maybe_none = current_app.config.get("BENCHMARK_RUN_DIR")
    if benchmark_run_dir_maybe_none is None:
        abort(500, "Benchmark run directory not configured.")
    benchmark_run_dir: str = (
        benchmark_run_dir_maybe_none  # Explicitly typed after check
    )
    prompts_dir = os.path.join(benchmark_run_dir, PROMPTS_SUBDIR)
    results_dir = os.path.join(benchmark_run_dir, RESULTS_SUBDIR)

    details = get_run_details(
        benchmark_case_prefix,
        model_name,  # Use original model name for path construction
        timestamp,
        prompts_dir,  # Pass derived prompts path
        results_dir,  # Pass derived results path
        benchmark_run_dir=benchmark_run_dir,  # Pass benchmark_run_dir explicitly
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


# Removed the entire orphaned get_plot_data function and its route decorator.


@app.route("/api/sliding-plot-data")
def get_sliding_plot_data():
    """
    Returns the sliding window plot data as JSON for the chart,
    optionally filtered by language.
    """
    # Get selected languages from query parameters
    languages_str = request.args.get("languages")
    selected_languages = (
        languages_str.split(",")
        if languages_str and languages_str.lower() != "null" and languages_str != ""
        else None
    )  # Handle empty/null string

    # We need benchmark metadata and results data to re-run analysis if filtered
    benchmark_metadata = current_app.config.get("BENCHMARK_METADATA")
    benchmark_run_dir = current_app.config.get("BENCHMARK_RUN_DIR")

    # Ensure metadata is loaded (should be loaded by index route first)
    if not benchmark_metadata:
        prompts_dir = os.path.join(benchmark_run_dir, PROMPTS_SUBDIR)
        current_app.config["BENCHMARK_METADATA"] = load_benchmark_metadata(prompts_dir)
        benchmark_metadata = current_app.config["BENCHMARK_METADATA"]

    if not benchmark_metadata or not benchmark_run_dir:
        print("Error: Benchmark metadata or run directory not available for API.")
        return jsonify({"error": "Benchmark data not loaded."}), 500

    # Re-scan results directory to get models and latest results data
    results_dir = os.path.join(benchmark_run_dir, RESULTS_SUBDIR)
    models_found, results_data = scan_results_directory(results_dir)

    if not models_found:
        print("Error: No models found in results directory for API.")
        return jsonify({"error": "No models found in results directory."}), 404

    # Re-run analysis with the selected languages
    try:
        print(
            f"Running filtered analysis for API with languages: {selected_languages}"
        )  # Debug log
        filtered_analysis = analyze_results(
            benchmark_metadata, models_found, results_data, selected_languages
        )
    except Exception as e:
        print(f"Error during filtered analysis for API: {e}")
        return jsonify({"error": f"Error during analysis: {e}"}), 500

    if (
        not filtered_analysis
        or "sliding_window" not in filtered_analysis
        or not filtered_analysis["sliding_window"]
    ):
        # If filtering resulted in no data, return empty structure
        print(
            "Filtered analysis resulted in no sliding window data for API."
        )  # Debug log
        return jsonify({"labels": [], "datasets": []})

    sliding_data = filtered_analysis["sliding_window"]
    models_data = sliding_data.get("models", {})
    window_centers_k = sliding_data.get("window_centers_k", [])

    # Format data for Chart.js
    labels = [f"{center}k" for center in window_centers_k]
    datasets = []

    # Get overall model costs from the initial (unfiltered) analysis stored in config
    # We show the *overall* cost, not cost filtered by language
    initial_analysis_results = current_app.config.get("ANALYSIS_RESULTS", {})
    overall_model_stats = (
        initial_analysis_results.get("models", {}) if initial_analysis_results else {}
    )

    # Sort models for consistent coloring based on the filtered results
    models_in_filtered_data = sorted(list(models_data.keys()))

    for model_name in models_in_filtered_data:
        model_window_data = models_data[model_name]
        rates = []
        wilson_lowers = []
        wilson_uppers = []
        totals = []
        successes = []

        # Ensure data aligns with window_centers_k
        window_centers_abs = [
            k * 1000 for k in window_centers_k
        ]  # Convert back for lookup
        for center_abs in window_centers_abs:
            stats = model_window_data.get(center_abs)
            if stats:
                rates.append(stats.get("rate"))  # Can be None if total is 0
                wilson_lowers.append(stats.get("wilson_lower"))
                wilson_uppers.append(stats.get("wilson_upper"))
                totals.append(stats.get("total"))
                successes.append(stats.get("successful"))
            else:
                # Append placeholders if no data for this center
                rates.append(None)
                wilson_lowers.append(None)
                wilson_uppers.append(None)
                totals.append(0)
                successes.append(0)

        # Get total cost for this model from the overall stats
        model_total_cost = overall_model_stats.get(model_name, {}).get(
            "total_cost_usd", 0.0
        )
        # Use .4f for cost precision as in the original unfiltered analysis display
        cost_str = f"{model_total_cost:.4f}" if model_total_cost is not None else "N/A"

        datasets.append(
            {
                "label": f"{model_name} (${cost_str})",  # Include overall cost in label
                "data": rates,
                "wilson_lower": wilson_lowers,
                "wilson_upper": wilson_uppers,
                "totals": totals,  # Add total counts for tooltip
                "successes": successes,  # Add success counts for tooltip
                "borderWidth": 2,  # Keep styling consistent
                "tension": 0.1,
                "fill": False,
                "spanGaps": True,
            }
        )

    return jsonify({"labels": labels, "datasets": datasets})


@app.route("/files/<path:filepath>")
def serve_file(filepath):
    """Serves files securely from allowed directories."""
    # Basic sanitization (Flask's path converter helps, but extra checks are good)
    if ".." in filepath or filepath.startswith("/"):
        abort(403, "Invalid file path.")

    # Define allowed base directories relative to app root (now within benchmark_run_dir)
    benchmark_run_dir_maybe_none = current_app.config.get("BENCHMARK_RUN_DIR")
    if benchmark_run_dir_maybe_none is None:
        abort(500, "Benchmark run directory not configured for file serving.")
    benchmark_run_dir: str = (
        benchmark_run_dir_maybe_none  # Explicitly typed after check
    )
    # Add assertion for pyright here as well
    assert benchmark_run_dir is not None, (
        "BENCHMARK_RUN_DIR cannot be None for file serving."
    )

    # Get the absolute path to the benchmark run directory
    run_dir_abs = os.path.abspath(benchmark_run_dir)
    allowed_subdirs_abs = [
        os.path.join(run_dir_abs, PROMPTS_SUBDIR),
        os.path.join(run_dir_abs, RESULTS_SUBDIR),
    ]

    # Construct the absolute path requested by the user, relative to the benchmark run dir
    # Assume filepath is relative to the benchmark_run_dir (e.g., "prompts/case_prompt.txt")
    requested_path_abs = os.path.abspath(os.path.join(run_dir_abs, filepath))

    # Security Check: Ensure the requested path is within one of the allowed subdirectories
    is_allowed = False  # Initialize before loop
    serving_directory = None
    filename = None
    for allowed_subdir in allowed_subdirs_abs:
        # Check if the requested path starts with the allowed subdirectory path + separator
        if os.path.normcase(requested_path_abs).startswith(
            os.path.normcase(allowed_subdir + os.sep)
        ):
            is_allowed = True
            serving_directory = allowed_subdir
            # Calculate filename relative to the serving subdirectory
            filename = os.path.relpath(requested_path_abs, allowed_subdir)
            # Double-check filename doesn't try to escape upwards
            if ".." in filename or filename.startswith(os.sep):
                is_allowed = False
                break
            break  # Found the allowed subdirectory

    if not is_allowed or serving_directory is None or filename is None:
        abort(
            403,
            "Access denied: File is outside allowed subdirectories (prompts/ or results/) or path calculation failed.",
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
        # The check `if serving_directory is None or filename is None:` happens before this try block.
        # Pyright should know they are strings here based on control flow.
        # Removed assertions and explicit casting as they didn't resolve the issue.

        # Use send_from_directory for safer serving
        # It requires the directory and the filename relative to that directory
        # print(f"Serving file: directory='{serving_directory}', filename='{filename}'") # Debugging
        # Use typing.cast after the check before the try block guarantees non-None
        return send_from_directory(
            typing.cast(str, serving_directory), typing.cast(str, filename), mimetype=mimetype, as_attachment=False
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
        benchmark_run_dir_maybe_none = current_app.config.get("BENCHMARK_RUN_DIR")
        if benchmark_run_dir_maybe_none is None:
            print(
                "Error: BENCHMARK_RUN_DIR not set in app config. Cannot run analysis.",
                file=sys.stderr,
            )
            current_app.config["ANALYSIS_RESULTS"] = None
            return
        # If we reach here, benchmark_run_dir_maybe_none is guaranteed to be a string.
        # If we reach here, benchmark_run_dir_maybe_none is guaranteed to be a string.
        # Use typing.cast for pyright clarity.
        checked_benchmark_run_dir = typing.cast(str, benchmark_run_dir_maybe_none)

        prompts_dir = os.path.join(checked_benchmark_run_dir, PROMPTS_SUBDIR)
        results_dir = os.path.join(checked_benchmark_run_dir, RESULTS_SUBDIR)

        print("\n--- Running Benchmark Data Analysis ---")
        print(f"Using prompts dir: {prompts_dir}")
        print(f"Using results dir: {results_dir}")

        benchmark_metadata = load_benchmark_metadata(prompts_dir)
        if not benchmark_metadata:
            print(
                f"Error: Benchmark metadata not found in {prompts_dir}. Cannot perform analysis.",
                file=sys.stderr,
            )
            current_app.config["ANALYSIS_RESULTS"] = None  # Ensure it's None
            return

        models_found, results_data = scan_results_directory(results_dir)
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
    # --- Argument Parsing for Direct Execution ---
    parser = argparse.ArgumentParser(
        description="Run the LoCoDiff Benchmark Explorer web app."
    )
    parser.add_argument(
        "--benchmark-run-dir",
        required=True,
        help="Path to the directory containing the benchmark run data (subdirectories: 'prompts/', 'results/').",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind the server to (default: 127.0.0.1). Use 0.0.0.0 for external access.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port number to run the server on (default: 5001).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode (enables auto-reloading).",
    )
    args = parser.parse_args()

    # --- Configure App ---
    app.config["BENCHMARK_RUN_DIR"] = args.benchmark_run_dir
    # Add assertion for pyright
    assert app.config["BENCHMARK_RUN_DIR"] is not None, "BENCHMARK_RUN_DIR must be set."
    # Ensure the static directory exists (needed for Chart.js etc.)
    static_folder = app.static_folder
    if static_folder is None:
        print("Error: Flask app static folder is not configured.", file=sys.stderr)
        sys.exit(1)
    assert isinstance(static_folder, str)  # Assure pyright it's a string path
    os.makedirs(static_folder, exist_ok=True)

    # --- Run Analysis and Start Server ---
    # Run data analysis before starting the server
    # This happens only once when the main process starts (not in reloader subprocess)
    if not args.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        run_data_analysis()

    # Open browser tab shortly after starting the server
    if not args.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        # Delay slightly to ensure server is likely up
        threading.Timer(1.5, lambda: open_browser(args.host, args.port)).start()

    # Run the Flask app
    app.run(debug=args.debug, host=args.host, port=args.port)
