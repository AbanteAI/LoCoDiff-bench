#!/usr/bin/env python3
"""
Generates GitHub Pages in the /docs directory based on benchmark results.

Purpose:
  This script analyzes benchmark results from a specified benchmark run directory
  and generates a static GitHub Pages site in the /docs directory to display
  summary statistics and performance metrics.

  The generated page includes:
  - Overall success rates across all models
  - Success rates by quartiles based on prompt size
  - Success rates by programming language
  - Cost information where available
  - Placeholder for future navigation to individual benchmark case details

Arguments:
  --benchmark-run-dir (required): Path to the directory containing benchmark run data
                                 (including 'prompts/' and 'results/' subdirectories).

Inputs:
  - Prompts and their metadata from `<benchmark_run_dir>/prompts/`
  - Results and their metadata from `<benchmark_run_dir>/results/`

Outputs:
  - Recreates the /docs directory (deleting any existing content)
  - Generates index.html with summary statistics tables
  - Adds a basic CSS file for styling

File Modifications:
  - Deletes and recreates the /docs directory and all its contents
  - Does not modify any files in the benchmark run directory
"""

import argparse
import copy
import glob
import json
import math
import os
import shutil
import sys
import yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set


# --- Helper Functions ---


def delete_and_recreate_dir(dir_path: Path) -> None:
    """Completely removes a directory and recreates it empty."""
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Recreated directory: {dir_path}")


def read_json_file(file_path: Path) -> Dict[str, Any]:
    """Reads a JSON file and returns the parsed content."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return {}


def determine_prompt_quartiles(
    prompt_tokens_list: List[int],
) -> List[Tuple[int, int]]:
    """
    Determines the token count ranges for each quartile.

    Args:
        prompt_tokens_list: List of token counts for all prompts

    Returns:
        List of tuples (min_tokens, max_tokens) for each quartile
    """
    if not prompt_tokens_list:
        return [(0, 0), (0, 0), (0, 0), (0, 0)]

    sorted_tokens = sorted(prompt_tokens_list)
    total_count = len(sorted_tokens)

    # Calculate quartile boundaries
    q1_idx = total_count // 4
    q2_idx = total_count // 2
    q3_idx = (3 * total_count) // 4

    # Handle edge cases for small lists
    if total_count < 4:
        return [(sorted_tokens[0], sorted_tokens[-1])] * 4

    # Create quartile ranges
    q1_range = (sorted_tokens[0], sorted_tokens[q1_idx])
    q2_range = (sorted_tokens[q1_idx], sorted_tokens[q2_idx])
    q3_range = (sorted_tokens[q2_idx], sorted_tokens[q3_idx])
    q4_range = (sorted_tokens[q3_idx], sorted_tokens[-1])

    return [q1_range, q2_range, q3_range, q4_range]


def determine_quartile(token_count: int, quartile_ranges: List[Tuple[int, int]]) -> int:
    """
    Determines which quartile a token count falls into.

    Args:
        token_count: The token count to classify
        quartile_ranges: List of (min, max) tuples for each quartile

    Returns:
        Quartile index (0-3) or -1 if not found
    """
    for i, (q_min, q_max) in enumerate(quartile_ranges):
        # First quartile includes both boundaries
        if i == 0 and q_min <= token_count <= q_max:
            return i
        # Middle quartiles have exclusive lower bound, inclusive upper bound
        elif i > 0 and q_min < token_count <= q_max:
            return i

    # Should not happen if quartile_ranges were calculated correctly
    print(f"Warning: Could not classify token count {token_count} into any quartile")
    return -1


def infer_language_from_filename(filename: str, ext_to_lang_map: Dict[str, str]) -> str:
    """
    Attempts to infer the programming language from a filename or path.

    Args:
        filename: The filename or path to analyze
        ext_to_lang_map: Dictionary mapping file extensions to language names

    Returns:
        The language name or "unknown"
    """
    # Extract extension if present
    if "." in filename:
        ext = "." + filename.split(".")[-1]
        if ext in ext_to_lang_map:
            return ext_to_lang_map[ext]

    # Look for known extension patterns in sanitized filenames
    for ext, lang in ext_to_lang_map.items():
        sanitized_ext = ext.replace(".", "_")
        if sanitized_ext in filename:
            return lang

    return "unknown"


def load_benchmark_config(config_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Loads the benchmark configuration including language mappings and model display names.

    Args:
        config_path: Path to the configuration YAML file

    Returns:
        Tuple containing:
        - Dictionary mapping file extensions to language names
        - Dictionary mapping original model names to display names
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Load language mappings
        ext_to_lang = {}
        for lang, settings in config["languages"].items():
            for ext in settings.get("extensions", []):
                ext_to_lang[ext] = lang

        # Load model display name mappings
        model_display_names = config["model_display_names"]

        return ext_to_lang, model_display_names
    except (yaml.YAMLError, IOError) as e:
        print(f"Error loading benchmark config from {config_path}: {e}")
        # Fallback to basic extensions and no model display names
        return {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".zig": "zig",
        }, {}


# --- Data Collection Functions ---


def collect_prompt_metadata(prompts_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Collects metadata for all prompts in the prompts directory.

    Args:
        prompts_dir: Path to the directory containing prompts and their metadata

    Returns:
        Dictionary mapping benchmark case prefixes to their metadata
    """
    prompt_metadata = {}
    metadata_path = prompts_dir / "metadata.json"

    if metadata_path.exists():
        try:
            metadata = read_json_file(metadata_path)
            if isinstance(metadata, list):
                for run_data in metadata:
                    if isinstance(run_data, dict):
                        cases_key = "benchmark_cases_added"
                        if cases_key not in run_data and "benchmark_cases" in run_data:
                            cases_key = "benchmark_cases"  # Legacy support

                        for case in run_data.get(cases_key, []):
                            if (
                                isinstance(case, dict)
                                and "benchmark_case_prefix" in case
                            ):
                                prefix = case["benchmark_case_prefix"]
                                prompt_metadata[prefix] = case
        except Exception as e:
            print(f"Error processing prompt metadata: {e}")

    # Fallback: If metadata.json is missing or invalid, scan the directory
    if not prompt_metadata:
        print("Falling back to directory scan for prompt files")
        prompt_files = glob.glob(str(prompts_dir / "*_prompt.txt"))
        for path in prompt_files:
            basename = os.path.basename(path)
            prefix = basename.replace("_prompt.txt", "")
            # Create minimal metadata
            prompt_metadata[prefix] = {
                "benchmark_case_prefix": prefix,
                "original_filename": basename,
            }

    return prompt_metadata


def collect_results_metadata(
    results_dir: Path, prompt_metadata: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
    """
    Collects result metadata for all benchmark cases and models.

    Args:
        results_dir: Path to the directory containing benchmark results
        prompt_metadata: Dictionary of prompt metadata by case prefix

    Returns:
        Tuple of:
        - Dictionary mapping (case_prefix, model) to result metadata
        - Set of all model names found
    """
    results_metadata = {}
    all_models = set()

    # Skip if results directory doesn't exist
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return {}, set()

    # Get all case directories
    case_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    for case_dir in case_dirs:
        case_prefix = case_dir.name

        # Skip if not in prompt metadata (could be old/invalid)
        if case_prefix not in prompt_metadata:
            print(f"Warning: Found results for unknown case: {case_prefix}")
            continue

        # Get all model directories for this case
        model_dirs = [d for d in case_dir.iterdir() if d.is_dir()]
        for model_dir in model_dirs:
            model_name = model_dir.name.replace("_", "/")  # Convert back from sanitized
            all_models.add(model_name)

            # Find the latest run for this case and model
            timestamp_dirs = sorted(
                [d for d in model_dir.iterdir() if d.is_dir()],
                key=lambda d: d.name,
                reverse=True,
            )

            if not timestamp_dirs:
                continue

            latest_dir = timestamp_dirs[0]
            metadata_path = latest_dir / "metadata.json"

            if metadata_path.exists():
                metadata = read_json_file(metadata_path)
                results_metadata[(case_prefix, model_name)] = metadata

    return results_metadata, all_models


# --- HTML Generation Functions ---


def create_html_header() -> str:
    """Creates the HTML header with basic metadata and CSS link."""
    warning = get_auto_generation_warning()
    return f"""<!DOCTYPE html>
<!--
{warning}
-->
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LoCoDiff Benchmark Results</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <h1>LoCoDiff Benchmark Results</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </header>
    <main>
"""


def create_html_footer(include_chart_js: bool = False) -> str:
    """
    Creates the HTML footer.

    Args:
        include_chart_js: Whether to include the chart JavaScript code.

    Returns:
        HTML string for the footer section
    """
    footer = """
    </main>
    <footer>
        <p>LoCoDiff-bench - <a href="https://github.com/AbanteAI/LoCoDiff-bench">GitHub Repository</a></p>
    </footer>
    """

    if include_chart_js:
        footer += create_chart_javascript()

    footer += """
</body>
</html>
"""
    return footer


def create_overall_stats_table(
    results_metadata: Dict[Any, Dict[str, Any]],
    all_models: Set[str],
    num_cases: int,
    model_display_names: Dict[str, str] = {},
) -> str:
    """
    Creates an HTML table showing overall statistics for each model.

    Args:
        results_metadata: Dictionary mapping (case_prefix, model) to result metadata
        all_models: Set of all model names
        num_cases: Total number of benchmark cases
        model_display_names: Optional dictionary mapping model names to display names

    Returns:
        HTML string containing the table
    """

    # Aggregate statistics by model
    model_stats = {}
    for (case_prefix, model), metadata in results_metadata.items():
        if model not in model_stats:
            model_stats[model] = {
                "total_attempts": 0,
                "successful": 0,
                "total_cost": 0.0,
            }

        model_stats[model]["total_attempts"] += 1
        if metadata.get("success", False):
            model_stats[model]["successful"] += 1

        cost = metadata.get("cost_usd")
        if cost is not None:
            model_stats[model]["total_cost"] += float(cost)

    # Create the HTML table
    html = """
    <section id="overall-stats">
        <h2>Overall Model Performance</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Success Rate</th>
                    <th>Cases Run</th>
                    <th>Total Cost</th>
                    <th>Avg Cost per Run</th>
                </tr>
            </thead>
            <tbody>
    """

    for model in sorted(all_models):
        stats = model_stats.get(
            model, {"total_attempts": 0, "successful": 0, "total_cost": 0.0}
        )
        attempts = stats["total_attempts"]
        success_rate = (stats["successful"] / attempts * 100) if attempts > 0 else 0
        total_cost = stats["total_cost"]
        avg_cost = total_cost / attempts if attempts > 0 else 0

        # Use display name if available
        display_name = model_display_names.get(model, model)

        html += f"""
                <tr>
                    <td>{display_name}</td>
                    <td>{success_rate:.2f}% ({stats["successful"]}/{attempts})</td>
                    <td>{attempts}/{num_cases} ({attempts / num_cases * 100:.2f}%)</td>
                    <td>${total_cost:.2f}</td>
                    <td>${avg_cost:.3f}</td>
                </tr>
        """

    html += """
            </tbody>
        </table>
    </section>
    """

    return html


def create_quartile_stats_table(
    results_metadata: Dict[Any, Dict[str, Any]],
    prompt_metadata: Dict[str, Dict[str, Any]],
    all_models: Set[str],
    model_display_names: Dict[str, str] = {},
) -> str:
    """
    Creates an HTML table showing success rates by prompt size quartiles.

    Args:
        results_metadata: Dictionary mapping (case_prefix, model) to result metadata
        prompt_metadata: Dictionary of prompt metadata by case prefix
        all_models: Set of all model names
        model_display_names: Optional dictionary mapping model names to display names

    Returns:
        HTML string containing the table
    """

    # Extract prompt token counts for quartile calculation
    prompt_tokens_list = []
    for case_prefix, metadata in prompt_metadata.items():
        token_count = metadata.get("prompt_tokens")
        if token_count is not None:
            prompt_tokens_list.append(token_count)

    # Determine quartile ranges
    quartile_ranges = determine_prompt_quartiles(prompt_tokens_list)
    quartile_labels = [
        f"Q1 ({quartile_ranges[0][0]}-{quartile_ranges[0][1]} tokens)",
        f"Q2 ({quartile_ranges[1][0]}-{quartile_ranges[1][1]} tokens)",
        f"Q3 ({quartile_ranges[2][0]}-{quartile_ranges[2][1]} tokens)",
        f"Q4 ({quartile_ranges[3][0]}-{quartile_ranges[3][1]} tokens)",
    ]

    # Aggregate statistics by model and quartile
    model_quartile_stats = {}
    for (case_prefix, model), result_metadata in results_metadata.items():
        if model not in model_quartile_stats:
            model_quartile_stats[model] = {
                i: {"attempts": 0, "successful": 0} for i in range(4)
            }

        # Get prompt token count from prompt metadata
        prompt_metadata_entry = prompt_metadata.get(case_prefix, {})
        token_count = prompt_metadata_entry.get("prompt_tokens")

        if token_count is None:
            # Try to get from result metadata if not in prompt metadata
            token_count = result_metadata.get("prompt_tokens")

        if token_count is not None:
            quartile = determine_quartile(token_count, quartile_ranges)
            if quartile >= 0:
                model_quartile_stats[model][quartile]["attempts"] += 1
                if result_metadata.get("success", False):
                    model_quartile_stats[model][quartile]["successful"] += 1

    # Create the HTML table
    html = """
    <section id="quartile-stats">
        <h2>Success Rates by Prompt Size Quartiles</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
    """

    for label in quartile_labels:
        html += f"<th>{label}</th>"

    html += """
                </tr>
            </thead>
            <tbody>
    """

    for model in sorted(all_models):
        # Use display name if available
        display_name = model_display_names.get(model, model)
        html += f"<tr><td>{display_name}</td>"

        quartile_stats = model_quartile_stats.get(model, {})
        for i in range(4):
            stats = quartile_stats.get(i, {"attempts": 0, "successful": 0})
            attempts = stats["attempts"]
            success_rate = (stats["successful"] / attempts * 100) if attempts > 0 else 0

            html += f"<td>{success_rate:.2f}% ({stats['successful']}/{attempts})</td>"

        html += "</tr>"

    html += """
            </tbody>
        </table>
    </section>
    """

    return html


def create_language_stats_table(
    results_metadata: Dict[Any, Dict[str, Any]],
    prompt_metadata: Dict[str, Dict[str, Any]],
    all_models: Set[str],
    ext_to_lang_map: Dict[str, str],
    model_display_names: Dict[str, str] = {},
) -> str:
    """
    Creates an HTML table showing success rates by programming language.

    Args:
        results_metadata: Dictionary mapping (case_prefix, model) to result metadata
        prompt_metadata: Dictionary of prompt metadata by case prefix
        all_models: Set of all model names
        ext_to_lang_map: Dictionary mapping file extensions to language names
        model_display_names: Optional dictionary mapping model names to display names

    Returns:
        HTML string containing the table
    """

    # First determine language for each case prefix
    case_languages = {}
    for case_prefix, metadata in prompt_metadata.items():
        # Try to get language from metadata
        language = metadata.get("language")

        # If not available, infer from filename
        if not language:
            filename = metadata.get("original_filename", case_prefix)
            language = infer_language_from_filename(filename, ext_to_lang_map)

        case_languages[case_prefix] = language

    # Count number of cases per language
    language_counts = defaultdict(int)
    for language in case_languages.values():
        language_counts[language] += 1

    # Aggregate statistics by model and language
    model_language_stats = {}
    for (case_prefix, model), result_metadata in results_metadata.items():
        if model not in model_language_stats:
            model_language_stats[model] = {}

        language = case_languages.get(case_prefix, "unknown")

        if language not in model_language_stats[model]:
            model_language_stats[model][language] = {"attempts": 0, "successful": 0}

        model_language_stats[model][language]["attempts"] += 1
        if result_metadata.get("success", False):
            model_language_stats[model][language]["successful"] += 1

    # Get all languages (sorted)
    all_languages = sorted(language_counts.keys())

    # Create the HTML table
    html = """
    <section id="language-stats">
        <h2>Success Rates by Programming Language</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
    """

    for language in all_languages:
        html += f"<th>{language} ({language_counts[language]})</th>"

    html += """
                </tr>
            </thead>
            <tbody>
    """

    for model in sorted(all_models):
        # Use display name if available
        display_name = model_display_names.get(model, model)
        html += f"<tr><td>{display_name}</td>"

        language_stats = model_language_stats.get(model, {})
        for language in all_languages:
            stats = language_stats.get(language, {"attempts": 0, "successful": 0})
            attempts = stats["attempts"]
            success_rate = (stats["successful"] / attempts * 100) if attempts > 0 else 0

            html += f"<td>{success_rate:.2f}% ({stats['successful']}/{attempts})</td>"

        html += "</tr>"

    html += """
            </tbody>
        </table>
    </section>
    """

    return html


def wilson_score_interval(
    successful: int, attempts: int, z: float = 1.96
) -> Tuple[float, float]:
    """
    Calculate Wilson score interval for a binomial proportion.

    This is used to compute confidence intervals for success rates.

    Args:
        successful: Number of successful attempts
        attempts: Total number of attempts
        z: Z-score for desired confidence level (default: 1.96 for 95% confidence)

    Returns:
        Tuple of (lower_bound, upper_bound) as proportions (not percentages)
    """
    if attempts == 0:
        return 0.0, 0.0

    # Observed proportion
    p_hat = successful / attempts

    # Wilson score calculation
    denominator = 1 + (z**2 / attempts)
    center = (p_hat + (z**2 / (2 * attempts))) / denominator
    interval = (
        z
        * math.sqrt((p_hat * (1 - p_hat) + (z**2 / (4 * attempts))) / attempts)
        / denominator
    )

    lower_bound = max(0.0, center - interval)
    upper_bound = min(1.0, center + interval)

    return lower_bound, upper_bound


def generate_chart_data(
    results_metadata: Dict[Any, Dict[str, Any]],
    prompt_metadata: Dict[str, Dict[str, Any]],
    all_models: Set[str],
    case_languages: Dict[str, str],
    model_display_names: Dict[str, str] = {},
) -> Dict[str, Any]:
    """
    Generates data for the token-based chart.

    Creates buckets based on case indices after sorting by prompt length:
    - Each bucket contains 25 cases
    - Buckets overlap and increment by 5 cases each
    - Bucket position is the average token count of all prompts in the bucket

    Args:
        results_metadata: Dictionary mapping (case_prefix, model) to result metadata
        prompt_metadata: Dictionary of prompt metadata by case prefix
        all_models: Set of all model names
        case_languages: Dictionary mapping case prefixes to language names
        model_display_names: Optional dictionary mapping model names to display names

    Returns:
        Dictionary containing the chart data
    """

    # Create a list of (case_prefix, token_count) sorted by token count
    case_token_pairs = []
    for case_prefix, metadata in prompt_metadata.items():
        token_count = metadata.get("prompt_tokens", 0)
        case_token_pairs.append((case_prefix, token_count))

    # Sort by token count
    case_token_pairs.sort(key=lambda pair: pair[1])

    # Define bucket parameters
    bucket_size = 30  # Number of cases in each bucket
    bucket_step = 10  # Increment between buckets

    # Initialize buckets
    buckets = []

    # Create buckets only if we have enough cases
    if len(case_token_pairs) >= bucket_size:
        num_buckets = 1 + (len(case_token_pairs) - bucket_size) // bucket_step

        for i in range(num_buckets):
            # Get the case range for this bucket
            start_idx = i * bucket_step
            end_idx = start_idx + bucket_size
            bucket_cases = case_token_pairs[start_idx:end_idx]

            # Calculate average token count for bucket position
            avg_token_count = sum(pair[1] for pair in bucket_cases) / len(bucket_cases)

            # Get the actual token range for this bucket
            min_tokens = min(pair[1] for pair in bucket_cases)
            max_tokens = max(pair[1] for pair in bucket_cases)

            # Calculate the bucket display value (for labels)
            bucket_location_k = round(avg_token_count / 1000, 1)

            # Create case set for quick membership testing
            case_set = {pair[0] for pair in bucket_cases}

            # Get the cases in this bucket with their languages
            bucket_case_languages = {
                case_prefix: case_languages.get(case_prefix, "unknown")
                for case_prefix in case_set
            }

            # Count cases per language in this bucket
            language_case_counts = defaultdict(int)
            for lang in bucket_case_languages.values():
                language_case_counts[lang] += 1

            # Initialize data for this bucket
            bucket_data = {
                "bucket_location": avg_token_count,
                "bucket_location_k": bucket_location_k,
                "bucket_min": min_tokens,
                "bucket_max": max_tokens,
                "bucket_range": f"{min_tokens // 1000}k-{max_tokens // 1000}k",
                "case_indices": f"{start_idx + 1}-{end_idx}",
                "models": {},
                "case_prefixes": case_set,  # Store set of cases in this bucket
                "language_case_counts": dict(
                    language_case_counts
                ),  # Store counts per language
            }

            # Initialize model data
            for model in all_models:
                bucket_data["models"][model] = {
                    "overall": {"attempts": 0, "successful": 0},
                    "languages": {},
                }

            buckets.append(bucket_data)

    # Populate buckets with result data
    for (case_prefix, model), result_metadata in results_metadata.items():
        # Get language for this case
        language = case_languages.get(case_prefix, "unknown")

        # Add this case to all buckets that contain it
        for bucket in buckets:
            if case_prefix in bucket["case_prefixes"]:
                # Initialize language if needed
                if language not in bucket["models"][model]["languages"]:
                    bucket["models"][model]["languages"][language] = {
                        "attempts": 0,
                        "successful": 0,
                    }

                # Add to counts
                bucket["models"][model]["overall"]["attempts"] += 1
                bucket["models"][model]["languages"][language]["attempts"] += 1

                if result_metadata.get("success", False):
                    bucket["models"][model]["overall"]["successful"] += 1
                    bucket["models"][model]["languages"][language]["successful"] += 1

    # Calculate confidence intervals for all data points
    for bucket in buckets:
        for model, model_data in bucket["models"].items():
            # Calculate confidence intervals for overall data
            successful = model_data["overall"]["successful"]
            attempts = model_data["overall"]["attempts"]
            lower, upper = wilson_score_interval(successful, attempts)
            model_data["overall"]["lower_bound"] = lower
            model_data["overall"]["upper_bound"] = upper

            # Calculate confidence intervals for each language
            for language_data in model_data["languages"].values():
                successful = language_data["successful"]
                attempts = language_data["attempts"]
                lower, upper = wilson_score_interval(successful, attempts)
                language_data["lower_bound"] = lower
                language_data["upper_bound"] = upper

    # Calculate all unique languages across all data
    all_languages = set()
    for bucket in buckets:
        for model_data in bucket["models"].values():
            for language in model_data["languages"].keys():
                all_languages.add(language)

    # Create model display name mapping for the chart
    model_display_map = {}
    for model in all_models:
        if model in model_display_names:
            model_display_map[model] = model_display_names[model]

    # Calculate max tokens from buckets or use default
    max_tokens_k = 75  # Default if no buckets
    if buckets:
        # Find the bucket with the highest average token count
        max_tokens = max(bucket["bucket_location"] for bucket in buckets)
        max_tokens_k = round(
            max_tokens / 1000, 1
        )  # Convert to k and round to 1 decimal

    # Create the chart data object
    chart_data = {
        "buckets": buckets,
        "models": list(sorted(all_models)),
        "model_display_names": model_display_map,  # Add model display names to chart data
        "languages": list(sorted(all_languages)),
        "min_tokens_k": 0,  # Always start at 0k
        "max_tokens_k": max_tokens_k,
    }

    return chart_data


def get_auto_generation_warning() -> str:
    """Returns a standard warning about auto-generated files."""
    return f"""
THIS FILE IS AUTOMATICALLY GENERATED BY benchmark_pipeline/3_generate_pages.py
DO NOT EDIT DIRECTLY - ANY CHANGES WILL BE OVERWRITTEN
Last generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""


def write_chart_data_to_file(chart_data: Dict[str, Any], output_dir: Path) -> None:
    """Writes chart data to a JSON file in the specified directory."""
    output_path = output_dir / "chart_data.json"
    try:
        # Process chart data to make it JSON serializable
        chart_data_copy = copy.deepcopy(chart_data)

        # Remove the case_prefixes set from each bucket as it's not JSON serializable
        # and only used internally for processing
        for bucket in chart_data_copy["buckets"]:
            if "case_prefixes" in bucket:
                del bucket["case_prefixes"]

        # Create a wrapper object that includes the warning
        wrapper = {
            "_warning": get_auto_generation_warning().strip(),
            "data": chart_data_copy,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, indent=2)
        print(f"Generated {output_path}")
    except IOError as e:
        print(f"Error writing chart data file: {e}")


def create_token_chart_section() -> str:
    """Creates an HTML section for the token-based chart."""
    return """
    <section id="token-chart">
        <h2>Success Rate by Prompt Size</h2>
        <div class="chart-controls">
            <div class="model-selection">
                <h3>Models</h3>
                <div id="model-checkboxes"></div>
            </div>
            <div class="language-selection">
                <h3>Languages</h3>
                <div id="language-checkboxes"></div>
            </div>
            <div class="display-options">
                <h3>Display Options</h3>
                <div class="checkbox-item">
                    <label>
                        <input type="checkbox" id="show-confidence-intervals" checked>
                        Show 95% Confidence Intervals
                    </label>
                </div>
            </div>
        </div>
        <div class="chart-container">
            <canvas id="token-success-chart"></canvas>
        </div>
    </section>
    """


def create_chart_javascript() -> str:
    """Creates JavaScript code for initializing and controlling the chart."""
    return """
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
// Load chart data
fetch('chart_data.json')
    .then(response => response.json())
    .then(wrapper => {
        // Use the data property which contains the actual chart data
        // The _warning property contains the auto-generation warning
        initializeChart(wrapper.data);
    });

// JavaScript implementation of Wilson score interval for confidence intervals
function wilson_score_interval(successful, attempts, z = 1.96) {
    if (attempts === 0) return [0.0, 0.0];
    
    // Observed proportion
    const p_hat = successful / attempts;
    
    // Wilson score calculation
    const denominator = 1 + (z * z / attempts);
    const center = (p_hat + (z * z / (2 * attempts))) / denominator;
    const interval = z * Math.sqrt((p_hat * (1 - p_hat) + (z * z / (4 * attempts))) / attempts) / denominator;
    
    const lower_bound = Math.max(0.0, center - interval);
    const upper_bound = Math.min(1.0, center + interval);
    
    return [lower_bound, upper_bound];
}

function initializeChart(chartData) {
    // Define chart colors
    const colors = [
        '#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f',
        '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab'
    ];
    
    // Create global variables for chart state that need to be accessed by callbacks
    let currentSelectedLanguages = [];
    let currentSelectedModels = [];
    
    // Get canvas context
    const ctx = document.getElementById('token-success-chart').getContext('2d');
    
    // Create a fixed color mapping for all models first
    const modelColorMap = {};
    chartData.models.forEach((model, index) => {
        modelColorMap[model] = colors[index % colors.length];
    });
    
    // Create model checkboxes using the fixed color mapping
    const modelCheckboxes = document.getElementById('model-checkboxes');
    chartData.models.forEach((model) => {
        const color = modelColorMap[model]; // Use consistent color from the mapping
        // Use display name if available, otherwise use original model name
        const displayName = chartData.model_display_names && chartData.model_display_names[model] 
            ? chartData.model_display_names[model] 
            : model;
        
        const checkbox = document.createElement('div');
        checkbox.className = 'checkbox-item';
        checkbox.innerHTML = `
            <label>
                <input type="checkbox" data-model="${model}" checked>
                <span class="checkbox-color" style="background-color: ${color};"></span>
                ${displayName}
            </label>
        `;
        modelCheckboxes.appendChild(checkbox);
    });
    
    // Create language checkboxes
    const languageCheckboxes = document.getElementById('language-checkboxes');
    chartData.languages.forEach(language => {
        const checkbox = document.createElement('div');
        checkbox.className = 'checkbox-item';
        checkbox.innerHTML = `
            <label>
                <input type="checkbox" data-language="${language}" checked>
                ${language}
            </label>
        `;
        languageCheckboxes.appendChild(checkbox);
    });
    
    // Create chart
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    // Simpler approach with basic configuration
                    ticks: {
                        precision: 1,
                        callback: function(value) {
                            return value + 'k';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Prompt Token Length (k)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Success Rate (%)'
                    },
                    min: 0,
                    max: 100
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'LoCoDiff: Natural Long Context Code Bench',
                    font: {
                        size: 18
                    },
                    padding: {
                        top: 10,
                        bottom: 30
                    }
                },
                legend: {
                    labels: {
                        // Custom filter function to exclude datasets with display: false from the legend
                        filter: (legendItem, data) => {
                            const dataset = data.datasets[legendItem.datasetIndex];
                            return dataset && dataset.display !== false;
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            // Safety check - make sure context and dataset exist
                            if (!context || !context.dataset || !context.dataset.label) {
                                return null;
                            }
                            
                            // Don't show tooltips for confidence interval datasets
                            if (context.dataset.label.includes('CI')) {
                                return null;
                            }
                            
                            try {
                                // Get display name (which is the label) and the original model name for data lookup
                                const displayName = context.dataset.label;
                                // Use originalModel property we added to dataset for data lookup
                                const originalModel = context.dataset.originalModel || displayName;
                                
                                // With x,y coordinates, we need to find the bucket by x value (token count)
                                // rather than by index
                                const xValue = context.raw.x; // This is the token count in thousands
                                
                                // Find the bucket with matching token count
                                const bucketData = chartData.buckets.find(bucket => 
                                    Math.abs((bucket.bucket_location / 1000) - xValue) < 0.01
                                );
                                
                                // Safety check for bucket
                                if (!bucketData) {
                                    return [`${displayName}`];
                                }
                                
                                // Safety check for model data
                                if (!bucketData.models || !bucketData.models[originalModel]) {
                                    return [`${displayName}: No data available`];
                                }
                                
                                const modelData = bucketData.models[originalModel];
                                
                                // Get basic stats - with x,y coordinate objects, y is the success rate
                                const successRate = context.raw.y;
                                
                                // Use filtered data if available, otherwise fall back to overall data
                                let successful, attempts;
                                
                                // Check if we have filtered data for this bucket and model
                                if (bucketData.filteredData && bucketData.filteredData[originalModel]) {
                                    successful = bucketData.filteredData[originalModel].successful;
                                    attempts = bucketData.filteredData[originalModel].attempts;
                                } else {
                                    // Fall back to overall data (before filtering)
                                    successful = modelData.overall.successful;
                                    attempts = modelData.overall.attempts;
                                }
                                
                                // Get confidence interval
                                let ciInfo = '';
                                const ciElement = document.getElementById('show-confidence-intervals');
                                if (ciElement && ciElement.checked && attempts > 0) {
                                    // Use the filtered counts we already have for CI calculation
                                    const [lower, upper] = wilson_score_interval(successful, attempts);
                                    ciInfo = `\n95% CI: ${(lower * 100).toFixed(2)}% - ${(upper * 100).toFixed(2)}%`;
                                }
                                
                                // Add case indices information
                                const caseInfo = bucketData.case_indices ? `\nCases: ${bucketData.case_indices}` : '';
                                
                                // Calculate filtered bucket size based on selected languages
                                let filteredBucketSize = 30; // Default size
                                if (bucketData.language_case_counts && currentSelectedLanguages.length > 0) {
                                    filteredBucketSize = 0;
                                    currentSelectedLanguages.forEach(language => {
                                        filteredBucketSize += bucketData.language_case_counts[language] || 0;
                                    });
                                }
                                
                                // If we have a filtered size of 0, use the default
                                if (filteredBucketSize === 0) {
                                    filteredBucketSize = 30;
                                }
                                
                                // Calculate untested cases
                                const untestedCases = filteredBucketSize - attempts;
                                let untestedInfo = '';
                                if (untestedCases > 0) {
                                    untestedInfo = `\nModel did not return result for ${untestedCases} case${untestedCases > 1 ? 's' : ''} in this bucket`;
                                }
                                
                                return [
                                    `${displayName}: ${successRate !== null && successRate !== undefined ? successRate.toFixed(2) : 'N/A'}% (${successful}/${filteredBucketSize})`,
                                    `Token Range: ${bucketData.bucket_range} (avg: ${(bucketData.bucket_location/1000).toFixed(1)}k)${caseInfo}${untestedInfo}${ciInfo}`
                                ];
                            } catch (error) {
                                console.error('Error in tooltip callback:', error);
                                return ['Error displaying tooltip'];
                            }
                        }
                    }
                }
            }
        }
    });
    
    // Function to update chart based on selected models and languages
    function updateChart() {
        // Get selected models and languages
        currentSelectedModels = Array.from(document.querySelectorAll('input[data-model]:checked'))
            .map(checkbox => checkbox.getAttribute('data-model'));
        
        currentSelectedLanguages = Array.from(document.querySelectorAll('input[data-language]:checked'))
            .map(checkbox => checkbox.getAttribute('data-language'));
        
        // Clear filteredData from previous filter selections
        chartData.buckets.forEach(bucket => {
            bucket.filteredData = {};
        });
        
        // Clear current datasets
        chart.data.datasets = [];
        
        // Create datasets for each selected model
        currentSelectedModels.forEach((model) => {
            // Use the consistent color from our color map
            const color = modelColorMap[model];
            
            // Calculate data points and store filtered data for tooltip
            const dataPoints = chartData.buckets.map((bucket, bucketIndex) => {
                const modelData = bucket.models[model];
                
                // Filter by selected languages
                let successful = 0;
                let attempts = 0;
                
                if (currentSelectedLanguages.length === 0) {
                    // No languages selected, show empty chart (consistent with model selection behavior)
                    return null;
                } else {
                    // Use only selected languages
                    currentSelectedLanguages.forEach(language => {
                        if (modelData.languages[language]) {
                            successful += modelData.languages[language].successful;
                            attempts += modelData.languages[language].attempts;
                        }
                    });
                }
                
                // Store filtered data for this bucket and model for tooltip access
                if (!bucket.filteredData) {
                    bucket.filteredData = {};
                }
                if (!bucket.filteredData[model]) {
                    bucket.filteredData[model] = {};
                }
                
                bucket.filteredData[model] = {
                    successful: successful,
                    attempts: attempts
                };
                
                // Calculate the filtered bucket size based on selected languages
                let filteredBucketSize = 0;
                if (bucket.language_case_counts) {
                    currentSelectedLanguages.forEach(language => {
                        filteredBucketSize += bucket.language_case_counts[language] || 0;
                    });
                }
                // Use filtered size as denominator if available, otherwise use default bucket size
                const denominator = filteredBucketSize > 0 ? filteredBucketSize : 30;
                
                // Calculate success rate - divide by appropriate denominator
                // This assumes any prompt not tested was a failure
                const y = successful > 0 ? (successful / denominator * 100) : 0;
                // Return x,y coordinates where x is the bucket location in thousands
                return {
                    x: bucket.bucket_location / 1000, // Convert to k
                    y: y
                };
            });
            
            // Calculate confidence interval data points if languages are selected
            let lowerBoundPoints = null;
            let upperBoundPoints = null;
            
            if (currentSelectedLanguages.length > 0) {
                lowerBoundPoints = chartData.buckets.map(bucket => {
                    const modelData = bucket.models[model];
                    
                    // Use only selected languages
                    let langSuccessful = 0;
                    
                    currentSelectedLanguages.forEach(language => {
                        if (modelData.languages[language]) {
                            langSuccessful += modelData.languages[language].successful;
                        }
                    });
                    
                    // Calculate bucket size based on selected languages
                    let filteredBucketSize = 0;
                    if (bucket.language_case_counts) {
                        currentSelectedLanguages.forEach(language => {
                            filteredBucketSize += bucket.language_case_counts[language] || 0;
                        });
                    }
                    // Use filtered size as denominator if available, otherwise use default bucket size
                    const denominator = filteredBucketSize > 0 ? filteredBucketSize : 30;
                    
                    // Recalculate Wilson interval using the appropriate denominator
                    const [lower, upper] = wilson_score_interval(langSuccessful, denominator);
                    // Return x,y coordinates for lower bound
                    return {
                        x: bucket.bucket_location / 1000, // Convert to k
                        y: lower * 100 // Convert to percentage
                    };
                });
                
                upperBoundPoints = chartData.buckets.map(bucket => {
                    const modelData = bucket.models[model];
                    
                    // Use only selected languages
                    let langSuccessful = 0;
                    
                    currentSelectedLanguages.forEach(language => {
                        if (modelData.languages[language]) {
                            langSuccessful += modelData.languages[language].successful;
                        }
                    });
                    
                    // Calculate bucket size based on selected languages
                    let filteredBucketSize = 0;
                    if (bucket.language_case_counts) {
                        currentSelectedLanguages.forEach(language => {
                            filteredBucketSize += bucket.language_case_counts[language] || 0;
                        });
                    }
                    // Use filtered size as denominator if available, otherwise use default bucket size
                    const denominator = filteredBucketSize > 0 ? filteredBucketSize : 30;
                    
                    // Recalculate Wilson interval using the appropriate denominator
                    const [lower, upper] = wilson_score_interval(langSuccessful, denominator);
                    // Return x,y coordinates for upper bound
                    return {
                        x: bucket.bucket_location / 1000, // Convert to k
                        y: upper * 100 // Convert to percentage
                    };
                });
            }
            
            // Get display name if available
            const displayName = chartData.model_display_names && chartData.model_display_names[model] 
                ? chartData.model_display_names[model] 
                : model;
                
            // Add main dataset
            chart.data.datasets.push({
                label: displayName,
                originalModel: model, // Store original model name for data lookup
                data: dataPoints,
                borderColor: color,
                backgroundColor: color + '33',
                fill: false,
                tension: 0.1,
                pointRadius: 4,
                pointHoverRadius: 6,
                parsing: false  // Tell Chart.js we're using x,y objects directly
            });
            
            // Add confidence interval datasets if enabled
            const showConfidenceIntervals = document.getElementById('show-confidence-intervals').checked;
            
            if (showConfidenceIntervals && currentSelectedLanguages.length > 0) {
                // Add lower bound line first (needed for reference by the area dataset)
                const lowerBoundIndex = chart.data.datasets.length;
                chart.data.datasets.push({
                    label: `${displayName} (95% CI lower)`,
                    data: lowerBoundPoints,
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    tension: 0.1,
                    fill: false,
                    parsing: false,  // Tell Chart.js we're using x,y objects directly
                    showLine: false, // Don't draw a line for this dataset
                    display: false   // Completely exclude from legend
                });
                
                // Add confidence interval area
                chart.data.datasets.push({
                    label: `${displayName} (95% CI)`,
                    data: upperBoundPoints,
                    borderColor: 'transparent',
                    backgroundColor: color + '22', // Very transparent version of the line color
                    pointRadius: 0,
                    tension: 0.1,
                    parsing: false,  // Tell Chart.js we're using x,y objects directly
                    fill: lowerBoundIndex, // Fill to the specific dataset index (the lower bound)
                    display: false         // Completely exclude from legend
                });
            }
        });
        
        // Update chart
        chart.update();
    }
    
    // Add event listeners to model and language checkboxes
    document.querySelectorAll('input[data-model], input[data-language]').forEach(checkbox => {
        checkbox.addEventListener('change', updateChart);
    });
    
    // Add event listener to confidence interval checkbox
    document.getElementById('show-confidence-intervals').addEventListener('change', updateChart);
    
    // Initial chart update
    updateChart();
}
</script>
"""


def create_cases_section(
    all_models: Set[str], model_display_names: Dict[str, str] = {}
) -> str:
    """Creates a section with links to model-specific benchmark case pages."""
    html = """
    <section id="explore-benchmarks">
        <h2>Explore Benchmark Prompts and Model Outputs</h2>
        <p>Select a model below to view its benchmark cases:</p>
        <ul class="model-list">
    """

    # Add links to model pages
    for model in sorted(all_models):
        # Create a safe filename for the model (sanitized)
        safe_model = model.replace("/", "_")
        # Use display name if available
        display_name = model_display_names.get(model, model)
        html += f"""
            <li>
                <a href="models/{safe_model}.html" class="model-link">
                    {display_name}
                </a>
            </li>
        """

    html += """
        </ul>
    </section>
    """
    return html


def generate_prompt_page(
    case_prefix: str,
    model: str,
    prompt_content: str,
    original_filename: str,
    docs_dir: Path,
    model_display_names: Dict[str, str] = {},
) -> None:
    """
    Generates a page displaying just the prompt content.

    Args:
        case_prefix: The benchmark case prefix
        model: The model name
        prompt_content: The prompt content to display
        original_filename: The original filename of the case
        docs_dir: Path to the docs directory
        model_display_names: Optional mapping of model names to display names
    """
    safe_model = model.replace("/", "_")
    safe_case = case_prefix.replace("/", "_")
    content_dir = docs_dir / "content" / safe_model / safe_case
    content_dir.mkdir(parents=True, exist_ok=True)

    prompt_page_path = content_dir / "prompt.html"
    display_name = model_display_names.get(model, model)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt: {original_filename} - {display_name}</title>
    <link rel="stylesheet" href="../../../../styles.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
</head>
<body>
    <header>
        <h1>Prompt: {original_filename}</h1>
        <p><a href="../../../cases/{safe_model}/{safe_case}.html">← Back to Case</a> | <a href="../../../index.html">Home</a></p>
    </header>
    <main>
        <section>
            <h2>Prompt Content</h2>
            <pre><code class="language-plaintext">{prompt_content}</code></pre>
        </section>
    </main>
    <footer>
        <p>LoCoDiff-bench - <a href="https://github.com/AbanteAI/LoCoDiff-bench">GitHub Repository</a></p>
    </footer>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            hljs.highlightAll();
        }});
    </script>
</body>
</html>
    """

    with open(prompt_page_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def generate_expected_output_page(
    case_prefix: str,
    model: str,
    expected_output: str,
    original_filename: str,
    docs_dir: Path,
    model_display_names: Dict[str, str] = {},
) -> None:
    """
    Generates a page displaying just the expected output content.

    Args:
        case_prefix: The benchmark case prefix
        model: The model name
        expected_output: The expected output content to display
        original_filename: The original filename of the case
        docs_dir: Path to the docs directory
        model_display_names: Optional mapping of model names to display names
    """
    safe_model = model.replace("/", "_")
    safe_case = case_prefix.replace("/", "_")
    content_dir = docs_dir / "content" / safe_model / safe_case
    content_dir.mkdir(parents=True, exist_ok=True)

    expected_page_path = content_dir / "expected.html"
    display_name = model_display_names.get(model, model)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Expected Output: {original_filename} - {display_name}</title>
    <link rel="stylesheet" href="../../../../styles.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
</head>
<body>
    <header>
        <h1>Expected Output: {original_filename}</h1>
        <p><a href="../../../cases/{safe_model}/{safe_case}.html">← Back to Case</a> | <a href="../../../index.html">Home</a></p>
    </header>
    <main>
        <section>
            <h2>Expected Output Content</h2>
            <pre><code class="language-plaintext">{expected_output}</code></pre>
        </section>
    </main>
    <footer>
        <p>LoCoDiff-bench - <a href="https://github.com/AbanteAI/LoCoDiff-bench">GitHub Repository</a></p>
    </footer>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            hljs.highlightAll();
        }});
    </script>
</body>
</html>
    """

    with open(expected_page_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def generate_actual_output_page(
    case_prefix: str,
    model: str,
    raw_response: str,
    original_filename: str,
    docs_dir: Path,
    model_display_names: Dict[str, str] = {},
    success: bool = False,
    expected_output: str = "",
    extracted_output: str = "",
) -> None:
    """
    Generates a page displaying the raw model response.

    Args:
        case_prefix: The benchmark case prefix
        model: The model name
        raw_response: The raw model response to display
        original_filename: The original filename of the case
        docs_dir: Path to the docs directory
        model_display_names: Optional mapping of model names to display names
        success: Whether the run was successful (matched expected output)
        expected_output: The expected output content (used as fallback)
        extracted_output: The extracted code from the raw response (used as fallback)
    """
    safe_model = model.replace("/", "_")
    safe_case = case_prefix.replace("/", "_")
    content_dir = docs_dir / "content" / safe_model / safe_case
    content_dir.mkdir(parents=True, exist_ok=True)

    actual_page_path = content_dir / "actual.html"
    display_name = model_display_names.get(model, model)

    # Handle output content based on available content and success status
    content_section = ""

    # If raw response is available, show it (with success notice if applicable)
    if raw_response and raw_response.strip():
        success_note = ""
        if success:
            success_note = """
            <div class="success-message">
                <p>✓ This model's extracted output matched the expected output exactly</p>
            </div>
            """

        content_section = f"""
        <section>
            <h2>Raw Model Response</h2>
            {success_note}
            <pre><code class="language-plaintext">{raw_response}</code></pre>
        </section>
        """
    # If no raw response but have extracted output, show it as fallback
    elif extracted_output and extracted_output.strip():
        success_note = "❌ This output did not match the expected output"
        if success:
            success_note = "✓ This output matched the expected output exactly"

        content_section = f"""
        <section>
            <h2>Extracted Model Output</h2>
            <div class="info-message">
                <p>Showing extracted code output (raw response unavailable)</p>
                <p>{success_note}</p>
            </div>
            <pre><code class="language-plaintext">{extracted_output}</code></pre>
        </section>
        """
    # For successful runs with no available outputs but expected output exists
    elif success and expected_output and expected_output.strip():
        content_section = f"""
        <section>
            <h2>Expected Output (Fallback)</h2>
            <div class="success-message">
                <p>✓ The model output matched this expected output exactly</p>
                <p>Raw model response unavailable - showing expected output as they are identical</p>
            </div>
            <pre><code class="language-plaintext">{expected_output}</code></pre>
        </section>
        """
    # For empty outputs, show appropriate message based on success status
    else:
        if success:
            content_section = """
            <section>
                <h2>Model Response</h2>
                <div class="success-message">
                    <p>✓ Model output matched expected output exactly</p>
                    <p>The raw response file is not available, but the benchmark was marked as successful.</p>
                </div>
            </section>
            """
        else:
            content_section = """
            <section>
                <h2>Model Response</h2>
                <div class="empty-content-notice">
                    <p>No model response available</p>
                    <p>This could be because the model failed to generate a response or the response files are missing.</p>
                </div>
            </section>
            """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Actual Output: {original_filename} - {display_name}</title>
    <link rel="stylesheet" href="../../../../styles.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <style>
        .empty-content-notice {{
            background-color: #f8f8f8;
            border: 1px dashed #ccc;
            border-radius: 4px;
            padding: 20px;
            text-align: center;
            color: #666;
        }}
        
        .empty-content-notice p:first-child {{
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .info-message {{
            background-color: #f1f8ff;
            border: 1px solid #c8e1ff;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 15px;
            color: #0366d6;
        }}
        
        .info-message p:first-child {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Actual Output: {original_filename}</h1>
        <p><a href="../../../cases/{safe_model}/{safe_case}.html">← Back to Case</a> | <a href="../../../index.html">Home</a></p>
    </header>
    <main>
        {content_section}
    </main>
    <footer>
        <p>LoCoDiff-bench - <a href="https://github.com/AbanteAI/LoCoDiff-bench">GitHub Repository</a></p>
    </footer>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            hljs.highlightAll();
        }});
    </script>
</body>
</html>
    """

    with open(actual_page_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def generate_model_page(
    model: str,
    prompt_metadata: Dict[str, Dict[str, Any]],
    results_metadata: Dict[Any, Dict[str, Any]],
    benchmark_run_dir: Path,
    docs_dir: Path,
    model_display_names: Dict[str, str] = {},
) -> None:
    """
    Generates a page for a specific model showing all its benchmark cases.

    Args:
        model: The model name
        prompt_metadata: Dictionary of prompt metadata by case prefix
        results_metadata: Dictionary mapping (case_prefix, model) to result metadata
        benchmark_run_dir: Path to the benchmark run directory
        docs_dir: Path to the docs directory
        model_display_names: Optional mapping of model names to display names
    """
    # Create the models directory if it doesn't exist
    models_dir = docs_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Get the sanitized model name for the filename
    safe_model = model.replace("/", "_")
    model_page_path = models_dir / f"{safe_model}.html"

    # Get display name if available
    display_name = model_display_names.get(model, model)

    # Collect cases for this model
    model_cases = []
    for (case_prefix, case_model), metadata in results_metadata.items():
        if case_model == model:
            # Get case metadata
            case_data = {
                "prefix": case_prefix,
                "success": metadata.get("success", False),
                "runtime_seconds": metadata.get("runtime_seconds", 0),
                "cost_usd": metadata.get("cost_usd", 0),
                "prompt_tokens": prompt_metadata.get(case_prefix, {}).get(
                    "prompt_tokens", 0
                ),
                "output_tokens": metadata.get("output_tokens", 0),
                "original_filename": prompt_metadata.get(case_prefix, {}).get(
                    "original_filename", case_prefix
                ),
            }
            model_cases.append(case_data)

    # Sort cases by prompt tokens (ascending)
    model_cases.sort(key=lambda x: x["prompt_tokens"])

    # Create page content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{display_name} - Benchmark Cases</title>
    <link rel="stylesheet" href="../styles.css">
</head>
<body>
    <header>
        <h1>{display_name} - Benchmark Cases</h1>
        <p><a href="../index.html">← Back to Overview</a></p>
    </header>
    <main>
        <section>
            <h2>All Benchmark Cases</h2>
            <p>{len(model_cases)} cases sorted by prompt token size (smallest to largest)</p>
            
            <div class="case-filter">
                <label>
                    <input type="checkbox" id="show-successful" checked> Show Successful
                </label>
                <label>
                    <input type="checkbox" id="show-failed" checked> Show Failed
                </label>
            </div>
            
            <table id="cases-table">
                <thead>
                    <tr>
                        <th>Case</th>
                        <th>Prompt Tokens</th>
                        <th>Status</th>
                        <th>Runtime</th>
                        <th>Cost</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Add case rows
    for case in model_cases:
        status_class = "success" if case["success"] else "failure"
        status_text = "Success" if case["success"] else "Failure"

        # Create a safe case page filename
        safe_case = case["prefix"].replace("/", "_")

        html_content += f"""
                    <tr class="case-row {status_class}">
                        <td>{case["original_filename"]}</td>
                        <td>{case["prompt_tokens"]}</td>
                        <td class="{status_class}">{status_text}</td>
                        <td>{case["runtime_seconds"]:.2f}s</td>
                        <td>${case["cost_usd"]:.6f}</td>
                        <td>
                            <a href="../cases/{safe_model}/{safe_case}.html" class="view-button">View Details</a>
                        </td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
        </section>
    </main>
    <footer>
        <p>LoCoDiff-bench - <a href="https://github.com/AbanteAI/LoCoDiff-bench">GitHub Repository</a></p>
    </footer>
    
    <script>
        // Add filtering functionality
        document.addEventListener('DOMContentLoaded', function() {
            const showSuccessfulCheckbox = document.getElementById('show-successful');
            const showFailedCheckbox = document.getElementById('show-failed');
            
            function updateFilters() {
                const showSuccessful = showSuccessfulCheckbox.checked;
                const showFailed = showFailedCheckbox.checked;
                
                document.querySelectorAll('.case-row').forEach(row => {
                    if (row.classList.contains('success')) {
                        row.style.display = showSuccessful ? '' : 'none';
                    } else {
                        row.style.display = showFailed ? '' : 'none';
                    }
                });
            }
            
            showSuccessfulCheckbox.addEventListener('change', updateFilters);
            showFailedCheckbox.addEventListener('change', updateFilters);
        });
    </script>
</body>
</html>
    """

    # Write the page
    with open(model_page_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def generate_case_page(
    case_prefix: str,
    model: str,
    prompt_metadata: Dict[str, Dict[str, Any]],
    results_metadata: Dict[Any, Dict[str, Any]],
    benchmark_run_dir: Path,
    docs_dir: Path,
    model_display_names: Dict[str, str] = {},
) -> None:
    """
    Generates a page for a specific benchmark case and model.

    Args:
        case_prefix: The benchmark case prefix
        model: The model name
        prompt_metadata: Dictionary of prompt metadata by case prefix
        results_metadata: Dictionary mapping (case_prefix, model) to result metadata
        benchmark_run_dir: Path to the benchmark run directory
        docs_dir: Path to the docs directory
        model_display_names: Optional mapping of model names to display names
    """
    # Create the cases directory structure if it doesn't exist
    safe_model = model.replace("/", "_")
    cases_dir = docs_dir / "cases" / safe_model
    cases_dir.mkdir(parents=True, exist_ok=True)

    # Get the sanitized case name for the filename
    safe_case = case_prefix.replace("/", "_")
    case_page_path = cases_dir / f"{safe_case}.html"

    # Get display name if available
    display_name = model_display_names.get(model, model)

    # Get metadata
    result_metadata = results_metadata.get((case_prefix, model), {})
    case_metadata = prompt_metadata.get(case_prefix, {})

    # Get file paths
    original_filename = case_metadata.get("original_filename", case_prefix)
    prompt_file_path = benchmark_run_dir / "prompts" / f"{case_prefix}_prompt.txt"
    expected_output_path = (
        benchmark_run_dir / "prompts" / f"{case_prefix}_expectedoutput.txt"
    )

    # Determine the paths for result files
    timestamp_dirs = []
    case_result_dir = benchmark_run_dir / "results" / case_prefix / safe_model
    if case_result_dir.exists():
        timestamp_dirs = sorted(
            [d for d in case_result_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )

    actual_output_path = None
    diff_file_path = None
    if timestamp_dirs:
        actual_output_path = timestamp_dirs[0] / "response.txt"
        diff_file_path = timestamp_dirs[0] / "output.diff"

    # Read file contents
    prompt_content = ""
    if prompt_file_path.exists():
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()

    expected_output = ""
    if expected_output_path.exists():
        with open(expected_output_path, "r", encoding="utf-8") as f:
            expected_output = f.read()

    # For actual output, try to find both raw response and extracted output
    actual_output = ""
    raw_response = ""

    # Look for the raw response file first
    if timestamp_dirs:
        raw_response_path = timestamp_dirs[0] / "raw_response.txt"
        if raw_response_path.exists():
            try:
                with open(raw_response_path, "r", encoding="utf-8") as f:
                    raw_response = f.read()
            except Exception as e:
                print(f"Error reading raw response file {raw_response_path}: {e}")

    # Then try to get the extracted output (the code from backticks)
    if actual_output_path and actual_output_path.exists():
        try:
            with open(actual_output_path, "r", encoding="utf-8") as f:
                actual_output = f.read()
        except Exception as e:
            print(f"Error reading actual output file {actual_output_path}: {e}")

    # Get success status
    success = result_metadata.get("success", False)

    # Generate content-specific pages
    generate_prompt_page(
        case_prefix,
        model,
        prompt_content,
        original_filename,
        docs_dir,
        model_display_names,
    )
    generate_expected_output_page(
        case_prefix,
        model,
        expected_output,
        original_filename,
        docs_dir,
        model_display_names,
    )
    generate_actual_output_page(
        case_prefix,
        model,
        raw_response,  # Use raw response as primary content
        original_filename,
        docs_dir,
        model_display_names,
        success=success,
        expected_output=expected_output,
        extracted_output=actual_output,  # Pass extracted output as fallback
    )

    # Load and format the precomputed diff file
    diff_content = ""
    if success:
        # For successful runs, show a success message instead of a diff
        diff_content = '<div class="success-message"><p>✓ No differences found (successful run)</p><p>Expected output matches the model output exactly.</p></div>'
    elif diff_file_path and diff_file_path.exists():
        try:
            # Read the precomputed diff file
            with open(diff_file_path, "r", encoding="utf-8") as f:
                diff_text = f.read()

            # Format the diff with syntax highlighting
            if diff_text:
                lines = diff_text.split("\n")
                html_lines = []

                for line in lines:
                    # Special handling for file header lines
                    if line.startswith("---") or line.startswith("+++"):
                        html_lines.append(f'<div class="diff-header">{line}</div>')
                    elif line.startswith("+"):
                        html_lines.append(f'<div class="diff-added">{line}</div>')
                    elif line.startswith("-"):
                        html_lines.append(f'<div class="diff-removed">{line}</div>')
                    elif line.startswith("@"):
                        html_lines.append(f'<div class="diff-info">{line}</div>')
                    else:
                        html_lines.append(f"<div>{line}</div>")

                diff_content = f'<pre class="diff">{"".join(html_lines)}</pre>'
            else:
                diff_content = '<div class="no-diff-message"><p>No diff content available for this case.</p></div>'
        except Exception as e:
            print(f"Error reading diff file {diff_file_path}: {e}")
            diff_content = f'<div class="no-diff-message"><p>Error loading diff file: {str(e)}</p></div>'
    else:
        # No diff file found
        diff_content = '<div class="no-diff-message"><p>No diff file found for this case.</p></div>'

    # Get status
    status_class = "success" if success else "failure"
    status_text = "Success" if success else "Failure"

    # Create the main case page content with links instead of embedded content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Case: {original_filename} - {display_name}</title>
    <link rel="stylesheet" href="../../styles.css">
</head>
<body>
    <header>
        <h1>Case: {original_filename}</h1>
        <p><a href="../../models/{safe_model}.html">← Back to {display_name} Cases</a> | <a href="../../index.html">Home</a></p>
    </header>
    <main>
        <section class="case-details">
            <div class="case-info">
                <h2>Benchmark Case Information</h2>
                <p><strong>Model:</strong> {display_name}</p>
                <p><strong>Status:</strong> <span class="{status_class}">{status_text}</span></p>
                <p><strong>Prompt Tokens:</strong> {case_metadata.get("prompt_tokens", "N/A")}</p>
                <p><strong>Output Tokens:</strong> {result_metadata.get("output_tokens", "N/A")}</p>
                <p><strong>Native Prompt Tokens:</strong> {result_metadata.get("native_prompt_tokens", "N/A")}</p>
                <p><strong>Native Completion Tokens:</strong> {result_metadata.get("native_completion_tokens", "N/A")}</p>
                <p><strong>Native Tokens Reasoning:</strong> {result_metadata.get("native_tokens_reasoning", "N/A")}</p>
                <p><strong>Native Finish Reason:</strong> {result_metadata.get("native_finish_reason", "N/A")}</p>
                <p><strong>Runtime:</strong> {result_metadata.get("runtime_seconds", "N/A")}s</p>
                <p><strong>Cost:</strong> ${result_metadata.get("cost_usd", "N/A")}</p>
            </div>
            
            <div class="content-links">
                <h2>View Content</h2>
                <ul>
                    <li><a href="../../content/{safe_model}/{safe_case}/prompt.html" class="content-link">View Prompt</a></li>
                    <li><a href="../../content/{safe_model}/{safe_case}/expected.html" class="content-link">View Expected Output</a></li>
                    <li><a href="../../content/{safe_model}/{safe_case}/actual.html" class="content-link">View Actual Output</a></li>
                </ul>
            </div>
            
            <div class="diff-section">
                <h2>Diff (Expected vs Actual)</h2>
                <div id="diff-output">
                    {diff_content}
                </div>
            </div>
        </section>
    </main>
    <footer>
        <p>LoCoDiff-bench - <a href="https://github.com/AbanteAI/LoCoDiff-bench">GitHub Repository</a></p>
    </footer>
</body>
</html>
    """

    # Write the page
    with open(case_page_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def create_css_file() -> str:
    """Creates a basic CSS stylesheet for the GitHub Pages site."""
    warning = get_auto_generation_warning()
    warning_comment = f"/*\n{warning}\n*/\n\n"

    css_content = """/* Basic Reset */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* General Styling */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

a {
    color: #0366d6;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Header */
header {
    margin-bottom: 30px;
    border-bottom: 1px solid #eaecef;
    padding-bottom: 10px;
}

header h1 {
    font-size: 32px;
    margin-bottom: 10px;
}

/* Sections */
section {
    margin-bottom: 40px;
}

section h2 {
    margin-bottom: 15px;
    border-bottom: 1px solid #eaecef;
    padding-bottom: 5px;
}

section h3 {
    margin-bottom: 10px;
    font-size: 18px;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
    font-size: 14px;
}

th, td {
    padding: 8px 12px;
    text-align: left;
    border: 1px solid #e1e4e8;
}

th {
    background-color: #f6f8fa;
    font-weight: 600;
}

tbody tr:nth-child(odd) {
    background-color: #f6f8fa;
}

tbody tr:hover {
    background-color: #f0f4f8;
}

/* Chart Section */
.chart-controls {
    display: flex;
    flex-wrap: wrap;
    margin-bottom: 20px;
    gap: 30px;
}

.model-selection, .language-selection, .display-options {
    flex: 1;
    min-width: 200px;
}

.checkbox-item {
    margin-bottom: 8px;
}

.checkbox-item label {
    display: flex;
    align-items: center;
    cursor: pointer;
}

.checkbox-item input[type="checkbox"] {
    margin-right: 8px;
}

.checkbox-color {
    display: inline-block;
    width: 12px;
    height: 12px;
    margin-right: 8px;
    border-radius: 2px;
}

.chart-container {
    width: 100%;
    height: 400px;
    margin-bottom: 30px;
}

/* Explore Benchmarks Section */
.model-list {
    list-style-type: none;
    padding: 0;
    margin: 20px 0;
}

.model-list li {
    margin-bottom: 10px;
}

.model-link {
    display: inline-block;
    padding: 10px 15px;
    background-color: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 4px;
    font-size: 14px;
    font-weight: 500;
    color: #0366d6;
    text-decoration: none;
    transition: background-color 0.2s;
    width: 300px;
}

.model-link:hover {
    background-color: #e1e4e8;
    text-decoration: none;
}

/* Case status indicators */
.success {
    color: #22863a;
    font-weight: bold;
}

.failure {
    color: #cb2431;
    font-weight: bold;
}

tr.success:hover, tr.failure:hover {
    background-color: #f0f4f8;
}

/* Case filter controls */
.case-filter {
    margin-bottom: 15px;
    display: flex;
    gap: 20px;
}

.case-filter label {
    display: flex;
    align-items: center;
    cursor: pointer;
}

.case-filter input[type="checkbox"] {
    margin-right: 8px;
}

/* View button */
.view-button {
    display: inline-block;
    padding: 5px 10px;
    background-color: #0366d6;
    color: white;
    border-radius: 4px;
    font-size: 12px;
    text-decoration: none;
    transition: background-color 0.2s;
}

.view-button:hover {
    background-color: #0255b3;
    text-decoration: none;
}

/* Case detail styles */
.case-details {
    margin-bottom: 30px;
}

.case-info {
    background-color: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 4px;
    padding: 15px;
    margin-bottom: 20px;
}

/* Content links section */
.content-links {
    margin: 20px 0;
}

.content-links h2 {
    margin-bottom: 15px;
}

.content-links ul {
    list-style-type: none;
    padding: 0;
}

.content-links li {
    margin-bottom: 10px;
}

.content-link {
    display: inline-block;
    padding: 10px 15px;
    background-color: #0366d6;
    color: white;
    border-radius: 4px;
    text-decoration: none;
    transition: background-color 0.2s;
    width: 250px;
    text-align: center;
}

.content-link:hover {
    background-color: #0255b3;
    text-decoration: none;
}

/* Diff section */
.diff-section {
    margin-top: 30px;
}

.diff-section h2 {
    margin-bottom: 15px;
}

#diff-output {
    background-color: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 4px;
    padding: 15px;
    overflow-x: auto;
}

/* Success and error messages in diff section */
.success-message {
    background-color: #e6ffec;
    color: #22863a;
    padding: 15px;
    border: 1px solid #22863a;
    border-radius: 4px;
    text-align: center;
}

.success-message p:first-child {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 5px;
}

.no-diff-message {
    background-color: #fffbdd;
    color: #735c0f;
    padding: 15px;
    border: 1px solid #d9d0a5;
    border-radius: 4px;
    text-align: center;
}

/* Diff formatting */
.diff {
    font-family: monospace;
    white-space: pre;
    font-size: 14px;
    line-height: 1.5;
    overflow-x: auto;
}

.diff-added {
    background-color: #e6ffec;
    color: #22863a;
}

.diff-removed {
    background-color: #ffebe9;
    color: #cb2431;
}

.diff-info {
    color: #6a737d;
    background-color: #f1f8ff;
}

.diff-header {
    color: #24292e;
    background-color: #f6f8fa;
    font-weight: bold;
}

/* Footer */
footer {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #eaecef;
    color: #586069;
    font-size: 14px;
    text-align: center;
}"""

    return warning_comment + css_content


# --- Main Function ---


def main():
    parser = argparse.ArgumentParser(
        description="Generate GitHub Pages for benchmark results."
    )
    parser.add_argument(
        "--benchmark-run-dir",
        type=Path,
        required=True,
        help="Path to the directory containing benchmark run data.",
    )

    args = parser.parse_args()
    benchmark_run_dir = args.benchmark_run_dir

    print("--- Starting GitHub Pages Generation ---")
    print(f"Benchmark Run Directory: {benchmark_run_dir}")

    # Define paths
    prompts_dir = benchmark_run_dir / "prompts"
    results_dir = benchmark_run_dir / "results"
    docs_dir = Path("docs")

    # Load benchmark configuration (languages and model display names)
    config_path = Path("benchmark_pipeline/benchmark_config.yaml")
    ext_to_lang_map, model_display_names = load_benchmark_config(str(config_path))

    if model_display_names:
        print(f"Loaded {len(model_display_names)} model display name mappings")

    # Check for required directories
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        print(f"Error: Prompts directory not found: {prompts_dir}")
        return 1

    # Delete and recreate docs directory
    delete_and_recreate_dir(docs_dir)

    # Collect metadata
    print("Collecting prompt metadata...")
    prompt_metadata = collect_prompt_metadata(prompts_dir)
    num_cases = len(prompt_metadata)
    print(f"Found {num_cases} benchmark cases")

    print("Collecting results metadata...")
    results_metadata, all_models = collect_results_metadata(
        results_dir, prompt_metadata
    )
    print(
        f"Found results for {len(results_metadata)} case-model combinations across {len(all_models)} models"
    )

    # Validate that all models have display names defined
    missing_models = [model for model in all_models if model not in model_display_names]
    if missing_models:
        print(
            "Error: The following models do not have display names defined in benchmark_config.yaml:"
        )
        for model in sorted(missing_models):
            print(f'  - "{model}"')
        print(
            "\nPlease add these models to the model_display_names section in benchmark_pipeline/benchmark_config.yaml"
        )
        return 1

    # Determine language for each case prefix
    print("Determining languages for benchmark cases...")
    case_languages = {}
    for case_prefix, metadata in prompt_metadata.items():
        # Try to get language from metadata
        language = metadata.get("language")

        # If not available, infer from filename
        if not language:
            filename = metadata.get("original_filename", case_prefix)
            language = infer_language_from_filename(filename, ext_to_lang_map)

        case_languages[case_prefix] = language

    # Generate chart data
    print("Generating chart data...")
    chart_data = generate_chart_data(
        results_metadata,
        prompt_metadata,
        all_models,
        case_languages,
        model_display_names,
    )

    # Write chart data to file
    print("Writing chart data file...")
    write_chart_data_to_file(chart_data, docs_dir)

    # Generate HTML content
    print("Generating HTML content...")
    html_content = create_html_header()
    html_content += create_token_chart_section()
    html_content += create_overall_stats_table(
        results_metadata, all_models, num_cases, model_display_names
    )
    html_content += create_quartile_stats_table(
        results_metadata, prompt_metadata, all_models, model_display_names
    )
    html_content += create_language_stats_table(
        results_metadata,
        prompt_metadata,
        all_models,
        ext_to_lang_map,
        model_display_names,
    )
    html_content += create_cases_section(all_models, model_display_names)
    html_content += create_html_footer(include_chart_js=True)

    # Generate model pages and case pages
    print("Generating model and case pages...")

    # Create model pages
    for model in all_models:
        print(f"Generating page for model: {model}")
        generate_model_page(
            model,
            prompt_metadata,
            results_metadata,
            benchmark_run_dir,
            docs_dir,
            model_display_names,
        )

        # Create case pages for this model
        model_case_count = 0
        for (case_prefix, case_model), metadata in results_metadata.items():
            if case_model == model:
                print(f"  - Generating case page: {case_prefix}")
                generate_case_page(
                    case_prefix,
                    model,
                    prompt_metadata,
                    results_metadata,
                    benchmark_run_dir,
                    docs_dir,
                    model_display_names,
                )
                model_case_count += 1

        print(f"Generated {model_case_count} case pages for model {model}")

    # Write HTML file
    index_path = docs_dir / "index.html"
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Generated {index_path}")
    except IOError as e:
        print(f"Error writing HTML file: {e}")
        return 1

    # Write CSS file
    css_path = docs_dir / "styles.css"
    try:
        with open(css_path, "w", encoding="utf-8") as f:
            f.write(create_css_file())
        print(f"Generated {css_path}")
    except IOError as e:
        print(f"Error writing CSS file: {e}")
        return 1

    print("--- GitHub Pages Generation Complete ---")
    print(f"Generated files in {docs_dir}")
    print("After committing these changes, they will be available on GitHub Pages")

    return 0


if __name__ == "__main__":
    sys.exit(main())
