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
                    <td>${total_cost:.6f}</td>
                    <td>${avg_cost:.6f}</td>
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

    Creates data points at 1k token increments from 0k to 75k,
    with each point representing a +/- 10k token bucket.
    Each prompt is included in all buckets whose range covers its length.

    Args:
        results_metadata: Dictionary mapping (case_prefix, model) to result metadata
        prompt_metadata: Dictionary of prompt metadata by case prefix
        all_models: Set of all model names
        case_languages: Dictionary mapping case prefixes to language names
        model_display_names: Optional dictionary mapping model names to display names

    Returns:
        Dictionary containing the chart data
    """

    # Find max token count (min is always 0)
    token_counts = [meta.get("prompt_tokens", 0) for meta in prompt_metadata.values()]
    max_tokens = 75000  # Default max
    if token_counts:
        max_tokens = max(
            max(token_counts), max_tokens
        )  # Use higher of actual max or default

    # Round max tokens up to the nearest thousand
    max_tokens_k = (max_tokens + 999) // 1000

    # Initialize data structure - start from 0k, go to max_tokens_k
    buckets = []
    for i in range(0, max_tokens_k + 1):  # Start from 0k
        bucket_location = i * 1000  # Renamed from bucket_center to bucket_location
        bucket_min = max(0, bucket_location - 10000)
        bucket_max = bucket_location + 10000

        # Initialize data for this bucket
        bucket_data = {
            "bucket_location": bucket_location,  # Renamed from bucket_center
            "bucket_location_k": i,  # Renamed from bucket_center_k
            "bucket_min": bucket_min,
            "bucket_max": bucket_max,
            "bucket_range": f"{bucket_min // 1000}k-{bucket_max // 1000}k",
            "models": {},
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
        # Get token count for this case
        token_count = prompt_metadata.get(case_prefix, {}).get("prompt_tokens", 0)

        # Get language for this case
        language = case_languages.get(case_prefix, "unknown")

        # Add this case to all buckets that cover its token count
        for bucket in buckets:
            if bucket["bucket_min"] <= token_count <= bucket["bucket_max"]:
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

                # Note: No break here, so we continue to add this case to all matching buckets

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
        # Create a wrapper object that includes the warning
        wrapper = {
            "_warning": get_auto_generation_warning().strip(),
            "data": chart_data,
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
            labels: chartData.buckets.map(bucket => bucket.bucket_location_k + 'k'),
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
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
                                
                                // Safety check for dataIndex
                                if (context.dataIndex === undefined || !chartData.buckets[context.dataIndex]) {
                                    return [`${displayName}`];
                                }
                                
                                const bucketData = chartData.buckets[context.dataIndex];
                                
                                // Safety check for model data
                                if (!bucketData.models || !bucketData.models[originalModel]) {
                                    return [`${displayName}: No data available`];
                                }
                                
                                const modelData = bucketData.models[originalModel];
                                
                                // Get basic stats
                                const successRate = context.raw;
                                
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
                                
                                return [
                                    `${displayName}: ${successRate !== null && successRate !== undefined ? successRate.toFixed(2) : 'N/A'}% (${successful}/${attempts})`,
                                    `Token Range: ${bucketData.bucket_range}${ciInfo}`
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
        const selectedModels = Array.from(document.querySelectorAll('input[data-model]:checked'))
            .map(checkbox => checkbox.getAttribute('data-model'));
        
        const selectedLanguages = Array.from(document.querySelectorAll('input[data-language]:checked'))
            .map(checkbox => checkbox.getAttribute('data-language'));
        
        // Clear filteredData from previous filter selections
        chartData.buckets.forEach(bucket => {
            bucket.filteredData = {};
        });
        
        // Clear current datasets
        chart.data.datasets = [];
        
        // Create datasets for each selected model
        selectedModels.forEach((model) => {
            // Use the consistent color from our color map
            const color = modelColorMap[model];
            
            // Calculate data points and store filtered data for tooltip
            const dataPoints = chartData.buckets.map((bucket, bucketIndex) => {
                const modelData = bucket.models[model];
                
                // Filter by selected languages
                let successful = 0;
                let attempts = 0;
                
                if (selectedLanguages.length === 0) {
                    // No languages selected, show empty chart (consistent with model selection behavior)
                    return null;
                } else {
                    // Use only selected languages
                    selectedLanguages.forEach(language => {
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
                
                // Calculate success rate
                return attempts > 0 ? (successful / attempts * 100) : null;
            });
            
            // Calculate confidence interval data points if languages are selected
            let lowerBoundPoints = null;
            let upperBoundPoints = null;
            
            if (selectedLanguages.length > 0) {
                lowerBoundPoints = chartData.buckets.map(bucket => {
                    const modelData = bucket.models[model];
                    
                    // Use only selected languages
                    let langSuccessful = 0;
                    let langAttempts = 0;
                    
                    selectedLanguages.forEach(language => {
                        if (modelData.languages[language]) {
                            langSuccessful += modelData.languages[language].successful;
                            langAttempts += modelData.languages[language].attempts;
                        }
                    });
                    
                    if (langAttempts > 0) {
                        // Recalculate Wilson interval for the combined languages
                        const [lower, upper] = wilson_score_interval(langSuccessful, langAttempts);
                        return lower * 100; // Convert to percentage
                    }
                    return null;
                });
                
                upperBoundPoints = chartData.buckets.map(bucket => {
                    const modelData = bucket.models[model];
                    
                    // Use only selected languages
                    let langSuccessful = 0;
                    let langAttempts = 0;
                    
                    selectedLanguages.forEach(language => {
                        if (modelData.languages[language]) {
                            langSuccessful += modelData.languages[language].successful;
                            langAttempts += modelData.languages[language].attempts;
                        }
                    });
                    
                    if (langAttempts > 0) {
                        // Recalculate Wilson interval for the combined languages
                        const [lower, upper] = wilson_score_interval(langSuccessful, langAttempts);
                        return upper * 100; // Convert to percentage
                    }
                    return null;
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
                pointHoverRadius: 6
            });
            
            // Add confidence interval datasets if enabled
            const showConfidenceIntervals = document.getElementById('show-confidence-intervals').checked;
            
            if (showConfidenceIntervals && selectedLanguages.length > 0) {
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


def create_cases_section(all_models: Set[str], model_display_names: Dict[str, str] = {}) -> str:
    """Creates a section with links to model-specific benchmark case pages."""
    html = """
    <section id="individual-cases">
        <h2>Individual Benchmark Cases</h2>
        <p>Select a model below to view its benchmark cases:</p>
        <div class="model-links">
    """
    
    # Add links to model pages
    for model in sorted(all_models):
        # Create a safe filename for the model (sanitized)
        safe_model = model.replace("/", "_")
        # Use display name if available
        display_name = model_display_names.get(model, model)
        html += f"""
            <a href="models/{safe_model}.html" class="model-link-button">
                {display_name}
            </a>
        """
    
    html += """
        </div>
    </section>
    """
    return html


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
                "prompt_tokens": prompt_metadata.get(case_prefix, {}).get("prompt_tokens", 0),
                "output_tokens": metadata.get("output_tokens", 0),
                "original_filename": prompt_metadata.get(case_prefix, {}).get("original_filename", case_prefix),
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
    expected_output_path = benchmark_run_dir / "prompts" / f"{case_prefix}_expectedoutput.txt"
    
    # Determine the paths for result files
    timestamp_dirs = []
    case_result_dir = benchmark_run_dir / "results" / case_prefix / safe_model
    if case_result_dir.exists():
        timestamp_dirs = sorted(
            [d for d in case_result_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True
        )
    
    actual_output_path = None
    if timestamp_dirs:
        actual_output_path = timestamp_dirs[0] / "response.txt"
    
    # Read file contents
    prompt_content = ""
    if prompt_file_path.exists():
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
    
    expected_output = ""
    if expected_output_path.exists():
        with open(expected_output_path, "r", encoding="utf-8") as f:
            expected_output = f.read()
    
    actual_output = ""
    if actual_output_path and actual_output_path.exists():
        with open(actual_output_path, "r", encoding="utf-8") as f:
            actual_output = f.read()
    
    # Get status
    success = result_metadata.get("success", False)
    status_class = "success" if success else "failure"
    status_text = "Success" if success else "Failure"
    
    # Create page content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Case: {original_filename} - {display_name}</title>
    <link rel="stylesheet" href="../../styles.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/diff/5.1.0/diff.min.js"></script>
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
                <p><strong>Runtime:</strong> {result_metadata.get("runtime_seconds", "N/A")}s</p>
                <p><strong>Cost:</strong> ${result_metadata.get("cost_usd", "N/A")}</p>
            </div>
            
            <div class="tabs">
                <div class="tab-buttons">
                    <button class="tab-button active" data-tab="prompt">Prompt</button>
                    <button class="tab-button" data-tab="expected">Expected Output</button>
                    <button class="tab-button" data-tab="actual">Actual Output</button>
                    <button class="tab-button" data-tab="diff">Diff</button>
                </div>
                
                <div class="tab-content active" id="prompt-tab">
                    <h3>Prompt</h3>
                    <pre><code class="language-plaintext">{prompt_content}</code></pre>
                </div>
                
                <div class="tab-content" id="expected-tab">
                    <h3>Expected Output</h3>
                    <pre><code class="language-plaintext">{expected_output}</code></pre>
                </div>
                
                <div class="tab-content" id="actual-tab">
                    <h3>Actual Output</h3>
                    <pre><code class="language-plaintext">{actual_output}</code></pre>
                </div>
                
                <div class="tab-content" id="diff-tab">
                    <h3>Diff (Expected vs Actual)</h3>
                    <div id="diff-output"></div>
                </div>
            </div>
        </section>
    </main>
    <footer>
        <p>LoCoDiff-bench - <a href="https://github.com/AbanteAI/LoCoDiff-bench">GitHub Repository</a></p>
    </footer>
    
    <script>
        // Initialize tabs
        document.addEventListener('DOMContentLoaded', function() {
            const tabButtons = document.querySelectorAll('.tab-button');
            const tabContents = document.querySelectorAll('.tab-content');
            
            tabButtons.forEach(button => {
                button.addEventListener('click', () => {
                    // Remove active class from all buttons and contents
                    tabButtons.forEach(btn => btn.classList.remove('active'));
                    tabContents.forEach(content => content.classList.remove('active'));
                    
                    // Add active class to clicked button and corresponding content
                    button.classList.add('active');
                    const tabId = button.getAttribute('data-tab');
                    document.getElementById(`${tabId}-tab`).classList.add('active');
                    
                    // If diff tab, generate diff if not already done
                    if (tabId === 'diff' && !document.getElementById('diff-output').innerHTML) {
                        generateDiff();
                    }
                });
            });
            
            // Initialize highlight.js for code highlighting
            hljs.highlightAll();
        });
        
        // Generate diff between expected and actual outputs
        function generateDiff() {
            const expectedOutput = document.querySelector('#expected-tab code').textContent;
            const actualOutput = document.querySelector('#actual-tab code').textContent;
            
            // Create a diff using the diff library
            const diff = Diff.createTwoFilesPatch('expected', 'actual', expectedOutput, actualOutput);
            
            // Format the diff for display
            const formattedDiff = formatDiff(diff);
            document.getElementById('diff-output').innerHTML = formattedDiff;
        }
        
        // Format the diff with syntax highlighting
        function formatDiff(diff) {
            if (!diff) return '<pre>No difference</pre>';
            
            const lines = diff.split('\\n');
            let html = '<pre class="diff">';
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                let className = '';
                
                if (line.startsWith('+')) {
                    className = 'diff-added';
                } else if (line.startsWith('-')) {
                    className = 'diff-removed';
                } else if (line.startsWith('@')) {
                    className = 'diff-info';
                }
                
                html += `<div class="${{className}}">${{escapeHtml(line)}}</div>`;
            }
            
            html += '</pre>';
            return html;
        }
        
        // Helper function to escape HTML
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
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

/* Individual Cases Section */
.model-links {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 20px;
}

.model-link-button {
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
}

.model-link-button:hover {
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

/* Tabs for case details */
.tabs {
    margin-top: 20px;
}

.tab-buttons {
    display: flex;
    border-bottom: 1px solid #e1e4e8;
    margin-bottom: 15px;
}

.tab-button {
    background: none;
    border: none;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}

.tab-button:hover {
    background-color: #f6f8fa;
}

.tab-button.active {
    border-bottom: 2px solid #0366d6;
    color: #0366d6;
}

.tab-content {
    display: none;
    padding: 15px 0;
}

.tab-content.active {
    display: block;
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
    html_content += create_token_chart_section()
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
            model_display_names
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
                    model_display_names
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
