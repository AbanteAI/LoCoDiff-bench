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
    Generates raw case data for the token-based chart.

    Instead of pre-calculating buckets, provides the raw case data for JavaScript
    to process with dynamic bucketing based on user selection.

    Args:
        results_metadata: Dictionary mapping (case_prefix, model) to result metadata
        prompt_metadata: Dictionary of prompt metadata by case prefix
        all_models: Set of all model names
        case_languages: Dictionary mapping case prefixes to language names
        model_display_names: Optional dictionary mapping model names to display names

    Returns:
        Dictionary containing the raw case data for chart generation
    """

    # Create a list of case data entries with token counts and results
    cases = []
    for case_prefix, case_metadata in prompt_metadata.items():
        token_count = case_metadata.get("prompt_tokens", 0)
        language = case_languages.get(case_prefix, "unknown")

        # Create case entry with basic info
        case_entry = {
            "prefix": case_prefix,
            "token_count": token_count,
            "language": language,
            "results": {},
        }

        # Add results for each model
        for model in all_models:
            result_key = (case_prefix, model)
            if result_key in results_metadata:
                result_metadata = results_metadata[result_key]
                case_entry["results"][model] = {
                    "success": result_metadata.get("success", False),
                    "attempted": True,
                }
            else:
                case_entry["results"][model] = {"success": False, "attempted": False}

        cases.append(case_entry)

    # Sort cases by token count
    cases.sort(key=lambda case: case["token_count"])

    # Calculate all unique languages
    all_languages = set(case_languages.values())

    # Create model display name mapping for the chart
    model_display_map = {}
    for model in all_models:
        if model in model_display_names:
            model_display_map[model] = model_display_names[model]

    # Calculate max tokens or use default
    max_tokens_k = 75  # Default
    if cases:
        # Find the case with the highest token count
        max_tokens = max(case["token_count"] for case in cases)
        max_tokens_k = round(
            max_tokens / 1000, 1
        )  # Convert to k and round to 1 decimal

    # Create the chart data object
    chart_data = {
        "cases": cases,
        "models": list(sorted(all_models)),
        "model_display_names": model_display_map,
        "languages": list(sorted(all_languages)),
        "min_tokens_k": 0,  # Always start at 0k
        "max_tokens_k": max_tokens_k,
        "default_bucket_count": 5,  # Default number of buckets
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
            <div class="bucketing-options">
                <h3>Bucketing Options</h3>
                <div class="bucket-count-control">
                    <label for="bucket-count">Number of Buckets: <span id="bucket-count-display">5</span></label>
                    <input type="range" id="bucket-count" min="1" max="20" value="5" step="1">
                </div>
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
    let currentBucketCount = chartData.default_bucket_count || 5;
    let buckets = []; // Will be calculated dynamically
    
    // Get canvas context
    const ctx = document.getElementById('token-success-chart').getContext('2d');
    
    // Create model checkboxes
    const modelCheckboxes = document.getElementById('model-checkboxes');
    chartData.models.forEach((model) => {
        // Use display name if available, otherwise use original model name
        const displayName = chartData.model_display_names && chartData.model_display_names[model] 
            ? chartData.model_display_names[model] 
            : model;
        
        const checkbox = document.createElement('div');
        checkbox.className = 'checkbox-item';
        checkbox.innerHTML = `
            <label>
                <input type="checkbox" data-model="${model}" checked>
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
    
    // Setup bucket count slider
    const bucketCountSlider = document.getElementById('bucket-count');
    const bucketCountDisplay = document.getElementById('bucket-count-display');
    
    // Make sure the elements exist before trying to access them
    if (bucketCountSlider && bucketCountDisplay) {
        bucketCountSlider.value = currentBucketCount;
        bucketCountDisplay.textContent = currentBucketCount;
        
        bucketCountSlider.addEventListener('input', function() {
            currentBucketCount = parseInt(this.value);
            bucketCountDisplay.textContent = currentBucketCount;
            calculateBuckets();
            updateChart();
        });
    } else {
        console.warn('Bucket count slider elements not found. Check that the HTML includes the proper elements.');
    }
    
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
                    min: 0,
                    max: chartData.max_tokens_k,
                    // We'll override ticks in updateChart
                    ticks: {
                        // Empty default ticks configuration - will be set dynamically
                    },
                    title: {
                        display: true,
                        text: 'Prompt Token Length (k)'
                    },
                    grid: {
                        // Make grid lines match our ticks
                        z: -1 // Draw grid lines behind the data
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
                        filter: (legendItem, data) => {
                            const dataset = data.datasets[legendItem.datasetIndex];
                            return dataset && dataset.display !== false;
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (!context || !context.dataset || !context.dataset.label) {
                                return null;
                            }
                            
                            if (context.dataset.label.includes('CI')) {
                                return null;
                            }
                            
                            try {
                                const displayName = context.dataset.label;
                                const originalModel = context.dataset.originalModel || displayName;
                                const xValue = context.raw.x; // Token count in thousands
                                
                                // Find the bucket with matching token count
                                const bucketData = buckets.find(bucket => 
                                    Math.abs((bucket.avgTokens / 1000) - xValue) < 0.01
                                );
                                
                                if (!bucketData) {
                                    return [`${displayName}`];
                                }
                                
                                const modelStats = bucketData.modelStats[originalModel];
                                if (!modelStats) {
                                    return [`${displayName}: No data available`];
                                }
                                
                                const successRate = context.raw.y;
                                const successful = modelStats.successful;
                                const attempts = modelStats.attempts;
                                const caseCount = bucketData.caseCount;
                                
                                // Get confidence interval
                                let ciInfo = '';
                                const ciElement = document.getElementById('show-confidence-intervals');
                                if (ciElement && ciElement.checked && attempts > 0) {
                                    const [lower, upper] = wilson_score_interval(successful, caseCount);
                                    ciInfo = `\n95% CI: ${(lower * 100).toFixed(2)}% - ${(upper * 100).toFixed(2)}%`;
                                }
                                
                                // Calculate untested cases
                                const untestedCases = caseCount - attempts;
                                let untestedInfo = '';
                                if (untestedCases > 0) {
                                    untestedInfo = `\nModel did not return result for ${untestedCases} case${untestedCases > 1 ? 's' : ''} in this bucket`;
                                }
                                
                                return [
                                    `${displayName}: ${successRate.toFixed(2)}% (${successful}/${caseCount})`,
                                    `Token Range: ${bucketData.minTokens/1000}k-${bucketData.maxTokens/1000}k (avg: ${(bucketData.avgTokens/1000).toFixed(1)}k)
Cases: ${bucketData.caseCount}${untestedInfo}${ciInfo}`
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
    
    // Function to filter cases based on selected languages
    function getFilteredCases() {
        if (currentSelectedLanguages.length === 0) {
            return [];
        }
        
        return chartData.cases.filter(caseData => 
            currentSelectedLanguages.includes(caseData.language)
        );
    }
    
    // Function to calculate buckets based on filtered cases
    function calculateBuckets() {
        const filteredCases = getFilteredCases();
        if (filteredCases.length === 0) {
            buckets = [];
            return;
        }
        
        // Sort cases by token count (should already be sorted, but just to be safe)
        filteredCases.sort((a, b) => a.token_count - b.token_count);
        
        // Calculate cases per bucket
        const casesPerBucket = Math.floor(filteredCases.length / currentBucketCount);
        const extraCases = filteredCases.length % currentBucketCount;
        
        // Create new buckets array
        buckets = [];
        
        let caseIndex = 0;
        for (let i = 0; i < currentBucketCount; i++) {
            // Calculate how many cases go in this bucket
            // Last bucket gets the extra cases if count isn't evenly divisible
            const bucketSize = (i === currentBucketCount - 1) 
                ? casesPerBucket + extraCases 
                : casesPerBucket;
            
            if (bucketSize === 0) continue;
            
            // Get cases for this bucket
            const bucketCases = filteredCases.slice(caseIndex, caseIndex + bucketSize);
            caseIndex += bucketSize;
            
            // Calculate token stats
            const minTokens = bucketCases[0].token_count;
            const maxTokens = bucketCases[bucketCases.length - 1].token_count;
            const avgTokens = bucketCases.reduce((sum, c) => sum + c.token_count, 0) / bucketCases.length;
            
            // Initialize bucket data
            const bucket = {
                minTokens,
                maxTokens,
                avgTokens,
                caseCount: bucketCases.length,
                cases: bucketCases,
                modelStats: {}
            };
            
            // Calculate model statistics for this bucket
            chartData.models.forEach(model => {
                const modelStats = {
                    successful: 0,
                    attempts: 0
                };
                
                bucketCases.forEach(caseData => {
                    const result = caseData.results[model];
                    if (result.attempted) {
                        modelStats.attempts++;
                        if (result.success) {
                            modelStats.successful++;
                        }
                    }
                });
                
                bucket.modelStats[model] = modelStats;
            });
            
            buckets.push(bucket);
        }
    }
    
    // Function to update chart based on selected models, languages, and bucket count
    function updateChart() {
        // Get selected models and languages
        currentSelectedModels = Array.from(document.querySelectorAll('input[data-model]:checked'))
            .map(checkbox => checkbox.getAttribute('data-model'));
        
        currentSelectedLanguages = Array.from(document.querySelectorAll('input[data-language]:checked'))
            .map(checkbox => checkbox.getAttribute('data-language'));
        
        // Recalculate buckets based on selected languages and bucket count
        calculateBuckets();
        
        // Clear current datasets
        chart.data.datasets = [];
        
        // If no buckets (e.g., no languages selected), don't update further
        if (buckets.length === 0) {
            chart.update();
            return;
        }
        
        // Update x-axis scale and ticks based on actual data
        const minToken = buckets[0].minTokens / 1000;
        const maxToken = buckets[buckets.length - 1].maxTokens / 1000;
        
        // Create custom ticks for each bucket
        const customTicks = buckets.map(bucket => {
            // Position tick at the bucket's average token count
            const tickValue = bucket.avgTokens / 1000;
            
            // Label shows the token range for this bucket
            const minK = (bucket.minTokens / 1000).toFixed(1);
            const maxK = (bucket.maxTokens / 1000).toFixed(1);
            const tickLabel = `${minK}k-${maxK}k`;
            
            return {
                value: tickValue,
                label: tickLabel
            };
        });
        
        // Set precise min/max values with small padding for visual appeal
        chart.options.scales.x.min = Math.max(0, minToken - 0.5);
        chart.options.scales.x.max = maxToken + 0.5;
        
        // Use our custom ticks
        chart.options.scales.x.ticks = {
            callback: function(val, index) {
                // Find the tick with this value
                const tick = customTicks.find(t => Math.abs(t.value - val) < 0.01);
                return tick ? tick.label : '';
            }
        };
        
        // Set up explicit ticks array to ensure our ticks are used
        chart.options.scales.x.afterBuildTicks = function(scale) {
            scale.ticks = customTicks;
            return;
        };
        
        // Create datasets for each selected model
        currentSelectedModels.forEach((model, index) => {
            const color = colors[index % colors.length];
            
            // Calculate data points for this model
            const dataPoints = buckets.map(bucket => {
                const modelStats = bucket.modelStats[model];
                const successful = modelStats.successful;
                const caseCount = bucket.caseCount;
                
                // Calculate success rate
                const successRate = (caseCount > 0) ? (successful / caseCount) * 100 : 0;
                
                return {
                    x: bucket.avgTokens / 1000, // Convert to k
                    y: successRate
                };
            });
            
            // Calculate confidence interval data points
            let lowerBoundPoints = null;
            let upperBoundPoints = null;
            
            if (document.getElementById('show-confidence-intervals').checked) {
                lowerBoundPoints = buckets.map(bucket => {
                    const modelStats = bucket.modelStats[model];
                    const successful = modelStats.successful;
                    const caseCount = bucket.caseCount;
                    
                    const [lower, upper] = wilson_score_interval(successful, caseCount);
                    
                    return {
                        x: bucket.avgTokens / 1000,
                        y: lower * 100
                    };
                });
                
                upperBoundPoints = buckets.map(bucket => {
                    const modelStats = bucket.modelStats[model];
                    const successful = modelStats.successful;
                    const caseCount = bucket.caseCount;
                    
                    const [lower, upper] = wilson_score_interval(successful, caseCount);
                    
                    return {
                        x: bucket.avgTokens / 1000,
                        y: upper * 100
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
                originalModel: model,
                data: dataPoints,
                borderColor: color,
                backgroundColor: color + '33',
                fill: false,
                tension: 0.1,
                pointRadius: 4,
                pointHoverRadius: 6,
                parsing: false
            });
            
            // Add confidence interval datasets if enabled
            if (document.getElementById('show-confidence-intervals').checked) {
                // Add lower bound line (needed for reference by the area dataset)
                const lowerBoundIndex = chart.data.datasets.length;
                chart.data.datasets.push({
                    label: `${displayName} (95% CI lower)`,
                    data: lowerBoundPoints,
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    tension: 0.1,
                    fill: false,
                    parsing: false,
                    showLine: false,
                    display: false
                });
                
                // Add confidence interval area
                chart.data.datasets.push({
                    label: `${displayName} (95% CI)`,
                    data: upperBoundPoints,
                    borderColor: 'transparent',
                    backgroundColor: color + '22',
                    pointRadius: 0,
                    tension: 0.1,
                    parsing: false,
                    fill: lowerBoundIndex,
                    display: false
                });
            }
        });
        
        // Update chart
        chart.update();
    }
    
    // Add event listeners
    document.querySelectorAll('input[data-model], input[data-language]').forEach(checkbox => {
        checkbox.addEventListener('change', updateChart);
    });
    
    document.getElementById('show-confidence-intervals').addEventListener('change', updateChart);
    
    // Calculate initial buckets and update chart
    calculateBuckets();
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

.model-selection, .language-selection, .display-options, .bucketing-options {
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

.bucket-count-control {
    margin: 15px 0;
}

.bucket-count-control label {
    display: block;
    margin-bottom: 8px;
}

.bucket-count-control input[type="range"] {
    width: 100%;
    cursor: pointer;
}

#bucket-count-display {
    font-weight: bold;
}

.chart-container {
    width: 100%;
    height: 450px;  /* Slightly taller for better visibility with many buckets */
    margin-bottom: 30px;
    padding: 0 10px; /* Add small padding to prevent labels from being cut off */
    box-sizing: border-box;
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
