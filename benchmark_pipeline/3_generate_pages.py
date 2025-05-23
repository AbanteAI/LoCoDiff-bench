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
  - Highlighting of top performers in each category with gold/silver/bronze indicators

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
from typing import Dict, List, Any, Tuple, Set, Optional


# --- Helper Functions ---


def delete_and_recreate_dir(dir_path: Path) -> None:
    """Completely removes a directory and recreates it empty."""
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Recreated directory: {dir_path}")


def copy_static_assets(docs_dir: Path) -> None:
    """Copies static assets to the docs directory."""
    # Create assets directory structure
    assets_dir = docs_dir / "assets" / "images"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Define the source directory for static assets
    static_dir = Path("benchmark_pipeline/static/images")

    # Copy each image file
    for image_file in static_dir.glob("*"):
        shutil.copy(image_file, assets_dir / image_file.name)

    print(f"Copied static assets to {assets_dir}")


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

    # Calculate quartile indices to ensure exact even split
    # For 200 items, this would give us indices [0, 50, 100, 150, 200]
    quartile_indices = []
    for i in range(5):
        idx = (i * total_count) // 4
        # Cap at the last index to avoid out of bounds
        idx = min(idx, total_count - 1) if i < 4 else total_count
        quartile_indices.append(idx)

    # Handle edge cases for small lists
    if total_count < 4:
        return [(sorted_tokens[0], sorted_tokens[-1])] * 4

    # Create quartile ranges using the indices to get token values
    # Important: The last index is total_count which is out of bounds, so -1 for the upper bound of Q4
    q1_range = (
        sorted_tokens[quartile_indices[0]],
        sorted_tokens[quartile_indices[1] - 1],
    )
    q2_range = (
        sorted_tokens[quartile_indices[1]],
        sorted_tokens[quartile_indices[2] - 1],
    )
    q3_range = (
        sorted_tokens[quartile_indices[2]],
        sorted_tokens[quartile_indices[3] - 1],
    )
    q4_range = (sorted_tokens[quartile_indices[3]], sorted_tokens[-1])

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
    # Find the quartile by checking each range inclusively
    for i, (q_min, q_max) in enumerate(quartile_ranges):
        if q_min <= token_count <= q_max:
            return i

    # Try a second pass with looser bounds to handle edge cases
    # This ensures we don't miss any cases that might fall on boundaries due to float rounding
    for i, (q_min, q_max) in enumerate(quartile_ranges):
        if abs(token_count - q_min) < 0.001 or abs(token_count - q_max) < 0.001:
            return i

    # Should not happen if quartile_ranges were calculated correctly
    print(f"Warning: Could not classify token count {token_count} into any quartile")
    return -1


def parse_success_rate(cell_content: str) -> float:
    """
    Extracts success rate percentage from a cell content string.

    Args:
        cell_content: String containing success rate info (e.g. "24.50% (49/200)")

    Returns:
        Float representing the success rate percentage
    """
    try:
        # Extract percentage value from the beginning of the string
        percentage_str = cell_content.split("%")[0].strip()
        return float(percentage_str)
    except (ValueError, IndexError):
        return 0.0


def find_top_performers(values: List[Tuple[int, float]]) -> Dict[int, str]:
    """
    Identifies the indices of the top 3 performers in a list of values.

    Args:
        values: List of tuples (index, value) to rank

    Returns:
        Dictionary mapping indices to rank classes ('gold', 'silver', 'bronze')
    """
    if not values:
        return {}

    # Sort values in descending order (higher is better)
    sorted_values = sorted(values, key=lambda x: x[1], reverse=True)

    # Initialize result dictionary
    rank_classes = {}

    # Assign medals if there are values to rank
    medal_classes = ["gold", "silver", "bronze"]

    # Handle ties by giving the same rank to equal values
    current_rank = 0
    prev_value = None

    for i, (idx, value) in enumerate(sorted_values):
        # If we've already assigned all medals, break
        if current_rank >= len(medal_classes):
            break

        # If this value is different from the previous one, increment rank
        if prev_value is not None and value < prev_value:
            current_rank = i

        # If we've reached the maximum rank, break
        if current_rank >= len(medal_classes):
            break

        # Assign medal class to this index
        rank_classes[idx] = medal_classes[current_rank]
        prev_value = value

    return rank_classes


def format_cell_with_rank(content: str, rank_class: Optional[str] = None) -> str:
    """
    Formats a table cell with medal emoji and border styling if ranked.

    Args:
        content: The cell content
        rank_class: Optional rank class ('gold', 'silver', 'bronze')

    Returns:
        HTML string for the formatted cell with medal emoji and border styling for top performers
    """
    if rank_class:
        medal_emoji = {"gold": "🥇 ", "silver": "🥈 ", "bronze": "🥉 "}.get(
            rank_class, ""
        )

        return f'<td class="{rank_class}">{medal_emoji}{content}</td>'
    else:
        return f"<td>{content}</td>"


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


def create_table_of_contents() -> str:
    """
    Creates a table of contents section for the main page.

    Returns:
        HTML string for the table of contents section
    """
    return """
    <section id="table-of-contents">
        <nav class="toc">
            <ul>
                <li><a href="#locodiff-summary">What is LoCoDiff?</a></li>
                <li><a href="#token-chart">Interactive Chart</a></li>
                <li><a href="#benchmark-example">Methodology Example</a></li>
                <li><a href="#overall-stats">Overall Model Performance</a></li>
                <li><a href="#quartile-stats">Performance by Prompt Size</a></li>
                <li><a href="#language-stats">Performance by Language</a></li>
                <li><a href="#explore-benchmarks">Explore Benchmark Cases</a></li>
            </ul>
        </nav>
    </section>
    """


def create_locodiff_summary() -> str:
    """
    Creates a summary section describing LoCoDiff and Mentat's role.

    Returns:
        HTML string for the summary section
    """
    return """
    <section id="locodiff-summary" style="background-color: transparent; border: none; padding: 0;">
        <p style="margin-bottom: 10px;">
            LoCoDiff is a novel <strong>lo</strong>ng-<strong>co</strong>ntext benchmark with several unique strengths:
        </p>
        <ul style="margin-top: 0; margin-bottom: 20px; margin-left: 20px;">
            <li>Utilizes <strong>naturally interconnected content</strong>, not artificially generated or padded context</li>
            <li><strong>No junk context</strong>: every part of the context is required for the task</li>
            <li><strong>Tests a real skill critical for coding agents</strong>: keeping track of the state of edited files</li>
            <li>Prompt generation and output evaluation are <strong>simple and easy to understand</strong></li>
            <li>Challenges models' capacity to generate <strong>long-form outputs</strong></li>
            <li>Surprisingly <strong>difficult for reasoning models</strong> to reason about</li>
            <li><strong>Easy to procedurally generate</strong>: any file in any git repo can be made into a benchmark case</li>
        </ul>
        <p style="margin-bottom: 20px;">
            100% of the code for the LoCoDiff was written by 
            <a href="https://mentat.ai">Mentat</a>, a coding agent developed by AbanteAI. Mentat also generated the prompts and ran the benchmark on the models, setup the github page hosting, and built this site. All benchmark code and results are on the <a href="https://github.com/AbanteAI/LoCoDiff-bench">Github repo</a>, and you can see the Mentat agent runs <a href="https://mentat.ai/gh/AbanteAI/LoCoDiff-bench/agents">here</a>.
        </p>
    </section>
    """


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
    <title>LoCoDiff Benchmark</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <h1>LoCoDiff: Natural <span style="font-weight:bold">Lo</span>ng <span style="font-weight:bold">Co</span>ntext Code Bench</h1>
        <p style="margin-top: 0; font-size: 0.9em; color: #666;"><a href="https://mentat.ai" target="_blank">Mentat AI Team</a> - May 8th 2025</p>
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
        <div class="footer-content">
            <p>LoCoDiff-bench - <a href="https://github.com/AbanteAI/LoCoDiff-bench" class="github-link"><img src="assets/images/github-logo.png" alt="GitHub" class="github-icon"></a></p>
            <p class="built-with">
                built with <a href="https://mentat.ai" target="_blank" class="mentat-link">mentat.ai</a> <a href="https://mentat.ai" target="_blank" class="mentat-link"><img src="assets/images/mentat-logo-transparent.png" alt="Mentat" class="mentat-icon"></a>
            </p>
        </div>
    </footer>
    """

    if include_chart_js:
        footer += create_chart_javascript()

    # Add JavaScript for syntax highlighting diffs
    footer += """
    <script>
        // Highlight diff lines (added/removed) on load
        document.addEventListener('DOMContentLoaded', function() {
            // Select all code blocks in the example prompt
            const codeBlocks = document.querySelectorAll('.example-prompt pre code.language-diff');
            
            codeBlocks.forEach(function(codeBlock) {
                // Get all lines in the code block
                const content = codeBlock.innerHTML;
                
                // Replace the content with highlighted version
                let highlightedContent = '';
                
                // Split by real newlines in the content
                const lines = content.split('\\n');
                
                // Process each line and preserve empty lines
                let highlightedLines = [];
                for (let i = 0; i < lines.length; i++) {
                    let line = lines[i];
                    
                    if (line === '') {
                        // Preserve blank lines by using a non-breaking space
                        highlightedLines.push('<span class="empty-line">&nbsp;</span>');
                    }
                    // Skip highlighting for file path indicators
                    else if (line.startsWith('+++') || line.startsWith('---')) {
                        highlightedLines.push('<span>' + line + '</span>');
                    }
                    // Highlight added lines - check for both '+' at start and '+' after whitespace (for merge conflicts)
                    else if (line.startsWith('+') || line.trim().startsWith('+')) {
                        highlightedLines.push('<span style="background-color: #e6ffec; color: #22863a;">' + line + '</span>');
                    }
                    // Highlight removed lines - check for both '-' at start and '-' after whitespace
                    // But exclude command-line options that start with '--'
                    else if ((line.startsWith('-') && !line.startsWith('--')) || 
                             (line.trim().startsWith('-') && !line.trim().startsWith('--'))) {
                        highlightedLines.push('<span style="background-color: #ffebe9; color: #cb2431;">' + line + '</span>');
                    }
                    // Normal line
                    else {
                        highlightedLines.push('<span>' + line + '</span>');
                    }
                }
                
                // Concatenate the spans directly without adding newlines
                highlightedContent = highlightedLines.join('');
                
                // Replace code block content
                codeBlock.innerHTML = highlightedContent;
            });
            
            // Initialize highlight.js 
            hljs.highlightAll();
        });
    </script>
    """

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

    # Create the HTML table header
    html = """
    <section id="overall-stats">
        <h2>Overall Model Performance</h2>
        <p style="margin-top: 0; margin-bottom: 10px; font-size: 0.9em; color: #666; text-align: center;"><img src="assets/images/mentat-logo-transparent.png" alt="Mentat" style="height: 24px; vertical-align: middle; margin-right: 5px;">Mentat.ai LoCoDiff Bench</p>
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

    # Collect models and their success rates for ranking
    success_rates = []
    model_rows = {}

    for i, model in enumerate(sorted(all_models)):
        stats = model_stats.get(
            model, {"total_attempts": 0, "successful": 0, "total_cost": 0.0}
        )
        attempts = stats["total_attempts"]
        success_rate = (stats["successful"] / attempts * 100) if attempts > 0 else 0
        total_cost = stats["total_cost"]
        avg_cost = total_cost / attempts if attempts > 0 else 0

        # Use display name if available
        display_name = model_display_names.get(model, model)

        # Store row content for this model
        model_rows[i] = (
            model,
            display_name,
            success_rate,
            stats,
            attempts,
            total_cost,
            avg_cost,
        )

        # Track success rate for ranking
        success_rates.append((i, success_rate))

    # Find top performers
    top_performers = find_top_performers(success_rates)

    # Generate HTML rows with ranking
    for i, (
        model,
        display_name,
        success_rate,
        stats,
        attempts,
        total_cost,
        avg_cost,
    ) in model_rows.items():
        # Get rank for this model (if it's a top performer)
        rank_class = top_performers.get(i)

        # Format success rate cell with appropriate ranking
        success_rate_cell = f"{success_rate:.2f}% ({stats['successful']}/{attempts})"
        formatted_success_cell = format_cell_with_rank(success_rate_cell, rank_class)

        html += f"""
                <tr>
                    <td>{display_name}</td>
                    {formatted_success_cell}
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
        <h2>Accuracy by Context Length Quartiles</h2>
        <p style="margin-top: 0; margin-bottom: 10px; font-size: 0.9em; color: #666; text-align: center;"><img src="assets/images/mentat-logo-transparent.png" alt="Mentat" style="height: 24px; vertical-align: middle; margin-right: 5px;">Mentat.ai LoCoDiff Bench</p>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
    """

    for label in quartile_labels:
        html += f"<th>{label}</th>"

    # Add Total Cost column header
    html += "<th>Total Cost</th>"

    html += """
                </tr>
            </thead>
            <tbody>
    """

    # Prepare model data rows and collect success rates for each quartile
    model_rows = {}
    quartile_success_rates = {i: [] for i in range(4)}  # For each quartile
    model_total_costs = {}  # To track total cost for each model

    # Calculate total cost for each model
    for (case_prefix, model), result_metadata in results_metadata.items():
        if model not in model_total_costs:
            model_total_costs[model] = 0.0

        cost = result_metadata.get("cost_usd", 0.0)
        if cost is not None:
            model_total_costs[model] += float(cost)

    for idx, model in enumerate(sorted(all_models)):
        display_name = model_display_names.get(model, model)
        quartile_stats = model_quartile_stats.get(model, {})
        total_cost = model_total_costs.get(model, 0.0)

        # Store model data for later use
        model_rows[idx] = (model, display_name, quartile_stats, total_cost)

        # Collect success rates for each quartile
        for q in range(4):
            stats = quartile_stats.get(q, {"attempts": 0, "successful": 0})
            attempts = stats["attempts"]
            success_rate = (stats["successful"] / attempts * 100) if attempts > 0 else 0
            quartile_success_rates[q].append((idx, success_rate))

    # Find top performers for each quartile
    top_performers = {}
    for q in range(4):
        top_performers[q] = find_top_performers(quartile_success_rates[q])

    # Generate HTML rows with rankings for each quartile
    for idx, (model, display_name, quartile_stats, total_cost) in model_rows.items():
        html += f"<tr><td>{display_name}</td>"

        for q in range(4):
            stats = quartile_stats.get(q, {"attempts": 0, "successful": 0})
            attempts = stats["attempts"]
            success_rate = (stats["successful"] / attempts * 100) if attempts > 0 else 0

            # Get rank for this model in this quartile (if it's a top performer)
            rank_class = top_performers[q].get(idx)

            # Format cell content
            cell_content = f"{success_rate:.2f}% ({stats['successful']}/{attempts})"
            html += format_cell_with_rank(cell_content, rank_class)

        # Add total cost column
        html += f"<td>${total_cost:.2f}</td>"

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
        <h2>Accuracy by Programming Language</h2>
        <p style="margin-top: 0; margin-bottom: 10px; font-size: 0.9em; color: #666; text-align: center;"><img src="assets/images/mentat-logo-transparent.png" alt="Mentat" style="height: 24px; vertical-align: middle; margin-right: 5px;">Mentat.ai LoCoDiff Bench</p>
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

    # Prepare model data rows and collect success rates for each language
    model_rows = {}
    language_success_rates = {lang: [] for lang in all_languages}  # For each language

    for idx, model in enumerate(sorted(all_models)):
        display_name = model_display_names.get(model, model)
        language_stats = model_language_stats.get(model, {})

        # Store model data for later use
        model_rows[idx] = (model, display_name, language_stats)

        # Collect success rates for each language
        for language in all_languages:
            stats = language_stats.get(language, {"attempts": 0, "successful": 0})
            attempts = stats["attempts"]
            success_rate = (stats["successful"] / attempts * 100) if attempts > 0 else 0
            language_success_rates[language].append((idx, success_rate))

    # Find top performers for each language
    top_performers = {}
    for language in all_languages:
        top_performers[language] = find_top_performers(language_success_rates[language])

    # Generate HTML rows with rankings for each language
    for idx, (model, display_name, language_stats) in model_rows.items():
        html += f"<tr><td>{display_name}</td>"

        for language in all_languages:
            stats = language_stats.get(language, {"attempts": 0, "successful": 0})
            attempts = stats["attempts"]
            success_rate = (stats["successful"] / attempts * 100) if attempts > 0 else 0

            # Get rank for this model in this language (if it's a top performer)
            rank_class = top_performers[language].get(idx)

            # Format cell content
            cell_content = f"{success_rate:.2f}% ({stats['successful']}/{attempts})"
            html += format_cell_with_rank(cell_content, rank_class)

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
        "default_bucket_count": 4,  # Default number of buckets
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
        <div class="chart-container">
            <canvas id="token-success-chart"></canvas>
        </div>
        <h3 style="margin-top: 15px; margin-bottom: 5px; font-size: 1.1em;">Chart Options</h3>
        <div class="chart-controls" style="font-size: 0.85em; margin-top: 10px; padding: 5px;">
            <div class="model-selection" style="margin-right: 15px;">
                <h3 style="font-size: 1em; margin: 5px 0;">Models</h3>
                <div id="model-checkboxes"></div>
            </div>
            <div class="language-selection" style="margin-right: 15px;">
                <h3 style="font-size: 1em; margin: 5px 0;">Languages</h3>
                <div id="language-checkboxes"></div>
            </div>
            <div class="bucketing-options" style="margin-right: 15px;">
                <h3 style="font-size: 1em; margin: 5px 0;">Bucketing Options</h3>
                <div class="bucket-count-control" style="margin: 5px 0;">
                    <label for="bucket-count" style="font-size: 0.9em;">Number of Buckets: <span id="bucket-count-display">4</span></label>
                    <input type="range" id="bucket-count" min="1" max="10" value="4" step="1" style="width: 100%;">
                </div>
            </div>
            <div class="display-options">
                <h3 style="font-size: 1em; margin: 5px 0;">Display Options</h3>
                <div class="checkbox-item" style="margin: 3px 0;">
                    <label style="font-size: 0.9em;">
                        <input type="checkbox" id="show-confidence-intervals">
                        95% Wilson Score Intervals
                    </label>
                </div>
            </div>
        </div>
    </section>
    """


def load_example_git_history() -> tuple[str, str]:
    """
    Loads the example git history and expected output from files.

    Returns:
        Tuple of (git_history, expected_output)
    """
    example_prompt_path = Path("benchmark_pipeline/locodiff-example/example_prompt.txt")
    example_expected_path = Path(
        "benchmark_pipeline/locodiff-example/example_expected.txt"
    )

    git_history = ""
    expected_output = ""

    if example_prompt_path.exists():
        with open(example_prompt_path, "r", encoding="utf-8") as f:
            git_history = f.read()
    else:
        print(f"Warning: Example prompt file not found at {example_prompt_path}")
        print("Run generate_example_prompt.py to create it")
        # Provide a placeholder
        git_history = "# Example git history not found. Run benchmark_pipeline/generate_example_prompt.py first."

    if example_expected_path.exists():
        with open(example_expected_path, "r", encoding="utf-8") as f:
            expected_output = f.read()
    else:
        expected_output = "# Expected output file not found. Run benchmark_pipeline/generate_example_prompt.py first."

    return git_history, expected_output


def create_example_section() -> str:
    """Creates an HTML section explaining the benchmark with a git merge conflict example."""
    # Load real git history and expected output
    git_history, expected_output = load_example_git_history()

    # Simple ASCII diagram of the git branch structure
    ascii_diagram = """
A
/  \\
B    C
\\  /
D
"""

    # Construct the HTML for the example section
    html = (
        """
    <section id="benchmark-example" style="background-color: transparent; border: none; padding: 0;">
        <h2>Methodology</h2>
        
        <p style="margin-bottom: 15px;">
            For each benchmark prompt, we show the model the commit history of a particular file, and ask the model to infer the exact current state of that file. This requires the model to track the state of the file as it changes, from the initial commit, diffs along various branches, and merge conflict resolutions. Accuracy is the percentage of files that matched exactly - there is no partial credit.
        </p>
        <p style="margin-bottom: 15px;">
            The exact command we use to generate the history for each file is: <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px; font-family: monospace;">git log -p --cc --reverse --topo-order -- path/to/file</code>. <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px; font-family: monospace;">-p</code> and <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px; font-family: monospace;">--cc</code> display the diffs for commits and show merge commit diffs with respect to each parent. <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px; font-family: monospace;">--reverse</code> and <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px; font-family: monospace;">--topo-order</code> make sure the commits are shown from oldest to newest, with parent commits always appearing before children. This is the cleanest, clearest way to present the history to the model.
        </p>
        
        <p style="margin-bottom: 15px;">
            The benchmark consists of 200 files, 40 each from 5 repos: 
            <a href="https://github.com/Aider-AI/aider" target="_blank">Aider</a> (Python), 
            <a href="https://github.com/ghostty-org/ghostty" target="_blank">Ghostty</a> (Zig), 
            <a href="https://github.com/tldraw/tldraw" target="_blank">tldraw</a> (TypeScript), 
            <a href="https://github.com/qdrant/qdrant" target="_blank">Qdrant</a> (Rust), and 
            <a href="https://github.com/facebook/react" target="_blank">React</a> (JavaScript). 
            For each repo, we filtered to files modified in the last 6 months that were no longer than 12k tokens long (in their final state - what the model needs to output). We then sampled, biasing the sampling to target an even distribution of prompt lengths, with a limit of 100k.
        </p>

        <p style="margin-bottom: 15px;">
            All case prompts, expected outputs, and model answers can be explored <a href="cases.html">here</a>. A <a href="/LoCoDiff-bench/content/anthropic_claude-3.7-sonnetthinking/qdrant_src_actix_api_snapshot_api.rs/prompt.html">typical prompt</a> consisting of 50k tokens can be extremely complex, containing 50-150 commits and ending up with several hundred lines to reproduce.
        </p>
        
        <p style="margin-bottom: 15px;">
            To quickly understand what the model sees, here is a minimal example:
        </p>

        <div class="example-timeline">
            <h3 style="text-align: left;">Toy Example with Shopping List</h3>
            <div class="branch-structure-container">
                <div class="branch-diagram-box">
                    <pre class="branch-diagram">"""
        + ascii_diagram
        + """</pre>
                </div>
                <div class="branch-explanation">
                    <p class="commit-description">
                        Commit A: Creates initial shopping list file<br>
                        Commit B: Changes "apples" to "oranges", and adds new item at the end<br>
                        Commit C: On a separate branch from B, changes "apples" to "bananas"<br>
                        Commit D: Merges B and C branches, resolving conflict by keeping both "oranges" and "bananas"
                    </p>
                </div>
            </div>
        </div>
        
        <div class="example-io-container">
            <div class="example-prompt" style="border: 1px solid #ddd; border-radius: 6px; padding: 15px; background-color: #f8f8f8;">
                <h3>Input: git log output for a file</h3>
                <pre style="background-color: #f1f1f1; border: 1px solid #e1e4e8; border-radius: 3px;"><code class="language-diff">"""
        + git_history
        + """</code></pre>
            </div>
            
            <div class="example-expected" style="border: 1px solid #ddd; border-radius: 6px; padding: 15px; background-color: #f8f8f8;">
                <h3>Target Output: Exact final state of the file</h3>
                <pre style="background-color: #f1f1f1; border: 1px solid #e1e4e8; border-radius: 3px;"><code class="language-text">"""
        + expected_output
        + """</code></pre>
            </div>
        </div>
        
    </section>
    """
    )
    return html


def create_key_takeaways_section() -> str:
    """Creates a section highlighting key findings from the benchmark."""
    return """
    <section id="key-takeaways">
        <h2>Key Takeaways</h2>
        <p style="margin-bottom: 15px;">
            <strong>Performance drops rapidly as context increases:</strong> While some models score near 100% for prompts <5k tokens, all drop significantly by 10k. All models drop to under 50% accuracy when prompts are just 25k tokens long. When we originally conceived of this benchmark, we were excited to put the million token long context limits of some models to the test, but it seems they are not yet ready for that.
        </p>
        <p style="margin-bottom: 15px;">
            <strong>Claude 3.7 Sonnet Thinking is the clear SOTA:</strong> It's the best for all context lengths and languages. We believe its ability to track the evolving state of files over long contexts is one of the reasons it makes such a strong model for coding agents.
        </p>
        <p style="margin-bottom: 15px;">
            <strong>Reasoning models, except for Sonnet do WORSE than their non-reasoning counterparts:</strong> DeepSeek's Chat v3 beats R1, Gemini 2.5 Flash Non-thinking beats Gemini 2.5 Flash Thinking, and GPT-4.1 beats o3 and o4-mini. The only exception to this trend is Sonnet 3.7 Thinking, which beats Sonnet 3.7 Non-thinking. It's unclear how reasoning models should best use their tokens to solve this task, but somehow Sonnet 3.7 uses them well.
        </p>
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
    let currentBucketCount = chartData.default_bucket_count || 4;
    let buckets = []; // Will be calculated dynamically
    
    // Define models to check by default
    const defaultSelectedModels = [
        "anthropic/claude-3.7-sonnetthinking",
        "anthropic/claude-sonnet-4",
        "deepseek/deepseek-chat-v3-0324",
        "deepseek/deepseek-r1",
        "google/gemini-2.5-pro-preview",
        "openai/gpt-4.1",
        "openai/o3",
        "x-ai/grok-3-beta"
    ];
    
    // Get canvas context
    const ctx = document.getElementById('token-success-chart').getContext('2d');
    
    // Create model checkboxes
    const modelCheckboxes = document.getElementById('model-checkboxes');
    chartData.models.forEach((model) => {
        // Use display name if available, otherwise use original model name
        const displayName = chartData.model_display_names && chartData.model_display_names[model] 
            ? chartData.model_display_names[model] 
            : model;
        
        // Check if this model should be checked by default
        const isChecked = defaultSelectedModels.includes(model);
        
        const checkbox = document.createElement('div');
        checkbox.className = 'checkbox-item';
        checkbox.innerHTML = `
            <label>
                <input type="checkbox" data-model="${model}" ${isChecked ? 'checked' : ''}>
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
    
    // Create watermark plugin
    const watermarkPlugin = {
        id: 'watermark',
        beforeDraw: (chart) => {
            const ctx = chart.ctx;
            const { chartArea: { top, left, right, bottom }, width, height } = chart;
            
            // Create an image object
            const image = new Image();
            image.src = 'assets/images/mentat-logo-transparent.png';
            
            // Only draw the image once it's loaded
            if (image.complete) {
                const logoWidth = width * 0.15;  // Reduced from 30% to 15% of chart width
                const logoHeight = logoWidth * (image.height / image.width);
                
                // Position the image in the top right of the chart with a margin
                const margin = 20;
                const x = right - logoWidth - margin;
                const y = top + margin + 40;  // Added 40px to move it down
                
                // Set transparency
                ctx.globalAlpha = 0.15;
                
                // Add "mentat.ai" text above the logo
                ctx.globalAlpha = 0.8;  // More visible text
                ctx.font = 'bold 16px Arial';
                ctx.fillStyle = '#666';
                ctx.textAlign = 'center';
                ctx.fillText('mentat.ai', x + logoWidth/2, y + 12);
                
                // Draw image
                ctx.globalAlpha = 0.15;  // Reset to logo transparency
                ctx.drawImage(image, x, y, logoWidth, logoHeight);
                
                // Reset transparency
                ctx.globalAlpha = 1.0;
            } else {
                // Set up image.onload if the image isn't loaded yet
                image.onload = () => chart.draw();
            }
        }
    };
    
    // Create chart
    const chart = new Chart(ctx, {
        type: 'line',
        plugins: [watermarkPlugin],  // Register the watermark plugin
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
                        text: 'Prompt Token Length'
                    },
                    grid: {
                        // Make grid lines match our ticks
                        z: -1 // Draw grid lines behind the data
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Success Rate'
                    },
                    min: 0,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                watermark: {
                    // Empty object enables the plugin
                },
                title: {
                    display: true,
                    text: 'LoCoDiff: Natural Long Context Code Bench',
                    font: {
                        size: 18
                    },
                    padding: {
                        top: 10,
                        bottom: 5
                    }
                },
                subtitle: {
                    display: true,
                    text: 'All Languages',
                    color: '#666',
                    font: {
                        size: 14,
                        style: 'italic'
                    },
                    padding: {
                        bottom: 20
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
                        // Add title callback to display bucket information once at the top
                        title: function(tooltipItems) {
                            if (!tooltipItems || tooltipItems.length === 0) return '';
                            
                            try {
                                const xValue = tooltipItems[0].raw.x; // Token count in thousands
                                
                                // Find the bucket with matching token count
                                const bucketData = buckets.find(bucket => 
                                    Math.abs((bucket.avgTokens / 1000) - xValue) < 0.01
                                );
                                
                                if (!bucketData) return 'Bucket Information';
                                
                                // Format bucket information
                                return [
                                    `Token Range: ${bucketData.minTokens/1000}k–${bucketData.maxTokens/1000}k`,
                                    `Average Tokens: ${(bucketData.avgTokens/1000).toFixed(1)}k`,
                                    `Total Cases: ${bucketData.caseCount}`
                                ];
                            } catch (error) {
                                console.error('Error in tooltip title callback:', error);
                                return 'Bucket Information';
                            }
                        },
                        
                        // Modify label callback to show only model-specific information
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
                                    return `${displayName}`;
                                }
                                
                                const modelStats = bucketData.modelStats[originalModel];
                                if (!modelStats) {
                                    return `${displayName}: No data available`;
                                }
                                
                                const successRate = context.raw.y;
                                const successful = modelStats.successful;
                                const attempts = modelStats.attempts;
                                const caseCount = bucketData.caseCount;
                                
                                // Format model success rate
                                let modelInfo = `${displayName}: ${successRate.toFixed(2)}% (${successful}/${caseCount})`;
                                
                                // Add model-specific notes if any
                                const untestedCases = caseCount - attempts;
                                if (untestedCases > 0) {
                                    modelInfo += ` (${untestedCases} untested)`;
                                }
                                
                                // Add confidence interval if enabled
                                const ciElement = document.getElementById('show-confidence-intervals');
                                if (ciElement && ciElement.checked && attempts > 0) {
                                    const [lower, upper] = wilson_score_interval(successful, caseCount);
                                    modelInfo += `\n  95% CI: ${(lower * 100).toFixed(2)}% - ${(upper * 100).toFixed(2)}%`;
                                }
                                
                                return modelInfo;
                            } catch (error) {
                                console.error('Error in tooltip label callback:', error);
                                return 'Error displaying model data';
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
    
    // Function to update the chart subtitle based on selected languages
    function updateChartSubtitle() {
        // Get language description
        let languageText = '';
        
        if (currentSelectedLanguages.length === 0) {
            languageText = 'no languages selected';
        } else if (currentSelectedLanguages.length === chartData.languages.length) {
            languageText = 'all languages';
        } else if (currentSelectedLanguages.length <= 3) {
            // Show all selected languages if there are 3 or fewer
            languageText = currentSelectedLanguages.join(', ');
        } else {
            // Show count if more than 3 languages selected
            languageText = `${currentSelectedLanguages.length} languages selected`;
        }
        
        // Count filtered cases
        const filteredCases = getFilteredCases();
        const caseCount = filteredCases.length;
        
        // Create the full subtitle with case count, bucket count, and language info
        let subtitleText = '';
        if (caseCount === 0) {
            subtitleText = `No cases to display (${languageText})`;
        } else {
            subtitleText = `${caseCount} cases, divided into ${currentBucketCount} buckets (${languageText})`;
        }
        
        // Update the chart subtitle
        chart.options.plugins.subtitle.text = subtitleText;
    }
    
    // Function to update chart based on selected models, languages, and bucket count
    function updateChart() {
        // Get selected models and languages
        currentSelectedModels = Array.from(document.querySelectorAll('input[data-model]:checked'))
            .map(checkbox => checkbox.getAttribute('data-model'));
        
        currentSelectedLanguages = Array.from(document.querySelectorAll('input[data-language]:checked'))
            .map(checkbox => checkbox.getAttribute('data-language'));
        
        // Update the chart subtitle based on selected languages
        updateChartSubtitle();
        
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
            
            // Label shows the token range for this bucket (rounded to whole numbers)
            const minK = Math.round(bucket.minTokens / 1000);
            const maxK = Math.round(bucket.maxTokens / 1000);
            const tickLabel = `${minK}-${maxK}k`;
            
            return {
                value: tickValue,
                label: tickLabel
            };
        });
        
        // Calculate min/max from bucket averages (not the absolute min/max prompt values)
        const firstBucketAvg = buckets[0].avgTokens / 1000;
        const lastBucketAvg = buckets[buckets.length - 1].avgTokens / 1000;
        
        // Set precise min/max values with small padding for visual appeal
        chart.options.scales.x.min = Math.max(0, firstBucketAvg - 1.5);
        chart.options.scales.x.max = lastBucketAvg + 1.5;
        
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
        <div class="explore-options">
            <div class="case-view">
                <h3>View by Case</h3>
                <p>View all benchmark cases with their results across models:</p>
                <a href="cases.html" class="view-all-cases-button">View All Cases</a>
            </div>
            <div class="model-view">
                <h3>View by Model</h3>
                <p>Select a model to view all its benchmark cases:</p>
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
            </div>
        </div>
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
        <h2>Model: {display_name}</h2>
        <p><a href="../../../cases/{safe_model}/{safe_case}.html">Back to Case</a> | <a href="../../../cases.html">All Cases</a> | <a href="../../../index.html">Home</a></p>
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
        <h2>Model: {display_name}</h2>
        <p><a href="../../../cases/{safe_model}/{safe_case}.html">Back to Case</a> | <a href="../../../cases.html">All Cases</a> | <a href="../../../index.html">Home</a></p>
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
        <h2>Model: {display_name}</h2>
        <p><a href="../../../cases/{safe_model}/{safe_case}.html">Back to Case</a> | <a href="../../../cases.html">All Cases</a> | <a href="../../../index.html">Home</a></p>
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


def generate_cases_overview_page(
    prompt_metadata: Dict[str, Dict[str, Any]],
    results_metadata: Dict[Any, Dict[str, Any]],
    all_models: Set[str],
    docs_dir: Path,
    model_display_names: Dict[str, str] = {},
) -> None:
    """
    Generates a page showing all benchmark cases with their results across all models.

    Args:
        prompt_metadata: Dictionary of prompt metadata by case prefix
        results_metadata: Dictionary mapping (case_prefix, model) to result metadata
        all_models: Set of all model names
        docs_dir: Path to the docs directory
        model_display_names: Optional mapping of model names to display names
    """
    # Define the path for the cases overview page
    cases_page_path = docs_dir / "cases.html"

    # Extract all cases and sort by prompt token count
    cases = []
    for case_prefix, metadata in prompt_metadata.items():
        token_count = metadata.get("prompt_tokens", 0)
        original_filename = metadata.get("original_filename", case_prefix)

        # Create case data structure
        case_data = {
            "prefix": case_prefix,
            "original_filename": original_filename,
            "token_count": token_count,
            "model_results": {},
        }

        # Collect results for this case across all models
        for model in all_models:
            result_key = (case_prefix, model)
            if result_key in results_metadata:
                result_metadata = results_metadata[result_key]
                case_data["model_results"][model] = {
                    "success": result_metadata.get("success", False),
                    "tested": True,
                }
            else:
                case_data["model_results"][model] = {"success": False, "tested": False}

        cases.append(case_data)

    # Sort cases by token count
    cases.sort(key=lambda x: x["token_count"])

    # Start building the HTML content
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>All Benchmark Cases - LoCoDiff Benchmark</title>
    <link rel="stylesheet" href="styles.css">
    <style>
        /* Custom styles for the cases page specifically */
        body {{
            max-width: 95%; /* Use more of the screen width */
        }}
        
        /* Redo the table layout completely to fix shading issues */
        #cases-table {{
            table-layout: fixed;
            width: 100%;
            border-collapse: collapse;
        }}
        
        /* Override the default cases table styles */
        #cases-table th, #cases-table td {{
            padding: 8px;
            border: 1px solid #e1e4e8;
        }}
        
        /* Case name column - matching model-specific pages */
        #cases-table th:first-child, #cases-table td:first-child {{
            width: 240px; /* Match model-specific pages */
        }}
        
        /* Prompt tokens column - matching model-specific pages */
        #cases-table th:nth-child(2), #cases-table td:nth-child(2) {{
            width: 120px; /* Match model-specific pages */
        }}
        
        /* Ensure background colors match the row shading */
        #cases-table tbody tr:nth-child(odd) td {{
            background-color: #f6f8fa;
        }}
        
        #cases-table tbody tr:nth-child(even) td {{
            background-color: white;
        }}
        
        /* Header row always has its own background */
        #cases-table thead th {{
            background-color: #f6f8fa;
            font-weight: 600;
            text-align: center;
            z-index: 3; /* Ensure header is above all */
        }}
        
        /* All headers are center-aligned for consistency */
        
        /* Ensure case names are center-aligned and wrap properly */
        .case-name {{
            white-space: normal;
            overflow-wrap: break-word;
            text-align: center; /* Center alignment to match model pages */
        }}
        
        /* Multi-column layout for model checkboxes */
        .multi-column-checkboxes {{
            display: flex;
            flex-wrap: wrap;
            max-height: none;
        }}
        .multi-column-checkboxes .checkbox-item {{
            width: 25%; /* Four columns */
            min-width: 250px;
            margin-bottom: 5px;
        }}
        /* Ensure case names wrap properly */
        .case-name {{
            max-width: 300px;
            white-space: normal;
            overflow-wrap: break-word;
        }}
    </style>
</head>
<body>
    <header>
        <h1>All Benchmark Cases</h1>
        <p><a href="index.html">← Back to Overview</a></p>
    </header>
    <main>
        <section>
            <h2>Benchmark Cases by Prompt Size</h2>
            
            <div class="model-selection">
                <h3>Select Models to Display</h3>
                <div id="model-checkboxes" class="multi-column-checkboxes">
"""

    # Add model selection checkboxes - initialized as unchecked
    sorted_models = sorted(all_models)
    for model in sorted_models:
        display_name = model_display_names.get(model, model)
        safe_model = model.replace("/", "_")
        html_content += f"""
                    <div class="checkbox-item">
                        <label>
                            <input type="checkbox" data-model="{safe_model}">
                            {display_name}
                        </label>
                    </div>
"""

    html_content += """
                </div>
            </div>
            
            <table id="cases-table">
                <thead>
                    <tr>
                        <th>Case</th>
                        <th>Prompt Tokens</th>
"""

    # Add column headers for each model
    for model in sorted_models:
        display_name = model_display_names.get(model, model)
        safe_model = model.replace("/", "_")
        html_content += f"""
                        <th class="model-col" data-model="{safe_model}">{display_name}</th>
"""

    html_content += """
                    </tr>
                </thead>
                <tbody>
"""

    # Add case rows
    for case in cases:
        case_prefix = case["prefix"]
        safe_case = case_prefix.replace("/", "_")
        truncated_name = truncate_case_name(case["original_filename"])
        html_content += f"""
                    <tr>
                        <td class="case-name">{truncated_name}</td>
                        <td>{case["token_count"]}</td>
"""

        # Add status buttons for each model
        for model in sorted_models:
            safe_model = model.replace("/", "_")
            result = case["model_results"][model]

            if result["tested"]:
                if result["success"]:
                    button_class = "success-button"
                    button_text = "Success"
                    html_content += f"""
                        <td class="model-col" data-model="{safe_model}">
                            <a href="cases/{safe_model}/{safe_case}.html" class="{button_class}">
                                {button_text}
                            </a>
                        </td>
"""
                else:
                    button_class = "failure-button"
                    button_text = "Failure"
                    html_content += f"""
                        <td class="model-col" data-model="{safe_model}">
                            <a href="cases/{safe_model}/{safe_case}.html" class="{button_class}">
                                {button_text}
                            </a>
                        </td>
"""
            else:
                # Don't add a link for untested cases
                button_class = "untested-button"
                button_text = "Not Tested"
                html_content += f"""
                        <td class="model-col" data-model="{safe_model}">
                            <span class="{button_class}">{button_text}</span>
                        </td>
"""

    # Complete the HTML
    html_content += """
                </tbody>
            </table>
        </section>
    </main>
    <footer>
        <p>LoCoDiff-bench - <a href="https://github.com/AbanteAI/LoCoDiff-bench">GitHub Repository</a></p>
    </footer>
    
    <script>
        // Add model column toggling functionality
        document.addEventListener('DOMContentLoaded', function() {
            // Function to save selected models to localStorage
            function saveSelectedModels() {
                const selectedModels = [];
                document.querySelectorAll('input[data-model]:checked').forEach(checkbox => {
                    selectedModels.push(checkbox.getAttribute('data-model'));
                });
                localStorage.setItem('locodiff_selected_models', JSON.stringify(selectedModels));
            }
                
            // Function to update visibility based on checkbox state
            function updateVisibility(checkbox) {
                const modelId = checkbox.getAttribute('data-model');
                const isVisible = checkbox.checked;
                    
                // Toggle visibility of corresponding table cells
                document.querySelectorAll(`th.model-col[data-model="${modelId}"], td.model-col[data-model="${modelId}"]`).forEach(cell => {
                    cell.style.display = isVisible ? '' : 'none';
                });
            }
                
            // Handle checkbox changes
            document.querySelectorAll('input[data-model]').forEach(checkbox => {
                checkbox.addEventListener('change', function() {
                    updateVisibility(this);
                    saveSelectedModels();
                });
            });
                
            // Restore selections from localStorage and apply visibility
            try {
                const savedModels = JSON.parse(localStorage.getItem('locodiff_selected_models') || '[]');
                    
                // First, uncheck all by default (in case there are no saved selections)
                document.querySelectorAll('input[data-model]').forEach(checkbox => {
                    checkbox.checked = false;
                });
                    
                // Check boxes for saved models if any exist
                if (savedModels.length > 0) {
                    savedModels.forEach(modelId => {
                        const checkbox = document.querySelector(`input[data-model="${modelId}"]`);
                        if (checkbox) checkbox.checked = true;
                    });
                } else {
                    // For first-time visitors, check these three models by default
                    const defaultModels = [
                        "anthropic_claude-3.7-sonnetthinking",
                        "anthropic_claude-sonnet-4",
                        "google_gemini-2.5-pro-preview",
                        "x-ai_grok-3-beta"
                    ];
                    
                    defaultModels.forEach(modelId => {
                        const checkbox = document.querySelector(`input[data-model="${modelId}"]`);
                        if (checkbox) {
                            checkbox.checked = true;
                            // Also update visibility to show these columns
                            updateVisibility(checkbox);
                        }
                    });
                }
                    
                // Apply visibility based on current checkbox states
                document.querySelectorAll('input[data-model]').forEach(checkbox => {
                    updateVisibility(checkbox);
                });
            } catch (error) {
                console.error('Error restoring model selections:', error);
                    
                // Fallback: ensure all columns have proper visibility
                document.querySelectorAll('input[data-model]').forEach(checkbox => {
                    updateVisibility(checkbox);
                });
            }
        });
    </script>
</body>
</html>
"""

    # Write the HTML file
    with open(cases_page_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated cases overview page: {cases_page_path}")


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
    <style>
        /* Custom styles for model-specific pages */
        #cases-table {{
            table-layout: fixed;
            width: 100%;
            border-collapse: collapse;
        }}
        
        #cases-table th, #cases-table td {{
            padding: 8px;
            border: 1px solid #e1e4e8;
            text-align: center;
        }}
        
        /* First column (case name) styling */
        #cases-table th:first-child, #cases-table td:first-child {{
            width: 240px; /* Fixed width as requested */
        }}
        
        /* Second column (prompt tokens) styling */
        #cases-table th:nth-child(2), #cases-table td:nth-child(2) {{
            width: 120px; /* Increased from 80px */
        }}
        
        /* Third column (Status) styling */
        #cases-table th:nth-child(3), #cases-table td:nth-child(3) {{
            width: 70px; /* Made smaller */
        }}
        
        /* Fourth column (Cost) styling */
        #cases-table th:nth-child(4), #cases-table td:nth-child(4) {{
            width: 80px; /* Made smaller */
        }}
        
        /* Fifth column (Actions) styling */
        #cases-table th:nth-child(5), #cases-table td:nth-child(5) {{
            width: 90px; /* Made smaller */
        }}
        
        /* Ensure text wraps properly */
        .case-name {{
            white-space: normal;
            overflow-wrap: break-word;
            text-align: center; /* Ensure center alignment */
        }}
        
        /* Zebra striping */
        #cases-table tbody tr:nth-child(odd) {{
            background-color: #f6f8fa;
        }}
    </style>
</head>
<body>
    <header>
        <h1>{display_name} - Benchmark Cases</h1>
        <p><a href="../index.html">← Back to Overview</a></p>
    </header>
    <main>
        <section>
            <h2>All Benchmark Cases</h2>
            
            <table id="cases-table">
                <thead>
                    <tr>
                        <th>Case</th>
                        <th>Prompt Tokens</th>
                        <th>Status</th>
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

        # Truncate case name if too long
        truncated_name = truncate_case_name(case["original_filename"])

        cost_display = (
            f"${case['cost_usd']:.6f}" if case["cost_usd"] is not None else "N/A"
        )
        html_content += f"""
                    <tr class="case-row {status_class}">
                        <td class="case-name">{truncated_name}</td>
                        <td>{case["prompt_tokens"]}</td>
                        <td class="{status_class}">{status_text}</td>
                        <td>{cost_display}</td>
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
        <h2>Model: {display_name}</h2>
        <p><a href="../../models/{safe_model}.html">All {display_name} Cases</a> | <a href="../../cases.html">All Cases</a> | <a href="../../index.html">Home</a></p>
    </header>
    <main>
        <section class="case-details">
            <div class="case-info">
                <h2>Benchmark Case Information</h2>
                <p><strong>Model:</strong> {display_name}</p>
                <p><strong>Status:</strong> <span class="{status_class}">{status_text}</span></p>
                <p><strong>Prompt Tokens:</strong> {case_metadata.get("prompt_tokens", "N/A")}</p>
                <p><strong>Native Prompt Tokens:</strong> {result_metadata.get("native_prompt_tokens", "N/A")}</p>
                <p><strong>Native Completion Tokens:</strong> {result_metadata.get("native_completion_tokens", "N/A")}</p>
                <p><strong>Native Tokens Reasoning:</strong> {result_metadata.get("native_tokens_reasoning", "N/A")}</p>
                <p><strong>Native Finish Reason:</strong> {result_metadata.get("native_finish_reason", "N/A")}</p>
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

    css_content = """/* Table of Contents styles */
#table-of-contents {
    margin: 20px 0 30px 0;
    padding: 15px 20px;
    background-color: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
}

.toc ul {
    list-style-type: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
}

.toc li {
    margin-bottom: 10px;
}

.toc a {
    display: inline-block;
    padding: 5px 12px;
    background-color: #eef2f5;
    border: 1px solid #d1d5da;
    border-radius: 4px;
    color: #0366d6;
    font-weight: 500;
    text-decoration: none;
    transition: background-color 0.2s;
}

.toc a:hover {
    background-color: #d1e4f6;
    text-decoration: none;
}

/* LoCoDiff Summary styles */
#locodiff-summary {
    margin: 30px 0;
    padding: 20px;
    background-color: #f8f9fa;
    border-radius: 6px;
    border: 1px solid #e1e4e8;
}

.locodiff-description p {
    font-size: 16px;
    line-height: 1.6;
    margin-bottom: 15px;
}

.mentat-contribution {
    background-color: #eef8ff;
    border: 1px solid #c8e1ff;
    border-radius: 6px;
    padding: 15px;
    margin-top: 20px;
}

.mentat-contribution p {
    margin: 0;
    color: #0366d6;
}

.mentat-contribution strong {
    color: #24292e;
}

/* Ranking styles for top performers - Using both borders and medal emojis */
td.gold {
    border: 3px solid #ffd700; /* Gold color */
    background-color: rgba(255, 215, 0, 0.1); /* Light gold background */
    font-weight: bold;
}

td.silver {
    border: 3px solid #c0c0c0; /* Silver color */
    background-color: rgba(192, 192, 192, 0.1); /* Light silver background */
    font-weight: bold;
}

td.bronze {
    border: 3px solid #cd7f32; /* Bronze color */
    background-color: rgba(205, 127, 50, 0.1); /* Light bronze background */
    font-weight: bold;
}

/* Basic Reset */
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

/* Explore options layout */
.explore-options {
    display: flex;
    flex-wrap: wrap;
    gap: 30px;
}

.model-view, .case-view {
    flex: 1;
    min-width: 300px;
}

.view-all-cases-button {
    display: inline-block;
    padding: 12px 20px;
    background-color: #2ea44f;
    color: white;
    border-radius: 4px;
    font-size: 16px;
    font-weight: 500;
    text-decoration: none;
    text-align: center;
    transition: background-color 0.2s;
    margin-top: 15px;
}

.view-all-cases-button:hover {
    background-color: #2c974b;
    text-decoration: none;
}

/* Benchmark Example Section Styles */
#benchmark-example {
    margin: 30px 0 50px 0;
    padding: 20px;
    background-color: #f8f9fa;
    border-radius: 8px;
    border: 1px solid #e1e4e8;
}

.intro-text {
    font-size: 16px;
    line-height: 1.6;
    margin-bottom: 20px;
}

.branch-diagram {
    font-family: monospace;
    text-align: center;
    font-size: 16px;
    line-height: 1.4;
    margin: 0;
    padding: 0;
    white-space: pre;
}

.example-timeline {
    margin: 20px 0;
}

.example-timeline h3 {
    text-align: center;
    margin-bottom: 15px;
}

.branch-structure-container {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    align-items: center;
    margin: 15px 0;
}

.branch-diagram-box {
    flex: 0 0 auto;
    background-color: white;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 15px;
    margin: 0 auto;
}

.branch-explanation {
    flex: 1;
    min-width: 300px;
    padding: 10px;
    text-align: left;
}

.branch-explanation p {
    line-height: 1.6;
    margin: 0 0 15px 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 14px;
}

.branch-explanation p:last-child {
    margin-bottom: 0;
}

/* Ensure consistent font styling for both paragraphs */
.branch-explanation p.commit-description,
.branch-explanation p.model-task {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #333;
}

.branch-explanation p.model-task {
    margin-top: 12px;
    font-size: 0.95em;
}

.branch-explanation p.model-task code {
    background-color: #f1f1f1;
    border-radius: 3px;
    padding: 2px 4px;
    font-family: Consolas, Monaco, 'Andale Mono', monospace;
    font-size: 0.9em;
}

/* Container for side-by-side display */
.example-io-container {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    margin: 20px 0;
    background-color: white;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 20px;
}

.example-prompt, .example-expected {
    flex: 1;
    min-width: 400px;
}

.example-task {
    margin: 20px 0;
    padding: 15px;
    background-color: white;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
}

.example-io-container h3 {
    padding-bottom: 5px;
    border-bottom: 1px solid #e1e4e8;
    margin-bottom: 15px;
    color: #24292e;
}

.example-prompt pre, .example-expected pre {
    /* No max-height restriction to show full content */
}

.example-task ul {
    margin-left: 20px;
    line-height: 1.6;
}

/* Cases overview table */
#cases-table {
    table-layout: fixed;
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
}

#cases-table th, #cases-table td {
    padding: 8px;
    text-align: center;
    white-space: nowrap;
}

/* The fixed-col classes have been replaced with first-child and nth-child selectors
   in the page-specific CSS for better control of backgrounds and positioning */

#cases-table th.model-col, #cases-table td.model-col {
    min-width: 120px;
}

.success-button, .failure-button, .untested-button {
    display: inline-block;
    width: 100px;
    padding: 5px 10px;
    border-radius: 4px;
    font-size: 12px;
    text-align: center;
    text-decoration: none;
    transition: opacity 0.2s;
}

.success-button {
    background-color: #22863a;
    color: white;
}

.failure-button {
    background-color: #cb2431;
    color: white;
}

.untested-button {
    background-color: #6a737d;
    color: white;
}

.success-button:hover, .failure-button:hover {
    opacity: 0.9;
    text-decoration: none;
    color: white;
}

.untested-button:hover {
    cursor: not-allowed;
    text-decoration: none;
    color: white;
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

/* Git diff syntax highlighting for the example */
/* Simple and clean CSS for diff highlighting */
.example-prompt pre code.language-diff {
    font-family: Consolas, Monaco, 'Andale Mono', monospace;
    font-size: 14px;
    line-height: 1.2;
    padding: 0;
    white-space: pre;
}

/* Make all spans display as blocks without extra spacing */
.example-prompt pre code.language-diff span {
    display: block;
    white-space: pre;
    margin: 0;
    padding: 0 8px;
}

/* Highlight.js generated classes */
.example-prompt pre code.language-diff span.hljs-addition {
    background-color: #e6ffec;
    color: #22863a;
    margin: 0;
    padding: 0 8px;
}

.example-prompt pre code.language-diff span.hljs-deletion {
    background-color: #ffebe9;
    color: #cb2431;
    margin: 0;
    padding: 0 8px;
}

/* Remove any extra spacing that might be causing gaps */
.example-prompt pre {
    margin: 0;
    padding: 0;
    white-space: pre;
}

.example-prompt pre code {
    margin: 0;
    padding: 8px;
    white-space: pre;
}

/* Make sure spans don't have extra margins/padding */
.example-prompt pre code.language-diff span:first-child {
    margin-top: 0;
}

.example-prompt pre code.language-diff span:last-child {
    margin-bottom: 0;
}

/* Style for empty lines to ensure they have height */
.example-prompt pre code.language-diff span.empty-line {
    height: 1.2em;
    line-height: 1.2;
    display: block;
}

/* Style for command note showing the git command */
.command-note {
    margin: 10px 0;
    background-color: #1e1e1e;
    border-radius: 4px;
    padding: 8px 12px;
    display: inline-block;
}

.command-note code {
    color: #f0f0f0;
    font-family: Consolas, Monaco, 'Andale Mono', monospace;
    font-size: 14px;
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
    position: fixed;
    bottom: 15px;
    right: 15px;
    padding: 10px;
    color: #586069;
    font-size: 14px;
    text-align: right;
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    z-index: 1000;
}

.footer-content {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
}

footer p {
    margin: 0;
    display: flex;
    align-items: center;
    gap: 6px;
}

.github-link, .mentat-link {
    color: #0366d6;
    font-weight: 500;
    transition: opacity 0.2s;
}

.github-link:hover, .mentat-link:hover {
    opacity: 0.8;
    text-decoration: none;
}

.github-icon, .mentat-icon {
    width: 16px;
    height: 16px;
    max-width: 16px;
    max-height: 16px;
    object-fit: contain;
}

.mentat-icon {
    border-radius: 3px;
    margin-left: 2px;
}

.built-with {
    font-weight: 400;
}"""

    return warning_comment + css_content


# --- Main Function ---


def truncate_case_name(name: str) -> str:
    """
    Extracts just the filename from a path, discarding directory information.

    Args:
        name: The case name or path

    Returns:
        Just the filename part of the path
    """
    # Extract just the filename (part after the last slash)
    return name.split("/")[-1].split("\\")[-1]


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

    # Copy static assets to the docs directory
    copy_static_assets(docs_dir)

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
    html_content += create_locodiff_summary()  # Add summary section first
    html_content += create_token_chart_section()
    html_content += (
        create_key_takeaways_section()
    )  # Add key takeaways before methodology
    html_content += create_example_section()
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

    # Generate cases overview page
    print("Generating cases overview page...")
    generate_cases_overview_page(
        prompt_metadata,
        results_metadata,
        all_models,
        docs_dir,
        model_display_names,
    )

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
