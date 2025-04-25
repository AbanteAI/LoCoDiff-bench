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
import os
import shutil
import sys
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


def load_language_config(config_path: str) -> Dict[str, str]:
    """
    Loads the language configuration and builds a map from extensions to language names.

    Args:
        config_path: Path to the languages.yaml file

    Returns:
        Dictionary mapping file extensions to language names
    """
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        ext_to_lang = {}
        for lang, settings in config.items():
            for ext in settings.get("extensions", []):
                ext_to_lang[ext] = lang

        return ext_to_lang
    except (yaml.YAMLError, IOError) as e:
        print(f"Error loading language config from {config_path}: {e}")
        # Fallback to basic extensions
        return {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".zig": "zig",
        }


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
    return f"""<!DOCTYPE html>
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


def create_html_footer() -> str:
    """Creates the HTML footer."""
    return """
    </main>
    <footer>
        <p>LoCoDiff-bench - <a href="https://github.com/AbanteAI/LoCoDiff-bench">GitHub Repository</a></p>
    </footer>
</body>
</html>
"""


def create_overall_stats_table(
    results_metadata: Dict[Tuple[str, str], Dict[str, Any]],
    all_models: Set[str],
    num_cases: int,
) -> str:
    """
    Creates an HTML table showing overall statistics for each model.

    Args:
        results_metadata: Dictionary mapping (case_prefix, model) to result metadata
        all_models: Set of all model names
        num_cases: Total number of benchmark cases

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

        html += f"""
                <tr>
                    <td>{model}</td>
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
    results_metadata: Dict[Tuple[str, str], Dict[str, Any]],
    prompt_metadata: Dict[str, Dict[str, Any]],
    all_models: Set[str],
) -> str:
    """
    Creates an HTML table showing success rates by prompt size quartiles.

    Args:
        results_metadata: Dictionary mapping (case_prefix, model) to result metadata
        prompt_metadata: Dictionary of prompt metadata by case prefix
        all_models: Set of all model names

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
        html += f"<tr><td>{model}</td>"

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
    results_metadata: Dict[Tuple[str, str], Dict[str, Any]],
    prompt_metadata: Dict[str, Dict[str, Any]],
    all_models: Set[str],
    ext_to_lang_map: Dict[str, str],
) -> str:
    """
    Creates an HTML table showing success rates by programming language.

    Args:
        results_metadata: Dictionary mapping (case_prefix, model) to result metadata
        prompt_metadata: Dictionary of prompt metadata by case prefix
        all_models: Set of all model names
        ext_to_lang_map: Dictionary mapping file extensions to language names

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
        html += f"<tr><td>{model}</td>"

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


def create_cases_placeholder() -> str:
    """Creates a placeholder section for individual benchmark cases."""
    return """
    <section id="individual-cases">
        <h2>Individual Benchmark Cases</h2>
        <p>Details for individual benchmark cases will be available in a future update.</p>
    </section>
    """


def create_css_file() -> str:
    """Creates a basic CSS stylesheet for the GitHub Pages site."""
    return """/* Basic Reset */
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

/* Footer */
footer {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #eaecef;
    color: #586069;
    font-size: 14px;
    text-align: center;
}
"""


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

    # Load language configuration
    languages_config_path = Path("benchmark_pipeline/languages.yaml")
    ext_to_lang_map = load_language_config(str(languages_config_path))

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

    # Generate HTML content
    print("Generating HTML content...")
    html_content = create_html_header()
    html_content += create_overall_stats_table(results_metadata, all_models, num_cases)
    html_content += create_quartile_stats_table(
        results_metadata, prompt_metadata, all_models
    )
    html_content += create_language_stats_table(
        results_metadata, prompt_metadata, all_models, ext_to_lang_map
    )
    html_content += create_cases_placeholder()
    html_content += create_html_footer()

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
