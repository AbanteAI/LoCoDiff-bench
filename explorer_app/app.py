#!/usr/bin/env python3
import os
import json
import glob
import re
from flask import Flask, render_template, abort, url_for
from markupsafe import escape
import shutil

# --- Configuration ---
# Assuming the app is run from the root of the repository
BENCHMARK_DIR = "generated_prompts"
RESULTS_BASE_DIR = "benchmark_results"
PLOT_FILENAME = "benchmark_success_rate.png"  # Original plot location
STATIC_DIR = "explorer_app/static"
PLOT_STATIC_PATH = os.path.join(STATIC_DIR, PLOT_FILENAME)

# --- Flask App Initialization ---
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["BENCHMARK_DIR"] = BENCHMARK_DIR
app.config["RESULTS_BASE_DIR"] = RESULTS_BASE_DIR
app.config["STATIC_DIR"] = (
    STATIC_DIR  # Make static dir accessible in templates if needed
)

# --- Helper Functions ---


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


def find_models_in_results(results_base_dir):
    """Finds all unique model names present in the results directory structure."""
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
        "prompt_content": None,
        "expected_content": None,
        "raw_response": None,
        "extracted_output": None,
        "diff_content": None,
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
    prompt_filepath = os.path.join(benchmark_dir, prompt_filename)
    expected_filepath = os.path.join(benchmark_dir, expected_filename)

    try:
        with open(prompt_filepath, "r", encoding="utf-8") as f:
            details["prompt_content"] = f.read()
    except IOError as e:
        details["error"] = details.get("error", "") + f" Error loading prompt file: {e}"

    try:
        with open(expected_filepath, "r", encoding="utf-8") as f:
            details["expected_content"] = f.read()
    except IOError as e:
        details["error"] = (
            details.get("error", "") + f" Error loading expected output file: {e}"
        )

    # Load files from run_dir
    def read_run_file(filename):
        filepath = os.path.join(run_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return f.read()
            except IOError as e:
                details["error"] = (
                    details.get("error", "") + f" Error loading {filename}: {e}"
                )
        return None  # Or indicate file not found?

    details["raw_response"] = read_run_file("raw_response.txt")
    details["extracted_output"] = read_run_file("extracted_output.txt")
    details["diff_content"] = read_run_file("output.diff")

    return details


def copy_plot_to_static():
    """Copies the benchmark plot to the static directory if it exists."""
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR, exist_ok=True)

    if os.path.exists(PLOT_FILENAME):
        try:
            shutil.copy2(PLOT_FILENAME, PLOT_STATIC_PATH)  # copy2 preserves metadata
            print(f"Copied {PLOT_FILENAME} to {PLOT_STATIC_PATH}")
            return True
        except Exception as e:
            print(f"Error copying plot file: {e}")
            return False
    else:
        print(
            f"Warning: Plot file {PLOT_FILENAME} not found. Cannot copy to static dir."
        )
        # Check if it already exists in static dir from a previous run
        return os.path.exists(PLOT_STATIC_PATH)


# --- Routes ---


@app.route("/")
def index():
    """Index page: Shows overall summary, models, and plot."""
    benchmark_metadata = load_benchmark_metadata(app.config["BENCHMARK_DIR"])
    models = find_models_in_results(app.config["RESULTS_BASE_DIR"])
    plot_exists = copy_plot_to_static()  # Ensure plot is in static dir for serving
    plot_url = url_for("static", filename=PLOT_FILENAME) if plot_exists else None

    # Basic stats (can be expanded)
    total_cases = 0
    if benchmark_metadata and "benchmark_buckets" in benchmark_metadata:
        for bucket_key, cases in benchmark_metadata["benchmark_buckets"].items():
            total_cases += len(cases)

    return render_template(
        "index.html",
        models=models,
        total_cases=total_cases,
        plot_url=plot_url,
        benchmark_metadata=benchmark_metadata,  # Pass metadata for potential display
    )


@app.route("/model/<path:model_name>")
def model_results(model_name):
    """Shows results for a specific model, grouped by bucket."""
    # model_name might have been sanitized (e.g., openai/gpt-4o -> openai_gpt-4o)
    # We need to find runs using the name found in the filesystem.
    # The find_models_in_results should return the filesystem name.
    safe_model_name = escape(model_name)  # Escape for display

    benchmark_metadata = load_benchmark_metadata(app.config["BENCHMARK_DIR"])
    all_runs = find_runs_for_model(
        model_name, app.config["RESULTS_BASE_DIR"]
    )  # Use original name for searching

    if not all_runs and not any(
        m == model_name for m in find_models_in_results(app.config["RESULTS_BASE_DIR"])
    ):
        abort(404, description=f"Model '{safe_model_name}' not found in results.")

    runs_by_bucket = {}
    if benchmark_metadata and "benchmark_buckets" in benchmark_metadata:
        # Initialize buckets based on metadata
        for bucket_key in benchmark_metadata["benchmark_buckets"]:
            runs_by_bucket[bucket_key] = []

        # Assign runs to buckets
        case_to_bucket = {}
        for bucket_key, cases in benchmark_metadata["benchmark_buckets"].items():
            for case_info in cases:
                case_to_bucket[case_info["benchmark_case_prefix"]] = bucket_key

        for run in all_runs:
            bucket_key = case_to_bucket.get(run["benchmark_case_prefix"])
            if bucket_key:
                if bucket_key not in runs_by_bucket:
                    runs_by_bucket[
                        bucket_key
                    ] = []  # Should exist from init, but safety check
                runs_by_bucket[bucket_key].append(run)
            else:
                # Handle runs for cases not found in current metadata?
                if "unknown" not in runs_by_bucket:
                    runs_by_bucket["unknown"] = []
                runs_by_bucket["unknown"].append(run)

        # Sort buckets numerically by lower bound for display
        sorted_bucket_keys = sorted(
            runs_by_bucket.keys(),
            key=lambda k: int(k.split("-")[0]) if k != "unknown" else float("inf"),
        )
        sorted_runs_by_bucket = {k: runs_by_bucket[k] for k in sorted_bucket_keys}

    else:
        # If no metadata, just list all runs without bucketing
        sorted_runs_by_bucket = {"all_runs": all_runs}

    return render_template(
        "model_results.html",
        model_name=safe_model_name,  # Display escaped name
        original_model_name=model_name,  # Pass original name for link generation
        runs_by_bucket=sorted_runs_by_bucket,
        benchmark_metadata=benchmark_metadata,  # Pass metadata for context
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


if __name__ == "__main__":
    # Ensure plot is available before starting
    copy_plot_to_static()
    # Add host='0.0.0.0' to make accessible on network
    app.run(debug=True, host="0.0.0.0")
