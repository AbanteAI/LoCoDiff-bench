#!/usr/bin/env python3
import argparse
import os
import sys
import difflib
import json
import re
import asyncio
import glob
from pathlib import Path
from datetime import datetime, timezone
from utils import (
    get_model_response_openrouter,
    get_generation_stats_openrouter,
)


def sanitize_filename(name):
    """Removes characters that are problematic for filenames/paths."""
    # Replace slashes with underscores
    name = name.replace(os.path.sep, "_").replace("/", "_")
    # Remove other potentially problematic characters
    name = re.sub(r'[<>:"|?*]', "", name)
    return name


def extract_code_from_backticks(text: str) -> str | None:
    """
    Extracts content wrapped in triple backticks, handling optional language identifiers
    and stripping leading/trailing whitespace.
    """
    match = re.search(r"```(?:\w+)?\s*?\n(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        match = re.search(r"```(?:\w+)?(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None


def find_benchmark_cases(benchmark_dir: str) -> list[str]:
    """Finds all benchmark case prefixes in the given directory."""
    prompt_files = glob.glob(os.path.join(benchmark_dir, "*_prompt.txt"))
    prefixes = set()
    for f in prompt_files:
        basename = os.path.basename(f)
        # Extract prefix by removing '_prompt.txt'
        prefix = basename[:-11]  # Length of '_prompt.txt' is 11
        prefixes.add(prefix)
    return sorted(list(prefixes))


def get_previous_run_status(
    benchmark_case_prefix: str, model: str, results_base_dir: str
) -> tuple[bool, float]:
    """
    Checks if a case/model has been run previously and returns its status and cost.

    Returns:
        A tuple (was_run: bool, cost: float).
        - was_run is True if any result directory exists.
        - cost is the cost_usd from the *latest* run's metadata, or 0.0 if
          no run exists, metadata is missing/unreadable, or cost is not recorded.
    """
    sanitized_model_name = sanitize_filename(model)
    pattern = os.path.join(
        results_base_dir, benchmark_case_prefix, sanitized_model_name, "*"
    )
    potential_dirs = glob.glob(pattern)

    latest_dir = None
    latest_timestamp = ""

    for result_dir in potential_dirs:
        if not os.path.isdir(result_dir):
            continue
        dir_name = os.path.basename(result_dir)
        # Check if it looks like a timestamp directory and find the latest
        if re.match(r"\d{8}_\d{6}", dir_name):
            if dir_name > latest_timestamp:
                latest_timestamp = dir_name
                latest_dir = result_dir

    if latest_dir is None:
        return False, 0.0  # Not run

    # Found at least one run attempt, try to get cost from the latest
    metadata_path = os.path.join(latest_dir, "metadata.json")
    cost = 0.0
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            # Get cost, default to 0.0 if key missing or value is None/invalid
            cost = float(metadata.get("cost_usd", 0.0) or 0.0)
        except (json.JSONDecodeError, IOError, ValueError, TypeError) as e:
            print(
                f"Warning: Could not read/parse metadata or cost for {latest_dir}: {e}"
            )
            cost = 0.0  # Treat as 0 cost if metadata is problematic

    return True, cost  # Was run, return cost (might be 0.0)


async def run_single_benchmark(
    benchmark_case_prefix: str,
    model: str,
    benchmark_dir: str,
    results_base_dir: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Runs a single benchmark case asynchronously."""
    async with semaphore:
        prompt_filename = f"{benchmark_case_prefix}_prompt.txt"
        expected_filename = f"{benchmark_case_prefix}_expectedoutput.txt"
        prompt_filepath = os.path.join(benchmark_dir, prompt_filename)
        expected_filepath = os.path.join(benchmark_dir, expected_filename)

        print(f"Starting benchmark: {benchmark_case_prefix} with model {model}")

        run_metadata = {
            "model": model,
            "benchmark_case": benchmark_case_prefix,
            "benchmark_dir": benchmark_dir,
            "prompt_file": prompt_filepath,
            "expected_file": expected_filepath,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "success": False,
            "error": None,
            "raw_response_length": 0,
            "extracted_output_length": None,
            "expected_output_length": 0,
            "results_dir": None,
            "generation_id": None,
            "cost_usd": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "native_prompt_tokens": None,
            "native_completion_tokens": None,
            "stats_error": None,
        }

        results_dir = None  # Define results_dir outside try block for finally

        try:
            # --- Setup Results Directory ---
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_model_name = sanitize_filename(model)
            results_dir = os.path.join(
                results_base_dir,
                benchmark_case_prefix,
                sanitized_model_name,
                timestamp_str,
            )
            # Use pathlib for potentially deeper paths and easier creation
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            run_metadata["results_dir"] = results_dir
            # print(f"Results for {benchmark_case_prefix} will be saved to: {results_dir}") # Less verbose

            # --- Read Input Files ---
            if not os.path.exists(prompt_filepath):
                raise FileNotFoundError(f"Prompt file not found: {prompt_filepath}")
            with open(prompt_filepath, "r", encoding="utf-8") as f_prompt:
                prompt_content = f_prompt.read()

            if not os.path.exists(expected_filepath):
                raise FileNotFoundError(
                    f"Expected output file not found: {expected_filepath}"
                )
            with open(expected_filepath, "r", encoding="utf-8") as f_expected:
                expected_content = f_expected.read()
            run_metadata["expected_output_length"] = len(expected_content)

            # --- Call Model API (Async) ---
            (
                raw_model_response,
                generation_id,
                api_error_message,
            ) = await get_model_response_openrouter(prompt_content, model)

            # --- Handle API Errors ---
            if api_error_message:
                run_metadata["error"] = api_error_message
                run_metadata["api_error"] = True  # Flag for specific handling
                # Print specific API error message and skip saving results
                print(
                    f"⚠️ API Error: {benchmark_case_prefix} - Error: {api_error_message} - Skipping results."
                )
                # DO NOT save metadata or other files for this run
                # Return metadata indicating API failure
                return run_metadata

            # --- Process Successful API Response ---
            # These steps only run if api_error_message is None
            run_metadata["raw_response_length"] = len(raw_model_response)
            run_metadata["generation_id"] = generation_id

            # --- Get Generation Stats (Async) ---
            if generation_id:
                # Add a small delay as stats might not be immediately available
                await asyncio.sleep(0.5)
                stats = await get_generation_stats_openrouter(generation_id)
                if stats:
                    run_metadata.update(stats)
                else:
                    run_metadata["stats_error"] = (
                        "Failed to retrieve stats from OpenRouter API"
                    )
            else:
                run_metadata["stats_error"] = (
                    "No generation ID received from chat completion"
                )

            # --- Save Raw Response ---
            raw_response_path = os.path.join(results_dir, "raw_response.txt")
            with open(raw_response_path, "w", encoding="utf-8") as f_raw:
                f_raw.write(raw_model_response)

            # --- Extract Content ---
            extracted_content = extract_code_from_backticks(raw_model_response)

            if extracted_content is None:
                run_metadata["error"] = "Extraction backticks not found"
                # Keep success=False
            else:
                run_metadata["extracted_output_length"] = len(extracted_content)
                extracted_output_path = os.path.join(
                    results_dir, "extracted_output.txt"
                )
                with open(extracted_output_path, "w", encoding="utf-8") as f_ext:
                    f_ext.write(extracted_content)

                # --- Compare Extracted vs Expected ---
                extracted_stripped = extracted_content.strip()
                expected_stripped = expected_content.strip()

                if extracted_stripped == expected_stripped:
                    run_metadata["success"] = True
                else:
                    # Keep the error message for logging/metadata
                    run_metadata["error"] = "Output mismatch"
                    # Success remains False

                # --- Always Generate Diff File ---
                diff_path = os.path.join(results_dir, "output.diff")
                try:
                    if run_metadata["success"]:
                        # If successful, write a "no diff" message
                        with open(diff_path, "w", encoding="utf-8") as f_diff:
                            f_diff.write("No differences found.\n")
                    else:
                        # If mismatch, generate and write the actual diff
                        diff = difflib.unified_diff(
                            expected_stripped.splitlines(keepends=True),
                            extracted_stripped.splitlines(keepends=True),
                            fromfile=f"{expected_filename} (expected)",
                            tofile=f"{benchmark_case_prefix}_extracted.txt (actual)",
                            lineterm="",
                        )
                        with open(diff_path, "w", encoding="utf-8") as f_diff:
                            f_diff.writelines(diff)
                except Exception as diff_e:
                    print(f"Warning: Failed to generate/save diff file: {diff_e}")
                    # Optionally, write the error to the diff file itself
                    try:
                        with open(diff_path, "w", encoding="utf-8") as f_diff_err:
                            f_diff_err.write(f"Error generating diff: {diff_e}\n")
                    except Exception:
                        pass  # Ignore errors writing the error message

            # --- Save Metadata (only if no API error occurred earlier) ---
            metadata_path = os.path.join(results_dir, "metadata.json")
            try:
                with open(metadata_path, "w", encoding="utf-8") as f_meta:
                    json.dump(run_metadata, f_meta, indent=4)
            except Exception as meta_e:
                print(
                    f"\nWarning: Failed to save metadata.json for {benchmark_case_prefix}: {meta_e}"
                )

        except FileNotFoundError as e:
            run_metadata["error"] = f"File Error: {e}"
            # Save metadata for file errors
            if results_dir:
                metadata_path = os.path.join(results_dir, "metadata.json")
                try:
                    with open(metadata_path, "w", encoding="utf-8") as f_meta:
                        json.dump(run_metadata, f_meta, indent=4)
                except Exception as meta_e:
                    print(
                        f"\nWarning: Failed to save metadata.json after File Error for {benchmark_case_prefix}: {meta_e}"
                    )
        except IOError as e:
            run_metadata["error"] = f"IOError: {e}"
            # Save metadata for IO errors
            if results_dir:
                metadata_path = os.path.join(results_dir, "metadata.json")
                try:
                    with open(metadata_path, "w", encoding="utf-8") as f_meta:
                        json.dump(run_metadata, f_meta, indent=4)
                except Exception as meta_e:
                    print(
                        f"\nWarning: Failed to save metadata.json after IO Error for {benchmark_case_prefix}: {meta_e}"
                    )
        except ValueError as e:  # Catches missing API key from utils
            run_metadata["error"] = f"Config Error: {e}"
            # Don't save metadata if config error prevented API call attempt
            print(f"Config Error for {benchmark_case_prefix}: {e} - Skipping results.")
        except Exception as e:  # Catch other unexpected issues during processing
            run_metadata["error"] = f"Runtime Error: {type(e).__name__}: {e}"
            # Save metadata for runtime errors during processing
            if results_dir:
                metadata_path = os.path.join(results_dir, "metadata.json")
                try:
                    with open(metadata_path, "w", encoding="utf-8") as f_meta:
                        json.dump(run_metadata, f_meta, indent=4)
                except Exception as meta_e:
                    print(
                        f"\nWarning: Failed to save metadata.json after Runtime Error for {benchmark_case_prefix}: {meta_e}"
                    )
            # Optional: Log traceback for debugging
            # import traceback
            # run_metadata["traceback"] = traceback.format_exc()

        # --- Print Final Status for this Case ---
        # Check if it was an API error (already printed specific message)
        if not run_metadata.get("api_error"):
            cost_str = (
                f"Cost: ${run_metadata.get('cost_usd', 0.0):.6f}"
                if run_metadata.get("cost_usd") is not None
                else "Cost: N/A"
            )
            if run_metadata["success"]:
                print(f"✅ Success: {benchmark_case_prefix} - {cost_str}")
            else:
                # Use the error stored in metadata (could be extraction, mismatch, file error, etc.)
                error_msg = run_metadata.get("error", "Unknown processing error")
                print(
                    f"❌ Failure: {benchmark_case_prefix} - Error: {error_msg} - {cost_str}"
                )

        return run_metadata


async def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark cases against a model using OpenRouter, checking for existing results and running concurrently."
    )
    default_benchmark_dir = "generated_prompts"
    default_results_dir = "benchmark_results"

    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier for OpenRouter (e.g., 'openai/gpt-4o').",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=default_benchmark_dir,
        help=f"Directory containing the benchmark prompt/expected files (default: '{default_benchmark_dir}').",
    )
    parser.add_argument(
        "--results-dir",
        default=default_results_dir,
        help=f"Base directory to save results (default: '{default_results_dir}').",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=0,
        help="Maximum number of new benchmarks to run. Set to -1 to run all remaining. (default: 0 - just show status).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of benchmarks to run concurrently (default: 1).",
    )

    args = parser.parse_args()

    print("--- Starting Benchmark Run ---")
    print(f"Model: {args.model}")
    print(f"Benchmark Directory: {args.benchmark_dir}")
    print(f"Results Base Directory: {args.results_dir}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max New Runs: {'All Remaining' if args.num_runs == -1 else args.num_runs}")
    print("-" * 30)

    all_cases = find_benchmark_cases(args.benchmark_dir)
    if not all_cases:
        print(f"Error: No benchmark cases found in '{args.benchmark_dir}'.")
        return 1

    print(f"Found {len(all_cases)} total benchmark cases.")

    already_run_cases = set()
    total_previous_cost = 0.0
    print("Checking for existing results and calculating previous costs...")
    for case_prefix in all_cases:
        was_run, cost = get_previous_run_status(
            case_prefix, args.model, args.results_dir
        )
        if was_run:
            already_run_cases.add(case_prefix)
            total_previous_cost += cost

    print(
        f"{len(already_run_cases)}/{len(all_cases)} cases have already been run for model '{args.model}'."
    )
    print(f"Total cost of previously run cases: ${total_previous_cost:.6f}")
    print("-" * 30)

    # Determine cases to run (those not in the already_run_cases set)
    cases_to_run_all = [case for case in all_cases if case not in already_run_cases]

    if not cases_to_run_all:
        print("No remaining benchmark cases to run for this model.")
        print("--- Benchmark Run Complete ---")
        return 0

    # Determine cases to run based on limit
    if args.num_runs == 0:
        cases_to_run_limited = []
        print(
            "Running in informational mode (num-runs=0). No new benchmarks will be executed."
        )
    elif args.num_runs == -1:
        cases_to_run_limited = cases_to_run_all
        print(
            f"Preparing to run all {len(cases_to_run_limited)} remaining benchmarks..."
        )
    else:
        cases_to_run_limited = cases_to_run_all[: args.num_runs]
        if not cases_to_run_limited:
            # This covers cases where num_runs > 0 but no cases are left within the slice
            print("Limit specified, but no remaining cases to run within that limit.")
        else:
            print(
                f"Preparing to run up to {args.num_runs} new benchmarks ({len(cases_to_run_limited)} available within limit)..."
            )

    if not cases_to_run_limited:
        # This handles both num_runs=0 and cases where the limit is met/exceeded
        print("--- Benchmark Run Complete ---")
        return 0

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(
            run_single_benchmark(
                case, args.model, args.benchmark_dir, args.results_dir, semaphore
            )
        )
        for case in cases_to_run_limited
    ]

    # Run tasks and collect results (metadata dictionaries or exceptions)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results of newly run benchmarks
    success_count = 0
    failure_count = 0  # Failures due to mismatch, extraction error, file error etc.
    api_error_count = 0  # Failures due to API call issues (credits, rate limits etc.)
    system_error_count = (
        0  # Failures due to unexpected exceptions in gather/task execution
    )
    total_new_cost = 0.0  # Cost for runs executed in this session

    for result in results:
        if isinstance(result, Exception):
            # Exception occurred within run_single_benchmark or gather itself
            print(
                f"❌ System Error: An unexpected error occurred during task execution: {result}"
            )
            system_error_count += 1
        elif isinstance(result, dict):
            # Got metadata back from run_single_benchmark
            # Accumulate cost only if it wasn't an API error (where cost might be irrelevant or missing)
            # and cost is actually present
            if not result.get("api_error") and result.get("cost_usd") is not None:
                total_new_cost += (
                    result.get("cost_usd") or 0.0
                )  # Use 'or 0.0' for safety

            # Categorize the result
            if result.get("api_error"):
                api_error_count += 1
                # Specific API error message already printed in run_single_benchmark
            elif result.get("success"):
                success_count += 1
                # Individual success/cost already printed in run_single_benchmark
            else:
                # This covers failures like mismatch, extraction, file errors, runtime errors during processing
                failure_count += 1
                # Individual failure/error already printed in run_single_benchmark
        else:
            # Should not happen if run_single_benchmark always returns dict
            print(f"❌ System Error: Unexpected result type from task: {type(result)}")
            system_error_count += 1

    print("\n--- Benchmark Run Summary ---")
    print(f"Model: {args.model}")
    print(f"Attempted in this run: {len(results)} benchmarks")
    print(f"  ✅ Successful: {success_count}")
    print(f"  ❌ Failed (Mismatch/Extraction/File Error): {failure_count}")
    print(f"  ⚠️ API Errors (Credits/Rate Limits/etc.): {api_error_count}")
    if system_error_count > 0:
        print(f"  🔥 System Errors (Unexpected Task Failures): {system_error_count}")
    print("-" * 20)
    print(f"Cost of this run (successful/failed runs only): ${total_new_cost:.6f}")
    print(f"Total cost of previous runs: ${total_previous_cost:.6f}")
    print(
        f"Overall total cost (previous + current): ${total_previous_cost + total_new_cost:.6f}"
    )
    print("--- Benchmark Run Complete ---")

    # Return failure if any benchmarks failed (non-API/System errors) or had API/System errors in this run
    return 1 if (failure_count + api_error_count + system_error_count) > 0 else 0


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
