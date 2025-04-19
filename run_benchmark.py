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


def check_if_already_run(
    benchmark_case_prefix: str, model: str, results_base_dir: str
) -> bool:
    """Checks if *any* result directory exists for this case/model, indicating it has been run."""
    sanitized_model_name = sanitize_filename(model)
    # Look for any timestamped directory within the case/model structure
    pattern = os.path.join(
        results_base_dir, benchmark_case_prefix, sanitized_model_name, "*"
    )
    potential_dirs = glob.glob(pattern)

    for result_dir in potential_dirs:
        # If we find any directory matching the pattern, it means a run was attempted.
        # We don't need to check metadata.json or success status anymore.
        if os.path.isdir(result_dir):
            # Check if it looks like a timestamp directory (e.g., YYYYMMDD_HHMMSS)
            # This is a basic check to avoid matching unrelated directories if any exist.
            dir_name = os.path.basename(result_dir)
            if re.match(r"\d{8}_\d{6}", dir_name):
                return True  # Found evidence of a previous run attempt

    # No directory indicating a previous run attempt was found
    return False


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
            raw_model_response, generation_id = await get_model_response_openrouter(
                prompt_content, model
            )
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
                    run_metadata["error"] = "Output mismatch"
                    # Optionally save diff
                    diff_path = os.path.join(results_dir, "output.diff")
                    try:
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

        except FileNotFoundError as e:
            run_metadata["error"] = f"File Error: {e}"
        except IOError as e:
            run_metadata["error"] = f"IOError: {e}"
        except ValueError as e:  # Catches missing API key
            run_metadata["error"] = f"Config Error: {e}"
        except Exception as e:  # Catch API errors or other unexpected issues
            run_metadata["error"] = f"Runtime Error: {type(e).__name__}: {e}"
            # Optional: Log traceback for debugging
            # import traceback
            # run_metadata["traceback"] = traceback.format_exc()
        finally:
            # --- Save Metadata ---
            if results_dir:  # Ensure results_dir was assigned
                metadata_path = os.path.join(results_dir, "metadata.json")
                try:
                    with open(metadata_path, "w", encoding="utf-8") as f_meta:
                        json.dump(run_metadata, f_meta, indent=4)
                except Exception as meta_e:
                    print(
                        f"\nWarning: Failed to save metadata.json for {benchmark_case_prefix}: {meta_e}"
                    )

        # Print result immediately after completion
        cost_str = (
            f"Cost: ${run_metadata.get('cost_usd', 0.0):.6f}"
            if run_metadata.get("cost_usd") is not None
            else "Cost: N/A"
        )
        if run_metadata["success"]:
            print(f"✅ Success: {benchmark_case_prefix} - {cost_str}")
        else:
            error_msg = run_metadata.get("error", "Unknown error")
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
    print("Checking for existing results (any previous run attempt)...")
    for case_prefix in all_cases:
        # Use the renamed function
        if check_if_already_run(case_prefix, args.model, args.results_dir):
            already_run_cases.add(case_prefix)

    print(
        f"{len(already_run_cases)}/{len(all_cases)} cases have already been run (attempted) for model '{args.model}'."
    )

    # Determine cases to run (those not in the already_run_cases set)
    cases_to_run_all = [case for case in all_cases if case not in already_run_cases]

    if not cases_to_run_all:
        print("All benchmark cases for this model are already completed.")
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

    # Process results
    success_count = 0
    failure_count = 0
    total_cost = 0.0

    for result in results:
        if isinstance(result, Exception):
            # Exception occurred within run_single_benchmark or gather itself
            print(
                f"❌ System Error: An unexpected error occurred during task execution: {result}"
            )
            failure_count += 1
        elif isinstance(result, dict):
            # Got metadata back from run_single_benchmark
            # Accumulate cost regardless of success/failure, if available
            total_cost += result.get("cost_usd", 0.0)

            if result.get("success"):
                success_count += 1
                # Individual success/cost already printed in run_single_benchmark
            else:
                failure_count += 1
                # Individual failure/error already printed in run_single_benchmark
        else:
            # Should not happen if run_single_benchmark always returns dict
            print(f"❌ System Error: Unexpected result type from task: {type(result)}")
            failure_count += 1

    print("\n--- Benchmark Run Summary ---")
    print(f"Model: {args.model}")
    print(f"Attempted: {len(results)} benchmarks")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Total Cost (All Attempted Runs): ${total_cost:.6f}")
    print("--- Benchmark Run Complete ---")

    return 1 if failure_count > 0 else 0


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
