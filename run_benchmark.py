#!/usr/bin/env python3
import argparse
import os
import sys
import difflib
import json
import re
import shutil  # Moved import to top
import time  # Added for retries
import openai  # Added for APIError exception
from datetime import datetime, timezone
from utils import (
    get_model_response_openrouter,
    InvalidResponseError,
)  # Added InvalidResponseError


def sanitize_filename(name):
    """Removes characters that are problematic for filenames/paths."""
    # Replace slashes with underscores
    name = name.replace(os.path.sep, "_")
    # Remove other potentially problematic characters (add more as needed)
    name = re.sub(r'[<>:"|?*]', "", name)
    return name


def extract_code_from_backticks(text: str) -> str | None:
    """
    Extracts content wrapped in triple backticks, handling optional language identifiers
    and stripping leading/trailing whitespace.
    """
    # Regex explanation:
    # ```                  - Match the opening triple backticks
    # (?:\w+)?           - Optionally match (non-capturing group) a language identifier (word characters)
    # \s*?\n             - Match optional whitespace and a newline (non-greedy)
    # (.*?)              - Capture the content (non-greedy) - this is group 1
    # \n?```             - Match an optional newline and the closing triple backticks
    # re.DOTALL          - Make '.' match newline characters
    match = re.search(r"```(?:\w+)?\s*?\n(.*?)\n?```", text, re.DOTALL)

    if match:
        # Return the captured group, stripped of leading/trailing whitespace
        return match.group(1).strip()
    else:
        # Fallback: Maybe the model just returned the code without the newlines after/before backticks,
        # or maybe no language identifier was present.
        match = re.search(r"```(?:\w+)?(.*?)```", text, re.DOTALL)
        if match:
            # Return the captured group, stripped of leading/trailing whitespace
            return match.group(1).strip()
        return None  # Indicate backticks not found or no content


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Run a specific benchmark case against a model using OpenRouter."
    )
    # Default directory for benchmark files
    default_benchmark_dir = "generated_prompts"

    # Required arguments
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier for OpenRouter (e.g., 'openai/gpt-4o').",
    )
    parser.add_argument(
        "--benchmark-case",
        required=True,
        help=f"The common prefix for the benchmark files (prompt and expected output) located in '{default_benchmark_dir}'. "
        "Example: 'MyRepo_subdir_myfile_py'",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=default_benchmark_dir,
        help=f"Directory containing the benchmark files (default: '{default_benchmark_dir}').",
    )

    # Parse arguments
    args = parser.parse_args()

    # Construct file paths
    prompt_filename = f"{args.benchmark_case}_prompt.txt"
    expected_filename = f"{args.benchmark_case}_expectedoutput.txt"
    prompt_filepath = os.path.join(args.benchmark_dir, prompt_filename)
    expected_filepath = os.path.join(args.benchmark_dir, expected_filename)

    print("--- Starting Benchmark Test ---")
    print(f"Model: {args.model}")
    print(f"Benchmark Case Prefix: {args.benchmark_case}")
    print(f"Benchmark Directory: {args.benchmark_dir}")
    print(f"Prompt File: {prompt_filepath}")
    print(f"Expected Output File: {expected_filepath}")
    print("-" * 30)

    exit_code = 1  # Default to failure
    api_error = (
        None  # Initialize api_error here to ensure it's defined for the finally block
    )
    raw_model_response = (
        None  # Initialize raw_model_response for the finally block check
    )
    run_metadata = {
        "model": args.model,
        "benchmark_case": args.benchmark_case,
        "benchmark_dir": args.benchmark_dir,
        "prompt_file": prompt_filepath,
        "expected_file": expected_filepath,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),  # Use timezone-aware UTC now
        "success": False,
        "error": None,
        "raw_response_length": 0,
        "extracted_output_length": None,  # None if extraction fails
        "expected_output_length": 0,
        "results_dir": None,
    }

    try:
        # --- Setup Results Directory ---
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_model_name = sanitize_filename(args.model)
        results_base_dir = "benchmark_results"
        results_dir = os.path.join(
            results_base_dir,
            args.benchmark_case,
            sanitized_model_name,
            timestamp_str,
        )
        os.makedirs(results_dir, exist_ok=True)
        run_metadata["results_dir"] = results_dir
        print(f"Results will be saved to: {results_dir}")

        # --- Read Input Files ---
        print(f"Reading prompt file: {prompt_filepath}")
        if not os.path.exists(prompt_filepath):
            raise FileNotFoundError(f"Prompt file not found: {prompt_filepath}")
        with open(prompt_filepath, "r", encoding="utf-8") as f_prompt:
            prompt_content = f_prompt.read()
        print(f"Prompt read successfully ({len(prompt_content)} characters).")

        print(f"Reading expected output file: {expected_filepath}")
        if not os.path.exists(expected_filepath):
            raise FileNotFoundError(
                f"Expected output file not found: {expected_filepath}"
            )
        with open(expected_filepath, "r", encoding="utf-8") as f_expected:
            expected_content = f_expected.read()
        run_metadata["expected_output_length"] = len(expected_content)
        print(
            f"Expected output read successfully ({len(expected_content)} characters)."
        )

        # --- Call Model API with Retries ---
        print(f"Sending prompt to model '{args.model}' via OpenRouter...")
        # raw_model_response and api_error are initialized outside this try block
        MAX_RETRIES = 3
        INITIAL_BACKOFF = 1.0  # seconds
        backoff = INITIAL_BACKOFF
        api_call_succeeded = False

        for attempt in range(MAX_RETRIES):
            try:
                # Attempt the API call
                current_response = get_model_response_openrouter(
                    prompt_content, args.model
                )
                # If successful, store response, clear error, mark success, and break loop
                raw_model_response = current_response
                run_metadata["raw_response_length"] = len(raw_model_response)
                print(
                    f"Received response from model ({len(raw_model_response)} characters)."
                )
                api_error = None  # Clear any previous error on success
                api_call_succeeded = True
                break  # Exit retry loop on success
            except (openai.APIError, InvalidResponseError) as e:
                # If APIError or InvalidResponse, store the error and handle retry logic
                api_error = e  # Store the last error
                print(
                    f"Warning: API Error/Invalid Response (Attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                )
                if attempt < MAX_RETRIES - 1:
                    print(f"Retrying in {backoff:.1f} seconds...")
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff
                else:
                    # If max retries reached, print error and let loop finish
                    print("Error: Max retries reached. Failing benchmark run.")
                    # api_error already holds the last error

        # Check if the API call ultimately failed after retries
        if not api_call_succeeded:
            # If the loop finished without success, record the final error and set exit code
            print("\n❌ Error: Failed to get valid response from API after retries.")
            run_metadata["error"] = (
                f"API Failure after {MAX_RETRIES} retries: {api_error}"
            )
            exit_code = 1
            # Skip subsequent processing by raising a specific exception or returning early?
            # For now, subsequent code relies on raw_model_response being None
        else:
            # --- Save Raw Response (only if API call succeeded) ---
            raw_response_path = os.path.join(results_dir, "raw_response.txt")
            print(f"Saving raw response to: {raw_response_path}")
            # Type check: raw_model_response should be str here due to api_call_succeeded logic
            # Add assert for explicit type narrowing for pyright
            assert raw_model_response is not None
            with open(raw_response_path, "w", encoding="utf-8") as f_raw:
                f_raw.write(raw_model_response)

            # --- Extract Content ---
            print("Extracting content using triple backticks (```)...")
            # Add assert for explicit type narrowing for pyright
            assert raw_model_response is not None
            extracted_content = extract_code_from_backticks(raw_model_response)

            if extracted_content is None:
                print(
                    "❌ Error: Could not find triple backticks (```) wrapping the code in the response."
                )
                run_metadata["error"] = "Extraction backticks not found"
                exit_code = (
                    1  # Indicate failure (even though API succeeded, processing failed)
                )
            else:
                run_metadata["extracted_output_length"] = len(extracted_content)
                print(
                    f"Extracted content successfully ({len(extracted_content)} characters)."
                )
                extracted_output_path = os.path.join(
                    results_dir, "extracted_output.txt"
                )
                print(f"Saving extracted output to: {extracted_output_path}")
                with open(extracted_output_path, "w", encoding="utf-8") as f_ext:
                    f_ext.write(extracted_content)

                # --- Compare Extracted vs Expected ---
                print(
                    "Comparing extracted content to expected output (ignoring leading/trailing whitespace)..."
                )
                # Strip both before comparing
                extracted_stripped = extracted_content.strip()
                expected_stripped = expected_content.strip()

                if extracted_stripped == expected_stripped:
                    print(
                        "\n✅ Success: Stripped model output matches stripped expected output."
                    )
                    run_metadata["success"] = True
                    exit_code = 0
                else:
                    print(
                        "\n❌ Failure: Stripped model output does not match stripped expected output."
                    )
                    print("-" * 30)
                    print(
                        "Diff (Stripped Expected -> Stripped Extracted Model Output):"
                    )
                    print("-" * 30)
                    # Show diff of the stripped content
                    diff = difflib.unified_diff(
                        expected_stripped.splitlines(keepends=True),
                        extracted_stripped.splitlines(keepends=True),
                        fromfile=f"{expected_filepath} (stripped)",
                        tofile=f"{extracted_output_path} (stripped)",
                        lineterm="",
                    )
                    diff_lines = list(diff)
                    if diff_lines:
                        sys.stdout.writelines(diff_lines)
                    else:
                        print(
                            "(No differences found in line-by-line diff, check internal whitespace/characters)"
                        )
                    print()
                    print("-" * 30)
                    exit_code = 1  # Indicate failure

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        run_metadata["error"] = str(e)
        exit_code = 1
    except IOError as e:
        print(f"\nError reading file: {e}")
        run_metadata["error"] = f"IOError: {e}"
        exit_code = 1
    except ValueError as e:  # Catches missing API key from utils function
        print(f"\nConfiguration Error: {e}")
        run_metadata["error"] = f"ValueError: {e}"
        exit_code = 1
    except Exception as e:  # Catch API errors or other unexpected issues
        error_message = f"An unexpected error occurred during testing: {e}"
        print(f"\n{error_message}")
        run_metadata["error"] = error_message
        # Consider adding traceback logging here for debugging
        # import traceback
        # traceback.print_exc()
        exit_code = 1
    finally:
        # --- Save Metadata ---
        # Only save metadata if the results directory was created AND
        # the API call didn't fail terminally (i.e., raw_model_response is not None)
        # or if a non-API error occurred before the API call (e.g., FileNotFoundError).
        # Simpler: Only save if results_dir exists and the error wasn't an API failure after retries.
        should_save_metadata = run_metadata.get("results_dir") and not (
            raw_model_response is None and api_error is not None
        )

        if should_save_metadata:
            metadata_path = os.path.join(run_metadata["results_dir"], "metadata.json")
            try:
                print(f"Saving run metadata to: {metadata_path}")
                with open(metadata_path, "w", encoding="utf-8") as f_meta:
                    json.dump(run_metadata, f_meta, indent=4)
            except Exception as meta_e:
                print(f"\nWarning: Failed to save metadata.json: {meta_e}")
        elif run_metadata.get("results_dir"):
            # Clean up the results directory if it was created but the API call failed
            try:
                print(
                    f"Cleaning up results directory due to API failure: {run_metadata['results_dir']}"
                )
                # Be cautious with rmtree, ensure path is correct
                # import shutil # Moved to top

                # Add a safety check to prevent accidental deletion outside the expected base directory
                results_base_dir = (
                    "benchmark_results"  # Define it here for safety check
                )
                if os.path.exists(run_metadata["results_dir"]) and run_metadata[
                    "results_dir"
                ].startswith(results_base_dir + os.sep):
                    shutil.rmtree(run_metadata["results_dir"])
                else:
                    print(
                        f"\nWarning: Skipped cleanup for potentially unsafe path: {run_metadata['results_dir']}"
                    )
            except Exception as clean_e:
                print(
                    f"\nWarning: Failed to clean up results directory {run_metadata['results_dir']}: {clean_e}"
                )

    print("\n--- Test Complete ---")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
