#!/usr/bin/env python3
import argparse
import os
import sys
import difflib
import json
import re
from datetime import datetime, timezone  # Added timezone
from utils import get_model_response_openrouter


def sanitize_filename(name):
    """Removes characters that are problematic for filenames/paths."""
    # Replace slashes with underscores
    name = name.replace(os.path.sep, "_")
    # Remove other potentially problematic characters (add more as needed)
    name = re.sub(r'[<>:"|?*]', "", name)
    return name


def extract_code_from_tags(text: str) -> str | None:
    """Extracts content between <final_state_of_file> tags."""
    # Regex to find content between the tags, handling potential leading/trailing whitespace
    # DOTALL flag allows '.' to match newlines
    match = re.search(
        r"<final_state_of_file>(.*?)</final_state_of_file>", text, re.DOTALL
    )
    if match:
        # Strip leading/trailing whitespace from the captured group
        return match.group(1).strip()
    else:
        return None  # Indicate tags not found or no content


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

        # --- Call Model API ---
        print(f"Sending prompt to model '{args.model}' via OpenRouter...")
        raw_model_response = get_model_response_openrouter(prompt_content, args.model)
        run_metadata["raw_response_length"] = len(raw_model_response)
        print(f"Received response from model ({len(raw_model_response)} characters).")

        # --- Save Raw Response ---
        raw_response_path = os.path.join(results_dir, "raw_response.txt")
        print(f"Saving raw response to: {raw_response_path}")
        with open(raw_response_path, "w", encoding="utf-8") as f_raw:
            f_raw.write(raw_model_response)

        # --- Extract Content ---
        print("Extracting content using <final_state_of_file> tags...")
        extracted_content = extract_code_from_tags(raw_model_response)

        if extracted_content is None:
            print(
                "❌ Error: Could not find <final_state_of_file> tags in the response."
            )
            run_metadata["error"] = "Extraction tags not found"
            # Keep exit_code = 1 (failure)
        else:
            run_metadata["extracted_output_length"] = len(extracted_content)
            print(
                f"Extracted content successfully ({len(extracted_content)} characters)."
            )
            extracted_output_path = os.path.join(results_dir, "extracted_output.txt")
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
                print("Diff (Original Expected -> Original Extracted Model Output):")
                print("-" * 30)
                # Show diff of the original, unstripped content
                diff = difflib.unified_diff(
                    expected_content.splitlines(keepends=True),
                    extracted_content.splitlines(
                        keepends=True
                    ),  # Use original extracted content for diff
                    fromfile=expected_filepath,
                    tofile=extracted_output_path,  # Use path for clarity
                    lineterm="",
                )
                # Check if diff is empty (only whitespace changes) before printing
                diff_lines = list(diff)
                if diff_lines:
                    sys.stdout.writelines(diff_lines)
                else:
                    # This case should be less likely now if the stripped versions differ,
                    # but could happen if internal whitespace differs.
                    print(
                        "(No differences found in line-by-line diff, check internal whitespace/characters)"
                    )
                print()  # Add newline before separator
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
        if run_metadata.get("results_dir"):
            metadata_path = os.path.join(run_metadata["results_dir"], "metadata.json")
            try:
                print(f"Saving run metadata to: {metadata_path}")
                with open(metadata_path, "w", encoding="utf-8") as f_meta:
                    json.dump(run_metadata, f_meta, indent=4)
            except Exception as meta_e:
                print(f"\nWarning: Failed to save metadata.json: {meta_e}")

    print("\n--- Test Complete ---")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
