#!/usr/bin/env python3
"""
Update existing benchmark results to use the new extraction logic.

This script searches for benchmark cases that were marked as failing due to
"Extraction backticks not found" and updates them to use the new extraction logic,
which treats the entire response as the extracted content when no backticks are found.

Usage:
    python benchmark_pipeline/update_extraction_results.py --benchmark-run-dir BENCHMARK_DIR

Arguments:
    --benchmark-run-dir (required): Path to the benchmark run directory containing results.
    --dry-run (optional): If set, only print what would be changed without making changes.
"""

import argparse
import difflib
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


def extract_code_from_backticks(text: str) -> str:
    """
    Updated extraction function that matches the new logic in 2_run_benchmark.py.

    If no backticks are found, returns the entire text as is.
    Returns empty string only if the input text is empty.
    """
    if not text or text.isspace():
        # Return empty string only if input is empty or only whitespace
        return ""

    try:
        # Find the start of the first ``` block
        start_outer = text.find("```")
        if start_outer == -1:
            # No backticks found, return the entire text
            return text.strip()

        # Find the end of the first ``` marker (including optional language and newline)
        # Use regex to find the end position after ```, optional language, and optional newline
        start_inner_match = re.search(r"```(?:\w+)?\s*?\n?", text[start_outer:])
        if start_inner_match:
            # Calculate the index in the original string where the actual content begins
            start_inner = start_outer + start_inner_match.end()
        else:
            # Fallback if regex fails (e.g., ``` immediately followed by content without newline)
            # Find the end of the initial ``` marker itself
            start_inner = start_outer + 3  # Length of ```

        # Find the start of the last ``` block using rfind
        end_outer = text.rfind("```")

        # Check if the last ``` was found and if it's after the first ``` marker ended
        if end_outer == -1 or end_outer < start_inner:
            # No closing backticks found, or they are invalid.
            # Extract from the start marker to the end of the string.
            extracted_content = text[start_inner:]
        else:
            # Both opening and closing backticks are valid
            # Extract the content between the end of the first marker and the start of the last marker
            extracted_content = text[start_inner:end_outer]

        return extracted_content.strip()

    except Exception:
        # Return the whole text on error instead of None
        return text.strip()


def _save_text_file(filepath: Path, content: str):
    """Helper to save text content to a file, creating parent dirs."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
    except IOError as e:
        # Raise the error to be caught by the caller
        raise IOError(f"Failed to write file {filepath}: {e}") from e


def find_affected_runs(results_base_dir: Path) -> List[Path]:
    """
    Find all benchmark runs that were marked as failing due to "Extraction backticks not found".

    Args:
        results_base_dir: Path to the benchmark results directory.

    Returns:
        List of paths to affected metadata.json files.
    """
    affected_runs = []

    # Find all metadata.json files recursively
    metadata_files = glob.glob(
        os.path.join(results_base_dir, "**", "metadata.json"), recursive=True
    )

    for metadata_path in metadata_files:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Check if this run was marked as failing due to "Extraction backticks not found"
            if (
                not metadata.get("success", False)
                and metadata.get("error") == "Extraction backticks not found"
            ):
                affected_runs.append(Path(metadata_path))
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error reading metadata file {metadata_path}: {e}")

    return affected_runs


def update_run(
    run_path: Path, expected_content: str, dry_run: bool = False
) -> Tuple[bool, str]:
    """
    Update a single benchmark run with the new extraction logic.

    Args:
        run_path: Path to the metadata.json file of the run to update.
        expected_content: Content of the expected output file.
        dry_run: If True, only print what would be changed without making changes.

    Returns:
        Tuple of (success_status, message) where success_status indicates if
        the run would now pass, and message contains details about the update.
    """
    results_dir = run_path.parent

    try:
        # Load metadata
        with open(run_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Path to raw response
        raw_response_path = results_dir / "raw_response.txt"
        if not raw_response_path.exists():
            return False, f"Raw response file not found at {raw_response_path}"

        # Read raw response
        raw_response = raw_response_path.read_text(encoding="utf-8")

        # Apply new extraction logic
        extracted_content = extract_code_from_backticks(raw_response)

        # Initialize stripped variables to avoid "possibly unbound" errors
        extracted_stripped = extracted_content.strip() if extracted_content else ""
        expected_stripped = expected_content.strip()

        # Check if extracted content would be empty
        if not extracted_content:
            new_error = "Model returned empty output"
            new_success = False
            message = (
                f"Would update error to '{new_error}'"
                if dry_run
                else f"Updated error to '{new_error}'"
            )
        else:
            # Check if output would match expected
            new_success = extracted_stripped == expected_stripped

            if new_success:
                new_error = None
                message = (
                    "Would change status to success"
                    if dry_run
                    else "Changed status to success"
                )
            else:
                new_error = "Output mismatch"
                message = (
                    f"Would update error to '{new_error}'"
                    if dry_run
                    else f"Updated error to '{new_error}'"
                )

        # Create or update extraction file
        extracted_output_path = results_dir / "extracted_output.txt"
        if not dry_run:
            _save_text_file(extracted_output_path, extracted_content or "")

        # Generate new diff
        diff_path = results_dir / "output.diff"
        if new_success:
            diff_content = "No differences found.\n"
        else:
            diff_lines = difflib.unified_diff(
                expected_stripped.splitlines(keepends=True),
                extracted_stripped.splitlines(keepends=True),
                fromfile=f"{metadata['benchmark_case']}_expectedoutput.txt (expected)",
                tofile=f"{metadata['benchmark_case']}_extracted.txt (actual)",
                lineterm="",
            )
            diff_content = "".join(diff_lines)
            if not diff_content:
                diff_content = "No differences found (after stripping whitespace).\n"

        if not dry_run:
            _save_text_file(diff_path, diff_content)

        # Update metadata
        updated_metadata = metadata.copy()
        updated_metadata["success"] = new_success
        updated_metadata["error"] = new_error

        if extracted_content:
            updated_metadata["extracted_output_length"] = len(extracted_content)

        if not dry_run:
            with open(run_path, "w", encoding="utf-8") as f:
                json.dump(updated_metadata, f, indent=4)

        return new_success, message

    except Exception as e:
        return False, f"Error updating run: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Update benchmark results to use the new extraction logic."
    )
    parser.add_argument(
        "--benchmark-run-dir",
        type=Path,
        required=True,
        help="Path to the benchmark run directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without making changes.",
    )

    args = parser.parse_args()

    # Define results directory
    results_base_dir = args.benchmark_run_dir / "results"
    prompts_dir = args.benchmark_run_dir / "prompts"

    if not results_base_dir.exists():
        print(f"Error: Results directory {results_base_dir} does not exist.")
        return 1

    print("--- Starting Update Process ---")
    print(f"Benchmark Run Directory: {args.benchmark_run_dir}")
    print(f"Dry Run: {args.dry_run}")

    # Find affected runs
    print("Finding affected runs...")
    affected_runs = find_affected_runs(results_base_dir)
    print(f"Found {len(affected_runs)} affected runs.")

    if not affected_runs:
        print("No runs need to be updated.")
        return 0

    # Group runs by benchmark case
    runs_by_case = {}
    for run_path in affected_runs:
        try:
            with open(run_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            case = metadata.get("benchmark_case")
            if case:
                if case not in runs_by_case:
                    runs_by_case[case] = []
                runs_by_case[case].append(run_path)
        except Exception as e:
            print(f"Error reading metadata at {run_path}: {e}")

    # Process each benchmark case
    success_count = 0
    failure_count = 0
    empty_output_count = 0
    output_mismatch_count = 0
    error_count = 0

    for case, run_paths in runs_by_case.items():
        # Get expected output for this case
        expected_output_path = prompts_dir / f"{case}_expectedoutput.txt"
        if not expected_output_path.exists():
            print(f"Warning: Expected output file not found for case {case}. Skipping.")
            error_count += len(run_paths)
            continue

        try:
            expected_content = expected_output_path.read_text(encoding="utf-8")
        except IOError as e:
            print(f"Error reading expected output for case {case}: {e}")
            error_count += len(run_paths)
            continue

        # Update each run for this case
        for run_path in run_paths:
            model_name = run_path.parent.parent.name
            timestamp = run_path.parent.name
            print(f"Processing: {case} / {model_name} / {timestamp}...")

            success, message = update_run(run_path, expected_content, args.dry_run)
            print(f"  {message}")

            if "error updating" in message.lower():
                error_count += 1
            elif success:
                success_count += 1
            else:
                failure_count += 1
                # Track specific failure types
                if "Model returned empty output" in message:
                    empty_output_count += 1
                elif "Output mismatch" in message:
                    output_mismatch_count += 1

    # Print summary
    print("\n--- Update Summary ---")
    print(f"Total affected metadata files: {len(affected_runs)}")

    # Count unique benchmark cases
    unique_cases = set()
    for run_path in affected_runs:
        try:
            with open(run_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            case = metadata.get("benchmark_case")
            if case:
                unique_cases.add(case)
        except Exception:
            pass

    # Count model runs processed
    total_runs_processed = sum(len(run_paths) for run_paths in runs_by_case.values())

    print(f"Representing {len(unique_cases)} unique benchmark cases")
    print(f"Total processing actions performed: {total_runs_processed}")
    print(f"  Now successful: {success_count}")
    print(f"  Still failing: {failure_count}")
    print(f"    - Empty output failures: {empty_output_count}")
    print(f"    - Output mismatch failures: {output_mismatch_count}")
    print(f"  Errors during update: {error_count}")

    if args.dry_run:
        print("\nThis was a dry run. No files were modified.")
        print("Run without --dry-run to apply these changes.")

    print("--- Update Process Complete ---")

    # After updating the results, suggest regenerating visualizations
    print(
        "\nNote: After updating the results, you should regenerate the visualization pages:"
    )
    print(
        f"python benchmark_pipeline/3_generate_pages.py --benchmark-run-dir {args.benchmark_run_dir}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
