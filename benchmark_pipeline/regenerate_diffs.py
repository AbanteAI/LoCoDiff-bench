#!/usr/bin/env python3
"""
Regenerate all diff files in benchmark results with proper line termination.

This script recreates diff files for all benchmark results by comparing
extracted_output.txt files with their corresponding expected output files.
It ensures proper line termination for correct visualization in HTML pages.

Usage:
    python benchmark_pipeline/regenerate_diffs.py --benchmark-run-dir BENCHMARK_DIR

Arguments:
    --benchmark-run-dir (required): Path to the benchmark run directory.
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


def _save_text_file(filepath: Path, content: str):
    """Helper to save text content to a file, creating parent dirs."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
    except IOError as e:
        # Raise the error to be caught by the caller
        raise IOError(f"Failed to write file {filepath}: {e}") from e


def find_all_results(results_base_dir: Path) -> List[Tuple[Path, Path, Path]]:
    """
    Find all benchmark results that have extracted_output.txt files.

    Args:
        results_base_dir: Path to the benchmark results directory.

    Returns:
        List of tuples containing (metadata_path, extracted_output_path, expected_output_path)
        for each result that has both extracted and expected output files.
    """
    results = []

    # Find all metadata.json files recursively
    metadata_files = glob.glob(
        os.path.join(results_base_dir, "**", "metadata.json"), recursive=True
    )

    print(f"Found {len(metadata_files)} metadata files to process.")

    for metadata_path in metadata_files:
        metadata_path = Path(metadata_path)
        results_dir = metadata_path.parent

        # Check if extracted_output.txt exists
        extracted_output_path = results_dir / "extracted_output.txt"
        if not extracted_output_path.exists():
            continue

        # Load metadata to get the expected output path
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            expected_output_path = Path(metadata.get("expected_file", ""))
            if not expected_output_path.exists():
                print(
                    f"Warning: Expected output file not found: {expected_output_path}"
                )
                continue

            results.append((metadata_path, extracted_output_path, expected_output_path))
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error reading metadata file {metadata_path}: {e}")

    return results


def regenerate_diff(
    metadata_path: Path, extracted_output_path: Path, expected_output_path: Path
) -> bool:
    """
    Regenerate a diff file for a benchmark result using `git diff`.

    Args:
        metadata_path: Path to the metadata.json file.
        extracted_output_path: Path to the extracted_output.txt file.
        expected_output_path: Path to the expected output file.

    Returns:
        True if the diff was successfully regenerated, False otherwise.
    """
    import subprocess
    import tempfile
    import os

    results_dir = metadata_path.parent
    diff_path = results_dir / "output.diff"

    try:
        # Load the metadata to get benchmark case info
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        benchmark_case = metadata.get("benchmark_case", "unknown")

        # Read the output files
        extracted_content = extracted_output_path.read_text(encoding="utf-8")
        expected_content = expected_output_path.read_text(encoding="utf-8")

        # Strip content for comparison
        extracted_stripped = extracted_content.strip()
        expected_stripped = expected_content.strip()

        # Check if outputs match
        is_success = extracted_stripped == expected_stripped

        # Generate diff
        if is_success:
            diff_content = "No differences found.\n"
        else:
            # Create temporary files for git diff
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix="_expected.txt"
            ) as expected_file:
                expected_file.write(expected_stripped)
                expected_file_path = expected_file.name

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix="_actual.txt"
            ) as actual_file:
                actual_file.write(extracted_stripped)
                actual_file_path = actual_file.name

            try:
                # Generate diff using git diff for proper formatting
                expected_label = expected_output_path.name + " (expected)"
                actual_label = f"{benchmark_case}_extracted.txt (actual)"

                # Use git diff for proper formatting
                process = subprocess.run(
                    [
                        "git",
                        "diff",
                        "--no-index",
                        "--no-color",
                        f"--src-prefix=a/{expected_label}:",
                        f"--dst-prefix=b/{actual_label}:",
                        expected_file_path,
                        actual_file_path,
                    ],
                    capture_output=True,
                    text=True,
                )

                # git diff returns exit code 1 if there are differences
                diff_output = process.stdout

                # Remove the temp file paths from the output if present
                # Replace a/tmp/whatever_expected.txt: -> a/expected_label:
                diff_output = re.sub(r"a/[^:]*:", f"a/{expected_label}:", diff_output)
                diff_output = re.sub(r"b/[^:]*:", f"b/{actual_label}:", diff_output)

                # Remove the "diff --git" line if present
                diff_output = re.sub(
                    r"^diff --git .*$", "", diff_output, flags=re.MULTILINE
                )

                # Consolidate empty lines
                diff_output = re.sub(r"\n\n+", "\n\n", diff_output)

                if diff_output.strip():
                    diff_content = diff_output
                else:
                    diff_content = (
                        "No differences found (after stripping whitespace).\n"
                    )

            finally:
                # Clean up temp files
                os.unlink(expected_file_path)
                os.unlink(actual_file_path)

        # Save the diff file
        _save_text_file(diff_path, diff_content)
        return True

    except Exception as e:
        print(f"Error regenerating diff for {metadata_path}: {e}")
        return False


def count_patterns_in_diff(diff_path: Path) -> Tuple[int, int, int]:
    """
    Count the number of pattern matches in a diff file.

    Returns:
        Tuple of (lines_starting_with_minus_plus, total_minus, total_plus)
    """
    if not diff_path.exists():
        return 0, 0, 0

    try:
        # We're specifically looking for lines that start with - and have a + somewhere after
        # This would indicate an improperly formatted diff line
        mixed_pattern = re.compile(r"^-[^-+]*\+")

        # Standard patterns for removed and added lines
        minus_pattern = re.compile(r"^-")
        plus_pattern = re.compile(r"^\+")

        content = diff_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Skip lines that start with --- or +++ (diff headers)
        filtered_lines = [
            line
            for line in lines
            if not line.startswith("---")
            and not line.startswith("+++")
            and not line.startswith("@@")
        ]

        mixed_count = sum(1 for line in filtered_lines if mixed_pattern.match(line))
        minus_count = sum(1 for line in filtered_lines if minus_pattern.match(line))
        plus_count = sum(1 for line in filtered_lines if plus_pattern.match(line))

        return mixed_count, minus_count, plus_count
    except Exception as e:
        print(f"Error counting patterns in {diff_path}: {e}")
        return 0, 0, 0


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate all diff files in benchmark results with proper line termination."
    )
    parser.add_argument(
        "--benchmark-run-dir",
        type=Path,
        required=True,
        help="Path to the benchmark run directory.",
    )

    args = parser.parse_args()

    # Define results directory
    results_base_dir = args.benchmark_run_dir / "results"

    if not results_base_dir.exists():
        print(f"Error: Results directory {results_base_dir} does not exist.")
        return 1

    print("--- Starting Diff Regeneration Process ---")
    print(f"Benchmark Run Directory: {args.benchmark_run_dir}")

    # Find all results
    results = find_all_results(results_base_dir)
    print(
        f"Found {len(results)} results with both extracted and expected output files."
    )

    if not results:
        print("No results to process.")
        print("--- Diff Regeneration Process Complete ---")
        return 0

    # Process results
    success_count = 0
    failure_count = 0

    # Track statistics about problematic diffs
    problematic_diffs = 0
    total_mixed_lines_before = 0
    total_mixed_lines_after = 0

    for metadata_path, extracted_output_path, expected_output_path in results:
        diff_path = metadata_path.parent / "output.diff"

        # Check for problematic diff before regeneration
        mixed_before, _, _ = count_patterns_in_diff(diff_path)
        total_mixed_lines_before += mixed_before
        if mixed_before > 0:
            problematic_diffs += 1

        # Regenerate diff
        success = regenerate_diff(
            metadata_path, extracted_output_path, expected_output_path
        )

        # Count patterns after regeneration
        mixed_after, _, _ = count_patterns_in_diff(diff_path)
        total_mixed_lines_after += mixed_after

        if success:
            success_count += 1
        else:
            failure_count += 1

    # Print summary
    print("\n--- Diff Regeneration Summary ---")
    print(f"Total results processed: {len(results)}")
    print(f"  Successfully regenerated diffs: {success_count}")
    print(f"  Failed to regenerate diffs: {failure_count}")
    print(
        f"Problematic diffs found (with lines starting with both - and +): {problematic_diffs}"
    )
    print(f"Total mixed lines before regeneration: {total_mixed_lines_before}")
    print(f"Total mixed lines after regeneration: {total_mixed_lines_after}")

    print("--- Diff Regeneration Process Complete ---")

    # After regenerating the diffs, suggest regenerating visualizations
    print(
        "\nNote: After regenerating the diffs, you should regenerate the visualization pages:"
    )
    print(
        f"python benchmark_pipeline/3_generate_pages.py --benchmark-run-dir {args.benchmark_run_dir}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
