#!/usr/bin/env python3
import argparse
import os
import sys
import difflib
from utils import get_model_response_openrouter


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

    try:
        # 1. Read prompt file
        print(f"Reading prompt file: {prompt_filepath}")
        if not os.path.exists(prompt_filepath):
            raise FileNotFoundError(f"Prompt file not found: {prompt_filepath}")
        with open(prompt_filepath, "r", encoding="utf-8") as f_prompt:
            prompt_content = f_prompt.read()
        print(f"Prompt read successfully ({len(prompt_content)} characters).")

        # 2. Read expected output file
        print(f"Reading expected output file: {expected_filepath}")
        if not os.path.exists(expected_filepath):
            raise FileNotFoundError(
                f"Expected output file not found: {expected_filepath}"
            )
        with open(expected_filepath, "r", encoding="utf-8") as f_expected:
            expected_content = f_expected.read()
        print(
            f"Expected output read successfully ({len(expected_content)} characters)."
        )

        # 3. Call OpenRouter API
        print(f"Sending prompt to model '{args.model}' via OpenRouter...")
        model_response = get_model_response_openrouter(prompt_content, args.model)
        print(f"Received response from model ({len(model_response)} characters).")

        # 4. Compare the model's response with the expected output
        print("Comparing model response to expected output...")
        # Use strip() to potentially ignore leading/trailing whitespace differences if desired,
        # but for exact reconstruction, direct comparison might be better. Let's stick to direct.
        # expected_stripped = expected_content.strip()
        # response_stripped = model_response.strip()

        if model_response == expected_content:
            print("\n✅ Success: Model output exactly matches expected output.")
            exit_code = 0
        else:
            print("\n❌ Failure: Model output does not exactly match expected output.")
            print("-" * 30)
            # print("Expected Output:")
            # print("-" * 30)
            # print(expected_content)
            # print("-" * 30)
            # print("Model Response:")
            # print("-" * 30)
            # print(model_response)
            # print("-" * 30)
            print("Diff (Expected -> Model Response):")
            print("-" * 30)
            # Generate and print a diff
            diff = difflib.unified_diff(
                expected_content.splitlines(keepends=True),
                model_response.splitlines(keepends=True),
                fromfile=expected_filepath,
                tofile="model_response",
                lineterm="",
            )
            sys.stdout.writelines(diff)
            if not any(diff):  # Check if the generator yields anything
                print("(No differences found, potentially only whitespace changes)")
            print("-" * 30)
            exit_code = 1  # Indicate failure

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        exit_code = 1
    except IOError as e:
        print(f"\nError reading file: {e}")
        exit_code = 1
    except ValueError as e:  # Catches missing API key from utils function
        print(f"\nConfiguration Error: {e}")
        exit_code = 1
    except Exception as e:  # Catch API errors or other unexpected issues
        print(f"\nAn unexpected error occurred during testing: {e}")
        # Consider adding traceback logging here for debugging
        # import traceback
        # traceback.print_exc()
        exit_code = 1

    print("\n--- Test Complete ---")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
