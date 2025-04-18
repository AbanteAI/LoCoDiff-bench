#!/usr/bin/env python3
import argparse
import sys
import difflib  # For showing differences
from utils import (
    clone_repo_to_cache,
    generate_prompts_and_expected,
    get_model_response_openrouter,
)


def print_stats_table(stats_list):
    """Prints a formatted table of the collected statistics."""
    if not stats_list:
        print("No statistics generated.")
        return

    # Sort by prompt tokens (descending) before printing
    stats_list.sort(key=lambda x: x["prompt_tokens"], reverse=True)

    # Determine column widths dynamically
    max_len_filename = max(len(s["filename"]) for s in stats_list) if stats_list else 10
    col_widths = {
        "filename": max(max_len_filename, 8),  # Min width 8 for "Filename"
        "prompt_tokens": 13,  # "Prompt Tokens"
        "expected_tokens": 15,  # "Expected Tokens"
        "num_commits": 8,  # "Commits"
        "lines_added": 7,  # "Added"
        "lines_deleted": 9,  # "Deleted"
        "final_lines": 11,  # "Final Lines"
    }

    # Header
    header = (
        f"{'Filename':<{col_widths['filename']}} | "
        f"{'Prompt Tokens':>{col_widths['prompt_tokens']}} | "
        f"{'Expected Tokens':>{col_widths['expected_tokens']}} | "
        f"{'Commits':>{col_widths['num_commits']}} | "
        f"{'Added':>{col_widths['lines_added']}} | "
        f"{'Deleted':>{col_widths['lines_deleted']}} | "
        f"{'Final Lines':>{col_widths['final_lines']}}"
    )
    print(header)
    print("-" * len(header))

    # Rows
    for stats in stats_list:
        row = (
            f"{stats['filename']:<{col_widths['filename']}} | "
            f"{stats['prompt_tokens']:>{col_widths['prompt_tokens']}} | "
            f"{stats['expected_tokens']:>{col_widths['expected_tokens']}} | "
            f"{stats['num_commits']:>{col_widths['num_commits']}} | "
            f"{stats['lines_added']:>{col_widths['lines_added']}} | "
            f"{stats['lines_deleted']:>{col_widths['lines_deleted']}} | "
            f"{stats['final_lines']:>{col_widths['final_lines']}}"
        )
        print(row)


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Generate prompts from git history or test a model on a specific prompt."
    )
    # Output directory is hardcoded for generation
    output_dir = "generated_prompts"

    # Mutually exclusive groups for generation vs testing
    group = parser.add_mutually_exclusive_group(required=True)

    # Group 1: Arguments for generating prompts
    gen_group = group.add_argument_group("Generation Options")
    gen_group.add_argument(
        "--repo",
        "-r",
        help="GitHub repository to clone for prompt generation (format: 'org/repo' or full URL)",
    )
    gen_group.add_argument(
        "--extensions",
        "-e",
        nargs="+",
        default=[".py"],
        help="File extensions to process for prompt generation (include the dot), e.g. .py .txt",
    )

    # Group 2: Arguments for testing a model on a specific prompt
    test_group = group.add_argument_group("Testing Options")
    test_group.add_argument(
        "--test-prompt",
        help="Path to the prompt file to test.",
    )
    test_group.add_argument(
        "--test-expected",
        help="Path to the expected output file for comparison.",
    )
    test_group.add_argument(
        "--model",
        help="Model identifier for OpenRouter (e.g., 'openai/gpt-4o').",
    )

    # Parse arguments
    args = parser.parse_args()

    # --- Mode 1: Generation ---
    if args.repo:
        if not args.extensions:
            parser.error("--extensions is required when --repo is provided.")
        # Clone the repository
        try:
            repo_path = clone_repo_to_cache(args.repo)
            print(f"Repository ready at: {repo_path}")

            # Generate prompts and expected outputs
            print(f"Generating prompts and expected outputs in '{output_dir}/'...")
            stats_list = generate_prompts_and_expected(
                repo_path, args.extensions, output_dir
            )

            # Print statistics table
            print("\n--- Statistics ---")
            print_stats_table(stats_list)
            print("\nPrompt generation complete.")

        except ValueError as e:
            print(f"Error during generation: {e}")
            return 1
        except Exception as e:
            print(f"An unexpected error occurred during generation: {e}")
            return 1

    # --- Mode 2: Testing ---
    elif args.test_prompt:
        if not args.test_expected or not args.model:
            parser.error(
                "--test-expected and --model are required when --test-prompt is provided."
            )

        print("--- Starting Test ---")
        print(f"Model: {args.model}")
        print(f"Prompt File: {args.test_prompt}")
        print(f"Expected Output File: {args.test_expected}")
        print("-" * 20)

        try:
            # 1. Read prompt file
            print(f"Reading prompt file: {args.test_prompt}")
            with open(args.test_prompt, "r", encoding="utf-8") as f_prompt:
                prompt_content = f_prompt.read()
            print(f"Prompt read successfully ({len(prompt_content)} characters).")

            # 2. Read expected output file
            print(f"Reading expected output file: {args.test_expected}")
            with open(args.test_expected, "r", encoding="utf-8") as f_expected:
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
            # Use strip() to ignore leading/trailing whitespace differences
            expected_stripped = expected_content.strip()
            response_stripped = model_response.strip()

            if response_stripped == expected_stripped:
                print("\n✅ Success: Model output matches expected output.")
            else:
                print("\n❌ Failure: Model output does not match expected output.")
                print("-" * 20)
                print("Expected Output:")
                print("-" * 20)
                print(expected_content)  # Show original for context
                print("-" * 20)
                print("Model Response:")
                print("-" * 20)
                print(model_response)  # Show original for context
                print("-" * 20)
                print("Diff (Expected -> Model):")
                print("-" * 20)
                # Generate and print a diff
                diff = difflib.unified_diff(
                    expected_content.splitlines(keepends=True),
                    model_response.splitlines(keepends=True),
                    fromfile="expected_output",
                    tofile="model_response",
                    lineterm="",
                )
                sys.stdout.writelines(diff)  # Write diff directly to stdout
                print("-" * 20)
                return 1  # Indicate failure with exit code

        except FileNotFoundError as e:
            print(f"\nError: File not found - {e}")
            return 1
        except IOError as e:
            print(f"\nError reading file: {e}")
            return 1
        except ValueError as e:  # Catches missing API key from utils function
            print(f"\nConfiguration Error: {e}")
            return 1
        except Exception as e:  # Catch API errors or other unexpected issues
            print(f"\nAn unexpected error occurred during testing: {e}")
            # Potentially log the full traceback here if needed
            return 1

        print("\n--- Test Complete ---")

    else:
        # This case should not be reachable due to the mutually exclusive group being required
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
