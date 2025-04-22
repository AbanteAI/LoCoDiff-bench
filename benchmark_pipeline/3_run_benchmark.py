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
import openai
import aiohttp  # For async requests
from dotenv import load_dotenv


# --- Model Interaction ---

# Global async client instance
_ASYNC_CLIENT = None


def _get_async_openai_client():
    """Initializes and returns the async OpenAI client for OpenRouter."""
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is None:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables. "
                "Ensure it's set in a .env file or exported."
            )
        _ASYNC_CLIENT = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _ASYNC_CLIENT


async def get_model_response_openrouter(
    prompt_content: str, model_name: str
) -> tuple[str | None, str | None, str | None]:
    """
    Sends a prompt to a specified model via OpenRouter asynchronously.

    Args:
        prompt_content: The full content of the prompt to send to the model.
        model_name: The identifier of the model on OpenRouter (e.g., 'openai/gpt-4o').

    Returns:
        A tuple containing:
        - The content of the model's response message (str) if successful, else None.
        - The generation ID (str) if available, else None.
        - An error message (str) if an API error occurred, else None.

    Raises:
        ValueError: If the OPENROUTER_API_KEY environment variable is not set (raised by _get_async_openai_client).
    """
    client = _get_async_openai_client()
    error_message = None

    try:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt_content,
                },
            ],
            # Optional: Add other parameters like temperature, max_tokens if needed
            # temperature=0.7,
            # max_tokens=2000,
        )

        response_content = ""
        generation_id = None

        # Check for API-level errors returned in the response body (e.g., credit limits)
        # Use getattr for safer access to potentially dynamic attributes
        error_payload = getattr(completion, "error", None)
        if error_payload:
            # Try to serialize the full error payload for more detail
            try:
                if isinstance(error_payload, dict):
                    error_details = json.dumps(error_payload)
                else:
                    error_details = str(error_payload)
                error_message = f"Provider error in response body: {error_details}"
            except Exception as serialize_err:
                # Fallback if serialization fails
                error_message = f"Provider error in response body (serialization failed: {serialize_err}): {str(error_payload)}"

            print(f"OpenRouter API reported an error: {error_message}")
            return None, None, error_message

        # Extract content if successful and no error in body
        if completion.choices and completion.choices[0].message:
            response_content = completion.choices[0].message.content or ""

        # Extract generation ID
        if hasattr(completion, "id") and isinstance(completion.id, str):
            generation_id = completion.id
        else:
            # Log the full response if ID extraction fails, might reveal structure changes
            print(
                f"Warning: Could not extract generation ID from OpenRouter response object: {completion}"
            )

        return response_content, generation_id, None  # Success

    except openai.APIError as e:
        # This catches errors where the API call itself failed (e.g., 4xx/5xx status codes)
        # Use getattr for status_code and body as they might not be statically typed
        status_code = getattr(e, "status_code", "Unknown")
        base_error_message = f"OpenRouter API Error: Status {status_code} - {e.message}"
        detailed_error_message = base_error_message  # Start with base message

        # Attempt to extract more detail from the body
        body = getattr(e, "body", None)
        if body:
            try:
                if isinstance(body, dict):
                    # Try to get nested message first
                    nested_message = body.get("error", {}).get("message")
                    if nested_message and nested_message != e.message:
                        detailed_error_message = (
                            f"{base_error_message} | Detail: {nested_message}"
                        )
                    # Include full body if it might be useful and isn't just repeating the message
                    body_str = json.dumps(body)
                    if body_str not in detailed_error_message:  # Avoid redundancy
                        detailed_error_message += f" | Body: {body_str}"
                else:
                    # If body is not a dict, include its string representation if informative
                    body_str = str(body)
                    if body_str and body_str not in detailed_error_message:
                        detailed_error_message += f" | Body: {body_str}"
            except Exception as serialize_err:
                detailed_error_message += (
                    f" (Failed to serialize body: {serialize_err})"
                )

        print(detailed_error_message)  # Print the most detailed message obtained
        return None, None, detailed_error_message  # Return the detailed message
    except Exception as e:
        error_message = f"Unexpected Error during API call: {type(e).__name__}: {e}"
        print(error_message)
        # Log traceback for unexpected errors
        # import traceback
        # traceback.print_exc()
        return None, None, error_message


async def get_generation_stats_openrouter(generation_id: str) -> dict | None:
    """
    Queries the OpenRouter Generation Stats API asynchronously for cost and token information.

    Args:
        generation_id: The ID of the generation to query (e.g., "gen-12345").

    Returns:
        A dictionary containing statistics like cost and token counts, or None if
        the query fails or the API key is missing.
        Example return format:
        {
            'cost_usd': float,
            'prompt_tokens': int,
            'completion_tokens': int,
            'total_tokens': int,
                    'native_prompt_tokens': int | None,
                    'native_completion_tokens': int | None,
                    'native_finish_reason': str | None
                }

    Raises:
        ValueError: If the OPENROUTER_API_KEY environment variable is not set.
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment variables for stats query."
        )

    stats_url = f"https://openrouter.ai/api/v1/generation?id={generation_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    max_retries = 3
    retry_delay_seconds = 1

    for attempt in range(max_retries):
        try:
            # Use aiohttp for the async request
            async with aiohttp.ClientSession() as session:
                async with session.get(stats_url, headers=headers) as response:
                    # Check for 404 specifically for retry
                    if response.status == 404:
                        print(
                            f"Attempt {attempt + 1}/{max_retries}: Stats not found (404) for {generation_id}. Retrying in {retry_delay_seconds}s..."
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay_seconds)
                            continue  # Go to next retry iteration
                        else:
                            print(
                                f"Max retries reached for {generation_id}. Giving up."
                            )
                            return None  # Failed after retries

                    # Raise HTTPError for other bad responses (4xx or 5xx, excluding 404 handled above)
                    response.raise_for_status()
                    response_data = await response.json()

            # If successful (status 200 and no exception raised), process data
            if "data" in response_data and response_data["data"] is not None:
                stats_data = response_data["data"]
                # Extract relevant fields, providing defaults or None if missing
                cost_usd = stats_data.get("total_cost", 0.0)
                prompt_tokens = stats_data.get("tokens_prompt", 0)
                completion_tokens = stats_data.get("tokens_completion", 0)
                native_prompt_tokens = stats_data.get(
                    "native_tokens_prompt"
                )  # Can be None
                native_completion_tokens = stats_data.get(
                    "native_tokens_completion"
                )  # Can be None
                native_finish_reason = stats_data.get(
                    "native_finish_reason"
                )  # Can be None

                return {
                    "cost_usd": float(cost_usd) if cost_usd is not None else 0.0,
                    "prompt_tokens": int(prompt_tokens)
                    if prompt_tokens is not None
                    else 0,
                    "completion_tokens": int(completion_tokens)
                    if completion_tokens is not None
                    else 0,
                    "total_tokens": (
                        int(prompt_tokens or 0) + int(completion_tokens or 0)
                    ),
                    "native_prompt_tokens": int(native_prompt_tokens)
                    if native_prompt_tokens is not None
                    else None,
                    "native_completion_tokens": int(native_completion_tokens)
                    if native_completion_tokens is not None
                    else None,
                    "native_finish_reason": str(native_finish_reason)
                    if native_finish_reason is not None
                    else None,
                }
            else:
                print(
                    f"Warning: 'data' field missing or null in OpenRouter stats response for ID {generation_id}."
                )
                print(f"Full response: {response_data}")
                return None  # Indicate stats could not be retrieved (even on success status)

        except aiohttp.ClientResponseError as e:
            # Catch non-404 HTTP errors after raise_for_status
            print(
                f"HTTP error querying OpenRouter generation stats API (Attempt {attempt + 1}/{max_retries}): {e.status} {e.message}"
            )
            # Don't retry on non-404 errors
            return None
        except aiohttp.ClientError as e:
            # Catch other aiohttp client errors (e.g., connection issues)
            print(
                f"Client error querying OpenRouter generation stats API (Attempt {attempt + 1}/{max_retries}): {e}"
            )
            # Don't retry on general client errors
            return None
        except json.JSONDecodeError as e:
            print(
                f"Error decoding JSON response from OpenRouter stats API (Attempt {attempt + 1}/{max_retries}): {e}"
            )
            # Don't retry on JSON errors
            return None
        except Exception as e:
            print(
                f"An unexpected error occurred during the async stats API call (Attempt {attempt + 1}/{max_retries}): {e}"
            )
            # Log traceback for unexpected errors
            # import traceback
            # traceback.print_exc()
            # Don't retry on unexpected errors
            return None

    # Should be unreachable if logic is correct, but as a fallback
    return None


# --- Benchmark Logic ---


def sanitize_filename(name):
    """Removes characters that are problematic for filenames/paths."""
    # Replace slashes with underscores
    name = name.replace(os.path.sep, "_").replace("/", "_")
    # Remove other potentially problematic characters
    name = re.sub(r'[<>:"|?*]', "", name)
    return name


def extract_code_from_backticks(text: str) -> str | None:
    """
    Extracts content between the first and the last triple backticks (```).
    Handles optional language identifiers after the first backticks and strips
    leading/trailing whitespace from the extracted content.
    """
    try:
        # Find the start of the first ``` block
        start_outer = text.find("```")
        if start_outer == -1:
            return None  # No opening backticks found

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

        # Find the start of the last ``` block using rfind
        end_outer = text.rfind("```")

        # Check if the last ``` was found and if it's after the first ``` marker ended
        if end_outer == -1 or end_outer < start_inner:
            # No closing backticks found, or they are invalid.
            # Be generous: extract from the start marker to the end of the string.
            print(
                "Warning: Closing backticks not found or invalid. Extracting from start marker to end."
            )
            extracted_content = text[start_inner:]
        else:
            # Both opening and closing backticks are valid
            # Extract the content between the end of the first marker and the start of the last marker
            extracted_content = text[start_inner:end_outer]

        return extracted_content.strip()

    except Exception as e:
        # Log unexpected errors during extraction
        print(f"Error during backtick extraction: {e}")
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
            "native_finish_reason": None,  # Added this line
            "stats_error": None,
        }

        results_dir = None  # Define results_dir path string early

        try:
            # --- Define Results Directory Path ---
            # Directory creation is deferred until after successful API call
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_model_name = sanitize_filename(model)
            results_dir = os.path.join(
                results_base_dir,
                benchmark_case_prefix,
                sanitized_model_name,
                timestamp_str,
            )
            run_metadata["results_dir_path_planned"] = (
                results_dir  # Store planned path for metadata
            )

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
            # Add assertion to satisfy type checker after the early return for API errors
            assert raw_model_response is not None
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

            # --- Create Results Directory (only after successful API call) ---
            # Use pathlib for potentially deeper paths and easier creation
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            run_metadata["results_dir"] = (
                results_dir  # Update metadata with actual created dir
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

            # --- Save Metadata (only if no API error occurred and directory was created) ---
            if results_dir and os.path.exists(
                results_dir
            ):  # Check if dir was actually created
                metadata_path = os.path.join(results_dir, "metadata.json")
                try:
                    # Remove the temporary planned path key before saving
                    run_metadata.pop("results_dir_path_planned", None)
                    with open(metadata_path, "w", encoding="utf-8") as f_meta:
                        json.dump(run_metadata, f_meta, indent=4)
                except Exception as meta_e:
                    print(
                        f"\nWarning: Failed to save metadata.json for {benchmark_case_prefix} in {results_dir}: {meta_e}"
                    )
            else:
                # This case should only be hit if an API error occurred earlier,
                # preventing directory creation. We already printed the API error message.
                pass

        except FileNotFoundError as e:
            run_metadata["error"] = f"File Error: {e}"
            # Don't attempt to save metadata here as the results dir likely wasn't created
            print(f"File Error for {benchmark_case_prefix}: {e} - Skipping results.")

        except IOError as e:
            run_metadata["error"] = f"IOError: {e}"
            # Don't attempt to save metadata here
            print(f"IO Error for {benchmark_case_prefix}: {e} - Skipping results.")

        except ValueError as e:  # Catches missing API key
            run_metadata["error"] = f"Config Error: {e}"
            # Don't save metadata if config error prevented API call attempt
            print(f"Config Error for {benchmark_case_prefix}: {e} - Skipping results.")
        except Exception as e:  # Catch other unexpected issues during processing
            run_metadata["error"] = f"Runtime Error: {type(e).__name__}: {e}"
            # Attempt to save metadata only if the directory was created before the error
            if results_dir and os.path.exists(results_dir):
                metadata_path = os.path.join(results_dir, "metadata.json")
                try:
                    # Remove the temporary planned path key before saving
                    run_metadata.pop("results_dir_path_planned", None)
                    with open(metadata_path, "w", encoding="utf-8") as f_meta:
                        json.dump(run_metadata, f_meta, indent=4)
                except Exception as meta_e:
                    print(
                        f"\nWarning: Failed to save metadata.json after Runtime Error for {benchmark_case_prefix} in {results_dir}: {meta_e}"
                    )
            else:
                print(
                    f"Runtime Error for {benchmark_case_prefix}: {e} - Skipping results (directory not created)."
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
    from collections import defaultdict  # Import here for clarity

    success_count = 0
    # Use a dictionary to count failures by error type
    failure_counts_by_type = defaultdict(int)
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
                error_msg = result.get("error", "Unknown processing error")
                # Simplify common error messages for better grouping
                if "File Error:" in error_msg:
                    error_type = "File Error"
                elif "IOError:" in error_msg:
                    error_type = "IO Error"
                elif "Runtime Error:" in error_msg:
                    error_type = "Runtime Error"
                elif "Extraction backticks not found" in error_msg:
                    error_type = "Extraction Error"
                elif "Output mismatch" in error_msg:
                    error_type = "Output Mismatch"
                else:
                    error_type = error_msg  # Keep less common errors as is

                failure_counts_by_type[error_type] += 1
                # Individual failure/error already printed in run_single_benchmark
        else:
            # Should not happen if run_single_benchmark always returns dict
            print(f"❌ System Error: Unexpected result type from task: {type(result)}")
            system_error_count += 1

    print("\n--- Benchmark Run Summary ---")
    print(f"Model: {args.model}")
    print(f"Attempted in this run: {len(results)} benchmarks")
    print(f"  ✅ Successful: {success_count}")
    # Print detailed failure counts
    if failure_counts_by_type:
        print("  --- Failures by Type ---")
        for error_type, count in sorted(failure_counts_by_type.items()):
            print(f"    ❌ {error_type}: {count}")
        print("  ------------------------")
    else:
        # Explicitly state if there were no processing failures
        print("  ❌ Failed (Processing Errors): 0")

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
    total_failures = sum(failure_counts_by_type.values())
    return 1 if (total_failures + api_error_count + system_error_count) > 0 else 0


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
