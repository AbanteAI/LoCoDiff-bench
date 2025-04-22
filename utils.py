import json
import os
import openai
import aiohttp  # For async requests
import asyncio  # For sleep
from dotenv import load_dotenv


# --- Model Interaction (Async) ---

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
