#!/usr/bin/env python3
"""
Backfill and update costs for OpenAI pass-through models (e.g., GPT-5) in LoCoDiff results.

Context:
- For certain OpenAI models routed via OpenRouter as pass-through, OpenAI bills the usage
  directly. Our prior runs may have stored OpenRouter's total_cost in metadata.json.
- This script recomputes and updates costs for GPT-5 runs to reflect OpenAI billing.

Pricing (provided by user):
- GPT-5:
  - Input tokens (including reasoning tokens) billed at $1.25 per 1,000,000 tokens
  - Output tokens billed at $10.00 per 1,000,000 tokens

What this script does:
- Scans <benchmark_run_dir>/results/** for sanitized model directories matching openai_gpt-5*
- For each run's metadata.json:
  - Reads native token counts:
      native_prompt_tokens, native_completion_tokens, native_tokens_reasoning
  - Computes OpenAI cost as:
      input_tokens = native_prompt_tokens + native_tokens_reasoning (treat missing as 0)
      output_tokens = native_completion_tokens
      cost_openai = input_tokens/1e6 * 1.25 + output_tokens/1e6 * 10.0
  - Writes fields:
      - cost_usd_openrouter: preserves prior cost_usd if present
      - cost_usd: updated to OpenAI computed total (primary)
      - cost_usd_openai: same as cost_usd for clarity
      - cost_breakdown_openai: {input_tokens, output_tokens, input_rate_per_million, output_rate_per_million}
      - input_tokens_openai, output_tokens_openai
  - Leaves other metadata untouched.

Usage:
  python benchmark_pipeline/update_costs_openai_pass_through.py --benchmark-run-dir locodiff-250425 [--dry-run]

Notes:
- The script skips runs that don't have the expected native_* fields.
- It is idempotent; re-running will overwrite cost_usd with the computed value again.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple


GPT5_INPUT_RATE_PER_MILLION = 1.25
GPT5_OUTPUT_RATE_PER_MILLION = 10.00


def is_gpt5_model_dir_name(sanitized_model_dir: str) -> bool:
    """
    Check if a sanitized model directory name corresponds to openai/gpt-5*.

    We store model directory names sanitized with '/' -> '_', so:
      - openai/gpt-5 -> openai_gpt-5
      - openai/gpt-5:minimal -> openai_gpt-5:minimal (colon preserved)
      - openai/gpt-5* variants start with "openai_gpt-5"
    """
    return sanitized_model_dir.startswith("openai_gpt-5")


def compute_openai_cost(
    native_prompt_tokens: Optional[int],
    native_completion_tokens: Optional[int],
    native_tokens_reasoning: Optional[int],
) -> Tuple[float, int, int]:
    """
    Compute OpenAI cost for GPT-5 using provided pricing.

    Returns:
        (cost_openai, input_tokens, output_tokens)
    """
    p = int(native_prompt_tokens or 0)
    c = int(native_completion_tokens or 0)
    r = int(native_tokens_reasoning or 0)

    input_tokens = p + r
    output_tokens = c

    cost_input = (input_tokens / 1_000_000) * GPT5_INPUT_RATE_PER_MILLION
    cost_output = (output_tokens / 1_000_000) * GPT5_OUTPUT_RATE_PER_MILLION
    cost_openai = cost_input + cost_output
    return cost_openai, input_tokens, output_tokens


def update_metadata_file(path: Path, dry_run: bool = False) -> bool:
    """
    Update a single metadata.json file. Returns True if updated, False if skipped.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[skip] Failed to read JSON {path}: {e}")
        return False

    native_prompt_tokens = data.get("native_prompt_tokens")
    native_completion_tokens = data.get("native_completion_tokens")
    native_tokens_reasoning = data.get("native_tokens_reasoning")

    # Require at least prompt or completion to exist; reasoning may be absent
    if native_prompt_tokens is None and native_completion_tokens is None:
        print(f"[skip] Missing native token fields in {path}")
        return False

    cost_openai, input_tokens, output_tokens = compute_openai_cost(
        native_prompt_tokens, native_completion_tokens, native_tokens_reasoning
    )

    # Preserve prior cost as OpenRouter's reported cost, if present
    prior_cost = data.get("cost_usd")
    if prior_cost is not None:
        data["cost_usd_openrouter"] = prior_cost

    # Write OpenAI recomputed cost as the primary cost
    data["cost_usd"] = round(float(cost_openai), 8)
    data["cost_usd_openai"] = data["cost_usd"]

    # Add breakdown fields
    data["cost_breakdown_openai"] = {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "input_rate_per_million": GPT5_INPUT_RATE_PER_MILLION,
        "output_rate_per_million": GPT5_OUTPUT_RATE_PER_MILLION,
    }
    data["input_tokens_openai"] = int(input_tokens)
    data["output_tokens_openai"] = int(output_tokens)

    if dry_run:
        print(f"[dry-run] Would update {path} -> cost_usd={data['cost_usd']}")
        return False

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=False)
            f.write("\n")
        print(f"[ok] Updated {path} -> cost_usd={data['cost_usd']}")
        return True
    except Exception as e:
        print(f"[error] Failed to write {path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update GPT-5 costs in metadata.json files to reflect OpenAI billing."
    )
    parser.add_argument(
        "--benchmark-run-dir",
        required=True,
        help="Benchmark run directory (e.g., locodiff-250425)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show changes without writing files"
    )
    args = parser.parse_args()

    bench_dir = Path(args.benchmark_run_dir)
    results_dir = bench_dir / "results"
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    updated = 0
    scanned = 0

    # Iterate case directories
    for case_dir in results_dir.iterdir():
        if not case_dir.is_dir():
            continue

        # Iterate model directories
        for model_dir in case_dir.iterdir():
            if not model_dir.is_dir():
                continue

            if not is_gpt5_model_dir_name(model_dir.name):
                continue

            # Iterate timestamp directories
            for ts_dir in model_dir.iterdir():
                if not ts_dir.is_dir():
                    continue

                metadata_path = ts_dir / "metadata.json"
                if not metadata_path.exists():
                    continue

                scanned += 1
                if update_metadata_file(metadata_path, dry_run=args.dry_run):
                    updated += 1

    print(f"Scanned {scanned} metadata files. Updated {updated}.")


if __name__ == "__main__":
    main()
