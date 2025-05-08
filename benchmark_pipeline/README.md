# LoCoDiff Benchmark Pipeline

This directory contains the scripts for running the LoCoDiff benchmark. The benchmarking process involves two main steps:

## Step 1: Generate Prompts

```bash
python benchmark_pipeline/1_generate_prompts.py --benchmark-run-dir BENCHMARK_DIR --repos REPO1 [REPO2 ...] --max-prompt-tokens MAX_TOKENS --add-prompts NUM_PROMPTS
```

**Key parameters:**
- `--benchmark-run-dir`: Directory where benchmark prompts will be stored (e.g., `locodiff-250425`)
- `--repos`: One or more GitHub repositories in the format `org/repo` (e.g., `ghostty-org/ghostty`)
- `--max-prompt-tokens`: Maximum token limit for prompts in thousands (e.g., `75` for 75K tokens)
- `--add-prompts`: Number of new prompts to generate and add to the benchmark set

**Optional parameters:**
- `--min-prompt-tokens`: Minimum token limit for prompts (default: 0)
- `--modified-within-months`: Only process files modified within specified months (default: 6)
- `--max-expected-tokens`: Skip files whose output exceeds this token count (default: 12000)

## Step 2: Run Benchmark

```bash
python benchmark_pipeline/2_run_benchmark.py --concurrency CONCURRENCY --num-runs NUM_RUNS --model MODEL_NAME --benchmark-run-dir BENCHMARK_DIR
```

**Key parameters:**
- `--concurrency`: Number of benchmark tasks to run simultaneously (e.g., 10)
- `--num-runs`: Number of benchmarks to run (-1 for all available)
- `--model`: Model identifier to benchmark (e.g., `anthropic/claude-3.7-sonnet:thinking`)
- `--benchmark-run-dir`: The benchmark directory (e.g., `locodiff-250425`)

You'll need an OpenRouter API key set as `OPENROUTER_API_KEY` in your environment.

Rerunning the command will retry any benchmark cases that hit API errors or haven't been run yet.
