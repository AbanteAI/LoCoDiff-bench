# LoCoDiff Benchmark

LoCoDiff (Local Code Diff) is a novel long-context benchmark for evaluating language models' ability to understand git history and reconstruct code. Developed by the [Mentat AI](https://mentat.ai) team, this benchmark offers several unique strengths:

- Tests comprehension of **naturally interconnected content** (not artificially generated or padded)
- Focused on code, can be constructed for any repo and language
- **Simple and easy to understand** prompt generation and output evaluation
- Strains models' abilities to handle long outputs
- Surprisingly **difficult for reasoning models** to reason about

# [🔍 Interactive Benchmark Results Site](https://abanteai.github.io/LoCoDiff-bench/)

## Running the Benchmark

The benchmarking process involves two main steps:

### Step 1: Generate Prompts

```bash
python benchmark_pipeline/1_generate_prompts.py --benchmark-run-dir BENCHMARK_DIR --repos REPO1 [REPO2 ...] --max-prompt-tokens MAX_TOKENS --add-prompts NUM_PROMPTS
```

**Key parameters:**
- `--benchmark-run-dir`: Directory where benchmark prompts will be stored (e.g., `locodiff-250425`)
- `--repos`: One or more GitHub repositories in the format `org/repo` (e.g., `ghostty-org/ghostty`)
- `--max-prompt-tokens`: Maximum token limit for prompts in thousands (e.g., `75` for 75K tokens)
- `--add-prompts`: Number of new prompts to generate and add to the benchmark set

### Step 2: Run Benchmark

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
