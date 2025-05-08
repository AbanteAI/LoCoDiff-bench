# LoCoDiff Benchmark

LoCoDiff (Local Code Diff) is a novel long-context benchmark for evaluating language models' ability to understand git history and reconstruct code. Developed by the [Mentat AI](https://mentat.ai) team, this benchmark offers several unique strengths:

- Tests comprehension of **naturally interconnected content** (not artificially generated or padded)
- Focused on code, can be constructed for any repo and language
- **Simple and easy to understand** prompt generation and output evaluation
- Strains models' abilities to handle long outputs
- Surprisingly **difficult for reasoning models** to reason about

## [View the Latest Results](https://abanteai.github.io/LoCoDiff-bench/)

## How the Benchmark Works

LoCoDiff evaluates a model's ability to understand a file's git history and produce its final state. The benchmark process:

1. **Prompt Generation**: Creates prompts from real GitHub repositories, containing the full git history (with diffs) for files
2. **Model Evaluation**: Tests models by asking them to reconstruct the final state of the file based on its history
3. **Visualization**: Analyzes results and generates a static website with performance metrics

This approach tests both a model's ability to understand interconnected content in a long context and produce accurate, often lengthy outputs.

## Running the Benchmark

The benchmarking process involves three main scripts in the `benchmark_pipeline` directory:

### Step 1: Generate Prompts

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

**Example:**
```bash
# Generate 30 prompts from the ghostty repo with max 75K tokens each
python benchmark_pipeline/1_generate_prompts.py --benchmark-run-dir locodiff-250425 --max-prompt-tokens 75 --repos ghostty-org/ghostty --add-prompts 30
```

### Step 2: Run Benchmark

```bash
python benchmark_pipeline/2_run_benchmark.py --concurrency CONCURRENCY --num-runs NUM_RUNS --model MODEL_NAME --benchmark-run-dir BENCHMARK_DIR
```

**Key parameters:**
- `--concurrency`: Number of benchmark tasks to run simultaneously (e.g., 10)
- `--num-runs`: Number of benchmarks to run (-1 for all available)
- `--model`: Model identifier to benchmark (e.g., `anthropic/claude-3.7-sonnet:thinking`)
- `--benchmark-run-dir`: The benchmark directory (e.g., `locodiff-250425`)

**Example:**
```bash
# Run all benchmarks with Claude 3.7 Sonnet in thinking mode with 10 concurrent tasks
python benchmark_pipeline/2_run_benchmark.py --concurrency 10 --num-runs -1 --model anthropic/claude-3.7-sonnet:thinking --benchmark-run-dir locodiff-250425
```

**Handling API errors:** If benchmarks encounter API errors, rerun with reduced concurrency to retry only the failed runs:
```bash
python benchmark_pipeline/2_run_benchmark.py --concurrency 1 --num-runs -1 --model MODEL_NAME --benchmark-run-dir BENCHMARK_DIR
```

### Step 3: Generate Visualization Pages

```bash
python benchmark_pipeline/3_generate_pages.py --benchmark-run-dir BENCHMARK_DIR
```

This script creates/updates files in the `docs/` directory with visualizations and statistics for all the models that have been benchmarked.

## Requirements

```
# Install dependencies
pip install -r requirements.txt
```

Key dependencies include:
- tiktoken, PyYAML for processing prompts
- openai, requests, aiohttp for model API integration
- pandas, matplotlib for data analysis and visualization

## Environment Requirements

- **API Keys**: Set the necessary API keys as environment variables:
  - For OpenRouter API access: `OPENROUTER_API_KEY`

## Important Notes

- Benchmark runs can be expensive depending on the model used. Always verify costs in the benchmark summary.
- Always run the visualization script (`3_generate_pages.py`) after completing benchmarks to update the results website.
- The benchmark script automatically sanitizes model names in directories (replacing `/` with `_`).
- **Never manually modify** the files created by running the benchmark to maintain result integrity.

## Complete Example Workflow

### Step 1: Generate Prompts (if needed)
```bash
# Generate 30 new prompts from the ghostty repository
python benchmark_pipeline/1_generate_prompts.py --benchmark-run-dir locodiff-250425 --max-prompt-tokens 75 --repos ghostty-org/ghostty --add-prompts 30
```

### Step 2: Run Benchmark
```bash
# Benchmark Claude 3.7 Sonnet with thinking mode
python benchmark_pipeline/2_run_benchmark.py --concurrency 10 --num-runs -1 --model anthropic/claude-3.7-sonnet:thinking --benchmark-run-dir locodiff-250425
```

### Step 3: Generate Visualizations
```bash
python benchmark_pipeline/3_generate_pages.py --benchmark-run-dir locodiff-250425
```

## Contributing

Contributions to LoCoDiff-bench are welcome! Please feel free to submit pull requests or open issues to discuss potential improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
