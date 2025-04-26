# Mentat Instructions for LoCoDiff-bench

This document provides instructions for running benchmarks with Mentat on the LoCoDiff-bench repository.

## Running a Benchmark

When you're asked to run a benchmark, follow these steps:

1. **Install dependencies** - The benchmark scripts require several dependencies that may not be installed by default:
   ```
   pip install -r requirements.txt
   ```

2. **Run the benchmark script** (step 2 in the pipeline) with appropriate arguments:
   ```
   python benchmark_pipeline/2_run_benchmark.py --concurrency 10 --num-runs -1 --model MODEL_NAME --benchmark-run-dir BENCHMARK_DIR
   ```
   
   Where:
   - `--concurrency 10`: Runs 10 benchmark tasks simultaneously for faster completion
   - `--num-runs -1`: Runs all available benchmarks
   - `--model MODEL_NAME`: The model identifier to benchmark (e.g., `anthropic/claude-3.7-sonnet:thinking`)
   - `--benchmark-run-dir BENCHMARK_DIR`: The benchmark directory (e.g., `locodiff-250425`)

3. **Handling API errors**: If any benchmarks encounter API errors or technical failures (not model output failures), rerun the script after completion:
   ```
   python benchmark_pipeline/2_run_benchmark.py --concurrency 1 --num-runs -1 --model MODEL_NAME --benchmark-run-dir BENCHMARK_DIR
   ```
   
   The script will automatically detect and retry only the benchmarks that had API errors or were not completed due to technical issues. It will NOT retry benchmarks where the model produced incorrect output (those are considered completed runs with failed results).

4. **Generate visualization pages** (step 3 in the pipeline) after the benchmark is complete:
   ```
   python benchmark_pipeline/3_generate_pages.py --benchmark-run-dir BENCHMARK_DIR
   ```
   
   This creates/updates files in the `docs/` directory with visualizations and statistics.

5. **Create a PR with the results**:
   - Create a new branch with an appropriate name
   - Add both the benchmark results and visualization files
   - Commit and push to create a PR:
   
   ```
   git checkout -b MODEL_NAME-benchmark
   git add BENCHMARK_DIR/results/*/MODEL_SANITIZED_NAME/
   git add docs/
   ```
   
   Use the `commit_and_push` action to create a PR with a detailed description of the benchmark results.

## Environment Requirements

- **API Keys**: Ensure the necessary API keys are set in environment variables:
  - For OpenRouter API access: `OPENROUTER_API_KEY`

## Important Notes

- Benchmark runs can be expensive depending on the model used. Always verify costs in the benchmark summary.
- Always run the 3rd script (generate_pages.py) after completing benchmarks to update visualizations.
- Include key metrics in your PR description (success rate, costs, comparison to other models).
- The benchmark script handles sanitizing model names in directories (replacing `/` with `_`).

## Example Benchmark Run

To benchmark Claude 3.7 Sonnet with thinking mode:

```
python benchmark_pipeline/2_run_benchmark.py --concurrency 10 --num-runs -1 --model anthropic/claude-3.7-sonnet:thinking --benchmark-run-dir locodiff-250425
python benchmark_pipeline/3_generate_pages.py --benchmark-run-dir locodiff-250425
```

Then create a PR with the results:

```
git checkout -b claude-3.7-sonnet-benchmark
git add locodiff-250425/results/*/anthropic_claude-3.7-sonnetthinking/
git add docs/
```

And then use `commit_and_push` to create the PR with a detailed description of the benchmark results.
