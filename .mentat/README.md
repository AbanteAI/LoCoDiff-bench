# Mentat Instructions for LoCoDiff-bench

This document provides instructions for running benchmarks with Mentat on the LoCoDiff-bench repository.

## Benchmark Pipeline Overview

The benchmarking process involves three main steps:
1. **Generate Prompts**: Create benchmark prompts from GitHub repositories
2. **Run Benchmark**: Test models against these prompts
3. **Generate Visualizations**: Create visualizations of benchmark results

## Generating Benchmark Prompts

Before running benchmarks, you may need to generate benchmark prompts. The script analyzes git history of files in specified repositories to create prompts that test a model's ability to reconstruct code from its history.

1. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Generate prompts** using the prompt generation script:
   ```
   python benchmark_pipeline/1_generate_prompts.py --benchmark-run-dir BENCHMARK_DIR --repos REPO1 [REPO2 ...] --max-prompt-tokens MAX_TOKENS --add-prompts NUM_PROMPTS
   ```

   Where:
   - `--benchmark-run-dir`: Directory where benchmark prompts will be stored (e.g., `locodiff-250425`)
   - `--repos`: One or more GitHub repositories in the format `org/repo` (e.g., `ghostty-org/ghostty`)
   - `--max-prompt-tokens`: Maximum token limit for prompts in thousands (e.g., `75` for 75K tokens)
   - `--add-prompts`: Number of new prompts to generate and add to the benchmark set

   Additional optional parameters:
   - `--min-prompt-tokens`: Minimum token limit for prompts (default: 0)
   - `--modified-within-months`: Only process files modified within specified months (default: 6)
   - `--max-expected-tokens`: Skip files whose output exceeds this token count (default: 12000)

3. **Example usage**:
   ```
   # Generate 30 prompts from the ghostty repo with max 75K tokens each
   python benchmark_pipeline/1_generate_prompts.py --benchmark-run-dir locodiff-250425 --max-prompt-tokens 75 --repos ghostty-org/ghostty --add-prompts 30
   ```

   This will:
   - Clone the specified repositories to the `cached-repos/` directory (if not already present)
   - Filter files based on modification date and token limits
   - Generate prompts and expected outputs in the `BENCHMARK_DIR/prompts/` directory
   - Update the metadata.json file with information about the generated prompts

4. **Commit the generated prompts**:
   ```
   git checkout -b add-new-prompts
   git add BENCHMARK_DIR/prompts/
   ```
   Use the `commit_and_push` action to create a PR with the new prompts.

## Running a Benchmark

After generating or with existing prompts, follow these steps to run benchmarks:

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
- **Never manually modify** the files created by running the benchmark, as this would compromise the integrity of the results. All benchmark files should be the direct, unmodified output of the benchmark scripts.

## Complete Example Workflow

### Step 1: Generate Prompts (if needed)

```
# Generate 30 new prompts from the ghostty repository
python benchmark_pipeline/1_generate_prompts.py --benchmark-run-dir locodiff-250425 --max-prompt-tokens 75 --repos ghostty-org/ghostty --add-prompts 30

# Create a PR with the new prompts
git checkout -b add-ghostty-prompts
git add locodiff-250425/prompts/
```

Use `commit_and_push` to create a PR with the new prompts.

### Step 2: Run Benchmark

To benchmark Claude 3.7 Sonnet with thinking mode:

```
python benchmark_pipeline/2_run_benchmark.py --concurrency 10 --num-runs -1 --model anthropic/claude-3.7-sonnet:thinking --benchmark-run-dir locodiff-250425
```

### Step 3: Generate Visualizations

```
python benchmark_pipeline/3_generate_pages.py --benchmark-run-dir locodiff-250425
```

### Step 4: Create PR with Results

```
git checkout -b claude-3.7-sonnet-benchmark
git add locodiff-250425/results/*/anthropic_claude-3.7-sonnetthinking/
git add docs/
```

Use `commit_and_push` to create the PR with a detailed description of the benchmark results.
