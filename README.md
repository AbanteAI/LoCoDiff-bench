# LoCoDiff: Natural Long Context Code Bench

LoCoDiff is a novel **lo**ng-**co**ntext benchmark for evaluating language models' ability to understand git history and reconstruct code. Developed by the [Mentat AI](https://mentat.ai) team, this benchmark offers several unique strengths:

- Utilizes **naturally interconnected content**, not artificially generated or padded context
- **No junk context**: every part of the context is required for the task
- **Tests a real skill critical for coding agents**: keeping track of the state of edited files
- Prompt generation and output evaluation are **simple and easy to understand**
- Challenges models' capacity to generate **long-form outputs**
- Surprisingly **difficult for reasoning models** to reason about
- **Easy to procedurally generate**: any file in any git repo can be made into a benchmark case

To see results, methodology, and analysis:

### 👉 [Explore the **interactive benchmark dashboard**](https://abanteai.github.io/LoCoDiff-bench/)

For instructions on running the benchmark yourself, see the [benchmark pipeline README](benchmark_pipeline/README.md).
