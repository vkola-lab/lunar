# Benchmark Evaluation Suite for LLMs

The scripts in this directory are designed to automatically run benchmarks (multiple choice questions) on multiple LLMs using vLLM, extract the answers, compute metrics, and plot the results.

## Basic Usage

A single "run" of a benchmark suite consists of running *one* model on (possibly) *multiple* benchmarks. The model and the benchmarks are specified via YAML configuration files, which by convention should be kept in `configs/`. 

The configuration is hierarchical. All options under `LLM` will be passed to the `vllm.LLM` constructor (documentation [here](https://docs.vllm.ai/en/stable/api/vllm/index.html#vllm.LLM)), so you can use any of the allowed keyword arguments. The same applies to `sampling_params`, whose elements will be passed to `vllm.SamplingParams` (documentation [here](https://docs.vllm.ai/en/stable/api/vllm/index.html?h=samplingparams#vllm.SamplingParams)). 

Expand the following block for an example. Templates are also available under `configs/`.

<details>
<summary>Example configuration file</summary>

```
run_readable_name: "Qwen3-4B" # Will be used to name the results folder

# passed to vllm.LLM, you can use any accepted keyword argument
LLM:
  model: "Qwen/Qwen3-4B" # can be a path
  tensor_parallel_size: 1 # Number of GPUs used. Passed to vllm.LLM.tensor_parallel_size. 

# passed to vllm.SamplingParams, you can use any accepted keyword argument
sampling_params:
  n: 2 # Number of sequences to generate per prompt
  temperature: 1.0
  top_p: 1.0 # consider candidate tokens until the cumulative sum of their probabilities is top_p
  max_tokens: 5000 # maximum number of tokens to generate

prompt:
  system_prompt: "Please reason step by step, and put your final answer within \\boxed{}."
  template_style: grpo # see src/prompt_templates.py

benchmarks:
  results_dir: 'results_sub' # Benchmarks outputs will be saved in this directory
  base_dir: "/projectnb/vkolagrp/projects/adrd_foundation_model/benchmarks" # Base directory for benchmarks
  max_questions: 2 # Read at most this many questions from the benchmark, so it's easy to run a subset 
  benchmark_list:
    - '${benchmarks.base_dir}/nacc_test/test_mci'
    - '${benchmarks.base_dir}/nacc_test/test_etpr'
    - '${benchmarks.base_dir}/nacc_test/test_cog'
```

</details>

To run the benchmarks navigate to the folder containing `run_benchmark.sh` and use 
```bash
$ ./run_benchmarks.sh configs/config.yml
```
This will run all benchmarks specified under `benchmarks.benchmark_list`, in that order. The results will be saved to `benchmarks.results_dir`.

If you're on SCC, edit the resources requests at the top of `run_benchmarks.sh` and submit it as a batch job instead:
```bash
$ qsub ./run_benchmarks.sh configs/config.yml
```

To run multiple models on multiple benchmarks, use 
```bash
$ ./submit.all configs
```
where `configs` is a directory with YAML config files. This will submit a `run_benchmarks.sh` job for each config file. The advantage is that these get run in parallel, and you can specify different hardware for each model.

# How to write your own benchmark

Each benchmark is a JSONL file: each line should be a valid JSON. Each line is expected to have at least the `question` and `option` keys. This will be configurable in the future.