# Benchmark Evaluation Suite for LLMs

This directory contains a collection of scripts to evaluate LLMs on 'tasks'. A task is defined for the purposes of this project as a collection of multiple choice questions. For example, 'determine the primary etiology given patient data' is a task (which we refer to as the ETPR task). 


# Creating the python environments

Before doing anything else, you should create the necessary python environments, using the `create_environments.sh` script. This should create two environments:
- `venvs/venv_gpu`, containing the GPU-capable version of PyTorch, vLLM, transformers, and everything else necessary to run an LLM in inference mode on a given task.
- `venvs/venv_cpu`, containing the cpu-only version of PyTorch, and all the libraries necessary for analysis and plotting (e.g. pandas, numpy, matplotlib, seaborn).

If this fails, manually create the environments using your favorite environment manager and make sure that all `.sh` scripts are aware of them.

# Running a task

A single "run" of a task consists of running *one* model on (possibly) *multiple* benchmarks. The model, its options, the task, and output directories are specified via YAML configuration files, which by convention should be kept in `configs/`.  

To run a task, submit the `run_benchmarks.sh` script to SCC, specifying a configuration file. For example,
```
$ qsub run_benchmarks.sh configs/NACC/qwen2.5-3B-Instruct.yml
```
will run the tasks specified in that `.yml` file. To request resources (i.e. GPUs), and associate the run with the correct SCC project, edit the corresponding section in `run_benchmarks.sh`.


The results of a task will be saved in the directory specified by the `results_dir` key in the configuration `.yml` file. The result directory will contain:
- a copy of the configuration file used for this run, for reproducibility purposes
- a newline-delimited JSON file (also known as JSONL) with the model responses. If you asked the model to generate more than one output for each question (using the `sampling_params.n` configuratoin key), they will be collected as an array in the corresponding JSONL key. In other words, if the task has N questions, you will have N lines in the output JSONL file. If you want one line per output generation, use something like `pd.explode` on the JSONL file.

All tasks are assumed to comprise multiple choice questions, but the model output is in natural language. To extract the answer, we look for the pattern `\boxed{...}` in the output, and extract the letter within braces. To automate this, use the `extract_answers.sh` script. This will use a regular expression to attempt to extract a letter answer from the model output (e.g. "the answer is \boxed{A}" -> "A"). If this fails, either because the model did not produce the answer in this format, or multiple valid answers are found, the script falls back to using anoher LLM to resolve the ambiguity. This means that `extract_answers.sh` should also be submitted to SCC requesting GPUs. The `extract_answers.sh` script will recursively look at all JSONL files in the specified directory, and run them through the extractor. It will then produce a parquet file with the ground truth and predicted answers. The parquet file will be in the same directory as each procesed JSONL file.

Example:
'''
$ qsub extract_answers.sh results/NACC
'''

If you also want to compute metrics from these answers (precision, recall, etc.) use 
```
$ ./compute_metrics.sh
```
Notice that it it not necessary to `qsub` this, as it does not use an LLM internally. It is a very lightweight operation. If you are going to make figures straight from the parquet files, you can skip this step.

# Modifying the configuration files

The configuration is hierarchical. All options under `LLM` will be passed to the `vllm.LLM` constructor (documentation [here](https://docs.vllm.ai/en/stable/api/vllm/index.html#vllm.LLM)), so you can use any of the allowed keyword arguments. The same applies to `sampling_params`, whose elements will be passed to `vllm.SamplingParams` (documentation [here](https://docs.vllm.ai/en/stable/api/vllm/index.html?h=samplingparams#vllm.SamplingParams)). 

Expand the following block for an example. 
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
  results_dir: 'results_sub' # outputs will be saved in this directory
  base_dir: "/projectnb/vkolagrp/projects/adrd_foundation_model/benchmarks" # Base directory for benchmarks
  max_questions: 100 # Read at most this many questions from the benchmark, so it's easy to run a subset 
  benchmark_list:
    - '${benchmarks.base_dir}/nacc_test/test_mci'
    - '${benchmarks.base_dir}/nacc_test/test_etpr'
    - '${benchmarks.base_dir}/nacc_test/test_cog'
```

</details>


## How to write your own benchmark

Each benchmark is a JSONL file: each line should be a valid JSON. Each line is expected to have at least the `question` and `option` keys. This will be configurable in the future.