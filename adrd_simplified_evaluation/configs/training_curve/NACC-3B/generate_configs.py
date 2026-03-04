# ============================================================================
# CONFIGURATION - Edit these values to apply to all generated config files
# ============================================================================

MAX_QUESTIONS = 10000000

BENCHMARK_LIST = [
    '${benchmarks.base_dir}/adni_test/test_cog',
    '${benchmarks.base_dir}/adni_test/test_pet',
    '${benchmarks.base_dir}/adni_test/test_etpr',
    '${benchmarks.base_dir}/adni_test/test_csf',
    '${benchmarks.base_dir}/brainlat_test/test_cog',
    '${benchmarks.base_dir}/brainlat_test/test_etpr',
    '${benchmarks.base_dir}/nacc_test_updated/test_cog',
    '${benchmarks.base_dir}/nacc_test_updated/test_csf',
    '${benchmarks.base_dir}/nacc_test_updated/test_dat',
    '${benchmarks.base_dir}/nacc_test_updated/test_etpr',
    # '${benchmarks.base_dir}/nacc_test_updated/test_mci',
    # '${benchmarks.base_dir}/nacc_test_updated/test_np',
    '${benchmarks.base_dir}/nacc_test_updated/test_np_mixed',
    '${benchmarks.base_dir}/nacc_test_updated/test_np_one',
    '${benchmarks.base_dir}/nacc_test_updated/test_pet',
    '${benchmarks.base_dir}/nifd_test/test_cog',
    '${benchmarks.base_dir}/nifd_test/test_etpr',
    # '${benchmarks.base_dir}/nifd_test/test_ftld',
    '${benchmarks.base_dir}/ppmi_test/test_cog',
    '${benchmarks.base_dir}/ppmi_test/test_dat',
    '${benchmarks.base_dir}/ppmi_test/test_etpr',
]

# ============================================================================
# Template configuration content
# ============================================================================

def format_benchmark_list(benchmark_list):
    """Format the benchmark list for YAML output"""
    return '\n'.join([f"    - '{item}'" for item in benchmark_list])

config_template = """run_readable_name: "NACC-3B-{step_padded}" # will be used to name the results folder

training_steps: {step}

# passed to vllm.LLM, you can use any accepted keyword argument
LLM:
  model: "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt/ckpt_access/qwen25_3B_drgrpo_gp16_nacc_inc_oversample_dedup/checkpoint-{step}" # can be a path
  tensor_parallel_size: 1 # Number of GPUs used. Passed to vllm.LLM.tensor_parallel_size. 
  gpu_memory_utilization: 0.9 # Fraction of gpu memory reserved for model + attention
  max_model_len: 30000
  distributed_executor_backend: 'mp'
  seed: 1 # seed to initialize RNG for sampling
  enable_lora: false
  # lora_path: # uncomment if want to use lora

# passed to vllm.SamplingParams, you can use any accepted keyword argument
sampling_params:
  n: 5 # Number of sequences to generate per prompt
  temperature: 1 # recommended
  top_p: 1 # consider candidate tokens until the cumulative sum of their probabilities is top_p
  max_tokens: 10000 # maximum number of tokens to generate

prompt:
  system_prompt: "Please reason step by step, and put your final answer within \\\\boxed{{}}."
  template_style: grpo

benchmarks:
  results_dir: '/projectnb/vkolagrp/projects/adrd_foundation_model/results/training_curve' # Benchmarks outputs will be saved in this directory
  base_dir: "/projectnb/vkolagrp/projects/adrd_foundation_model/benchmarks" # Base directory for benchmarks
  run_dir: "NACC-3B" # Run folder
  max_questions: {max_questions} # Read at most this many questions from the benchmark
  benchmark_list:
{benchmark_list}
"""

config_final = """run_readable_name: "NACC-3B" # will be used to name the results folder

training_steps: 1115

# passed to vllm.LLM, you can use any accepted keyword argument
LLM:
  model: "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt/ckpt_access/qwen25_3B_drgrpo_gp16_nacc_inc_oversample_dedup" # can be a path
  tensor_parallel_size: 1 # Number of GPUs used. Passed to vllm.LLM.tensor_parallel_size. 
  gpu_memory_utilization: 0.9 # Fraction of gpu memory reserved for model + attention
  max_model_len: 30000
  distributed_executor_backend: 'mp'
  seed: 1 # seed to initialize RNG for sampling
  enable_lora: false
  # lora_path: # uncomment if want to use lora

# passed to vllm.SamplingParams, you can use any accepted keyword argument
sampling_params:
  n: 5 # Number of sequences to generate per prompt
  temperature: 1 # recommended
  top_p: 1 # consider candidate tokens until the cumulative sum of their probabilities is top_p
  max_tokens: 10000 # maximum number of tokens to generate

prompt:
  system_prompt: "Please reason step by step, and put your final answer within \\\\boxed{{}}."
  template_style: grpo

benchmarks:
  results_dir: '/projectnb/vkolagrp/projects/adrd_foundation_model/results/training_curve' # Benchmarks outputs will be saved in this directory
  base_dir: "/projectnb/vkolagrp/projects/adrd_foundation_model/benchmarks" # Base directory for benchmarks
  run_dir: "NACC-3B" # Run folder
  max_questions: {max_questions} # Read at most this many questions from the benchmark
  benchmark_list:
{benchmark_list}
"""

# ============================================================================
# Generate configuration files
# ============================================================================

# Generate configuration files for steps 100 through 1400 (increments of 100)
for step in range(200, 1100, 200):
    step_padded = str(step).zfill(4)
    
    config_content = config_template.format(
        step=step,
        step_padded=step_padded,
        max_questions=MAX_QUESTIONS,
        benchmark_list=format_benchmark_list(BENCHMARK_LIST)
    )
    
    filename = f"NACC-3B-{step_padded}.yml"
    with open(filename, "w") as f:
        f.write(config_content)
    
    print(f"Generated: {filename}")

# # Generate 3B baseline configuration file
# filename_3b = "Qwen2.5-3B-Instruct.yml"
# with open(filename_3b, "w") as f:
#     f.write(config_3b.format(
#         max_questions=MAX_QUESTIONS,
#         benchmark_list=format_benchmark_list(BENCHMARK_LIST)
#     ))
# print(f"Generated: {filename_3b}")

# # Generate 7B baseline configuration file
# filename_7b = "Qwen2.5-7B-Instruct.yml"
# with open(filename_7b, "w") as f:
#     f.write(config_7b.format(
#         max_questions=MAX_QUESTIONS,
#         benchmark_list=format_benchmark_list(BENCHMARK_LIST)
#     ))
# print(f"Generated: {filename_7b}")

# Generate final model configuration file
filename_final = "NACC-3B-1115.yml"
with open(filename_final, "w") as f:
    f.write(config_final.format(
        max_questions=MAX_QUESTIONS,
        benchmark_list=format_benchmark_list(BENCHMARK_LIST)
    ))
print(f"Generated: {filename_final}")