#!/bin/bash -l

# SCC project Setup
#$ -P vkolagrp
#$ -l h_rt=48:00:00
#$ -pe omp 4 
#$ -l gpus=2 -l gpu_c=8

# module load python3/3.8.10
module load miniconda/23.11.0
conda activate /projectnb/vkolagrp/skowshik/conda_envs/fmadrd_vllm
python -V
export HF_HOME=/projectnb/vkolagrp/skowshik/.cache/

# Login using "huggingface-cli login" before running this script "huggingface-cli login"

# With Batch Processing 
# CUDA_LAUNCH_BLOCKING=1 python ./code/data_preparation/patient_summary_vllm.py --batch_size 512 --json_name "nacc_unique_with_llama_summaries_4.json" --start_id 150000 --end_id 250000 #--distributed
# CUDA_LAUNCH_BLOCKING=1 python ./code/data_preparation/diagnostic_summary_vllm.py --batch_size 512 --model_id "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16" --json_name "nacc_unique_with_llama_summaries_test_quant.json" --start_id 0 --end_id 512 #--distributed

# CUDA_LAUNCH_BLOCKING=1 python ./code/data_preparation/diagnostic_summary_vllm_json_format_output.py --batch_size 512 --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" --json_name "nacc_summaries_json_output_test.json" --start_id 0 --end_id 1024 #--distributed

CUDA_LAUNCH_BLOCKING=1 python ./code/data_preparation/diagnostic_summary_vllm_json_format_output.py --batch_size 2048 --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" --json_name "nacc_summaries_json_output_3.json" --start_id 155669 --end_id 500000 #--distributed

# CUDA_LAUNCH_BLOCKING=1 python ./code/data_preparation/eeg_summary_vllm.py --batch_size 512 --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" --json_name "eeg_summary.json" 