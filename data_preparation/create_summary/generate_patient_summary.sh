#!/bin/bash -l

# SCC project Setup
#$ -P vkolagrp
#$ -l h_rt=12:00:00
#$ -pe omp 16 
#$ -l gpus=2 -l gpu_c=8

# module load python3/3.8.10
python -V
module load miniconda
conda activate /projectnb/vkolagrp/projects/adrd_foundation_model/envs/fmadrd
cd /projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/fm_adrd/data_preparation/create_summary
export HF_HOME=/projectnb/vkolagrp/skowshik/.cache/

# CUDA_LAUNCH_BLOCKING=1 python nacc_summary_generation_step_1.py

# CUDA_LAUNCH_BLOCKING=1 python generate_patient_summary.py --json_name "qwen3_14B_etpr_cog_mci_park_nocop.csv" --start_id 0 --end_id 100000 --directory_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/final/nacc/train_with_summary" --model_id "Qwen/Qwen3-14B" --n_devices 4 --batch_size 1000 --max_new_tokens 2000 --data_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/grpo/data_etpr_cog_mci_park.csv"

# CUDA_LAUNCH_BLOCKING=1 python generate_patient_summary.py --json_name "COG_STATUS_test.csv" --start_id 0 --end_id 1000 --directory_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/fm_adrd/adrd_simplified_evaluation/benchmarks/adrd_cog_status_summary" --model_id "Qwen/Qwen3-14B" --n_devices 1 --batch_size 1000 --max_new_tokens 2000 --data_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/fm_adrd/adrd_simplified_evaluation/benchmarks/adrd_cog_status/COG_STATUS.csv"

# CUDA_LAUNCH_BLOCKING=1 python generate_patient_summary.py --json_name "NP_MAJOR_test.csv" --start_id 0 --end_id 1000 --directory_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/fm_adrd/adrd_simplified_evaluation/benchmarks/adrd_neuropath_summary" --model_id "skowshik/foundation_adrd/adrd-foundation-model/open-r1/recipes/Qwen2.5-3B-Instruct/grpo/config_demo.yaml" --n_devices 1 --batch_size 1000 --max_new_tokens 2000 --data_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/fm_adrd/adrd_simplified_evaluation/benchmarks/adrd_neuropath/NP_MAJOR.csv"

CUDA_LAUNCH_BLOCKING=1 python generate_patient_summary.py --json_name "train_summary.csv" --start_id 0 --end_id 1000000 --directory_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/training_data_grpo" --model_id "Qwen/Qwen3-30B-A3B" --n_devices 4 --batch_size 1000 --max_new_tokens 2000 --data_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/training_data_grpo/train.csv"

CUDA_LAUNCH_BLOCKING=1 python generate_patient_summary.py --json_name "test_summary.csv" --start_id 0 --end_id 1000000 --directory_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/training_data_grpo" --model_id "Qwen/Qwen3-30B-A3B" --n_devices 4 --batch_size 1000 --max_new_tokens 2000 --data_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/test.csv"

# CUDA_LAUNCH_BLOCKING=1 python generate_patient_summary.py --json_name "train_summary.csv" --start_id 0 --end_id 1000000 --directory_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nifd/training_data/training_data_grpo" --model_id "Qwen/Qwen3-30B-A3B" --n_devices 4 --batch_size 1000 --max_new_tokens 2000 --data_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nifd/training_data/training_data_grpo/train.csv"

CUDA_LAUNCH_BLOCKING=1 python generate_patient_summary.py --json_name "train_summary_32B.csv" --start_id 0 --end_id 1000000 --directory_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/training_data_grpo" --model_id "Qwen/Qwen3-32B" --n_devices 4 --batch_size 1000 --max_new_tokens 2000 --data_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/training_data_grpo/train.csv"


CUDA_LAUNCH_BLOCKING=1 python generate_patient_summary.py --json_name "test_summary_32B.csv" --start_id 0 --end_id 1000000 --directory_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/training_data_grpo" --model_id "Qwen/Qwen3-32B" --n_devices 4 --batch_size 1000 --max_new_tokens 2000 --data_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/test.csv"