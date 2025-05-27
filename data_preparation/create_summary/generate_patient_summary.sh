#!/bin/bash -l

# SCC project Setup
#$ -P vkolagrp
#$ -l h_rt=2:00:00
#$ -pe omp 16 
#$ -l gpus=2 -l gpu_c=8

# module load python3/3.8.10
python -V
module load miniconda
conda activate /projectnb/vkolagrp/projects/adrd_foundation_model/envs/fmadrd
cd /projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/fm_adrd/data_preparation/create_summary
export HF_HOME=/projectnb/vkolagrp/skowshik/.cache/

# CUDA_LAUNCH_BLOCKING=1 python generate_patient_summary.py --json_name "train_summary.csv" --start_id 0 --end_id 1000000 --directory_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/training_data_grpo" --model_id "Qwen/Qwen3-32B" --n_devices 4 --batch_size 1000 --max_new_tokens 2000 --data_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/training_data_grpo/train_with_questions.csv"

# CUDA_LAUNCH_BLOCKING=1 python generate_patient_summary.py --json_name "test_summary.csv" --start_id 0 --end_id 1000000 --directory_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/testing_data_grpo" --model_id "Qwen/Qwen3-32B" --n_devices 4 --batch_size 1000 --max_new_tokens 2000 --data_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/test.csv"


CUDA_LAUNCH_BLOCKING=1 python generate_patient_summary.py --json_name "train_summary.csv" --start_id 0 --end_id 1000000 --directory_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nifd/training_data/training_data_grpo" --model_id "Qwen/Qwen3-32B" --n_devices 4 --batch_size 1000 --max_new_tokens 2000 --data_path "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nifd/training_data/training_data_grpo/train_with_questions.csv"