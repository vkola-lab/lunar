#!/bin/bash -l

# SCC project Setup
#$ -P vkolagrp
#$ -l h_rt=48:00:00
#$ -pe omp 32 
#$ -l gpus=4 -l gpu_c=8

# module load python3/3.8.10
python -V
module load miniconda
module load cuda
conda activate /projectnb/vkolagrp/skowshik/conda_envs/open_r1
cd /projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1

export HF_HOME=/projectnb/vkolagrp/skowshik/.cache/
export WANDB_CACHE_DIR=/projectnb/vkolagrp/skowshik/.cache/

pip install .

CUDA_VISIBLE_DEVICES=1,2,3 ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 3 \
src/open_r1/grpo.py --config recipes/Qwen2.5-3B-Instruct/grpo/config_demo.yaml > /projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/logs/log_files/temp.log

# CUDA_VISIBLE_DEVICES=1,2,3 ACCELERATE_LOG_LEVEL=info \
# accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 3 \
# src/open_r1/grpo.py --config recipes/Qwen2.5-7B-Instruct/grpo/config_demo.yaml > /projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/logs/qwen25_7B_filtered.log


