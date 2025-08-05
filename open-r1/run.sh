#!/bin/bash -l

# SCC project Setup
#$ -P vkolagrp
#$ -l h_rt=24:00:00
#$ -pe omp 32 
#$ -l gpus=4 -l gpu_c=8
#$ -m bea

# module load python3/3.8.10
python -V
module load miniconda
module load cuda
conda activate /projectnb/vkolagrp/skowshik/conda_envs/open_r1
cd /projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1

export HF_HOME=/projectnb/vkolagrp/skowshik/.cache/
export WANDB_CACHE_DIR=/projectnb/vkolagrp/skowshik/.cache/

pip install .

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 4 \
src/open_r1/grpo.py --config recipes/Qwen2.5-3B-Instruct/grpo/config_demo.yaml > logs_using_subset/log_files/qwen25_3B_drgrpo_gp8_train_filtered_sub.log

sleep 60

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 4 \
src/open_r1/grpo.py --config recipes/Qwen2.5-3B-Instruct/grpo/config_demo_1.yaml > logs_using_subset/log_files/qwen25_3B_drgrpo_gp8_train_filtered_sub_no_KL.log

sleep 60

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 4 \
src/open_r1/grpo.py --config recipes/Qwen2.5-3B-Instruct/grpo/config_demo_2.yaml > logs_using_subset/log_files/qwen25_3B_drgrpo_gp8_stage_wise_sub.log

sleep 60

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 4 \
src/open_r1/grpo.py --config recipes/Qwen2.5-3B-Instruct/grpo/config_demo_3.yaml > logs_using_subset/log_files/qwen25_3B_drgrpo_gp8_stage_wise_sub_no_KL.log


