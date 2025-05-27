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

CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-3B-Instruct
# CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt/qwen25_3B_filtered_no_kl_corrected/checkpoint-2601 
# CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt/qwen25_3B_filtered_corrected/checkpoint-2601