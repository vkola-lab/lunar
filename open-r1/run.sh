#!/bin/bash -l

# SCC project Setup
#$ -P vkolagrp
#$ -l h_rt=48:00:00
#$ -pe omp 4
#$ -l gpus=2 -l gpu_c=9
#$ -m bea

module load miniconda
module load cuda
conda activate /projects/beze/skowshik/conds_envs/open_r1
cd /u/skowshik/reasoning_framework/adrd-foundation-model/open-r1
python -V

export HF_HOME=/projects/beze/skowshik/.cache/
export WANDB_CACHE_DIR=/projects/beze/skowshik/.cache/

# pip install .


# ACCELERATE_LOG_LEVEL=info \
# accelerate launch -config_file recipes/accelerate_configs/zero2.yaml --num_processes 4 \
# src/open_r1/grpo.py --config recipes/Qwen2.5-3B-Instruct/grpo/config_demo.yaml >  /projects/beze/skowshik/logs/logs_modifed_cog_np_subset/log_files/qwen25_3B_drgrpo_gp8.log