#!/bin/bash -l

# SCC project Setup
#$ -P vkolagrp
#$ -l h_rt=48:00:00
#$ -pe omp 4 
#$ -l gpus=1 -l gpu_type=L40S

# module load python3/3.8.10
module load miniconda/23.11.0
conda activate /projectnb/vkolagrp/skowshik/conda_envs/adrd
python -V
export HF_HOME=/projectnb/vkolagrp/skowshik/.cache/

# Login with using "huggingface-cli login" before running this script "huggingface-cli login"

# Without DDP 
CUDA_LAUNCH_BLOCKING=1 python ./code/training/finetune.py --n 10000

# With DDP (Doesn't work right now)
# CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --master_port=29757 ./code/training/finetune.py --distributed

