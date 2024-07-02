#!/bin/bash -l

# SCC project Setup
#$ -P vkolagrp
#$ -l h_rt=24:00:00
#$ -pe omp 4 
#$ -l gpus=2 -l gpu_c=7.0 -l gpu_memory=32G

# module load python3/3.8.10
module load miniconda/23.11.0
conda activate adrd
python -V
export HF_HOME=/projectnb/vkolagrp/skowshik/.cache/

# Login with using "huggingface-cli login" before running this script "huggingface-cli login"

# Without DDP 
# CUDA_LAUNCH_BLOCKING=1 python ./code/data_preperation/diagnostic_summary.py

# With DDP 
CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --master_port=29757 ./code/data_preperation/diagnostic_summary.py --distributed

