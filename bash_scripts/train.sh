#!/bin/bash -l

# SCC project Setup
#$ -P vkolagrp
#$ -l h_rt=48:00:00
#$ -pe omp 4 
#$ -l gpus=2 -l gpu_c=8

# module load python3/3.8.10
module load miniconda/23.11.0
conda activate /projectnb/vkolagrp/projects/adrd_foundation_model/envs/fmadrd
python -V
export HF_HOME=/projectnb/vkolagrp/skowshik/.cache/hub/

# Login using "huggingface-cli login" before running this script "huggingface-cli login"


CUDA_LAUNCH_BLOCKING=1 python ./code/training/train.py --n 1000000000 --mode 1 --quant --wandb

