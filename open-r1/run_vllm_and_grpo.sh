#!/bin/bash -l

# SCC project Setup
#$ -P vkolagrp
#$ -l h_rt=48:00:00
#$ -pe omp 32
#$ -l gpu_c=8
#$ -l gpus=4
#$ -l gpu_type=A100

python -V
module load miniconda
module load cuda
conda activate /projectnb/vkolagrp/skowshik/conda_envs/open_r1
cd /projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1

export HF_HOME=/projectnb/vkolagrp/skowshik/.cache/
export WANDB_CACHE_DIR=/projectnb/vkolagrp/skowshik/.cache/


#--------------Block 1-------------#
# Run the model serving in the background
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-3B-Instruct &

# wait for 30 seconds to allow model serving to initiate completely.
sleep 60

# The 30 seconds delay above can be adjusted as necessary.

# Alternatively use the following block by uncommenting it.
#--------------Block 2-------------#
# Run the model serving in the background
#CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-3B-Instruct &

## Initiate the shell variable PORT and assign value 8000 to it. 
#PORT=8000

## PORT 8000 is default for vLLM.
## Now we will check if the port is active. 
#while ! ss -ltn | grep -q ":$PORT"; do
#    sleep 2 # If the PORT 8000 is not active, wait for 2 seconds and check again.
#done
##---------------------------#

# If the model is served, then move to the next step.
pip install .

CUDA_VISIBLE_DEVICES=1,2,3 ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 3 \
src/open_r1/grpo.py --config recipes/Qwen2.5-3B-Instruct/grpo/config_demo.yaml > /projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/logs/log_files/qwen25_3B_option_shuffling.log
