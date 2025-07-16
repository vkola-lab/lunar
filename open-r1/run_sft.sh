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

# CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
# accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 4 \
# src/open_r1/sft.py --config recipes/Qwen2.5-3B-Instruct/sft/config_demo.yaml > /projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/logs/log_files/qwen25_3B_sft_with_eval.log

while true; do
    count=$(nvidia-smi | grep -c python)
    if [ "$count" -lt 4 ]; then
        echo "GPU is idle with $count Python processes. Starting next script..."
        pids=$(nvidia-smi | grep python | awk '{print $5}')
        for pid in $pids; do
            echo "Killing process with PID $pid"
            kill -9 "$pid"
        done

        # Now start your script
        # echo "Starting SFT script..."

        CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
        accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 4 \
        src/open_r1/sft.py --config recipes/Qwen2.5-3B-Instruct/sft/config_demo.yaml > /projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/logs/log_files/qwen25_3B_sft_with_eval.log

        break
    else
        echo "GPU still busy with $count Python processes. Checking again in 20 minutes..."
        sleep 1200
    fi
done

