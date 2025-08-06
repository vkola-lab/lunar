#!/bin/bash -l

# This script is set up so that you can either qsub it or run it interactively.
# It runs main.py with the specifiied configuration file.
# Example usage:
#   Interactive: $ ./run_benchmarks.sh config.yml
#   Batch job: $ qsub ./run_benchmarks.sh config.yml

# Make sure you're logged in to huggingface before running, if you're not sure
# you should login using "huggingface-cli login" before running this script

# Requesting resources from SCC
#$ -P vkolagrp
#$ -l h_rt=24:00:00
#$ -pe omp 8
#$ -l mem_per_core=2G
#$ -l gpus=2
# GPU capability, must be at least 8 for this project
#$ -l gpu_c=8
#$ -m bea
# We can in theory request a minimum amount of GPU memory, but setting
# capability to 8 means that whatever GPU we get it will definitely have enough
# memory for our purposes

module load miniconda
module load cuda

# Using Sahana's cache to save some space
export HF_HOME=/projectnb/vkolagrp/skowshik/.cache
# export HF_HOME=/projectnb/vkolagrp/bellitti/hf_cache

export VLLM_SKIP_P2P_CHECK=1

# Set to 1 to execute gpu operations synchronously. I suspect SCC enforces this
# anyway, only one user can use each GPU at any time.
# export CUDA_LAUNCH_BLOCKING=1 

# We can probably do this using the -cwd option for qsub
cd /projectnb/vkolagrp/bellitti/adrd-foundation-model/adrd_simplified_evaluation 

conda activate /projectnb/vkolagrp/skowshik/conda_envs/vllm_env

python -V

while true; do
    count=$(nvidia-smi | grep -c python)
    if [ "$count" -lt 4 ]; then

        echo "GPU is idle with $count Python processes. Starting next script..."

        pids=$(nvidia-smi | grep python | awk '{print $5}')

        # for pid in $pids; do
        #     echo "Killing process with PID $pid"
        #     kill -9 "$pid"
        # done

        python src/main.py config_file=$1

        break
    else
        echo "GPU still busy with $count Python processes. Checking again in 20 minutes..."
        sleep 1200
    fi
done

