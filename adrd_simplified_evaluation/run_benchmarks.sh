#!/bin/bash -l

# This script is set up so that you can either qsub it or run it interactively.
# Example usage:
#   Interactive: $ ./run_benchmarks.sh config.yml
#   Batch job:   $ qsub ./run_benchmarks.sh config.yml

# Make sure you're logged in to huggingface before running, if you're not sure
# you should login using "huggingface-cli login" before running this script

# Requesting resources from SCC
#$ -P vkolagrp
#$ -l h_rt=24:00:00
#$ -pe omp 8
#$ -l mem_per_core=2G
#$ -l gpus=1
#$ -l gpu_c=8 # GPU capability, must be at least 8 for this project
# -l gpu_type=H200
# -m bea
#$ -e logs/$JOB_ID.stderr
#$ -o logs/$JOB_ID.stdout
# We can in theory request a minimum amount of GPU memory, but setting
# capability to 8 means that whatever GPU we get it will definitely have enough
# memory for our purposes

# Check that exactly one config file was passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 config_file.yml"
    exit 1
fi

module load python3
module load cuda

source venvs/venv_gpu/bin/activate

# Using Sahana's cache to save some space
# export HF_HOME=/projectnb/vkolagrp/skowshik/.cache
export HF_HOME=/projectnb/vkolagrp/bellitti/hf_cache

# If this env var is set to 1, vLLM will skip the peer-to-peer check,
# and trust the driver's peer-to-peer capability report.
export VLLM_SKIP_P2P_CHECK=1

python -V

python src/run_benchmarks.py config_file=$1