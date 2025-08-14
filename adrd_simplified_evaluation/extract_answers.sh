#!/bin/bash -l

# This script is set up so that you can either qsub it or run it interactively.
# Example usage:
#   Interactive: $ ./extract_answers.sh 
#   Batch job:   $ qsub ./extract_answers.sh 

# Make sure you're logged in to huggingface before running, if you're not sure
# you should login using "huggingface-cli login" before running this script

# Requesting resources from SCC
#$ -P vkolagrp
#$ -l h_rt=2:00:00
#$ -pe omp 8
#$ -l mem_per_core=2G
#$ -l gpus=1
#$ -l gpu_c=8 # GPU capability, must be at least 8 for this project
# -l gpu_type=H200
# -m bea
#$ -e logs/$JOB_ID.stderr
#$ -o logs/$JOB_ID.stdout

module load python3
module load cuda

# Using Sahana's cache to save some space
export HF_HOME=/projectnb/vkolagrp/skowshik/.cache
# export HF_HOME=/projectnb/vkolagrp/bellitti/hf_cache

# If this env var is set to 1, vLLM will skip the peer-to-peer check,
# and trust the driver's peer-to-peer capability report.
export VLLM_SKIP_P2P_CHECK=1

# we need the gpu version to extract answers because we try to recover invalid answers via LLM
source venvs/venv_gpu/bin/activate

python -V

RESULTS_DIR="results"
EXTRACTOR_CONFIG="src/extractor_config.yml"

python src/extract_answers.py $RESULTS_DIR $EXTRACTOR_CONFIG