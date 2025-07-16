#!/bin/bash -l

# This script is set up so that you can either qsub it or run it interactively

#$ -P vkolagrp
#$ -l h_rt=4:00:00
#$ -pe omp 4
#$ -l mem_per_core=2G
#$ -l gpus=2
# GPU capability, must be at least 8 for this project
#$ -l gpu_c=8
# We can in theory request a minimum amount of GPU memory, but setting
# capability to 8 means that whatever GPU we get it will definitely have enough
# memory for our purposes

# module load python3

module load miniconda
module load cuda

# Login using "huggingface-cli login" before running this script "huggingface-cli login"
export HF_HOME=/projectnb/vkolagrp/skowshik/.cache

export VLLM_SKIP_P2P_CHECK=1


cd /projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/llm_answer_extractor
conda activate /projectnb/vkolagrp/skowshik/conda_envs/vllm_env

python -V

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
        echo "Starting benchmark evaluation script..."
        python main.py config_file=config/config_np.yml 
        python main.py config_file=config/config_etpr.yml 
        python main.py config_file=config/config_mci.yml 
        python main.py config_file=config/config_cog_stat.yml 
        # python main.py config_file=config/config_cbd.yml
        # python main.py config_file=config/config_prio.yml
        # python main.py config_file=config/config_pick.yml
        # python main.py config_file=config/config_prog.yml
        # python main.py config_file=config.yml
        break
    else
        echo "GPU still busy with $count Python processes. Checking again in 20 minutes..."
        sleep 1200
    fi
done

# python main.py

# echo "Starting answer extaction script..."
# python main.py config_file=config/config_np.yml
# python main.py config_file=config/config_etpr.yml
# python main.py config_file=config/config_mci.yml
# python main.py config_file=config/config_cog_stat.yml
# # python main.py config_file=config/config_cbd.yml
# # python main.py config_file=config/config_prio.yml
# # python main.py config_file=config/config_pick.yml
# # python main.py config_file=config/config_prog.yml