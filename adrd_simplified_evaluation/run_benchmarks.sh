#!/bin/bash -l

# This script is set up so that you can either qsub it or run it interactively

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

# module load python3

module load miniconda
module load cuda

# Login using "huggingface-cli login" before running this script "huggingface-cli login"
export HF_HOME=/projectnb/vkolagrp/skowshik/.cache

export VLLM_SKIP_P2P_CHECK=1

# set to 1 to execute gpu operations synchronously
# export CUDA_LAUNCH_BLOCKING=1 

cd /projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation
# conda activate /projectnb/vkolagrp/projects/adrd_foundation_model/envs/fmadrd
conda activate /projectnb/vkolagrp/skowshik/conda_envs/vllm_env

# source venv/bin/activate

python -V

# for STEPS in 600 1000 2400 3000; do
# for SEED in 6 7 8 9 10; do
    # python src/main.py config_file=config.yml llm_sampling_seed=$SEED lora_path="/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/fm_adrd/ckpt/llama31_8B_pretraining_deepspeed_cleaned_lr_5e7_bs_4/checkpoint-$STEPS" 
# done
# done

# while nvidia-smi | grep -q 'python'; do
#     echo "GPU still in use. Checking again in 20 minutes..."
#     sleep 1200  # wait 20 minutes
# done

# python src/main.py config_file=config.yml

while true; do
    count=$(nvidia-smi | grep -c python)
    if [ "$count" -lt 4 ]; then
        echo "GPU is idle with $count Python processes. Starting next script..."
        pids=$(nvidia-smi | grep python | awk '{print $5}')
        # for pid in $pids; do
        #     echo "Killing process with PID $pid"
        #     kill -9 "$pid"
        # done

        # Now start your script
        # echo "Starting benchmark evaluation script..."
        # python src/main.py config_file=config.yml

        python src/main.py config_file=config.yml  \
        model_name='["/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt_using_subset/qwen25_3B_drgrpo_gp8_train_filtered_sub", "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt_using_subset/qwen25_3B_drgrpo_gp8_train_filtered_sub_no_KL"]' \
        benchmarks='["benchmarks/nacc_test/test_np", "benchmarks/nacc_test/test_mci", "benchmarks/nacc_test/test_cog"]' \
        template_style="grpo" \
        system_prompt="Please reason step by step, and put your final answer within \\boxed{}."


        # python src/main.py config_file=config.yml  \
        # model_name='["Qwen/Qwen2.5-3B-Instruct", "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt_using_subset/qwen25_3B_drgrpo_gp8_all_train_sub_old_prompt"]' \
        # benchmarks='["benchmarks/nacc_test/test_cog"]' \
        # template_style="grpo_think" \
        # system_prompt="Please reason step by step, and put your final answer within \\boxed{}. The reasoning process should be enclosed within <think> </think> and the answer within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Reply in English only, do not use other languages."

        break
    else
        echo "GPU still busy with $count Python processes. Checking again in 20 minutes..."
        sleep 1200
    fi
done

