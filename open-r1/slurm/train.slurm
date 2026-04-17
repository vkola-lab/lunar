#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuH200x8
#SBATCH --time=00:30:00
#SBATCH --job-name=test_run
#SBATCH --account=beze-delta-gpu
### GPU options ###
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skowshik@bu.edu
#SBATCH --output=%x-%j.out


if [[ "$*" == *"--help"* ]]; then
  echo "Usage: sbatch slurm/train.slurm [options]"
  echo "Options:"
  echo "  --model MODEL            Model name"
  echo "  --task TASK              Task name (e.g. sft, grpo)"
  echo "  --config SUFFIX          Configuration suffix (e.g. demo, v00.00)"
  echo "  --accelerator CONFIG     Accelerator configuration name (e.g. zero3)"
  echo "  --dp N                   Data parallelism for vLLM server (default: 1)"
  echo "  --tp N                   Tensor parallelism for vLLM server (default: 1)"
  echo "  --args \"ARGS\"          Optional arguments to pass to the training script"
  exit 0
fi

# Specific configuration optimized for the Hugging Face Compute Cluster
# module load cuda/12.4.0

echo $CONDA_DEFAULT_ENV
python -V

export HF_HOME=/projects/beze/skowshik/.cache/
export WANDB_CACHE_DIR=/projects/beze/skowshik/.cache/

cd /u/skowshik/reasoning_framework/adrd-foundation-model/open-r1
uv pip install -e ".[dev]"

START_TIME=$(date +%s)
echo "START TIME: $(date)"

# Refresh Weka on h4 cache
# echo "Refreshing Weka filesystem..."
# find -L /fsx/h4/ -type f | xargs -d '\n' -r -n512 -P64 weka fs tier fetch

# Default values
MODEL=""
TASK=""
CONFIG_SUFFIX=""
ACCELERATOR=""
DP=1
TP=1
OPTIONAL_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --config)
      CONFIG_SUFFIX="$2"
      shift 2
      ;;
    --accelerator)
      ACCELERATOR="$2"
      shift 2
      ;;
    --dp)
      DP="$2"
      shift 2
      ;;
    --tp)
      TP="$2"
      shift 2
      ;;
    --args)
      OPTIONAL_ARGS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "$MODEL" || -z "$TASK" || -z "$CONFIG_SUFFIX" || -z "$ACCELERATOR" ]]; then
  echo "Error: Missing required arguments"
  echo "Run with --help for usage information"
  exit 1
fi


CONFIG_FILE=recipes/$MODEL/$TASK/config_$CONFIG_SUFFIX.yaml
GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps' $CONFIG_FILE | awk '{print $2}')

# Split the string into individual arguments
IFS=' ' read -ra ARGS <<< "$OPTIONAL_ARGS"
# Loop through the arguments and find the one with "--gradient_accumulation_steps"
for arg in "${ARGS[@]}"; do
    if [[ "$arg" == "--gradient_accumulation_steps="* ]]; then
        # Extract the value after the equals sign
        GRAD_ACC_STEPS="${arg#*=}"
        break  # Exit the loop once we find the desired argument
    fi
done

echo "Gradient accumulation steps: $GRAD_ACC_STEPS"

MODEL=$(grep 'model_name_or_path:' $CONFIG_FILE | awk '{print $2}')
REVISION=$(grep 'model_revision:' $CONFIG_FILE | head -n 1 | awk '{print $2}')

# Distributed configuration
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}  # First node for main process
MASTER_PORT=6000
TRAIN_NODES=("${NODELIST[@]}")

USE_VLLM="false"
if [[ -f "$CONFIG_FILE" ]] && grep -qE '^\s*use_vllm:\s*true' "$CONFIG_FILE"; then
    USE_VLLM="true"
fi
# if using vllm
if [[ "$USE_VLLM" == "true" ]] && grep -qE '^\s*vllm_mode:\s*server' "$CONFIG_FILE"; then
     TRAIN_NODES=("${NODELIST[@]:0:$((NUM_NODES - 1))}")
     VLLM_NODE=${NODELIST[-1]} # Last node
     WORLD_SIZE=$((WORLD_SIZE - GPUS_PER_NODE))
     NUM_NODES=$((NUM_NODES - 1))
     srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE trl vllm-serve --model $MODEL --revision $REVISION --tensor_parallel_size $TP --data_parallel_size $DP &

     OPTIONAL_ARGS="$OPTIONAL_ARGS --vllm_server_host=$VLLM_NODE"
fi

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

export CMD=" \
    src/open_r1/$TASK.py --config $CONFIG_FILE $OPTIONAL_ARGS
    "

export LAUNCHER="ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file recipes/accelerate_configs/$ACCELERATOR.yaml  \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend=c10d \
    --max_restarts 0 \
    --tee 3 \
    "
# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
NODELIST=$(IFS=,; echo "${TRAIN_NODES[*]}")

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --nodes=$NUM_NODES \
    --ntasks=$NUM_NODES \
    --nodelist=$NODELIST
    "
srun $SRUN_ARGS bash -c "$LAUNCHER $CMD" 2>&1

# srun $SRUN_ARGS --export=ALL,ACCELERATE_LOG_LEVEL=info,TRANSFORMERS_VERBOSITY=info \
#   accelerate launch \
#     --config_file "recipes/accelerate_configs/$ACCELERATOR.yaml" \
#     --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
#     --num_machines "$NUM_NODES" \
#     --num_processes "$WORLD_SIZE" \
#     --main_process_ip "$MASTER_ADDR" \
#     --main_process_port "$MASTER_PORT" \
#     --machine_rank "$SLURM_PROCID" \
#     --rdzv_backend c10d \
#     --max_restarts 0 \
#     --tee 3 \
#     "src/open_r1/${TASK}.py" --config "$CONFIG_FILE" $OPTIONAL_ARGS

END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"


# How to run this
# conda activate conds_envs_dt/open_r1
# cd reasoning_framework/adrd-foundation-model/open-r1/logs/

# sbatch --job-name=qwen3b_nacc_inc_oversample --nodes=1 --time=16:00:00 ../slurm/train.slurm --model Qwen2.5-3B-Instruct --task grpo --config nacc_inc_oversample --accelerator zero2

# sbatch --job-name=qwen3b_nacc_inc_oversample_sce_tanh --nodes=1 --time=16:00:00 ../slurm/train.slurm --model Qwen2.5-3B-Instruct --task grpo --config nacc_inc_oversample_sce_tanh --accelerator zero2

# sbatch --job-name=qwen3b_nacc_inc_oversample_dedup --nodes=1 --time=16:00:00 ../slurm/train.slurm --model Qwen2.5-3B-Instruct --task grpo --config nacc_inc_oversample_dedup --accelerator zero2

# sbatch --job-name=qwen3b_nacc_inc_oversample_dedup_sce_tanh --nodes=1 --time=16:00:00 ../slurm/train.slurm --model Qwen2.5-3B-Instruct --task grpo --config nacc_inc_oversample_dedup_sce_tanh --accelerator zero2


# 7B
# sbatch --job-name=qwen7b_nacc_inc_oversample --nodes=1 --time=24:00:00 --mem=200g ../slurm/train.slurm --model Qwen2.5-7B-Instruct --task grpo --config nacc_inc_oversample --accelerator zero2

# sbatch --job-name=qwen7b_nacc_inc_oversample_sce_tanh --nodes=1 --time=24:00:00 --mem=200g ../slurm/train.slurm --model Qwen2.5-7B-Instruct --task grpo --config nacc_inc_oversample_sce_tanh --accelerator zero2


# SFT

# sbatch --job-name=qwen3b_nacc_inc_oversample_sft --nodes=1 --time=24:00:00 ../slurm/train.slurm --model Qwen2.5-3B-Instruct --task sft --config nacc_inc_oversample --accelerator zero2