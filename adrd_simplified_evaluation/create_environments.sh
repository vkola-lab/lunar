#!/bin/bash -l 
set -e  # Exit on first error

VENV_DIR="venvs"
CPU_ENV="$VENV_DIR/venv_cpu"
GPU_ENV="$VENV_DIR/venv_gpu"

REQ_CPU="requirements_cpu.txt"
REQ_GPU="requirements_gpu.txt"

mkdir -p "$VENV_DIR"

create_env() {
    local env_path=$1
    local req_file=$2

    if [ -d "$env_path" ]; then
        echo "=== Skipping: $env_path already exists ==="
        return
    fi

    echo "=== Creating environment: $env_path ==="
    python3 -m venv "$env_path"
    source "$env_path/bin/activate"
    pip install --upgrade pip
    pip install -r "$req_file"
    deactivate
}

# Always create CPU environment
create_env "$CPU_ENV" "$REQ_CPU"

# Create GPU environment only if GPU is present
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    echo "=== GPU detected ==="
    module load cuda # vllm needs some libraries from here, it compiles something internally while installing
    create_env "$GPU_ENV" "$REQ_GPU"
else
    echo "=== No GPU detected, skipping GPU environment creation ==="
    echo "=== Run this script on a GPU node to create the GPU environment ==="
fi

echo "=== Done! ==="
echo "Activate CPU env: source $CPU_ENV/bin/activate"
echo "Activate GPU env: source $GPU_ENV/bin/activate"
