#!/bin/bash -l

module load python3

# we do not need gpus to compute metrics
source venvs/venv_cpu/bin/activate

python -V

RESULTS_DIR="results/nacc_test_updated"

python src/compute_metrics.py $RESULTS_DIR