#!/bin/bash -l

module load python3

# we do not need gpus to compute metrics
source venvs/venv_cpu/bin/activate

python -V

RESULTS_DIR="results"

python src/make_figures.py $RESULTS_DIR