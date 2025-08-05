# LLM Answer Extractor

Framework for extracting multiple-choice answers from LLM-generated responses and evaluating them against clinician-labeled data. This framework supports multiple benchmarks including Neuropathology, MCI, Cognitive Status, and ETPR evaluations.

---

## Overview

- Use regular expression to extract the answers.
- If regex failed, use an LLM to extract answers from responses
- Compare model performance against clinician ground truth (for Neuropath)
- Evaluate multiple models across different medical benchmarks
- Generate pass@1 and cons@k metrics and save the final plots

## Project Structure

```bash
llm_answer_extractor/
│
├── config.yml                  # Main configuration for LLM models and benchmarks
├── main.py                     # Pipeline entry
├── extract_answers.sh          # Batch script for running evaluations on SCC
│
├── utils/                      # Core utilities and data processing
│   ├── config_loader.py        # Configuration management and loading
│   ├── data_utils.py           # Data loading, preprocessing, and clinician label processing
│   └── prompts.py              # LLM prompts for answer extraction
│
├── models/                     # LLM interface and answer extraction logic
│   ├── llm_interface.py        # Load the model used to extract answers
│   └── answer_extractor.py     # Answer extraction (regex + LLM)
│
├── pipeline/                   # Evaluation and scoring pipeline
│   └── evaluator.py            # Performance metrics calculation (pass@k, cons@k)
│
├── plots/                      # Visualization and reporting
│   └── plot_results.py         # Seaborn plot for model comparison
│
├── config/                     # Benchmark-specific configurations
│   ├── config_np.yml           # Neuropathology benchmark configuration
│   ├── config_mci.yml          # MCI benchmark configuration
│   ├── config_cog_stat.yml     # Cognitive Status benchmark configuration
│   ├── config_etpr.yml         # ETPR benchmark configuration
│   ├── config_train.yml        # Training data configuration
│
├── outputs/                    # Generated plots and visualizations
│   ├── full/                   # Full dataset results
│   └── subgroups/              # Subgroup analysis results
│
└── extracted_results/          # Extracted answer results by benchmark
    ├── Neuropath/              # Neuropathology results
    ├── MCI/                    # MCI results
    ├── COGSTAT/                # Cognitive Status results
    ├── ETPR/                   # ETPR results
    ├── Train/                  # Training data results
    └── result_csv/             # CSV summary files
 
```


## Usage

Update config.yml and the config files under config/
```bash
qsub -N run_name extract_answers.sh
```


## Outputs

### 1. **extracted_results/**
- CSV files with extracted answers for each model
- Each file has a column to indicate the extraction method used (regex vs LLM)

### 2. **extracted_results/result_csv**
- CSV summaries with all metrics
- Organized by benchmark type

### 3. **Outputs/**
- Bar plots comparing model performance
- Red - Baseline models, Blue - trained model

---
