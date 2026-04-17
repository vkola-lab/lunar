# data_preparation

This directory contains all scripts for processing raw cohort data into training and testing datasets for LUNAR. Each cohort (`nacc`, `adni`, `brainlat`, `nifd`, `ppmi`) has its own subfolder with a `data_preparation.ipynb` notebook that handles cohort-specific harmonization. Shared utilities and summary generation scripts live at the top level and in `create_summary/`.

---

## Directory Structure

```
data_preparation/
├── nacc/                          # NACC cohort processing
├── adni/                          # ADNI cohort processing
├── brainlat/                      # BrainLat cohort processing
├── nifd/                          # NIFD cohort processing
├── ppmi/                          # PPMI cohort processing
├── create_summary/                # Summary generation and quality check scripts
├── clinical_validation/           # Scripts for the neurologist validation study
├── Inclusion_criterion.txt        # Participant inclusion/exclusion criteria
├── csv_to_jsonl.py                # Converts CSV datasets to JSONL format for training
├── jsonl_to_csv.py                # Converts JSONL back to CSV
└── stats.ipynb                    # Population statistics and cohort breakdowns
```

Each cohort folder contains a `data_preparation.ipynb` specific to that cohort's data dictionary and variable naming conventions. See the notebook inside each subfolder for cohort-specific instructions.

---

## Data

Raw cohort files must be downloaded separately and are not included in this repository. See the [Data Access](../README.md#data-access) section in the top-level README for download links and instructions for each cohort.

Once downloaded, place raw data files in the expected paths as referenced in each cohort's `data_preparation.ipynb`.

---

## Pipeline Overview

The pipeline runs in the following order. Steps marked **(NACC only)** apply only to the training cohort; all other cohorts follow the testing path only.

### Step 1 — Cohort-specific data preparation

Run the `data_preparation.ipynb` inside each cohort folder. This harmonizes raw tabular data into a structured CSV with patient history and diagnosis JSON fields.

```
data/{cohort}/data_preparation.ipynb
→ data/{cohort}/training_data/{cohort}_wjson.csv   (or equivalent)
```

### Step 2 — Initial train/test split **(NACC only)**

Select the last visit per participant, incorporate neuropathology cases into the training set, and produce an initial split.

```
nacc/create_train_test_splits.ipynb
→ data/nacc/training_data/train.csv
→ data/nacc/training_data/test.csv
```

### Step 3 — Final training subset selection **(NACC only)**

The initial split from Step 2 is not the final training set. Run `sample_nacc_data.ipynb` to select the final training subset from `train.csv`.

```
nacc/sample_nacc_data.ipynb
→ final training subset used for GRPO and SFT
```

### Step 4 — Training data **(NACC only)**

**4a.** Create GRPO training questions (multiple-choice, with randomly shuffled answer choices):

```
nacc/create_grpo_training_data.ipynb
→ data/nacc/training_data/training_data_grpo/train_with_questions.csv
```

**4b.** Generate visit summaries (prose narratives from structured JSON using Qwen3-32B):

```
create_summary/generate_patient_summary.sh
→ data/nacc/training_data/training_data_grpo/train_summary.csv
```

**4c.** Check summary quality:

```
create_summary/check_generated_summaries.ipynb
```

Repeat 4b–4c until summary quality is satisfactory.

### Step 5 — Testing data (all cohorts)

For testing, summaries are generated first so the same summaries can be reused across all question types.

**5a.** Generate visit summaries:

```
create_summary/generate_patient_summary.sh
→ data/{cohort}/training_data/testing_data_grpo/test_summary.csv
```

**5b.** Check summary quality:

```
create_summary/check_generated_summaries.ipynb
```

Repeat 5a–5b until quality is satisfactory.

**5c.** Create GRPO testing questions. For NACC, this step also merges the remaining training data (cases not selected in Step 3) into the test set:

```
{cohort}/create_grpo_testing_data.ipynb
→ data/{cohort}/training_data/testing_data_grpo/with_summary/test_{task}.csv
```

where `{task}` is one of: `cog`, `etpr`, `csf`, `pet`, `dat`, `np_one`, `np_mixed`.

---

## Output Structure

After running the full pipeline, the expected output layout is:

```
data/
├── nacc/
│   └── training_data/
│       ├── train.csv
│       ├── test.csv
│       ├── training_data_grpo/
│       │   ├── train_with_questions.csv
│       │   └── train_summary.csv
│       └── testing_data_grpo/
│           ├── test_summary.csv
│           └── with_summary/
│               ├── test_cog.csv
│               ├── test_etpr.csv
│               ├── test_csf.csv
│               ├── test_pet.csv
│               ├── test_dat.csv
│               ├── test_np_one.csv
│               └── test_np_mixed.csv
├── adni/
│   └── training_data/
│       └── testing_data_grpo/
│           └── with_summary/
│               ├── test_cog.csv
│               ├── test_etpr.csv
│               ├── test_csf.csv
│               └── test_pet.csv
└── {brainlat, nifd, ppmi}/
    └── (same testing_data_grpo structure, tasks vary by cohort)
```

---

## Utility Scripts

**`csv_to_jsonl.py`** — converts a processed CSV dataset to JSONL format expected by the evaluation pipeline:

```bash
python csv_to_jsonl.py --input data/nacc/training_data/training_data_grpo/train_with_questions.csv \
                       --output data/nacc/training_data/training_data_grpo/train.jsonl
```

**`jsonl_to_csv.py`** — converts JSONL back to CSV:

```bash
python jsonl_to_csv.py --input data/nacc/training_data/training_data_grpo/train.jsonl \
                       --output data/nacc/training_data/training_data_grpo/train_check.csv
```

**`stats.ipynb`** — computes population statistics and cohort breakdowns (demographics, cognitive status, etiology distributions).

**`Inclusion_criterion.txt`** — documents the inclusion and exclusion criteria applied during participant selection for each cohort.

---

## Dependencies

```
python      3.11
pandas
numpy
jupyter
tqdm
```

The summary generation step (`generate_patient_summary.sh`) additionally requires access to Qwen3-32B via a vLLM server or API endpoint. See `create_summary/` for configuration details.
