# LLM Answer Extractor

A modular Python framework for extracting multiple-choice answers using LLMs (e.g. Qwen) with comparison against clinician-labeled data.

---

## 📁 Project Structure

```bash
llm_answer_extractor/
│
├── config.yml                  # Configuration for models, data, and paths
├── main.py                     # Pipeline entry point
│
├── utils/                      # Utilities: config, data, prompts
│   ├── config_loader.py
│   ├── data_utils.py
│   ├── prompts.py
│   └── __init__.py
│
├── models/                     # LLM model interface + answer extractor
│   ├── llm_interface.py
│   ├── answer_extractor.py
│   └── __init__.py
│
├── pipeline/                   # Evaluation metrics and scoring
│   ├── evaluator.py
│   └── __init__.py
│
├── plots/                      # Seaborn/Matplotlib visualization
│   ├── plot_results.py
│   └── __init__.py
│
└── outputs/
    └── np.pdf                  # Output plot comparing models
```

## Run the Pipeline
Update `config.yml`
```
bash extract_answers.sh
```