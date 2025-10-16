import json
from pathlib import Path
import pandas as pd

readable_model_names = {
        "Qwen2.5-3B-Instruct": "Qwen2.5-3B",
        # "Qwen2.5-3B-DrGRPO-subset",
        # "Qwen2.5-3B-DrGRPO-Stratified",
        "Qwen2.5-3B-DrGRPO-Strat-NACC": "NACC",
        "Qwen2.5-3B-DrGRPO-Strat-MedQA-NACC": "MedQA-NACC",
        "Qwen2.5-3B-DrGRPO-Strat-MedQA-NACC-filtered": "MedQA-NACC-filt",
        "Qwen2.5-3B-DrGRPO-Strat-MedQA-NACC-sce": "MedQA-NACC-sce",
        "Qwen2.5-3B-DrGRPO-Strat-MedQA-NACC-sce-scaled": "MedQA-NACC-sce-scale",
        "Qwen3-4B": "Qwen3-4B",
        "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
        "HuatuoGPT-o1-8B": "HuatuoGPT-o1-8B",
        "NACC_Inc-sce-tanh-1000": "Ours+SCE",
        "NACC_Inc-1000": "Ours",
        "NACC-inc-os-sce": "Ours-SCE-OS",
        "NACC-inc-os": "Ours-OS",
    }

cat_order = [
        "Qwen2.5-3B-Instruct",
        # "Qwen2.5-3B-DrGRPO-subset",
        # "Qwen2.5-3B-DrGRPO-Stratified",
        # "Qwen2.5-3B-DrGRPO-Strat-NACC",
        # "Qwen2.5-3B-DrGRPO-Strat-MedQA-NACC",
        # "Qwen2.5-3B-DrGRPO-Strat-MedQA-NACC-filtered",
        # "Qwen2.5-3B-DrGRPO-Strat-MedQA-NACC-sce",
        # "Qwen2.5-3B-DrGRPO-Strat-MedQA-NACC-sce-scaled",
        "NACC_Inc-1000",
        "NACC_Inc-sce-tanh-1000",
        "NACC-inc-os",
        "NACC-inc-os-sce",
        "Qwen2.5-7B-Instruct",
        "HuatuoGPT-o1-8B",
        "Qwen3-4B",
    ]

cat_order_readable = [readable_model_names[model] for model in cat_order]

def load_metrics(results_dir, readable_names=True):
    """
    Load metrics JSON files from a directory, concatenate them into a single DataFrame,
    and return a "tall" format DataFrame ready for plotting.
    
    Args:
        results_dir (str or Path): Directory containing metrics.json files.

    Returns:
        pd.DataFrame: Tall DataFrame with columns ['benchmark_name', 'model', 'metric', 'value'].
    """

    results_dir = Path(results_dir)

    data = []
    for f in results_dir.rglob("metrics.json"):
        with open(f) as file:
            data.append(json.load(file))
    
    # Concatenate all results
    summary = pd.concat([pd.DataFrame(f) for f in data])
    summary = summary.set_index(["benchmark_name", "model"]).sort_index()
    summary.columns.name = "metric"
    
    # Convert to tall format
    tall = summary.stack().to_frame(name="value").reset_index()

    if readable_names:
        tall['model'] = tall['model'].replace(readable_model_names)
    
    return tall

def load_class_metrics(results_dir):
    """
    Load class metrics JSON files from a directory, concatenate them into a single DataFrame,
    and return a "tall" format DataFrame ready for plotting.
    
    Args:
        results_dir (str or Path): Directory containing metrics.json files.

    Returns:
        pd.DataFrame: Tall DataFrame with columns ['benchmark_name', 'model', 'metric', 'value'].
    """

    results_dir = Path(results_dir)

    data = []
    for f in results_dir.rglob("class_metrics.json"):
        with open(f) as file:
            data.append(json.load(file))
    
    # Concatenate all results
    tall = pd.concat([pd.DataFrame(f) for f in data],ignore_index=True)

    tall['model'] = tall['model'].replace(readable_model_names)
    
    return tall

def get_cat_order():
    """Return the predefined model order for plotting."""
    return cat_order_readable

# Example usage if run as script
if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1]
    df = load_metrics(results_dir)
    print(df.head())
