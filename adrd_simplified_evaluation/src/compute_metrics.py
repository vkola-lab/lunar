from pathlib import Path
import textwrap
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import yaml
import re
import sys

import json

def option_string_to_dict(options):
    pattern = r'([A-Z])\. ([^\n]+)'
    matches = re.findall(pattern, options)
    return {key: value for key, value in matches}

def wrap_labels(labels, width):
    return ['\n'.join(textwrap.wrap(label, width)) for label in labels]


if __name__ == "__main__":

    ans_dir = Path(sys.argv[1])

    for ans_path in ans_dir.rglob("*_processed.parquet"):

        ans_df = pd.read_parquet(ans_path)

        if 'full_question' not in ans_df:
            ans_df['full_question'] = ans_df['question'] + ans_df['options']

        ans_df['attempt'] = ans_df.groupby('full_question').cumcount()

        benchmark_name = ans_path.parent.parent.stem

        metric_results = []

        for attempt,group in ans_df.groupby('attempt'):

            # NACC-based benchmarks are treated differently, we need to look at the ground_truth text
            # if we want meaningful class-wise metrics: for example, in the NC/MCI/DE problem the answers are shuffled:
            # sometimes 1=MCI and sometimes 2=MCI, so if we want a confusion matrix in terms of NC/MCI/DE instead of the
            # non informative 1/2/3 we have to map the options back to text. This does not apply to the MCQ benchmarks
            if benchmark_name in ['test_cog','test_etpr','test_mci','test_np']: 
                y_pred = group.apply(lambda row: option_string_to_dict(row['options']).get(row['prediction'],'Invalid'), axis=1)
                y_true = group['ground_truth_text']
                labels = sorted(group['ground_truth_text'].unique().astype(str))
            else:
                y_pred = group["prediction"].astype(str)
                y_true = group["ground_truth"].astype(str)
                labels = sorted(group['ground_truth'].unique().astype(str))

            metrics = {
                "benchmark_name": benchmark_name,
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "precision_macro": precision_score( y_true, y_pred, average="macro", zero_division=0,labels=labels),
                "recall_macro": recall_score( y_true, y_pred, average="macro", zero_division=0,labels=labels),
                "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0,labels=labels),
                "precision_weighted": precision_score( y_true, y_pred, average="weighted", zero_division=0,labels=labels),
                "recall_weighted": recall_score( y_true, y_pred, average="weighted", zero_division=0,labels=labels),
                "f1_weighted": f1_score( y_true, y_pred, average="weighted", zero_division=0,labels=labels),
                "invalid_percent": 100*len(y_pred[~y_pred.isin(labels)])/(len(y_pred))
            }

            with open(ans_path.parent / "config.yml", "r") as f:
                config = yaml.safe_load(f)
                metrics['model'] = config['run_readable_name']
            
            fig,ax = plt.subplots(1,1,figsize=(len(labels),len(labels)),layout='tight')

            ConfusionMatrixDisplay.from_predictions(y_true,y_pred,colorbar=False,labels=labels,cmap='Blues',ax=ax)

            # Wrap x-axis labels
            x_labels = [tick.get_text() for tick in ax.get_xticklabels()]
            wrapped_x = wrap_labels(x_labels, width=30)
            ax.set_xticklabels(wrapped_x, rotation=90, ha='right', va='center', rotation_mode='anchor',fontsize=7)

            # Wrap y-axis labels
            y_labels = [tick.get_text() for tick in ax.get_yticklabels()]
            wrapped_y = wrap_labels(y_labels, width=30)
            ax.set_yticklabels(wrapped_y,fontsize=7)   

            fig.savefig(ans_path.parent / f"confusion_matrix_attempt{attempt}.pdf")

            metric_results.append(metrics)

        with open(ans_path.parent / "metrics.json", "w") as f:
            json.dump(metric_results, f, indent=4)