from pathlib import Path
import textwrap
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import yaml
import re
import sys
from tqdm import tqdm
import warnings

import json

# ignore warning about unseen classes in y_pred
warnings.filterwarnings("ignore", category=UserWarning)


def option_string_to_dict(options):
    pattern = r"([A-Z])\. ([^\n]+)"
    matches = re.findall(pattern, options)
    return {key: value for key, value in matches}


def wrap_labels(labels, width):
    return ["\n".join(textwrap.wrap(label, width)) for label in labels]


if __name__ == "__main__":

    ans_dir = Path(sys.argv[1])

    for ans_path in tqdm(list(ans_dir.rglob("*.parquet"))):

        metrics_path = ans_path.parent / "metrics.json"
        class_metrics_path = None

        if metrics_path.is_file():
            continue
        else:
            ans_df = pd.read_parquet(ans_path)

            if "full_question" not in ans_df:
                ans_df["full_question"] = ans_df["question"] + ans_df["options"]

            if "ID" in ans_df:
                ans_df["attempt"] = ans_df.groupby("ID").cumcount()
            else:
                ans_df["attempt"] = ans_df.groupby("full_question").cumcount()

            benchmark_name = ans_path.parent.parent.stem

            metric_results = []
            class_metric_results = []

            for attempt, group in ans_df.groupby("attempt"):

                # NACC-based benchmarks are treated differently, we need to look at the ground_truth text
                # if we want meaningful class-wise metrics: for example, in the NC/MCI/DE problem the answers are shuffled:
                # sometimes 1=MCI and sometimes 2=MCI, so if we want a confusion matrix in terms of NC/MCI/DE instead of the
                # non informative 1/2/3 we have to map the options back to text. This does not apply to the MCQ benchmarks
                if benchmark_name in [
                    "test_cog",
                    "test_csf",
                    "test_dat",
                    "test_etpr",
                    "test_mci",
                    "test_np",
                    "test_np_mixed",
                    "test_np_one",
                    "test_pet",
                ]:
                    class_metrics_path = ans_path.parent / "class_metrics.json"
                    y_pred = group.apply(
                        lambda row: option_string_to_dict(row["options"]).get(
                            row["prediction"], "Invalid"
                        ),
                        axis=1,
                    )
                    y_true = group["ground_truth_text"]
                    labels = sorted(group["ground_truth_text"].unique().astype(str))
                else:
                    y_pred = group["prediction"].astype(str)
                    y_true = group["ground_truth"].astype(str)
                    labels = sorted(group["ground_truth"].unique().astype(str))

                metrics = {
                    "benchmark_name": benchmark_name,
                    "invalid_frac": len(y_pred[~y_pred.isin(labels)]) / (len(y_pred)),
                    "invalid_num": len(y_pred[~y_pred.isin(labels)]),
                    "accuracy": accuracy_score(y_true, y_pred),
                }

                with open(ans_path.parent / "config.yml", "r") as f:
                    config = yaml.safe_load(f)
                    metrics["model"] = config["run_readable_name"]

                # if the length of all labels is 1, they are multiple choice questions, it's not meaningful
                if any([len(label) != 1 for label in labels]):

                    # recall_weighted is the same thing as accuracy
                    # metrics["recall_weighted"] = recall_score( y_true, y_pred, average="weighted", zero_division=0,labels=labels)
                    metrics["precision_weighted"] = precision_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                        labels=labels,
                    )
                    metrics["f1_weighted"] = f1_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                        labels=labels,
                    )

                    # recall_macro is the same thing as balanced accuracy
                    metrics["recall_macro"] = recall_score(
                        y_true, y_pred, average="macro", zero_division=0, labels=labels
                    )
                    metrics["precision_macro"] = precision_score(
                        y_true, y_pred, average="macro", zero_division=0, labels=labels
                    )
                    metrics["f1_macro"] = f1_score(
                        y_true, y_pred, average="macro", zero_division=0, labels=labels
                    )

                    # clas specific metrics
                    precisions = precision_score(
                        y_true, y_pred, average=None, labels=labels
                    )
                    precision_df = pd.DataFrame(
                        {"class": labels, "value": precisions}
                    ).assign(metric="precision", benchmark_name=benchmark_name)

                    recalls = recall_score(y_true, y_pred, average=None, labels=labels)
                    recall_df = pd.DataFrame(
                        {"class": labels, "value": recalls}
                    ).assign(metric="recall", benchmark_name=benchmark_name)

                    class_metric_df = pd.concat(
                        (precision_df, recall_df), ignore_index=True
                    ).assign(model=config["run_readable_name"])

                    # make confusion matrices for all attempts
                    # fig, ax = plt.subplots(
                    #     1, 1, figsize=(len(labels), len(labels)), layout="tight"
                    # )

                    # ConfusionMatrixDisplay.from_predictions(
                    #     y_true,
                    #     y_pred,
                    #     colorbar=False,
                    #     labels=labels,
                    #     cmap="Blues",
                    #     ax=ax,
                    # )

                    # # Wrap x-axis labels
                    # x_labels = [tick.get_text() for tick in ax.get_xticklabels()]
                    # wrapped_x = wrap_labels(x_labels, width=30)
                    # ax.set_xticklabels(
                    #     wrapped_x,
                    #     rotation=90,
                    #     ha="right",
                    #     va="center",
                    #     rotation_mode="anchor",
                    #     fontsize=7,
                    # )

                    # # Wrap y-axis labels
                    # y_labels = [tick.get_text() for tick in ax.get_yticklabels()]
                    # wrapped_y = wrap_labels(y_labels, width=30)
                    # ax.set_yticklabels(wrapped_y, fontsize=7)

                    # fig.savefig(
                    #     ans_path.parent / f"confusion_matrix_attempt{attempt}.pdf"
                    # )

                    # plt.close(fig=fig)

                metric_results.append(metrics)

                if class_metrics_path is not None:
                    class_metric_results.append(class_metric_df)

            with open(metrics_path, "w") as f:
                json.dump(metric_results, f, indent=4)

            if class_metrics_path is not None:
                pd.concat(class_metric_results, ignore_index=True).to_json(
                    class_metrics_path
                )
