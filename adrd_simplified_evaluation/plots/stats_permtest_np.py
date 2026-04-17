import os
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


plt.rcParams["font.family"] = "Arial"
sns.set_style("whitegrid")


# Configuration

MODEL_MAP = {
    "Qwen2.5-3B-Instruct": "Q3B",
    # "NACC-3B": "LUNAR-OS-SCe",
    # "NACC-3B-SCE": "LUNAR-OS",
    # "NACC-3B-OS": "LUNAR-SCe",
    "NACC-3B-OS-SCE": "LUNAR",
    "Qwen2.5-7B-Instruct": "Q7B",
}

CLASS_MAP = {
    "Alzheimer's disease pathology (AD)": "NP-AD",
    "Frontotemporal Lobar Degeneration with tau pathology or TDP-43 pathology (FTLD)": "NP-FTLD",
    "Lewy body pathology (LBD)": "NP-LBD",
    "No listed option is correct": "None",
}

# CLASS_MAP = {
#     "Alzheimer's disease pathology (AD) and Frontotemporal Lobar Degeneration with tau pathology or TDP-43 pathology (FTLD)": 'AD and FTLD',
#     "Alzheimer's disease pathology (AD) and Lewy body pathology (LBD)": 'AD and LBD',
#     "Alzheimer's disease pathology (AD), Lewy body pathology (LBD) and Frontotemporal Lobar Degeneration with tau pathology or TDP-43 pathology (FTLD)": 'AD, LBD and FTLD',
#     "Lewy body pathology (LBD) and Frontotemporal Lobar Degeneration with tau pathology or TDP-43 pathology (FTLD)": "LBD and FTLD",
#     "No listed option is correct": "None"
# }

DATASET_PATHS = {
    "NACC": Path(
        "/projectnb/vkolagrp/projects/adrd_foundation_model/results/NACC/test_np_one"
    ),
}

MODEL_ORDER_FOR_PLOTS = [
    "Q3B",
    # "LUNAR-OS-SCe",
    # "LUNAR-SCe",
    # "LUNAR-OS",
    "LUNAR",
    "Q7B",
]

CLASS_ORDER = ["None", "NP-AD", "NP-LBD", "NP-FTLD"]
# CLASS_ORDER = ['None', 'AD and LBD', 'AD and FTLD', 'LBD and FTLD', 'AD, LBD and FTLD']

DATASET_ORDER = ["NACC"]

FONTSIZE = 7

FIGSIZE_PR = (4.0, 4.0)   # precision/recall
FIGSIZE_F1 = (3.0, 2.3)   # F1-only

N_BOOT = 1000
N_PERM = 10000
N_JOBS_BOOT = 20
N_JOBS_PERM = 20

OUTPUT_FIG_DIR = "../figures"
OUTPUT_FIGNAME_PR = "fig3_np_one_prec_rec"
OUTPUT_FIGNAME_F1 = "fig3_np_one_f1"

SAVE_LATEX = True
LATEX_OUTPUT_PATH = "../figures/fig3_np_one_table.tex"


# Data loading

def option_string_to_dict(options: str) -> dict[str, str]:
    """
    Parse multiple-choice options string like 'A. ...\\nB. ...' into a dict mapping letter -> text.
    """
    import re

    pattern = r"([A-Z])\. ([^\n]+)"
    matches = re.findall(pattern, options)
    return {key: value for key, value in matches}


def load_answers(dir_path: Path, dataset_name: str) -> pd.DataFrame:
    """
    Load all NP-one parquet files from a dataset directory into a tall DataFrame.
    """
    fpaths = list(dir_path.rglob("*.parquet"))

    dfs: list[pd.DataFrame] = []
    cols_to_read = [
        "ID",
        "ground_truth",
        "prediction",
        "ground_truth_text",
        "options",
    ]

    for fpath in tqdm(fpaths, desc=f"Loading {dataset_name}"):
        model = fpath.parent.name.split("-", 3)[-1]
        benchmark = fpath.parent.parent.name.split("_", 1)[-1].upper()

        df = pd.read_parquet(fpath, columns=cols_to_read)

        df = df.assign(model=model, benchmark=benchmark)
        df["correct"] = (df["ground_truth"] == df["prediction"]).astype(int)

        df["prediction_text"] = df.apply(
            lambda row: option_string_to_dict(row["options"]).get(
                row["prediction"], "invalid"
            ),
            axis=1,
        )

        dfs.append(df)

    if not dfs:
        return pd.DataFrame(
            columns=cols_to_read
            + ["model", "benchmark", "correct", "prediction_text", "dataset"]
        )

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["dataset"] = dataset_name

    group_cols = [
        "dataset",
        "benchmark",
        "model",
        "ground_truth_text",
        "prediction_text",
    ]
    for col in group_cols:
        df_all[col] = pd.Categorical(df_all[col])

    df_all = df_all[df_all["model"].isin(list(MODEL_MAP.keys()))]
    return df_all


def build_results_df() -> pd.DataFrame:
    """
    Load all datasets, concatenate, and add trial index.
    """
    nacc_res = load_answers(DATASET_PATHS["NACC"], dataset_name="NACC")

    results_df = nacc_res.copy()
    results_df["trial"] = results_df.index
    return results_df


# Metric utilities

def _vectorized_metric_calc(y_true, y_pred, label_code, metric: str):
    """Core vectorized math for both bootstrap and permutation."""
    tp = np.sum((y_pred == label_code) & (y_true == label_code), axis=-1)

    if metric == "precision":
        den = np.sum(y_pred == label_code, axis=-1)
    elif metric == "recall":
        den = np.sum(y_true == label_code, axis=-1)
    elif metric == "f1":
        precision_den = np.sum(y_pred == label_code, axis=-1)
        recall_den = np.sum(y_true == label_code, axis=-1)
        precision = np.divide(
            tp,
            precision_den,
            out=np.zeros_like(tp, dtype=float),
            where=precision_den != 0,
        )
        recall = np.divide(
            tp,
            recall_den,
            out=np.zeros_like(tp, dtype=float),
            where=recall_den != 0,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros_like(tp, dtype=float),
                where=(precision + recall) != 0,
            )
        return f1
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return np.divide(
        tp,
        den,
        out=np.zeros_like(tp, dtype=float),
        where=den != 0,
    )


# Bootstrap

def _single_bootstrap_task(group_info, y_true_b, y_pred_b, lbl_code, m_type):
    """Worker for individual bootstrap metric tasks."""
    boot_values = _vectorized_metric_calc(y_true_b, y_pred_b, lbl_code, m_type)
    low, med, high = np.percentile(boot_values, [2.5, 50, 97.5])
    return {
        **group_info,
        "class_code": lbl_code,
        "metric": m_type,
        "mean": np.mean(boot_values),
        "median": med,
        "low": low,
        "high": high,
    }


def optimized_bootstrap_parallel(
    df: pd.DataFrame,
    n_boot: int = N_BOOT,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Bootstrap CIs for precision/recall/F1 per (dataset, benchmark, model, class).
    """
    df_grouped = df[
        [
            "ID",
            "trial",
            "dataset",
            "benchmark",
            "model",
            "ground_truth_text",
            "prediction_text",
        ]
    ].copy()

    groups = list(df_grouped.groupby(["dataset", "benchmark", "model"], observed=True))
    main_rng = np.random.default_rng(seed)
    all_tasks = []
    all_int_to_label: dict[tuple[str, str, str], dict[int, str]] = {}

    print(f"Preparing Bootstrap data for {len(groups)} groups...")

    for g_id, group in groups:
        group = group.sort_values(["ID", "trial"]).reset_index(drop=True)

        gt_cats = (
            group["ground_truth_text"].astype("category").cat.categories.tolist()
        )
        pred_cats = (
            group["prediction_text"].astype("category").cat.categories.tolist()
        )
        all_cats = sorted(set(gt_cats + pred_cats))
        if "invalid" not in all_cats:
            all_cats.append("invalid")

        cat_dtype = pd.CategoricalDtype(categories=all_cats, ordered=False)
        invalid_code = all_cats.index("invalid")

        y_true = (
            group["ground_truth_text"]
            .astype(cat_dtype)
            .cat.codes.astype(np.int16)
            .to_numpy()
        )
        y_pred = (
            group["prediction_text"]
            .astype(cat_dtype)
            .cat.codes.astype(np.int16)
            .to_numpy()
        )

        labels_to_process = [i for i in range(len(all_cats)) if i != invalid_code]
        group_key = (g_id[0], g_id[1], g_id[2])
        all_int_to_label[group_key] = {i: cat for i, cat in enumerate(all_cats)}

        group_seed = int(main_rng.integers(0, 2**32))
        rng = np.random.default_rng(group_seed)

        indices = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
        y_true_b = y_true[indices]
        y_pred_b = y_pred[indices]

        for lbl_code in labels_to_process:
            for m_type in ["precision", "recall", "f1"]:
                all_tasks.append(
                    (
                        {
                            "dataset": g_id[0],
                            "benchmark": g_id[1],
                            "model": g_id[2],
                        },
                        y_true_b,
                        y_pred_b,
                        lbl_code,
                        m_type,
                    )
                )

    print(f"Executing Bootstrap on {len(all_tasks)} tasks across {n_jobs} cores...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_bootstrap_task)(*t) for t in all_tasks
    )

    res_df = pd.DataFrame(results)

    res_df["class"] = res_df.apply(
        lambda row: all_int_to_label[
            (row["dataset"], row["benchmark"], row["model"])
        ][row["class_code"]],
        axis=1,
    )

    return res_df.drop(columns=["class_code"])


# Permutation tests

def _permutation_worker(task, n_perms: int, seed: int):
    """Worker for individual permutation tasks."""
    rng = np.random.default_rng(seed)
    id_array = task["id_array"]
    yt, yp1, yp2, lbl_code, m_type = (
        task["yt"],
        task["yp1"],
        task["yp2"],
        task["class_code"],
        task["metric"],
    )

    obs_diff = _vectorized_metric_calc(
        yt, yp1, lbl_code, m_type
    ) - _vectorized_metric_calc(yt, yp2, lbl_code, m_type)

    unique_ids, id_indices = np.unique(id_array, return_inverse=True)
    n_ids = len(unique_ids)

    swap_ids = rng.integers(0, 2, size=(n_perms, n_ids), dtype=bool)
    swap = swap_ids[:, id_indices]

    p1 = np.where(swap[:, None], yp2, yp1)
    p2 = np.where(swap[:, None], yp1, yp2)

    null1 = _vectorized_metric_calc(yt, p1, lbl_code, m_type)
    null2 = _vectorized_metric_calc(yt, p2, lbl_code, m_type)

    p_val = np.mean(np.abs(null1 - null2) >= np.abs(obs_diff))
    return {
        k: v for k, v in task.items() if k not in ["yt", "yp1", "yp2"]
    } | {"p_value": p_val, "observed_diff": obs_diff}


def compute_pairwise_comparisons_optimized(
    df: pd.DataFrame,
    n_permutations: int = N_PERM,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Permutation tests on precision/recall/F1 between model pairs within each (dataset, benchmark).
    Applies BH correction within each (dataset, class, metric).
    """
    df_grouped = df[
        [
            "ID",
            "trial",
            "dataset",
            "benchmark",
            "model",
            "ground_truth_text",
            "prediction_text",
        ]
    ].copy()

    all_int_to_label: dict[tuple[str, str], dict[int, str]] = {}
    tasks: list[dict] = []

    for (ds, bench), group in df_grouped.groupby(
        ["dataset", "benchmark"], observed=True
    ):
        gt_cats = (
            group["ground_truth_text"].astype("category").cat.categories.tolist()
        )
        pred_cats = (
            group["prediction_text"].astype("category").cat.categories.tolist()
        )
        all_cats = sorted(set(gt_cats + pred_cats))
        if "invalid" not in all_cats:
            all_cats.append("invalid")

        cat_dtype = pd.CategoricalDtype(categories=all_cats, ordered=False)
        invalid_code = all_cats.index("invalid")
        labels_to_process = [i for i in range(len(all_cats)) if i != invalid_code]

        all_int_to_label[(ds, bench)] = {i: cat for i, cat in enumerate(all_cats)}

        group_int = pd.DataFrame(
            {
                "ID": group["ID"],
                "trial": group["trial"],
                "model": group["model"],
                "y_true": group["ground_truth_text"]
                .astype(cat_dtype)
                .cat.codes.astype(np.int16),
                "y_pred": group["prediction_text"]
                .astype(cat_dtype)
                .cat.codes.astype(np.int16),
            }
        )

        models = sorted(group_int["model"].unique())

        for m1, m2 in combinations(models, 2):
            d1 = (
                group_int[group_int["model"] == m1]
                .sort_values(["ID", "trial"])
                .reset_index(drop=True)
            )
            d2 = (
                group_int[group_int["model"] == m2]
                .sort_values(["ID", "trial"])
                .reset_index(drop=True)
            )
            if len(d1) != len(d2):
                continue

            yt = d1["y_true"].to_numpy()
            yp1 = d1["y_pred"].to_numpy()
            yp2 = d2["y_pred"].to_numpy()
            id_array = d1["ID"].to_numpy()

            for lbl_code in labels_to_process:
                for m_type in ["precision", "recall", "f1"]:
                    tasks.append(
                        {
                            "dataset": ds,
                            "benchmark": bench,
                            "model1": m1,
                            "model2": m2,
                            "class_code": lbl_code,
                            "metric": m_type,
                            "yt": yt,
                            "yp1": yp1,
                            "yp2": yp2,
                            "id_array": id_array,
                        }
                    )

    print(f"Executing Permutation Tests on {len(tasks)} tasks...")
    seeds = np.random.default_rng(seed).integers(0, 2**32, size=len(tasks))
    results = Parallel(n_jobs=n_jobs)(
        delayed(_permutation_worker)(tasks[i], n_permutations, int(seeds[i]))
        for i in range(len(tasks))
    )

    res_df = pd.DataFrame(results)

    res_df["class"] = res_df.apply(
        lambda row: all_int_to_label[(row["dataset"], row["benchmark"])][
            row["class_code"]
        ],
        axis=1,
    )

    res_df["p_value_bh"] = np.nan
    for keys, sub_idx in res_df.groupby(["dataset", "class", "metric"]).groups.items():
        print(keys, len(sub_idx))
        p = res_df.loc[sub_idx, "p_value"].to_numpy()
        _, p_bh, _, _ = multipletests(p, method="fdr_bh")
        res_df.loc[sub_idx, "p_value_bh"] = p_bh

    res_df["Significant_bh"] = res_df["p_value_bh"] < 0.05
    return res_df.drop(columns=["class_code"])


# LaTeX table

def generate_latex_table(
    all_metric: pd.DataFrame,
    model_map: dict[str, str],
    class_map: dict[str, str],
) -> str:
    df = all_metric.copy()

    df["model"] = df["model"].map(model_map).fillna(df["model"])
    df["class"] = df["class"].map(class_map).fillna(df["class"])

    stats = df.pivot_table(
        index=["class", "model"],
        columns="metric",
        values=["median", "low", "high"],
    ).reset_index()

    stats.columns = [
        f"{col[1]}_{col[0]}" if col[1] else col[0] for col in stats.columns
    ]

    model_order = list(model_map.values())

    stats["class"] = pd.Categorical(
        stats["class"], categories=CLASS_ORDER, ordered=True
    )
    stats["model"] = pd.Categorical(
        stats["model"], categories=model_order, ordered=True
    )

    stats = stats.sort_values(["class", "model"]).reset_index(drop=True)

    metrics = ["precision", "recall", "f1"]
    best_lookup: set[tuple[int, str]] = set()
    second_best_lookup: set[tuple[int, str]] = set()

    for m in metrics:
        col = f"{m}_median"
        ranks = stats.groupby(
            ["class"], observed=False
        )[col].rank(method="first", ascending=False)

        best_idx = stats.index[(ranks == 1) & stats[col].notna()]
        for i in best_idx:
            best_lookup.add((i, m))

        second_idx = stats.index[(ranks == 2) & stats[col].notna()]
        for i in second_idx:
            second_best_lookup.add((i, m))

    headers = ["Class", "Model", "Precision", "Recall", "F1-score"]
    latex_lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llccc}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline",
    ]

    prev_cl = None

    for i, row in stats.iterrows():
        if prev_cl is not None and row["class"] != prev_cl:
            latex_lines.append("\\hline")

        cl_disp = row["class"] if row["class"] != prev_cl else ""

        formatted_metrics = []
        for m in metrics:
            val_str = (
                f"{row[f'{m}_median']:.3f} "
                f"[{row[f'{m}_low']:.3f}, {row[f'{m}_high']:.3f}]"
            )
            if (i, m) in best_lookup:
                val_str = f"\\textbf{{{val_str}}}"
            if (i, m) in second_best_lookup:
                val_str = f"\\underline{{{val_str}}}"
            formatted_metrics.append(val_str)

        row_str = (
            f"{cl_disp} & {row['model']} & "
            + " & ".join(formatted_metrics)
            + " \\\\"
        )
        latex_lines.append(row_str)

        prev_cl = row["class"]

    latex_lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(latex_lines)


# Plotting

def get_significance_marker(p_value) -> str:
    if isinstance(p_value, str):
        return p_value
    if pd.isna(p_value):
        return ""
    if p_value < 0.0001:
        return "****"
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value <= 1.0:
        return "ns"
    else:
        return str(p_value)


def plot_classwise_with_pvalues(
    all_metrics: pd.DataFrame,
    pairwise_pvalues: pd.DataFrame,
    model_map: dict[str, str],
    class_map: dict[str, str],
    metrics=("precision", "recall"),
    output_dir: str = OUTPUT_FIG_DIR,
    show_all_comparisons: bool = False,
    p_threshold: float = 0.05,
    figname: str = "fig3_np_one",
    figsize=(4.0, 4.0),
) -> None:
    """
    Plot classwise metrics with p-value annotations using a dot/errorbar style.
    """
    metrics = list(metrics)
    metric_names = {"precision": "Precision", "recall": "Recall", "f1": "F1"}

    all_metrics = all_metrics.copy()
    all_metrics["model_abbrev"] = all_metrics["model"].map(model_map)
    all_metrics["class_abbrev"] = all_metrics["class"].map(class_map).fillna(
        all_metrics["class"]
    )

    pvalues = pairwise_pvalues.copy()
    pvalues["model1_abbrev"] = pvalues["model1"].apply(
        lambda x: model_map.get(x, x)
    )
    pvalues["model2_abbrev"] = pvalues["model2"].apply(
        lambda x: model_map.get(x, x)
    )
    pvalues["class_abbrev"] = pvalues["class"].map(class_map).fillna(pvalues["class"])

    df = all_metrics[all_metrics["metric"].isin(metrics)].copy()

    models = list(model_map.values())
    palette = dict(
        zip(models, sns.color_palette("colorblind", n_colors=len(models)))
    )
    marker_styles = {m: mk for m, mk in zip(models, ["o", "s", "^", "D", "v"])}

    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    n_datasets = len(datasets)
    n_metrics = len(metrics)
    n_models = len(models)

    # Point plot layout parameters
    group_gap = 0.3       # distance between class groups
    point_offset = 0.1    # spacing between models within a group

    fig, axes = plt.subplots(
        n_metrics,
        n_datasets,
        figsize=figsize,
        squeeze=False,
    )

    for row_idx, metric in enumerate(metrics):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]

            present = df[df["dataset"] == dataset]["class_abbrev"].unique()
            dataset_classes = [c for c in CLASS_ORDER if c in present]
            n_classes = len(dataset_classes)

            group_centers = np.arange(n_classes) * group_gap

            df_subset = df[
                (df["dataset"] == dataset) & (df["metric"] == metric)
            ]

            # Light vertical separators between class groups
            for cls_idx in range(n_classes - 1):
                midpoint = (group_centers[cls_idx] + group_centers[cls_idx + 1]) / 2
                ax.plot(
                    [midpoint, midpoint], [0, 1.0],
                    color="#e0e0e0", linewidth=0.8, linestyle=":", zorder=0,
                )

            point_info = {model: {"x": [], "height": []} for model in models}

            for i, model in enumerate(models):
                df_model = df_subset[df_subset["model_abbrev"] == model]

                values, errors_low, errors_high = [], [], []
                for cls in dataset_classes:
                    cls_data = df_model[df_model["class_abbrev"] == cls]
                    if len(cls_data) > 0:
                        med = cls_data["median"].values[0]
                        low = cls_data["low"].values[0]
                        high = cls_data["high"].values[0]
                        values.append(med)
                        errors_low.append(med - low)
                        errors_high.append(high - med)
                    else:
                        values.append(0.0)
                        errors_low.append(0.0)
                        errors_high.append(0.0)

                x_pos = group_centers + (i - (n_models - 1) / 2) * point_offset
                point_info[model]["x"] = x_pos
                point_info[model]["height"] = np.array(values) + np.array(errors_high)

                label = model if (row_idx == 0 and col_idx == n_datasets - 1) else ""
                ax.errorbar(
                    x_pos,
                    values,
                    yerr=[errors_low, errors_high],
                    fmt=marker_styles[model],
                    color=palette[model],
                    markersize=2,
                    capsize=2,
                    linewidth=0.8,
                    markeredgecolor=palette[model],
                    markerfacecolor=palette[model],
                    label=label,
                    zorder=5,
                )

                for x, val, err_high in zip(x_pos, values, errors_high):
                    if val > 0:
                        ax.text(
                            x, val + err_high + 0.02,
                            f"{val:.2f}",
                            ha="center", va="bottom",
                            fontsize=FONTSIZE - 1, color="black",
                        )

            # P-value annotations
            pval_subset = pvalues[
                (pvalues["dataset"] == dataset)
                & (pvalues["metric"] == metric)
                & (pvalues["p_value_bh"] < p_threshold)
            ]

            max_point_height = max(
                (
                    np.max(point_info[m]["height"])
                    for m in models
                    if len(point_info[m]["height"]) > 0
                ),
                default=0,
            )

            y_offset = 0.15
            y_step = 0.12

            for cls_idx, cls in enumerate(dataset_classes):
                pval_cls = pval_subset[pval_subset["class_abbrev"] == cls]
                if len(pval_cls) == 0:
                    continue

                if not show_all_comparisons:
                    baseline_model = models[0]
                    comparisons_to_show = pval_cls[
                        (pval_cls["model1_abbrev"] == baseline_model)
                        | (pval_cls["model2_abbrev"] == baseline_model)
                    ]
                else:
                    comparisons_to_show = pval_cls

                comparisons_to_show = comparisons_to_show.sort_values("p_value_bh")

                for comp_idx, (_, row_pval) in enumerate(
                    comparisons_to_show.iterrows()
                ):
                    model1 = row_pval["model1_abbrev"]
                    model2 = row_pval["model2_abbrev"]
                    if model1 not in models or model2 not in models:
                        continue

                    x1 = point_info[model1]["x"][cls_idx]
                    x2 = point_info[model2]["x"][cls_idx]
                    y_bracket = max_point_height + y_offset + comp_idx * y_step

                    ax.plot(
                        [x1, x1, x2, x2],
                        [y_bracket - 0.01, y_bracket, y_bracket, y_bracket - 0.01],
                        "k-", linewidth=0.6,
                    )
                    ax.text(
                        (x1 + x2) / 2, y_bracket + 0.005,
                        get_significance_marker(row_pval["p_value_bh"]),
                        ha="center", va="bottom",
                        fontsize=FONTSIZE, fontweight="bold",
                    )

            # Formatting
            ax.set_ylabel(metric_names[metric], fontsize=FONTSIZE)
            ax.yaxis.set_label_coords(-0.1, 0.35)
            ax.set_xticks(group_centers)
            ax.set_xticklabels(
                dataset_classes, rotation=30, ha="center", fontsize=FONTSIZE
            )
            ax.set_xlim(
                group_centers[0] - group_gap * 0.55,
                group_centers[-1] + group_gap * 0.55,
            )

            max_y = 1.2
            if len(pval_subset[pval_subset["class_abbrev"].isin(dataset_classes)]) > 0:
                max_brackets = 0
                for cls in dataset_classes:
                    pval_cls = pval_subset[pval_subset["class_abbrev"] == cls]
                    if not show_all_comparisons:
                        baseline_model = models[0]
                        n_brackets = len(
                            pval_cls[
                                (pval_cls["model1_abbrev"] == baseline_model)
                                | (pval_cls["model2_abbrev"] == baseline_model)
                            ]
                        )
                    else:
                        n_brackets = len(pval_cls)
                    max_brackets = max(max_brackets, n_brackets)
                max_y = 0.9 + max_brackets * y_step
            ax.set_ylim(0, max_y)

            yticks = ax.get_yticks()
            yticks_filtered = yticks[yticks <= 1.0]
            ax.set_yticks(yticks_filtered)
            ax.set_yticklabels(
                [f"{y:.1f}" for y in yticks_filtered], fontsize=FONTSIZE
            )

            ax.grid(True, color="#e0e0e0", linewidth=0.5, linestyle=":", alpha=0.3, axis="y")
            ax.grid(False, axis="x")

            ax.spines["left"].set_bounds(0, 1.0)
            sns.despine(ax=ax, right=True, top=True)

    # Legend using Line2D markers
    handles = [
        plt.Line2D(
            [0], [0],
            marker=marker_styles[m], color="w",
            markerfacecolor=palette[m], markeredgecolor=palette[m],
            markersize=5, label=m,
        )
        for m in models
    ]
    fig.legend(
        handles=handles,
        title="Model",
        loc="upper center",
        ncol=len(models),
        bbox_to_anchor=(0.5, 0.05),
        frameon=True,
        fontsize=FONTSIZE,
        title_fontsize=FONTSIZE,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{figname}.pdf")
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    print(f"Saved {filename}")
    plt.show()


# Main

def main() -> None:
    print("Loading NP-one results...")
    results_df = build_results_df()

    print("Running bootstrap...")
    all_metrics = optimized_bootstrap_parallel(
        results_df, n_boot=N_BOOT, seed=42, n_jobs=N_JOBS_BOOT
    )

    print("Running permutation tests...")
    pairwise_comparisons = compute_pairwise_comparisons_optimized(
        results_df, n_permutations=N_PERM, seed=42, n_jobs=N_JOBS_PERM
    )

    if SAVE_LATEX:
        print("Generating LaTeX table...")
        latex_output = generate_latex_table(
            all_metrics, model_map=MODEL_MAP, class_map=CLASS_MAP
        )
        out_dir = os.path.dirname(LATEX_OUTPUT_PATH)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(LATEX_OUTPUT_PATH, "w") as f:
            f.write(latex_output)
        print(f"Saved LaTeX table to {LATEX_OUTPUT_PATH}")

    print("Plotting F1...")
    plot_classwise_with_pvalues(
        all_metrics=all_metrics,
        pairwise_pvalues=pairwise_comparisons,
        model_map=MODEL_MAP,
        class_map=CLASS_MAP,
        output_dir=OUTPUT_FIG_DIR,
        show_all_comparisons=True,
        p_threshold=1.0,
        figname=OUTPUT_FIGNAME_F1,
        metrics=["f1"],
        figsize=FIGSIZE_F1,
    )

    print("Plotting Precision and Recall...")
    plot_classwise_with_pvalues(
        all_metrics=all_metrics,
        pairwise_pvalues=pairwise_comparisons,
        model_map=MODEL_MAP,
        class_map=CLASS_MAP,
        output_dir=OUTPUT_FIG_DIR,
        show_all_comparisons=True,
        p_threshold=1.0,
        figname=OUTPUT_FIGNAME_PR,
        metrics=["precision", "recall"],
        figsize=FIGSIZE_PR,
    )


if __name__ == "__main__":
    main()