import os
import re
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

BASE_RESULTS = "/projectnb/vkolagrp/projects/adrd_foundation_model/results"
DATASET_PATHS = {
    "NIFD": Path(f"{BASE_RESULTS}/NIFD/test_etpr"),
    "NACC": Path(f"{BASE_RESULTS}/NACC/test_etpr"),
    "ADNI": Path(f"{BASE_RESULTS}/ADNI/test_etpr"),
    "PPMI": Path(f"{BASE_RESULTS}/PPMI/test_etpr"),
    "BrainLat": Path(f"{BASE_RESULTS}/brainlat/test_etpr"),
}

DATASET_ORDER = ["NACC", "NIFD", "PPMI", "ADNI", "BrainLat"]
MODEL_ORDER = ("Q3B", "LUNAR", "Q7B")

FONTSIZE = 7
FIGSIZE_PR = (6, 3.2)   # macro precision + recall
FIGSIZE_F1 = (3.2, 3.2)  # macro F1 only

N_BOOT = 1000
N_PERM = 10000
N_JOBS = 40

OUTPUT_FIG_DIR = "../figures"
OUTPUT_FIGNAME_PR = "fig3_macro_circular_bar_plot_sidebyside_id_level"
OUTPUT_FIGNAME_F1 = "fig3_macro_circular_bar_plot_sidebyside_id_level_f1"

SAVE_LATEX = True
LATEX_OUTPUT_PATH = "../tables/fig3_etpr_macro_table.tex"


# Data loading

def option_string_to_dict(options: str) -> dict[str, str]:
    pattern = r"([A-Z])\. ([^\n]+)"
    matches = re.findall(pattern, options)
    return {key: value for key, value in matches}


def load_answers(dir_path: Path, dataset_name: str) -> pd.DataFrame:
    fpaths = list(dir_path.rglob("*.parquet"))
    dfs: list[pd.DataFrame] = []

    cols_to_read = ["ID", "ground_truth", "prediction",
                    "ground_truth_text", "options"]

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

    group_cols = ["dataset", "benchmark", "model",
                  "ground_truth_text", "prediction_text"]
    for col in group_cols:
        df_all[col] = pd.Categorical(df_all[col])

    df_all = df_all[df_all["model"].isin(list(MODEL_MAP.keys()))]
    return df_all


def build_results_df() -> pd.DataFrame:
    adni_res = load_answers(DATASET_PATHS["ADNI"], dataset_name="ADNI")
    brainlat_res = load_answers(DATASET_PATHS["BrainLat"], dataset_name="BrainLat")
    nifd_res = load_answers(DATASET_PATHS["NIFD"], dataset_name="NIFD")
    nacc_res = load_answers(DATASET_PATHS["NACC"], dataset_name="NACC")
    ppmi_res = load_answers(DATASET_PATHS["PPMI"], dataset_name="PPMI")

    results_df = pd.concat(
        [adni_res, brainlat_res, nifd_res, nacc_res, ppmi_res],
        ignore_index=True,
    )
    results_df["dataset_raw"] = results_df["dataset"]
    results_df["trial"] = results_df.index
    return results_df


def check_alignment(df: pd.DataFrame) -> bool:
    issues_found = False
    print("=" * 80)
    print("CHECKING DATA ALIGNMENT FOR PERMUTATION TESTS")
    print("=" * 80)

    for (ds, bench), group in df.groupby(["dataset", "benchmark"], observed=True):
        models = sorted(group["model"].unique())
        print(f"\n📊 {ds} / {bench}")
        for m1, m2 in combinations(models, 2):
            d1 = group[group["model"] == m1].copy()
            d2 = group[group["model"] == m2].copy()
            d1_sorted = d1.sort_values("ID").reset_index(drop=True)
            d2_sorted = d2.sort_values("ID").reset_index(drop=True)
            ids_match = d1_sorted["ID"].equals(d2_sorted["ID"])
            ids1, ids2 = set(d1["ID"]), set(d2["ID"])

            if ids_match:
                print(f"   ✅ {m1} vs {m2}: {len(d1)} samples, perfectly aligned")
            else:
                issues_found = True
                print(f"   ❌ {m1} vs {m2}: MISALIGNED! ({m1}: {len(d1)}, {m2}: {len(d2)})")
                if ids1 == ids2 and not ids_match:
                    print("      ⚠ SAME IDs but DIFFERENT ORDER")

    print("\n" + "=" * 80)
    print("❌ ALIGNMENT ISSUES FOUND" if issues_found else "✅ ALL DATA PROPERLY ALIGNED")
    print("=" * 80)
    return issues_found


# Metric utilities

def _vectorized_metric_calc(y_true, y_pred, label_code, metric: str):
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


def _vectorized_macro_metric_calc(y_true, y_pred, label_codes, metric: str):
    """
    Macro average over labels in `label_codes`.
    For bootstrap: returns shape (n_boot,)
    For non-bootstrap: scalar if inputs are 1D.
    """
    label_codes = np.asarray(label_codes, dtype=np.int16)
    if label_codes.size == 0:
        return np.zeros(y_true.shape[0], dtype=float) if y_true.ndim > 1 else 0.0

    acc = None
    used = []
    for lbl in label_codes:
        v = _vectorized_metric_calc(y_true, y_pred, int(lbl), metric)
        used.append(lbl)
        if acc is None:
            acc = v.astype(float, copy=False)
        else:
            acc = acc + v
    return acc / float(len(used))


# Bootstrap (macro metrics)

def _single_bootstrap_task_macro(group_info, y_true_b, y_pred_b, label_codes, m_type):
    boot_values = _vectorized_macro_metric_calc(y_true_b, y_pred_b, label_codes, m_type)
    low, med, high = np.quantile(boot_values, [0.025, 0.5, 0.975])
    return {
        **group_info,
        "metric": f"macro_{m_type}",
        "mean": float(np.mean(boot_values)),
        "median": float(med),
        "low": float(low),
        "high": float(high),
    }


def optimized_bootstrap_parallel_macro(
    df: pd.DataFrame,
    n_boot: int = N_BOOT,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    df_grouped = df[
        [
            "ID",
            "trial",
            "dataset_raw",
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

    print(f"Preparing Bootstrap data for {len(groups) * 3} macro tasks...")

    for g_id, group in groups:
        group = group.sort_values(["ID", "trial"]).reset_index(drop=True)

        gt_cats = group["ground_truth_text"].astype("category").cat.categories.tolist()
        pred_cats = group["prediction_text"].astype("category").cat.categories.tolist()
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

        label_codes_group = np.unique(y_true)
        label_codes_group = label_codes_group[label_codes_group != invalid_code]
        if label_codes_group.size == 0:
            continue

        group_seed = int(main_rng.integers(0, 2**32))
        rng = np.random.default_rng(group_seed)
        indices = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))

        y_true_b = y_true[indices]
        y_pred_b = y_pred[indices]

        group_info = {"dataset": g_id[0], "benchmark": g_id[1], "model": g_id[2]}

        for m_type in ["precision", "recall", "f1"]:
            all_tasks.append(
                (group_info, y_true_b, y_pred_b, label_codes_group, m_type)
            )

    print(f"Executing Bootstrap on {len(all_tasks)} tasks across {n_jobs} cores...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_bootstrap_task_macro)(*t) for t in all_tasks
    )

    return pd.DataFrame(results)


# Permutation tests (macro metrics)

def _permutation_worker_macro(task, n_perms: int, seed: int):
    rng = np.random.default_rng(seed)
    yt, yp1, yp2 = task["yt"], task["yp1"], task["yp2"]
    id_array = task["id_array"]
    label_codes, m_type = task["label_codes"], task["metric"]

    obs1 = _vectorized_macro_metric_calc(yt, yp1, label_codes, m_type)
    obs2 = _vectorized_macro_metric_calc(yt, yp2, label_codes, m_type)
    obs_diff = obs1 - obs2

    unique_ids, id_indices = np.unique(id_array, return_inverse=True)
    n_ids = len(unique_ids)
    swap_ids = rng.integers(0, 2, size=(n_perms, n_ids), dtype=bool)
    swap = swap_ids[:, id_indices]

    p1 = np.where(swap, yp2, yp1)
    p2 = np.where(swap, yp1, yp2)

    null1 = _vectorized_macro_metric_calc(yt, p1, label_codes, m_type)
    null2 = _vectorized_macro_metric_calc(yt, p2, label_codes, m_type)

    p_val = float(np.mean(np.abs(null1 - null2) >= np.abs(obs_diff)))

    out = {
        k: v
        for k, v in task.items()
        if k not in ["yt", "yp1", "yp2", "label_codes", "id_array"]
    }
    out.update(
        {
            "p_value": p_val,
            "observed_diff": float(obs_diff),
            "metric": f"macro_{m_type}",
            "obs1": obs1,
            "obs2": obs2,
        }
    )
    return out


def compute_pairwise_comparisons_macro(
    df: pd.DataFrame,
    n_permutations: int = N_PERM,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
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

    tasks: list[dict] = []

    for (ds, bench), group in df_grouped.groupby(["dataset", "benchmark"], observed=True):
        gt_cats = group["ground_truth_text"].astype("category").cat.categories.tolist()
        pred_cats = group["prediction_text"].astype("category").cat.categories.tolist()
        all_cats = sorted(set(gt_cats + pred_cats))
        if "invalid" not in all_cats:
            all_cats.append("invalid")

        cat_dtype = pd.CategoricalDtype(categories=all_cats, ordered=False)
        invalid_code = all_cats.index("invalid")

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
            
            assert np.array_equal(d1["y_true"].to_numpy(), d2["y_true"].to_numpy())
            yt = d1["y_true"].to_numpy()
            yp1 = d1["y_pred"].to_numpy()
            yp2 = d2["y_pred"].to_numpy()
            id_array = d1["ID"].to_numpy()

            label_codes = np.unique(yt)
            label_codes = label_codes[label_codes != invalid_code]
            if label_codes.size == 0:
                continue

            for m_type in ["precision", "recall", "f1"]:
                tasks.append(
                    {
                        "dataset": ds,
                        "benchmark": bench,
                        "model1": m1,
                        "model2": m2,
                        "metric": m_type,
                        "label_codes": label_codes,
                        "yt": yt,
                        "yp1": yp1,
                        "yp2": yp2,
                        "id_array": id_array,
                    }
                )

    print(f"Executing Permutation Tests on {len(tasks)} macro tasks...")
    seeds = np.random.default_rng(seed).integers(0, 2**32, size=len(tasks))
    results = Parallel(n_jobs=n_jobs)(
        delayed(_permutation_worker_macro)(tasks[i], n_permutations, int(seeds[i]))
        for i in range(len(tasks))
    )

    res_df = pd.DataFrame(results)

    res_df["p_value_bh"] = np.nan
    for keys, sub_idx in res_df.groupby(["dataset", "metric"]).groups.items():
        p = res_df.loc[sub_idx, "p_value"].to_numpy()
        _, p_bh, _, _ = multipletests(p, method="fdr_bh")
        res_df.loc[sub_idx, "p_value_bh"] = p_bh

    res_df["Significant_bh"] = res_df["p_value_bh"] < 0.05
    return res_df


# LaTeX table (macro)

def generate_latex_table_macro(
    all_metric: pd.DataFrame,
    model_map: dict[str, str],
    dataset_map: dict[str, str] | None = None,
) -> str:
    df = all_metric.copy()

    df["model"] = df["model"].map(model_map).fillna(df["model"])
    if dataset_map is not None and "dataset" in df.columns:
        df["dataset"] = df["dataset"].map(dataset_map).fillna(df["dataset"])

    idx_cols = ["dataset", "model"]

    stats = df.pivot_table(
        index=idx_cols,
        columns="metric",
        values=["median", "low", "high"],
    ).reset_index()

    stats.columns = [
        f"{col[1]}_{col[0]}" if col[1] else col[0] for col in stats.columns
    ]

    model_order = list(model_map.values())
    stats["model"] = pd.Categorical(
        stats["model"], categories=model_order, ordered=True
    )

    stats = stats.sort_values(["dataset", "model"]).reset_index(drop=True)

    metrics = ["macro_precision", "macro_recall", "macro_f1"]
    group_cols = ["dataset"]

    best_lookup: set[tuple[int, str]] = set()
    second_best_lookup: set[tuple[int, str]] = set()

    for m in metrics:
        col = f"{m}_median"
        ranks = stats.groupby(group_cols, observed=False)[col].rank(
            method="first", ascending=False
        )
        best_idx = stats.index[(ranks == 1) & stats[col].notna()]
        for i in best_idx:
            best_lookup.add((i, m))
        second_idx = stats.index[(ranks == 2) & stats[col].notna()]
        for i in second_idx:
            second_best_lookup.add((i, m))

    headers = ["Dataset", "Model", "Macro Precision", "Macro Recall", "Macro F1"]
    colspec = "l" + "lccc"

    latex_lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline",
    ]

    prev_group = None
    for i, row in stats.iterrows():
        this_group = tuple(row[c] for c in group_cols)
        if prev_group is not None and this_group != prev_group:
            latex_lines.append("\\hline")

        ds_disp = row["dataset"] if prev_group is None or row["dataset"] != prev_group[0] else ""
        parts = [ds_disp, str(row["model"])]

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

        row_str = " & ".join(parts) + " & " + " & ".join(formatted_metrics) + " \\\\"
        latex_lines.append(row_str)
        prev_group = this_group

    latex_lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(latex_lines)


# Circular bar plot

table_positions = {
    "macro_precision": {
        "NIFD": (-0.25, 1.07),
        "NACC": (-0.5, 0.9),
        "BrainLat": (0.34, 0.98),
        "ADNI": (0.3, 1.2),
        "PPMI": (-0.15, 0.92),
        "Other": (0.15, 1.08),
    },
    "macro_recall": {
        "NIFD": (-0.25, 0.9),
        "NACC": (-0.5, 0.9),
        "BrainLat": (0.34, 0.98),
        "ADNI": (0.3, 1.1),
        "PPMI": (-0.15, 0.92),
        "Other": (0.15, 1.08),
    },
    "macro_f1": {
        "NIFD": (-0.25, 0.9),
        "NACC": (-0.5, 0.9),
        "BrainLat": (0.34, 0.98),
        "ADNI": (0.3, 1.1),
        "PPMI": (-0.15, 0.92),
        "Other": (0.15, 1.08),
    },
}

value_label_offsets = {
    "NIFD": {"r_offset": 0.06, "theta_offset": 0.0},
    "NACC": {"r_offset": 0.12, "theta_offset": -0.2},
    "BrainLat": {"r_offset": 0.15, "theta_offset": 0.02},
    "ADNI": {"r_offset": 0.14, "theta_offset": -0.03},
    "PPMI": {"r_offset": 0.14, "theta_offset": 0.1},
}


def get_significance_marker(p_value):
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


def plot_macro_circular(
    all_metrics,
    pairwise_pvalues,
    model_map,
    output_dir=".",
    show_all_comparisons=False,
    p_threshold=0.05,
    model_order=None,
    dataset_order=None,
    filename="fig3_macro_circular_bar_plot_sidebyside",
    metrics=("macro_precision", "macro_recall"),
    figsize=(2.3, 2.3),
    fontsize=None,
):
    if model_order is None:
        model_order = MODEL_ORDER
    if dataset_order is None:
        dataset_order = DATASET_ORDER
    if fontsize is None:
        fontsize = FONTSIZE

    metrics = list(metrics)
    metric_names = {
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro F1",
    }

    df = all_metrics.copy()
    df["model_abbrev"] = df["model"].map(model_map).fillna(df["model"])

    pv = pairwise_pvalues.copy()
    pv["model1_abbrev"] = pv["model1"].map(model_map).fillna(pv["model1"])
    pv["model2_abbrev"] = pv["model2"].map(model_map).fillna(pv["model2"])

    df = df[df["metric"].isin(metrics)].copy()
    pv = pv[pv["metric"].isin(metrics)].copy()

    has_benchmark = ("benchmark" in df.columns) and ("benchmark" in pv.columns)

    datasets_present = sorted(df["dataset"].unique())
    datasets = [d for d in dataset_order if d in datasets_present] or datasets_present

    models = list(model_order)
    palette = dict(
        zip(models, sns.color_palette("colorblind", n_colors=len(models)))
    )

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=figsize,
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )

    for metric_idx, metric in enumerate(metrics):
        ax = axes[0, metric_idx]
        n_groups = len(datasets)
        group_width = 2 * np.pi / n_groups
        dataset_gap = group_width * 0.1
        bar_width = (group_width - dataset_gap) / len(models) * 0.95

        dataset_centers = {}

        for group_idx, dataset in enumerate(datasets):
            panel = df[
                (df["dataset"] == dataset) & (df["metric"] == metric)
            ].copy()

            if has_benchmark and "benchmark" in panel.columns and panel["benchmark"].nunique() > 1:
                first_bench = sorted(panel["benchmark"].unique())[0]
                panel = panel[panel["benchmark"] == first_bench]

            base_angle = group_idx * group_width + dataset_gap / 2

            for model_idx, model in enumerate(models):
                mdata = panel[panel["model_abbrev"] == model]
                if len(mdata) == 0:
                    val = low = high = 0.0
                else:
                    val = float(mdata["median"].iloc[0])
                    low = float(mdata["low"].iloc[0])
                    high = float(mdata["high"].iloc[0])
                err_low = val - low
                err_high = high - val

                theta = base_angle + model_idx * bar_width + bar_width / 2

                label = (
                    model if (metric_idx == 0 and group_idx == 0) else ""
                )
                ax.bar(
                    theta,
                    val,
                    width=bar_width * 0.98,
                    bottom=0.0,
                    color=palette[model],
                    alpha=0.8,
                    label=label,
                    edgecolor="white",
                    linewidth=0.5,
                )

                ax.errorbar(
                    theta,
                    val,
                    yerr=[[err_low], [err_high]],
                    fmt="none",
                    ecolor="black",
                    capsize=3,
                    capthick=0.6,
                    elinewidth=0.6,
                )

                if val > 0.1:
                    offsets = value_label_offsets.get(
                        dataset, {"r_offset": 0.08, "theta_offset": 0.0}
                    )
                    ax.text(
                        theta + offsets["theta_offset"],
                        val + err_high + offsets["r_offset"],
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=fontsize,
                        rotation=0,
                        fontweight="bold",
                    )

            group_center = base_angle + (len(models) * bar_width) / 2
            dataset_centers[dataset] = group_center

        for dataset, theta in dataset_centers.items():
            ax.text(
                theta,
                1.25,
                dataset,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="lightgray",
                    alpha=0.7,
                ),
            )

            psubset = pv[
                (pv["dataset"] == dataset) & (pv["metric"] == metric)
            ].copy()
            if has_benchmark and "benchmark" in psubset.columns:
                panel = df[
                    (df["dataset"] == dataset) & (df["metric"] == metric)
                ]
                if panel["benchmark"].nunique() == 1:
                    b = panel["benchmark"].iloc[0]
                    psubset = psubset[psubset["benchmark"] == b]

            if len(psubset) > 0:
                table_lines = []
                for _, prow in psubset.iterrows():
                    m1 = prow["model1_abbrev"]
                    m2 = prow["model2_abbrev"]
                    pval = float(prow["p_value_bh"])
                    sig_marker = get_significance_marker(pval)
                    if (m1 in models) and (m2 in models):
                        table_lines.append(f"{m1}-{m2}: {sig_marker}")
                if table_lines:
                    theta_offset, radius = table_positions.get(metric, {}).get(
                        dataset, (0.15, 1.08)
                    )
                    table_text = "\n".join(table_lines[:6])
                    ax.text(
                        theta + theta_offset,
                        radius,
                        table_text,
                        ha="center",
                        va="bottom",
                        fontsize=fontsize - 2,
                        fontfamily="monospace",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor="gray",
                            alpha=0.95,
                            linewidth=0.8,
                        ),
                    )

        ax.set_ylim(0, 1.28)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(
            ["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=fontsize - 2
        )
        ax.set_xticks([])
        ax.grid(True, alpha=0.35, axis="y")
        ax.grid(False, axis="x")

        for i in range(len(datasets)):
            sep_theta = i * group_width
            ax.plot(
                [sep_theta, sep_theta],
                [0, 1.28],
                "k--",
                alpha=0.3,
                linewidth=1.5,
            )

        if "macro_f1" not in metrics:
            ax.set_title(
                metric_names.get(metric, metric),
                fontsize=fontsize,
                fontweight="bold",
                pad=20,
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=palette[m], alpha=0.8, label=m)
        for m in models
    ]
    fig.legend(
        handles=handles,
        title="Model",
        title_fontsize=fontsize,
        loc="lower center",
        ncol=len(models),
        bbox_to_anchor=(0.5, -0.03),
        frameon=True,
        fontsize=fontsize,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{filename}.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")


# Main

def main() -> None:
    print("Loading ETPR results...")
    results_df = build_results_df()
    check_alignment(results_df)

    print("Running bootstrap...")
    all_macro_metrics = optimized_bootstrap_parallel_macro(
        results_df, n_boot=N_BOOT, seed=42, n_jobs=N_JOBS
    ).sort_values(["dataset", "benchmark", "model", "metric"])

    print("Running permutation tests...")
    pairwise_macro = compute_pairwise_comparisons_macro(
        results_df, n_permutations=N_PERM, seed=42, n_jobs=N_JOBS
    ).sort_values(["dataset", "benchmark", "metric", "model1", "model2"])

    if SAVE_LATEX:
        print("Generating LaTeX table...")
        latex_output = generate_latex_table_macro(
            all_macro_metrics, model_map=MODEL_MAP
        )
        os.makedirs(os.path.dirname(LATEX_OUTPUT_PATH) or ".", exist_ok=True)
        with open(LATEX_OUTPUT_PATH, "w") as f:
            f.write(latex_output)
        print(f"Saved LaTeX table to {LATEX_OUTPUT_PATH}")

    print("Plotting macro precision + recall...")
    plot_macro_circular(
        all_metrics=all_macro_metrics,
        pairwise_pvalues=pairwise_macro,
        model_map=MODEL_MAP,
        output_dir=OUTPUT_FIG_DIR,
        show_all_comparisons=True,
        p_threshold=1,
        dataset_order=DATASET_ORDER,
        filename=OUTPUT_FIGNAME_PR,
        metrics=("macro_precision", "macro_recall"),
        figsize=FIGSIZE_PR,
    )

    print("Plotting macro F1...")
    plot_macro_circular(
        all_metrics=all_macro_metrics,
        pairwise_pvalues=pairwise_macro,
        model_map=MODEL_MAP,
        output_dir=OUTPUT_FIG_DIR,
        show_all_comparisons=True,
        p_threshold=1,
        dataset_order=DATASET_ORDER,
        filename=OUTPUT_FIGNAME_F1,
        metrics=("macro_f1",),
        figsize=FIGSIZE_F1,
    )


if __name__ == "__main__":
    main()