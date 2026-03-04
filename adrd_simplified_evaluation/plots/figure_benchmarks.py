import re
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = "Arial"
mpl.rcParams["hatch.linewidth"] = 0.5
sns.set_style("whitegrid")

# -----------------
# Configuration
# -----------------

BENCH_PATH = Path(
    "/projectnb/vkolagrp/projects/adrd_foundation_model/results/standard_benchmarks"
)

benchmark_name_dict = {
    "medmcqa": "MedMCQA",
    "medqa_test": "MedQA",
    "clinical_knowledge": "MMLU - clinical knowledge",
    "professional_medicine": "MMLU - professional medicine",
    "anatomy": "MMLU - anatomy",
    "medexpqa": "MedExpQA",
}

model_name_dict = {
    "Qwen2.5-3B-Instruct": "Q3B",
    "Qwen2.5-7B-Instruct": "Q7B",
    "NACC-3B": "LUNAR-OS-SCE",
    "NACC-3B-OS": "LUNAR-SCE",
    "NACC-3B-SCE": "LUNAR-OS",
    "NACC-3B-OS-SCE": "LUNAR",
    "NACC-3B-OS-SFT": "SFT",
}

model_order = [
    "Q3B",
    "SFT",
    "LUNAR-OS-SCE",
    "LUNAR-SCE",
    "LUNAR-OS",
    "LUNAR",
    "Q7B",
]

color_order = [
    "Q3B",
    "LUNAR-OS-SCE",
    "LUNAR-SCE",
    "LUNAR-OS",
    "LUNAR",
    "Q7B",
    "random",
    "SFT",
]


# -----------------
# Data loading
# -----------------

def load_answers(dir_path: Path) -> pd.DataFrame:
    """Load benchmark answers from parquet files into a DataFrame."""
    fpaths = list(dir_path.rglob("*.parquet"))

    dfs: list[pd.DataFrame] = []
    cols_to_read = ["ID", "ground_truth", "prediction"]

    for fpath in tqdm(fpaths, desc="Loading parquet files"):
        model = fpath.parent.name.split("-", 3)[-1]
        benchmark = fpath.parent.parent.name

        df = pd.read_parquet(fpath, columns=cols_to_read)
        df = df.assign(model=model, benchmark=benchmark)
        df["correct"] = (df["ground_truth"] == df["prediction"]).astype(int)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df["benchmark"] = df["benchmark"].replace(benchmark_name_dict)
    df["model"] = df["model"].replace(model_name_dict)

    group_cols = ["benchmark", "model", "prediction", "ground_truth"]
    for col in group_cols:
        df[col] = pd.Categorical(df[col])

    return df


def print_id_coverage(df: pd.DataFrame) -> None:
    """Print ID coverage per model and benchmark (diagnostic)."""
    for benchmark in df["benchmark"].unique():
        print(f"\n=== {benchmark} ===")
        bench_data = df[df["benchmark"] == benchmark]

        for model in bench_data["model"].unique():
            model_data = bench_data[bench_data["model"] == model]
            unique_ids = model_data["ID"].unique()
            print(f"{model}: {len(unique_ids)} unique IDs")

        models = bench_data["model"].unique()
        if len(models) >= 2:
            m1_ids = set(
                bench_data[bench_data["model"] == models[0]]["ID"].unique()
            )
            m2_ids = set(
                bench_data[bench_data["model"] == models[1]]["ID"].unique()
            )
            common = m1_ids & m2_ids
            print(f"Common IDs between {models[0]} and {models[1]}: {len(common)}")
            print(f"Only in {models[0]}: {len(m1_ids - m2_ids)}")
            print(f"Only in {models[1]}: {len(m2_ids - m1_ids)}")


# -----------------
# Bootstrap accuracy
# -----------------

def _process_group_with_bootstrap_samples(
    model: str, group: pd.DataFrame, n_boot: int, seed: int
):
    """
    Compute bootstrap samples for accuracy for a single (model) group.
    Returns CI summary and raw bootstrap samples.
    """
    n_samples = len(group)
    if n_samples == 0:
        return [], {}

    rng = np.random.default_rng(seed)

    bootstrap_indices = rng.integers(0, n_samples, size=(n_boot, n_samples))
    correct = group["correct"].to_numpy()

    accuracy_samples = []
    for indices in bootstrap_indices:
        correct_boot = correct[indices]
        accuracy_samples.append(np.mean(correct_boot))

    accuracy_samples = np.array(accuracy_samples)

    low_idx = int(0.025 * n_boot)
    med_idx = int(0.5 * n_boot)
    high_idx = int(0.975 * n_boot)

    partitioned = np.partition(accuracy_samples, [low_idx, med_idx, high_idx])

    res_list = [
        {
            "model": model,
            "metric": "accuracy",
            "median": partitioned[med_idx],
            "low": partitioned[low_idx],
            "high": partitioned[high_idx],
            "n_questions": n_samples // 5,
        }
    ]

    bootstrap_samples = {
        (model, "accuracy"): accuracy_samples,
    }

    return res_list, bootstrap_samples


def compute_bootstrap_accuracy(
    df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int | None = None,
    n_jobs: int = -1,
):
    """
    Compute bootstrap CIs for accuracy grouped by model.
    """
    from joblib import Parallel, delayed  # local import to avoid clutter

    main_rng = np.random.default_rng(seed)

    df_copy = df.copy()
    group_col = "model"
    groups = list(df_copy.groupby(group_col, observed=True))

    n_groups = len(groups)
    group_seeds = main_rng.integers(0, 2**32, size=n_groups)

    results_with_samples = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_process_group_with_bootstrap_samples)(
            model,
            group,
            n_boot,
            int(group_seeds[i]),
        )
        for i, (model, group) in enumerate(groups)
    )

    final_results: list[dict] = []
    all_bootstrap_samples: dict[tuple[str, str], np.ndarray] = {}

    for res_list, boot_samples in results_with_samples:
        final_results.extend(res_list)
        all_bootstrap_samples.update(boot_samples)

    results_df = pd.DataFrame(final_results)
    results_df = results_df.sort_values("model").reset_index(drop=True)

    return results_df, all_bootstrap_samples


# -----------------
# Permutation tests
# -----------------

def _permutation_worker_accuracy(task: dict, n_perms: int, seed: int) -> dict:
    """
    Permutation test worker for accuracy comparison between two models.
    Permutes at ID level (all 5 trials of an ID swap together).
    """
    rng = np.random.default_rng(seed)

    correct1 = task["correct1"]
    correct2 = task["correct2"]
    id_array = task["id_array"]

    obs_acc1 = np.mean(correct1)
    obs_acc2 = np.mean(correct2)
    obs_diff = obs_acc1 - obs_acc2

    unique_ids, id_indices = np.unique(id_array, return_inverse=True)
    n_ids = len(unique_ids)

    swap_ids = rng.integers(0, 2, size=(n_perms, n_ids), dtype=bool)
    swap = swap_ids[:, id_indices]

    c1 = np.where(swap, correct2, correct1)
    c2 = np.where(swap, correct1, correct2)

    null_acc1 = np.mean(c1, axis=1)
    null_acc2 = np.mean(c2, axis=1)
    null_diff = null_acc1 - null_acc2

    p_val = float(np.mean(np.abs(null_diff) >= np.abs(obs_diff)))

    return {
        "model1": task["model1"],
        "model2": task["model2"],
        "metric": "accuracy",
        "p_value": p_val,
        "observed_diff": float(obs_diff),
        "obs_acc1": float(obs_acc1),
        "obs_acc2": float(obs_acc2),
        "n_ids": n_ids,
        "n_samples": len(id_array),
    }


def compute_pairwise_comparisons_accuracy(
    df: pd.DataFrame,
    n_permutations: int = 1000,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Permutation tests on accuracy between all model pairs (ID-level permutation).
    """
    from joblib import Parallel, delayed  # local import

    df_grouped = df[["ID", "model", "correct"]].copy()
    df_grouped["ID"] = df_grouped["ID"].astype(str)

    tasks: list[dict] = []
    models = sorted(df_grouped["model"].unique())

    for m1, m2 in combinations(models, 2):
        d1 = df_grouped[df_grouped["model"] == m1].copy()
        d2 = df_grouped[df_grouped["model"] == m2].copy()

        ids1 = set(d1["ID"].unique())
        ids2 = set(d2["ID"].unique())
        common_ids = sorted(ids1 & ids2)

        if len(common_ids) == 0:
            print(f"Warning: {m1} vs {m2} have NO common IDs")
            continue

        d1 = d1[d1["ID"].isin(common_ids)].sort_values("ID").reset_index(drop=True)
        d2 = d2[d2["ID"].isin(common_ids)].sort_values("ID").reset_index(drop=True)

        if not np.array_equal(d1["ID"].values, d2["ID"].values):
            print(f"ERROR: {m1} vs {m2} STILL misaligned after filtering!")
            continue

        if len(d1) != len(d2):
            print(f"ERROR: {m1} vs {m2} have different lengths!")
            continue

        correct1 = d1["correct"].to_numpy()
        correct2 = d2["correct"].to_numpy()
        id_array = d1["ID"].to_numpy()

        tasks.append(
            {
                "model1": m1,
                "model2": m2,
                "correct1": correct1,
                "correct2": correct2,
                "id_array": id_array,
            }
        )

    if len(tasks) == 0:
        print("ERROR: No valid tasks created!")
        return pd.DataFrame()

    print(f"Executing Permutation Tests on {len(tasks)} accuracy tasks...")

    main_rng = np.random.default_rng(seed)
    seeds = main_rng.integers(0, 2**32, size=len(tasks))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_permutation_worker_accuracy)(
            tasks[i], n_permutations, int(seeds[i])
        )
        for i in range(len(tasks))
    )

    res_df = pd.DataFrame(results)

    if len(res_df) > 0:
        _, res_df["p_value_bh"], _, _ = multipletests(
            res_df["p_value"], method="fdr_bh"
        )
        res_df["Significant_bh"] = res_df["p_value_bh"] < 0.05
    else:
        res_df["p_value_bh"] = []
        res_df["Significant_bh"] = []

    return res_df


# -----------------
# Plotting helpers
# -----------------

def map_values(value):
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    if value < 0.0001:
        return "****"
    elif value < 0.001:
        return "***"
    elif value < 0.01:
        return "**"
    elif value < 0.05:
        return "*"
    elif value <= 1.0:
        return "ns"
    else:
        return str(value)


def get_annotate_matrix(matrix_content: pd.DataFrame):
    """Build upper triangular matrix text for annotation."""
    n = len(matrix_content)
    row_labels = matrix_content.index.tolist()
    col_labels = matrix_content.columns.tolist()[1:]

    main: list[str] = []
    header = ["  "] + ["{: <5}".format(label) for label in col_labels]
    main.append("  ".join(header))

    for i in range(n - 1):
        row_data = []
        for j in range(i + 1, n):
            val = matrix_content.iloc[i, j]
            row_data.append("{: <5}".format(map_values(val)))

        padding = ["     "] * i
        row_values = padding + row_data
        row_line = "  ".join(row_values) + "  " + row_labels[i]
        main.append(row_line)

    return main


def annotate_pmatrix(
    ax,
    matrix_content: pd.DataFrame,
    xy=(0.995, 0.98),
    title: str = "p-values",
):
    """Annotate plot with p-value matrix."""
    matrix_text_lines = get_annotate_matrix(matrix_content)
    matrix_text_lines.insert(0, f"{title}")
    matrix_text = "\n".join(matrix_text_lines)

    ax.annotate(
        matrix_text,
        xy=xy,
        xycoords="axes fraction",
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            edgecolor="black",
            facecolor=(1, 1, 1, 0.0),
            lw=0.5,
        ),
        fontsize=4,
        family="monospace",
    )


def plot_bootstrap_accuracy(
    results_df: pd.DataFrame,
    perm_results_df: pd.DataFrame | None = None,
    model_order: list[str] | None = None,
    color_order: list[str] | None = None,
    figsize=None,
    palette="Set2",
    bar_width: float = 0.8,
    fontsize: int = 8,
):
    """
    Plot bootstrap accuracy results with error bars for all models in a single panel,
    optionally with permutation test p-value matrix.
    """
    df = results_df[results_df["metric"] == "accuracy"].copy()

    if model_order is None:
        model_order = sorted(df["model"].unique())
    else:
        df = df[df["model"].isin(model_order)].copy()

    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values("model")
    df = df.set_index("model").reindex(model_order).reset_index()

    yerr_low = df["median"] - df["low"]
    yerr_high = df["high"] - df["median"]
    yerr = np.array([yerr_low, yerr_high])

    if figsize is None:
        figsize = (max(2, 0.2 * len(model_order)), 5.5)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[20, 1], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])

    if isinstance(palette, str):
        colors = sns.color_palette(
            palette,
            n_colors=len(color_order if color_order else model_order),
        )
    else:
        colors = palette

    if color_order:
        color_map = {m: c for m, c in zip(color_order, colors)}
        colors = [color_map[m] for m in model_order]

    hatch_patterns = ["///", "\\\\\\", "|||", "---", "+++", "xxx", "ooo"]
    n_hatch = len(hatch_patterns)
    hatches = [hatch_patterns[i % n_hatch] for i in range(len(model_order))]

    x_pos = np.arange(len(model_order))

    bars = []
    for i in range(len(model_order)):
        bar = ax.bar(
            x_pos[i],
            df["median"].iloc[i],
            width=bar_width,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            hatch=hatches[i],
        )
        bars.append(bar[0])

    ax.errorbar(
        x_pos,
        df["median"],
        yerr=yerr,
        fmt="none",
        ecolor="black",
        capsize=5,
        capthick=1,
        linewidth=1,
    )

    for bar, median_val, high_val in zip(bars, df["median"], df["high"]):
        if pd.notna(median_val) and pd.notna(high_val):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                high_val + 0.01,
                f"{median_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                rotation=0,
                zorder=10,
            )

    ax.set_xlabel("", fontsize=fontsize)
    ax.set_ylabel("Accuracy", fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.grid(axis="y", alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.4)
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    model_to_letter: dict[str, str] | None = None

    if perm_results_df is not None:
        letters = [chr(65 + i) for i in range(len(model_order))]
        model_to_letter = {m: l for m, l in zip(model_order, letters)}

        letter_labels = [model_to_letter[m] for m in model_order]
        p_matrix = pd.DataFrame(
            np.nan, index=letter_labels, columns=letter_labels
        )

        for _, row in perm_results_df.iterrows():
            m1, m2 = row["model1"], row["model2"]
            if m1 in model_to_letter and m2 in model_to_letter:
                l1 = model_to_letter[m1]
                l2 = model_to_letter[m2]
                p_val = row["p_value_bh"]
                p_matrix.loc[l1, l2] = p_val
                p_matrix.loc[l2, l1] = p_val

        annotate_pmatrix(ax, p_matrix, xy=(0.985, 0.98), title="p-values               ")

    ax_legend.axis("off")

    from matplotlib.patches import Rectangle

    if perm_results_df is not None and model_to_letter is not None:
        legend_labels = [f"{m} ({model_to_letter[m]})" for m in model_order]
    else:
        legend_labels = model_order

    handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=colors[i],
            edgecolor="black",
            linewidth=0.5,
            hatch=hatches[i],
            alpha=0.85,
        )
        for i in range(len(model_order))
    ]

    ncol = max(1, int(np.ceil(len(model_order) / 3)))

    ax_legend.legend(
        handles,
        legend_labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize=fontsize - 2,
        handlelength=1.5,
        handleheight=1.5,
        bbox_to_anchor=(0.5, 0),
    )

    return fig, [ax, ax_legend]


# -----------------
# Main pipeline
# -----------------

def main() -> None:
    np.random.seed(42)

    print("Loading benchmark answers...")
    ans = load_answers(BENCH_PATH)

    # Drop one benchmark as in original script
    ans = ans[ans["benchmark"] != "MMLU - professional medicine"].reset_index(
        drop=True
    )

    print_id_coverage(ans)

    # Collapse all benchmarks into one “benchmark” (as original code does)
    ans["benchmark"] = "benchmark"

    print("Computing bootstrap accuracy CIs...")
    results, bootstrap_samples = compute_bootstrap_accuracy(
        ans,
        n_boot=1000,
        seed=42,
        n_jobs=-1,
    )

    print("Computing permutation tests...")
    perm_results = compute_pairwise_comparisons_accuracy(
        ans,
        n_permutations=10000,
        seed=42,
        n_jobs=40,
    )

    print("Plotting figure...")
    fig, axes = plot_bootstrap_accuracy(
        results,
        perm_results_df=perm_results,
        model_order=model_order,
        color_order=color_order,
        figsize=(2.3, 2),
        palette="colorblind",
        bar_width=0.7,
        fontsize=7,
    )

    out_path = "../figures/fig2_benchmarks.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()