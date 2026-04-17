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

# Configuration

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


# Data loading

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
    df = df[df['model'].isin(model_order)].reset_index(drop=True)

    for col in ["benchmark", "model", "prediction", "ground_truth"]:
        df[col] = pd.Categorical(df[col])

    return df


def print_id_coverage(df: pd.DataFrame) -> None:
    for benchmark in df["benchmark"].unique():
        print(f"\n=== {benchmark} ===")
        bench_data = df[df["benchmark"] == benchmark]
        for model in bench_data["model"].unique():
            unique_ids = bench_data[bench_data["model"] == model]["ID"].unique()
            print(f"{model}: {len(unique_ids)} unique IDs")


# Metric utility

def _vectorized_macro_accuracy(correct: np.ndarray,
                              benchmark_codes: np.ndarray,
                              benchmark_ids: np.ndarray):
    """
    Compute macro-averaged accuracy over benchmarks.

    Supports:
        correct: (n,) with benchmark_codes (n,)
        correct: (B, n) with benchmark_codes (B, n)

    Returns:
        scalar (1D case) or (B,) array (2D case)
    """
    # bench_accs = []
    # for b_code in benchmark_ids:
    #     mask = benchmark_codes == b_code
    #     if correct.ndim == 2:
    #         # Bootstrap case: shape (n_boot, n_bench_samples)
    #         bench_accs.append(correct[:, mask].mean(axis=1))
    #     else:
    #         bench_accs.append(correct[mask].mean())

    # # Stack and mean across benchmarks: shape (n_boot,) or scalar
    # return np.stack(bench_accs, axis=-1).mean(axis=-1)

    
    correct = np.asarray(correct)
    benchmark_codes = np.asarray(benchmark_codes)
    benchmark_ids = np.asarray(benchmark_ids)
    
    # print(correct.shape, benchmark_codes.shape)

    # 1D case (point estimate)
    if correct.ndim == 1:
        bench_accs = []
        for b in benchmark_ids:
            m = (benchmark_codes == b)
            if m.any():
                bench_accs.append(correct[m].mean())
        return np.mean(bench_accs)

    # 2D case (bootstrap/permutation)
    if correct.ndim == 2:
        B, n = correct.shape

        # If benchmark codes are 1D, broadcast them
        if benchmark_codes.ndim == 1:
            benchmark_codes = np.broadcast_to(benchmark_codes, (B, n))
        bench_accs = []

        for b in benchmark_ids:
            m = (benchmark_codes == b)        # (B, n)
            counts = m.sum(axis=1)            # (B,)
            sums = (correct * m).sum(axis=1)  # (B,)
            acc = np.divide(
                sums,
                counts,
                out=np.full(B, np.nan, dtype=float),
                where=(counts > 0),
            )
            bench_accs.append(acc)

        return np.nanmean(np.stack(bench_accs, axis=1), axis=1)

    raise ValueError("correct must be 1D or 2D")


# Bootstrap (macro accuracy)

def _single_bootstrap_task_macro_accuracy(group_info, correct_b, benchmark_codes, benchmark_ids):
    """Bootstrap worker: compute macro-averaged accuracy over n_boot resamples."""
    boot_values = _vectorized_macro_accuracy(correct_b, benchmark_codes, benchmark_ids)
    low, med, high = np.quantile(boot_values, [0.025, 0.5, 0.975])
    return {
        **group_info,
        "metric": "accuracy",
        "mean": float(np.mean(boot_values)),
        "median": float(med),
        "low": float(low),
        "high": float(high),
    }


def optimized_bootstrap_parallel_macro(
    df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Compute bootstrap CIs for macro-averaged accuracy grouped by model.
    Resamples questions within each model group, then macro-averages over benchmarks.
    """
    groups = list(df.groupby("model", observed=True))
    main_rng = np.random.default_rng(seed)
    all_tasks = []

    print(f"Preparing bootstrap data for {len(groups)} model groups...")

    # for model, group in groups:
    #     group = group.reset_index(drop=True)

    #     # Encode benchmarks as integer codes
    #     bench_cat = group["benchmark"].astype("category")
    #     benchmark_codes = bench_cat.cat.codes.to_numpy()
    #     benchmark_ids = np.unique(benchmark_codes)

    #     correct = group["correct"].to_numpy()
    #     n = len(correct)

    #     group_seed = int(main_rng.integers(0, 2**32))
    #     rng = np.random.default_rng(group_seed)
    #     indices = rng.integers(0, n, size=(n_boot, n))

    #     correct_b = correct[indices]  # shape (n_boot, n)
    #     bench_b = benchmark_codes[indices] 

    #     group_info = {"model": model}
    #     all_tasks.append((group_info, correct_b, bench_b, benchmark_ids))
    
    for model, group in groups:
        group = group.reset_index(drop=True)

        # --- collapse to one row per (benchmark, ID) ---
        # id_level = (
        #     group.groupby(["benchmark", "ID"], observed=True)["correct"]
        #     .mean()
        #     .reset_index(name="id_acc")
        # )

        # Encode benchmarks as integer codes on the collapsed table
        # bench_cat = id_level["benchmark"].astype("category")
        bench_cat = group["benchmark"].astype("category")
        benchmark_codes = bench_cat.cat.codes.to_numpy()
        benchmark_ids = np.unique(benchmark_codes)

        # id_acc = id_level["id_acc"].to_numpy()
        correct = group["correct"].to_numpy()

        group_seed = int(main_rng.integers(0, 2**32))
        rng = np.random.default_rng(group_seed)

        # Indices for each benchmark (variable sizes, now in ID-space)
        idx_by_bench = {b: np.flatnonzero(benchmark_codes == b) for b in benchmark_ids}

        # Bootstrap: resample IDs within each benchmark, then concatenate
        boot_idx_parts = []
        for b in benchmark_ids:
            idx_b = idx_by_bench[b]
            n_b = len(idx_b)  # number of IDs in this benchmark for this model
            boot_idx_parts.append(idx_b[rng.integers(0, n_b, size=(n_boot, n_b))])  # (n_boot, n_b)

        indices = np.concatenate(boot_idx_parts, axis=1)  # (n_boot, n_total_ids_across_benchmarks)

        correct_b = correct[indices]                 # (n_boot, n_total_ids)
        bench_b = benchmark_codes[indices]          # (n_boot, n_total_ids)

        group_info = {"model": model}
        all_tasks.append((group_info, correct_b, bench_b, benchmark_ids))
        
    print(f"Executing bootstrap on {len(all_tasks)} tasks across {n_jobs} cores...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_bootstrap_task_macro_accuracy)(*t) for t in all_tasks
    )

    return pd.DataFrame(results)


# Permutation tests (macro accuracy)

def _permutation_worker_macro_accuracy(task, n_perms: int, seed: int):
    """
    Permutation test worker for macro-averaged accuracy.
    Swaps predictions at the ID level, computes macro accuracy across benchmarks.
    """
    rng = np.random.default_rng(seed)

    correct1 = task["correct1"]
    correct2 = task["correct2"]
    id_array = task["id_array"]
    benchmark_codes = task["benchmark_codes"]
    benchmark_ids = task["benchmark_ids"]

    # Observed macro accuracy
    obs1 = _vectorized_macro_accuracy(correct1, benchmark_codes, benchmark_ids)
    obs2 = _vectorized_macro_accuracy(correct2, benchmark_codes, benchmark_ids)
    obs_diff = obs1 - obs2

    # ID-level swap null distribution
    unique_ids, id_indices = np.unique(id_array, return_inverse=True)
    n_ids = len(unique_ids)
    swap_ids = rng.integers(0, 2, size=(n_perms, n_ids), dtype=bool)
    swap = swap_ids[:, id_indices]  # shape (n_perms, n_samples)

    c1 = np.where(swap, correct2, correct1)  # shape (n_perms, n_samples)
    c2 = np.where(swap, correct1, correct2)

    null1 = _vectorized_macro_accuracy(c1, benchmark_codes, benchmark_ids)
    null2 = _vectorized_macro_accuracy(c2, benchmark_codes, benchmark_ids)

    p_val = float(np.mean(np.abs(null1 - null2) >= np.abs(obs_diff)))

    out = {k: v for k, v in task.items()
           if k not in ["correct1", "correct2", "id_array", "benchmark_codes", "benchmark_ids"]}
    out.update({
        "p_value": p_val,
        "observed_diff": float(obs_diff),
        "metric": "accuracy",
        "obs1": float(obs1),
        "obs2": float(obs2),
    })
    return out


def compute_pairwise_comparisons_macro(
    df: pd.DataFrame,
    n_permutations: int = 10000,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Permutation tests on macro-averaged accuracy between all model pairs.
    ID-level swaps preserve within-subject correlation across trials.
    """
    df_grouped = df[["ID", "model", "benchmark", "correct"]].copy()
    df_grouped["ID"] = df_grouped["ID"].astype(str)

    # Encode benchmarks globally
    bench_cat = pd.CategoricalDtype(categories=sorted(df_grouped["benchmark"].unique()))
    df_grouped["bench_code"] = df_grouped["benchmark"].astype(bench_cat).cat.codes
    benchmark_ids = np.unique(df_grouped["bench_code"].to_numpy())

    tasks: list[dict] = []
    models = sorted(df_grouped["model"].unique())

    for m1, m2 in combinations(models, 2):
        d1 = df_grouped[df_grouped["model"] == m1].copy()
        d2 = df_grouped[df_grouped["model"] == m2].copy()

        ids1 = set(d1["ID"].unique())
        ids2 = set(d2["ID"].unique())
        common_ids = sorted(ids1 & ids2)

        if len(common_ids) == 0:
            print(f"Warning: {m1} vs {m2} have NO common IDs, skipping")
            continue

        d1 = d1[d1["ID"].isin(common_ids)].sort_values(["ID", "benchmark"]).reset_index(drop=True)
        d2 = d2[d2["ID"].isin(common_ids)].sort_values(["ID", "benchmark"]).reset_index(drop=True)

        if len(d1) != len(d2) or not np.array_equal(d1["ID"].values, d2["ID"].values):
            print(f"ERROR: {m1} vs {m2} misaligned after filtering, skipping")
            continue

        tasks.append({
            "model1": m1,
            "model2": m2,
            "correct1": d1["correct"].to_numpy(),
            "correct2": d2["correct"].to_numpy(),
            "id_array": d1["ID"].to_numpy(),
            "benchmark_codes": d1["bench_code"].to_numpy(),
            "benchmark_ids": benchmark_ids,
        })

    if len(tasks) == 0:
        print("ERROR: No valid tasks created!")
        return pd.DataFrame()

    print(f"Executing permutation tests on {len(tasks)} model pairs...")
    main_rng = np.random.default_rng(seed)
    seeds = main_rng.integers(0, 2**32, size=len(tasks))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_permutation_worker_macro_accuracy)(tasks[i], n_permutations, int(seeds[i]))
        for i in range(len(tasks))
    )

    res_df = pd.DataFrame(results)

    if len(res_df) > 0:
        _, res_df["p_value_bh"], _, _ = multipletests(res_df["p_value"], method="fdr_bh")
        res_df["Significant_bh"] = res_df["p_value_bh"] < 0.05

    return res_df


# Plotting helpers

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
        row_line = "  ".join(padding + row_data) + "  " + row_labels[i]
        main.append(row_line)

    return main


def annotate_pmatrix(ax, matrix_content: pd.DataFrame, xy=(0.995, 0.98), title: str = "p-values"):
    matrix_text_lines = get_annotate_matrix(matrix_content)
    matrix_text_lines.insert(0, title)
    ax.annotate(
        "\n".join(matrix_text_lines),
        xy=xy,
        xycoords="axes fraction",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black",
                  facecolor=(1, 1, 1, 0.0), lw=0.5),
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
    df = results_df[results_df["metric"] == "accuracy"].copy()

    if model_order is None:
        model_order = sorted(df["model"].unique())
    else:
        df = df[df["model"].isin(model_order)].copy()

    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values("model").set_index("model").reindex(model_order).reset_index()

    yerr_low = df["median"] - df["low"]
    yerr_high = df["high"] - df["median"]
    yerr = np.array([yerr_low, yerr_high])

    if figsize is None:
        figsize = (max(2, 0.2 * len(model_order)), 5.5)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[20, 1], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])

    colors = sns.color_palette(palette, n_colors=len(color_order if color_order else model_order))
    if color_order:
        color_map = {m: c for m, c in zip(color_order, colors)}
        colors = [color_map[m] for m in model_order]

    hatch_patterns = ["///", "\\\\\\", "|||", "---", "+++", "xxx", "ooo"]
    hatches = [hatch_patterns[i % len(hatch_patterns)] for i in range(len(model_order))]
    x_pos = np.arange(len(model_order))

    bars = []
    for i in range(len(model_order)):
        bar = ax.bar(x_pos[i], df["median"].iloc[i], width=bar_width,
                     color=colors[i], edgecolor="black", linewidth=0.5,
                     alpha=0.85, hatch=hatches[i])
        bars.append(bar[0])

    ax.errorbar(x_pos, df["median"], yerr=yerr, fmt="none",
                ecolor="black", capsize=5, capthick=1, linewidth=1)

    for bar, median_val, high_val in zip(bars, df["median"], df["high"]):
        if pd.notna(median_val) and pd.notna(high_val):
            ax.text(bar.get_x() + bar.get_width() / 2.0, high_val + 0.01,
                    f"{median_val:.2f}", ha="center", va="bottom",
                    fontsize=fontsize, zorder=10)

    ax.set_ylabel("Macro-averaged accuracy", fontsize=fontsize)
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

        p_matrix = pd.DataFrame(np.nan, index=letter_labels, columns=letter_labels)
        for _, row in perm_results_df.iterrows():
            m1, m2 = row["model1"], row["model2"]
            if m1 in model_to_letter and m2 in model_to_letter:
                l1, l2 = model_to_letter[m1], model_to_letter[m2]
                p_matrix.loc[l1, l2] = row["p_value_bh"]
                p_matrix.loc[l2, l1] = row["p_value_bh"]

        annotate_pmatrix(ax, p_matrix, xy=(0.985, 0.98), title="p-values               ")

    ax_legend.axis("off")

    from matplotlib.patches import Rectangle
    legend_labels = (
        [f"{m} ({model_to_letter[m]})" for m in model_order]
        if model_to_letter else model_order
    )
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor="black",
                  linewidth=0.5, hatch=hatches[i], alpha=0.85)
        for i in range(len(model_order))
    ]
    ncol = max(1, int(np.ceil(len(model_order) / 3)))
    ax_legend.legend(handles, legend_labels, loc="center", ncol=ncol,
                     frameon=False, fontsize=fontsize - 2,
                     handlelength=1.5, handleheight=1.5,
                     bbox_to_anchor=(0.5, 0))

    return fig, [ax, ax_legend]


# Main pipeline

def main() -> None:
    np.random.seed(42)

    print("Loading benchmark answers...")
    ans = load_answers(BENCH_PATH)
    ans = ans[ans["benchmark"] != "MMLU - professional medicine"].reset_index(drop=True)

    print_id_coverage(ans)

    print("Computing macro-averaged bootstrap accuracy CIs...")
    results = optimized_bootstrap_parallel_macro(ans, n_boot=1000, seed=42, n_jobs=-1)

    print("Computing permutation tests (macro-averaged)...")
    perm_results = compute_pairwise_comparisons_macro(ans, n_permutations=10000, seed=42, n_jobs=-1)

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

    out_path = "../figures/fig2_benchmarks_macro.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()