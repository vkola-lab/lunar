from pathlib import Path
import math
import warnings
from yaml import safe_load

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.legend_handler import HandlerTuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Arial"
mpl.rcParams["hatch.linewidth"] = 0.5
sns.set_style("whitegrid")

RES_PATH = Path("/projectnb/vkolagrp/projects/adrd_foundation_model/results/training_curve")
COLS_TO_READ = ["ID", "ground_truth", "prediction"]
N_BOOT = 1000
INCLUDED_FILES = {}
IGNORED_FILES = {}


def load_data(res_path: Path, include_model: str = "SFT") -> pd.DataFrame:
    ignored_files: list[Path] = []
    dfs: list[pd.DataFrame] = []
    skipped: set[Path] = set()
    INCLUDED_FILES[include_model] = []
    IGNORED_FILES[include_model] = []

    for fpath in tqdm(res_path.rglob("*.parquet"), desc="Loading parquet files"):
        if include_model not in str(fpath):
            ignored_files.append(fpath)
            IGNORED_FILES[include_model].append(fpath)
            continue
        
        INCLUDED_FILES[include_model].append(fpath)
        config_path = fpath.parent / "config.yml"
        try:
            with config_path.open() as config_file:
                config = safe_load(config_file)
        except Exception:
            skipped.add(fpath)
            continue

        df = pd.read_parquet(fpath, columns=COLS_TO_READ)

        df["benchmark"] = fpath.parent.parent.parent.name
        df["cohort"] = fpath.parent.parent.parent.parent.name
        df["correct"] = (df["ground_truth"] == df["prediction"]).astype(np.int8)
        df["training_steps"] = int(config["training_steps"])
        df["model"] = str(config["run_readable_name"])

        dfs.append(df)

    print(f"Ignored: {len(ignored_files)} files")
    print(f"Skipped: {len(skipped)} files")
    if len(dfs) == 0:
        return pd.DataFrame(columns=COLS_TO_READ + ["benchmark", "cohort", "correct", "training_steps", "model"])
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    merged_df = merged_df.loc[
        ~merged_df["benchmark"].isin(["test_mci", "test_np_mixed", "test_np", "test_ftld"])
    ]
    
    print(f"Unique benchmarks {include_model}: {dict(merged_df[merged_df['model'] == 'Qwen2.5-7B-Instruct']['benchmark'].value_counts())}")
    print(f"Unique cohort {include_model}: {dict(merged_df[merged_df['model'] == 'Qwen2.5-7B-Instruct']['cohort'].value_counts())}")
    print(f"Unique model {include_model}: {dict(merged_df['model'].value_counts())}")
    

    return merged_df

# ID level
# def bootstrap_macro_cell_mean_cluster_id(
#     df: pd.DataFrame,
#     model_name: str,
#     id_col: str = "ID",
#     n_boot: int = 1000,
#     seed: int | None = None,
# ) -> tuple[float, float, float, float]:
#     df_m = df.loc[df["model"] == model_name].copy()
#     if df_m.empty:
#         return (float("nan"), float("nan"), float("nan"), float("nan"))

#     # one row per (cohort, benchmark, ID)
#     id_level = (
#         df_m.groupby(["cohort", "benchmark", id_col], observed=True)["correct"]
#         .mean()
#         .reset_index(name="id_acc")
#     )

#     # per-cell (cohort, benchmark) mean over IDs
#     cell_base = (
#         id_level.groupby(["cohort", "benchmark"], observed=True)["id_acc"]
#         .mean()
#         .reset_index(name="cell_acc")
#     )
#     point = float(cell_base["cell_acc"].mean())

#     # assign cell_id
#     cells = cell_base[["cohort", "benchmark"]].drop_duplicates().reset_index(drop=True)
#     cells["cell_id"] = np.arange(len(cells), dtype=int)

#     id_level2 = id_level.merge(cells, on=["cohort", "benchmark"], how="inner")

#     cell_id = id_level2["cell_id"].to_numpy()
#     id_acc = id_level2["id_acc"].to_numpy()

#     C = int(cells.shape[0])
#     per_cell = [id_acc[cell_id == c] for c in range(C)]

#     rng = np.random.default_rng(seed)
#     bs = np.empty(n_boot, dtype=float)

#     for b in range(n_boot):
#         cell_means = np.empty(C, dtype=float)
#         for c in range(C):
#             a = per_cell[c]
#             m = a.size
#             idx = rng.integers(0, m, size=m)
#             cell_means[c] = a[idx].mean()
#         bs[b] = cell_means.mean()

#     return point, float(bs.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

# ID level
# def macro_ci_over_training_steps(
#     df: pd.DataFrame,
#     model_pattern: str = "NACC",
#     id_col: str = "ID",
#     n_boot: int = 1000,
#     seed: int = 0,
# ) -> pd.DataFrame:
#     d = df.loc[df["model"].astype(str).str.contains(model_pattern, regex=True, na=False)].copy()
#     if d.empty:
#         return pd.DataFrame(columns=["training_steps", "in_distribution", "point", "low", "high"])

#     d["in_distribution"] = np.where(
#         d["cohort"] == "nacc_test_updated",
#         "Internal validation\n(NACC)",
#         "External validation\n(All other cohorts)",
#     )

#     # ID-level accuracy within each (step, in_dist, cohort, benchmark, ID)
#     id_level = (
#         d.groupby(["training_steps", "in_distribution", "cohort", "benchmark", id_col], observed=True)["correct"]
#         .mean()
#         .reset_index(name="id_acc")
#     )

#     rng = np.random.default_rng(seed)
#     out_rows: list[tuple[int, str, float, float, float]] = []

#     steps_groups = (
#         id_level[["training_steps", "in_distribution"]]
#         .drop_duplicates()
#         .sort_values(["training_steps", "in_distribution"])
#         .itertuples(index=False, name=None)
#     )

#     for step, grp in steps_groups:
#         slice_df = id_level.loc[
#             (id_level["training_steps"] == step) & (id_level["in_distribution"] == grp)
#         ].copy()

#         if slice_df.empty:
#             continue

#         cells = slice_df[["cohort", "benchmark"]].drop_duplicates().reset_index(drop=True)
#         cells["cell_id"] = np.arange(len(cells), dtype=int)
#         s2 = slice_df.merge(cells, on=["cohort", "benchmark"], how="inner")

#         cell_id = s2["cell_id"].to_numpy()
#         id_acc = s2["id_acc"].to_numpy()

#         C = int(cells.shape[0])
#         per_cell = [id_acc[cell_id == c] for c in range(C)]

#         cell_means = np.array([a.mean() for a in per_cell], dtype=float)
#         point = float(cell_means.mean())

#         bs = np.empty(n_boot, dtype=float)
#         for b in range(n_boot):
#             boot_cell_means = np.empty(C, dtype=float)
#             for c in range(C):
#                 a = per_cell[c]
#                 m = a.size
#                 idx = rng.integers(0, m, size=m)
#                 boot_cell_means[c] = a[idx].mean()
#             bs[b] = boot_cell_means.mean()

#         low = float(np.quantile(bs, 0.025))
#         high = float(np.quantile(bs, 0.975))

#         out_rows.append((int(step), str(grp), point, low, high))

#     return pd.DataFrame(out_rows, columns=["training_steps", "in_distribution", "point", "low", "high"])

# Row level
def bootstrap_macro_cell_mean_row_level(
    df: pd.DataFrame,
    model_name: str,
    id_col: str = "ID",
    n_boot: int = 1000,
    seed: int | None = None,
) -> tuple[float, float, float, float]:
    df_m = df.loc[df["model"] == model_name].copy()
    if df_m.empty:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    # assign cell_id per (cohort, benchmark)
    cells = (
        df_m[["cohort", "benchmark"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    cells["cell_id"] = np.arange(len(cells), dtype=int)
    df_m = df_m.merge(cells, on=["cohort", "benchmark"], how="inner")

    cell_id = df_m["cell_id"].to_numpy()
    correct = df_m["correct"].to_numpy(dtype=float)

    C = int(cells.shape[0])
    per_cell = [correct[cell_id == c] for c in range(C)]

    # point estimate: mean(correct) per cell, then macro-average
    cell_means_point = np.array([a.mean() for a in per_cell], dtype=float)
    point = float(cell_means_point.mean())

    # bootstrap: resample rows within each cell
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        cell_means = np.empty(C, dtype=float)
        for c in range(C):
            a = per_cell[c]
            m = a.size
            idx = rng.integers(0, m, size=m)
            cell_means[c] = a[idx].mean()
        bs[b] = cell_means.mean()

    return point, float(bs.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

# Row level
def macro_ci_over_training_steps_row_bootstrap(
    df: pd.DataFrame,
    model_pattern: str = "NACC",
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    d = df.loc[df["model"].astype(str).str.contains(model_pattern, regex=True, na=False)].copy()
    if d.empty:
        return pd.DataFrame(columns=["training_steps", "in_distribution", "point", "low", "high"])

    d["in_distribution"] = np.where(
        d["cohort"] == "nacc_test_updated",
        "Internal validation\n(NACC)",
        "External validation\n(All other cohorts)",
    )

    rng = np.random.default_rng(seed)
    out_rows: list[tuple[int, str, float, float, float]] = []

    steps_groups = (
        d[["training_steps", "in_distribution"]]
        .drop_duplicates()
        .sort_values(["training_steps", "in_distribution"])
        .itertuples(index=False, name=None)
    )

    for step, grp in steps_groups:
        slice_df = d.loc[
            (d["training_steps"] == step) & (d["in_distribution"] == grp)
        ].copy()

        if slice_df.empty:
            continue

        # Define cells as (cohort, benchmark)
        cells = slice_df[["cohort", "benchmark"]].drop_duplicates().reset_index(drop=True)
        cells["cell_id"] = np.arange(len(cells), dtype=int)
        s2 = slice_df.merge(cells, on=["cohort", "benchmark"], how="inner")

        cell_id = s2["cell_id"].to_numpy()
        correct = s2["correct"].to_numpy(dtype=float)

        C = int(cells.shape[0])
        per_cell = [correct[cell_id == c] for c in range(C)]

        # Point estimate: mean(correct) within each cell, then macro-average across cells
        cell_means = np.array([a.mean() for a in per_cell], dtype=float)
        point = float(cell_means.mean())

        # Bootstrap: resample ROWS within each cell
        bs = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            boot_cell_means = np.empty(C, dtype=float)
            for c in range(C):
                a = per_cell[c]
                m = a.size
                idx = rng.integers(0, m, size=m)
                boot_cell_means[c] = a[idx].mean()
            bs[b] = boot_cell_means.mean()

        low = float(np.quantile(bs, 0.025))
        high = float(np.quantile(bs, 0.975))

        out_rows.append((int(step), str(grp), point, low, high))

    return pd.DataFrame(out_rows, columns=["training_steps", "in_distribution", "point", "low", "high"])


def select_every_200_and_last(df: pd.DataFrame, step_col: str = "training_steps") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    step_vals_sorted = np.sort(df[step_col].unique())
    every_200 = set(step_vals_sorted[step_vals_sorted % 200 == 0])
    every_200.add(step_vals_sorted[-1])
    selection = df[df[step_col].isin(list(every_200))].copy()
    return selection.sort_values(step_col)


def make_main_training_curve_figure(
    ci_sft: pd.DataFrame,
    ci_sce: pd.DataFrame,
    q3b_sft: tuple[float, float, float, float],
    q7b_sft: tuple[float, float, float, float],
    q3b_sce: tuple[float, float, float, float],
    q7b_sce: tuple[float, float, float, float],
) -> plt.Figure:
    fig = plt.figure(figsize=(4.6, 1.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)

    fontsize = 7
    linewidth = 0.8
    markersize = 2

    markers = {
        "External validation\n(All other cohorts)": "^",
        "Internal validation\n(NACC)": "o",
    }
    colors = {
        "Internal validation\n(NACC)": "#ff7f00",
        "External validation\n(All other cohorts)": "#377eb8",
    }

    plot_configs = [
        {
            "ax": ax1,
            "ci_pd": ci_sft.sort_values(["training_steps", "in_distribution"]),
            "baselines": {"3B": q3b_sft, "7B": q7b_sft},
            "ylabel": "Macro-averaged\naccuracy",
        },
        {
            "ax": ax2,
            "ci_pd": ci_sce.sort_values(["training_steps", "in_distribution"]),
            "baselines": {"3B": q3b_sce, "7B": q7b_sce},
            "ylabel": "",
        },
    ]

    hatch_styles = {"3B": "\\\\\\", "7B": "///"}
    linestyles = {"3B": "-", "7B": "--"}

    baseline_handles = {}
    for size in ["3B", "7B"]:
        line = mlines.Line2D([], [], color="black", linewidth=0.5, linestyle=linestyles[size])
        patch = mpatches.Patch(
            facecolor="darkgray",
            edgecolor="black",
            hatch=hatch_styles[size],
            alpha=0.8,
            linewidth=0.5,
        )
        baseline_handles[size] = (line, patch)

    for cfg in plot_configs:
        ax = cfg["ax"]

        for label, group_df in cfg["ci_pd"].groupby("in_distribution", sort=False):
            x = group_df["training_steps"].to_numpy()
            y = group_df["point"].to_numpy()
            yerr = np.vstack([y - group_df["low"].to_numpy(), group_df["high"].to_numpy() - y])

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                linewidth=linewidth,
                marker=markers[label],
                markersize=markersize,
                alpha=0.9,
                color=colors.get(label),
                label=label,
            )

        for size, (mean, _bs_mean, low, high) in cfg["baselines"].items():
            ax.axhline(y=mean, color="black", linewidth=0.5, linestyle=linestyles[size])
            ax.axhspan(
                low,
                high,
                alpha=0.5,
                hatch=hatch_styles[size],
                edgecolor="black",
                facecolor="darkgray",
                zorder=0,
                linewidth=0.5,
            )

        ax.set_ylabel(cfg["ylabel"], fontsize=fontsize)
        ax.set_xlabel("Training steps", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)

    ax1.set_ylim(0.4, 0.75)

    line_handles, line_labels = ax1.get_legend_handles_labels()
    all_handles = line_handles + [baseline_handles["3B"], baseline_handles["7B"]]
    all_labels = line_labels + ["Q3B", "Q7B"]

    for ax in [ax1, ax2]:
        if ax.get_legend():
            ax.get_legend().remove()

    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        ncol=4,
        fontsize=fontsize,
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
        handlelength=3,
    )

    plt.tight_layout()
    return fig


def make_ablation_figure(
    ci_nacc_3b: pd.DataFrame,
    ci_nacc_3b_sce: pd.DataFrame,
    ci_nacc_3b_os: pd.DataFrame,
    ci_sce: pd.DataFrame,
    q3b_sce: tuple[float, float, float, float],
    q7b_sce: tuple[float, float, float, float],
) -> plt.Figure:
    fontsize = 10
    linewidth = 1.5
    markersize = 4

    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharey=True)

    panel_titles = ["LUNAR-OS-SCe", "LUNAR-OS", "LUNAR-SCe", "LUNAR"]
    ci_list = [ci_nacc_3b, ci_nacc_3b_sce, ci_nacc_3b_os, ci_sce]

    markers = {
        "External validation\n(All other cohorts)": "^",
        "Internal validation\n(NACC)": "o",
    }
    colors = {
        "Internal validation\n(NACC)": "#ff7f00",
        "External validation\n(All other cohorts)": "#377eb8",
    }

    hatch_styles = {"3B": "\\\\\\", "7B": "///"}
    linestyles = {"3B": "-", "7B": "--"}

    baselines = {"3B": q3b_sce, "7B": q7b_sce}
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        ci_df = ci_list[i].copy()

        if "in_distribution" in ci_df.columns:
            for label, group_df in ci_df.groupby("in_distribution"):
                group_df_sub = select_every_200_and_last(group_df, step_col="training_steps")
                x = group_df_sub["training_steps"].to_numpy()
                y = group_df_sub["point"].to_numpy()
                yerr = np.vstack([y - group_df_sub["low"].to_numpy(), group_df_sub["high"].to_numpy() - y])

                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    linewidth=linewidth,
                    marker=markers.get(label, "x"),
                    markersize=markersize,
                    alpha=0.9,
                    color=colors.get(label),
                    label=label if i == 0 else None,
                )

        for size, (mean, _bs_mean, low, high) in baselines.items():
            ax.axhline(y=mean, color="black", linewidth=0.5, linestyle=linestyles[size])
            ax.axhspan(
                low,
                high,
                alpha=0.5,
                hatch=hatch_styles[size],
                edgecolor="black",
                facecolor="darkgray",
                zorder=0,
                linewidth=0.5,
            )

        ax.set_title(panel_titles[i], fontsize=fontsize + 1)
        ax.set_ylabel("Macro-averaged\naccuracy" if i % 2 == 0 else "", fontsize=fontsize)
        ax.set_xlabel("Training steps" if i in [2, 3] else "", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)

    baseline_handles = {
        size: (
            mlines.Line2D([], [], color="black", linewidth=0.5, linestyle=linestyles[size]),
            mpatches.Patch(facecolor="darkgray", edgecolor="black", hatch=hatch_styles[size], alpha=0.4),
        )
        for size in ["3B", "7B"]
    }

    line_handles, line_labels = axes_flat[0].get_legend_handles_labels()
    all_handles = line_handles + [baseline_handles["3B"], baseline_handles["7B"]]
    all_labels = line_labels + ["Q3B", "Q7B"]

    for ax in axes_flat:
        ax.set_ylim(0.4, 0.8)
        if ax.get_legend():
            ax.get_legend().remove()

    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        ncol=4,
        fontsize=fontsize,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
        handlelength=3,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    return fig


def make_per_task_figure(
    df_sce_all: pd.DataFrame,
    base_values_sce: dict[str, dict[str, dict[str, float]]],
) -> plt.Figure:
    benchmark_label_map = {
        "test_cog": "COG",
        "test_etpr": "ETPR",
        "test_pet": "PET",
        "test_csf": "CSF",
        "test_dat": "DAT",
        "test_np_one": "NP_ONE",
        "test_np_mixed": "NP_MIXED",
    }
    benchmarks = list(benchmark_label_map.keys())
    n_bench = len(benchmarks)
    ncols = 2
    nrows = math.ceil(n_bench / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 8), sharey=True)

    fontsize = 8
    linewidth = 1.5
    markersize = 4

    markers = {
        "External validation\n(All other cohorts)": "^",
        "Internal validation\n(NACC)": "o",
    }
    colors = {
        "Internal validation\n(NACC)": "#ff7f00",
        "External validation\n(All other cohorts)": "#377eb8",
    }
    hatch_styles = {"3B": "\\\\\\", "7B": "///"}
    linestyles = {"3B": "-", "7B": "--"}

    for idx, benchmark in enumerate(benchmarks):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col] if nrows > 1 else axes[col]

        base_sce = base_values_sce.get(benchmark, {})

        ci_bench = macro_ci_over_training_steps_row_bootstrap(
            df_sce_all.loc[df_sce_all["benchmark"] == benchmark],
            model_pattern="NACC",
            # id_col="ID",
            n_boot=1000,
            seed=0,
        ).sort_values(["training_steps", "in_distribution"])

        if "in_distribution" in ci_bench.columns:
            for label, group_df in ci_bench.groupby("in_distribution"):
                x = group_df["training_steps"].to_numpy()
                y = group_df["point"].to_numpy()
                yerr = np.vstack([y - group_df["low"].to_numpy(), group_df["high"].to_numpy() - y])

                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    linewidth=linewidth,
                    marker=markers.get(label, "x"),
                    markersize=markersize,
                    alpha=0.9,
                    color=colors.get(label),
                    label=label if idx == 0 else None,
                )

        for size in ["3B", "7B"]:
            cur = base_sce.get(size)
            if cur is not None:
                ax.axhline(
                    y=cur["mean"],
                    color="black",
                    linewidth=0.9,
                    linestyle=linestyles[size],
                    alpha=0.8,
                )
                ax.axhspan(
                    cur["low"],
                    cur["high"],
                    alpha=0.18,
                    facecolor="darkgray",
                    edgecolor="black",
                    hatch=hatch_styles[size],
                    zorder=0,
                )

        ax.set_title(benchmark_label_map.get(benchmark, benchmark), fontsize=fontsize + 1, fontweight="bold")
        ax.set_xlabel("Training steps", fontsize=fontsize)
        ax.set_ylabel("Macro-averaged\naccuracy" if col == 0 else "", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize - 1)
        ax.tick_params(axis="y", labelsize=fontsize - 1)

    total_axes = nrows * ncols
    for idx2 in range(len(benchmarks), total_axes):
        ax = axes[idx2 // ncols, idx2 % ncols] if nrows > 1 else axes[idx2 % ncols]
        ax.axis("off")

    baseline_handles = {
        size: (
            mlines.Line2D([], [], color="black", linewidth=0.9, linestyle=linestyles[size], alpha=0.8),
            mpatches.Patch(facecolor="darkgray", edgecolor="black", hatch=hatch_styles[size], alpha=0.4),
        )
        for size in ["3B", "7B"]
    }

    first_ax = axes[0, 0] if nrows > 1 else axes[0]
    line_handles, line_labels = first_ax.get_legend_handles_labels()
    all_handles = line_handles + [baseline_handles["3B"], baseline_handles["7B"]]
    all_labels = line_labels + ["Q3B", "Q7B"]

    axs = axes.flatten() if nrows > 1 else axes
    for ax in axs:
        if ax.get_legend():
            ax.get_legend().remove()

    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        ncol=4,
        fontsize=fontsize,
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
        handlelength=3,
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


def compute_baselines_sce(df_sce_all: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    base_values_sce: dict[str, dict[str, dict[str, float]]] = {}

    for model_size, model_name in [("3B", "Qwen2.5-3B-Instruct"), ("7B", "Qwen2.5-7B-Instruct")]:
        benches = (
            df_sce_all.loc[df_sce_all["model"] == model_name, "benchmark"]
            .dropna()
            .unique()
            .tolist()
        )

        for bench in benches:
            df_slice = df_sce_all.loc[
                (df_sce_all["model"] == model_name) & (df_sce_all["benchmark"] == bench)
            ].copy()

            mean, bs_mean, low, high = bootstrap_macro_cell_mean_row_level(
                df_slice,
                model_name=model_name,
                n_boot=N_BOOT,
                seed=0,
                id_col="ID",
            )

            base_values_sce.setdefault(bench, {})
            base_values_sce[bench][model_size] = {"mean": mean, "bs_mean": bs_mean, "low": low, "high": high}

    return base_values_sce


def main() -> None:
    # SFT / SCE main curves
    print("Loading SFT data...")
    df_sft_all = load_data(RES_PATH, include_model="NACC-3B-OS-SFT-ES")
    df_sft = df_sft_all.loc[
        ~df_sft_all["benchmark"].isin(["test_mci", "test_np_mixed", "test_np", "test_ftld"])
    ].copy()

    print("Loading SCE data...")
    df_sce_all = load_data(RES_PATH, include_model="NACC-3B-OS-SCE")
    df_sce = df_sce_all.loc[
        ~df_sce_all["benchmark"].isin(["test_mci", "test_np_mixed", "test_np", "test_ftld"])
    ].copy()
    
    # # Save INCLUDED_FILES to CSV for inspection
    # for model_key, file_list in INCLUDED_FILES.items():
    #     file_paths = [str(f) for f in file_list]
    #     df_files = pd.DataFrame({"model": model_key, "file_path": file_paths})
    #     out_csv = f"training_curve_files/INCLUDED_FILES_{model_key}.csv"
    #     df_files.to_csv(out_csv, index=False)
    #     print(f"Saved INCLUDED_FILES for {model_key} to {out_csv}")
        
    # for model_key, file_list in IGNORED_FILES.items():
    #     file_paths = [str(f) for f in file_list]
    #     df_files = pd.DataFrame({"model": model_key, "file_path": file_paths})
    #     out_csv = f"training_curve_files/IGNORED_FILES_{model_key}.csv"
    #     df_files.to_csv(out_csv, index=False)
    #     print(f"Saved IGNORED_FILES for {model_key} to {out_csv}")
        
    # raise ValueError

    boot_fn = bootstrap_macro_cell_mean_row_level

    q3b_mean_sft, q3b_bs_mean_sft, q3b_low_sft, q3b_high_sft = boot_fn(
        df_sft, "Qwen2.5-3B-Instruct", n_boot=N_BOOT, id_col="ID"
    )
    q7b_mean_sft, q7b_bs_mean_sft, q7b_low_sft, q7b_high_sft = boot_fn(
        df_sft, "Qwen2.5-7B-Instruct", n_boot=N_BOOT, id_col="ID"
    )

    q3b_mean_sce, q3b_bs_mean_sce, q3b_low_sce, q3b_high_sce = boot_fn(
        df_sce, "Qwen2.5-3B-Instruct", n_boot=N_BOOT, id_col="ID"
    )
    q7b_mean_sce, q7b_bs_mean_sce, q7b_low_sce, q7b_high_sce = boot_fn(
        df_sce, "Qwen2.5-7B-Instruct", n_boot=N_BOOT, id_col="ID"
    )

    ci_sft = macro_ci_over_training_steps_row_bootstrap(df_sft, n_boot=N_BOOT, seed=0)
    ci_sce = macro_ci_over_training_steps_row_bootstrap(df_sce, n_boot=N_BOOT, seed=0)
    
    # Save the desired variables to files for inspection or future use
    # import pickle

    # # Save ci_sft and ci_sce as CSV files
    # ci_sft.to_csv("../figures/ci_sft.csv", index=False)
    # ci_sce.to_csv("../figures/ci_sce.csv", index=False)

    # # Save scalar stats as a dictionary and pickle them
    # stats_dict = {
    #     "q3b_mean_sft": q3b_mean_sft,
    #     "q3b_bs_mean_sft": q3b_bs_mean_sft,
    #     "q3b_low_sft": q3b_low_sft,
    #     "q3b_high_sft": q3b_high_sft,
    #     "q7b_mean_sft": q7b_mean_sft,
    #     "q7b_bs_mean_sft": q7b_bs_mean_sft,
    #     "q7b_low_sft": q7b_low_sft,
    #     "q7b_high_sft": q7b_high_sft,
    #     "q3b_mean_sce": q3b_mean_sce,
    #     "q3b_bs_mean_sce": q3b_bs_mean_sce,
    #     "q3b_low_sce": q3b_low_sce,
    #     "q3b_high_sce": q3b_high_sce,
    #     "q7b_mean_sce": q7b_mean_sce,
    #     "q7b_bs_mean_sce": q7b_bs_mean_sce,
    #     "q7b_low_sce": q7b_low_sce,
    #     "q7b_high_sce": q7b_high_sce,
    # }
    # with open("../figures/scalar_stats.pkl", "wb") as f:
    #     pickle.dump(stats_dict, f)
    
    # raise ValueError

    fig_main = make_main_training_curve_figure(
        ci_sft,
        ci_sce,
        (q3b_mean_sft, q3b_bs_mean_sft, q3b_low_sft, q3b_high_sft),
        (q7b_mean_sft, q7b_bs_mean_sft, q7b_low_sft, q7b_high_sft),
        (q3b_mean_sce, q3b_bs_mean_sce, q3b_low_sce, q3b_high_sce),
        (q7b_mean_sce, q7b_bs_mean_sce, q7b_low_sce, q7b_high_sce),
    )
    fig_main.savefig(
        "../figures/fig2_test_perf_over_training_trail_bootstrap.pdf",
        dpi=200,
        format="pdf",
        bbox_inches="tight",
    )

    # Ablations
    print("Loading ablation data...")
    df_nacc_3b = load_data(RES_PATH, include_model="/NACC-3B/")
    df_nacc_3b = df_nacc_3b.loc[
        ~df_nacc_3b["benchmark"].isin(["test_mci", "test_np_mixed", "test_np", "test_ftld"])
    ].copy()

    df_nacc_3b_sce = load_data(RES_PATH, include_model="/NACC-3B-SCE/")
    df_nacc_3b_sce = df_nacc_3b_sce.loc[
        ~df_nacc_3b_sce["benchmark"].isin(["test_mci", "test_np_mixed", "test_np", "test_ftld"])
    ].copy()

    df_nacc_3b_os = load_data(RES_PATH, include_model="/NACC-3B-OS/")
    df_nacc_3b_os = df_nacc_3b_os.loc[
        ~df_nacc_3b_os["benchmark"].isin(["test_mci", "test_np_mixed", "test_np", "test_ftld"])
    ].copy()

    ci_nacc_3b = macro_ci_over_training_steps_row_bootstrap(df_nacc_3b, n_boot=N_BOOT, seed=0)
    ci_nacc_3b_sce = macro_ci_over_training_steps_row_bootstrap(df_nacc_3b_sce, n_boot=N_BOOT, seed=0)
    ci_nacc_3b_os = macro_ci_over_training_steps_row_bootstrap(df_nacc_3b_os, n_boot=N_BOOT, seed=0)

    fig_ablation = make_ablation_figure(
        ci_nacc_3b,
        ci_nacc_3b_sce,
        ci_nacc_3b_os,
        ci_sce,
        (q3b_mean_sce, q3b_bs_mean_sce, q3b_low_sce, q3b_high_sce),
        (q7b_mean_sce, q7b_bs_mean_sce, q7b_low_sce, q7b_high_sce),
    )
    fig_ablation.savefig(
        "../figures/sup_test_perf_over_training_trial_bootstrap_ablations.pdf",
        dpi=200,
        format="pdf",
        bbox_inches="tight",
    )

    # Per-task curves (SCE only)
    df_sce_all_tasks = df_sce_all.loc[~df_sce_all["benchmark"].isin(["test_mci", "test_ftld"])].copy()
    base_values_sce = compute_baselines_sce(df_sce_all_tasks)
    fig_per_task = make_per_task_figure(df_sce_all_tasks, base_values_sce)
    fig_per_task.savefig(
        "../figures/sup_test_perf_over_training_trial_bootstrap_sep_benchmarks.pdf",
        dpi=200,
        format="pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()