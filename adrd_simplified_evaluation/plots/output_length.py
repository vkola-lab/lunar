import os
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from transformers import AutoTokenizer

plt.rcParams["font.family"] = "Arial"
mpl.rcParams["hatch.linewidth"] = 0.5
sns.set_style("whitegrid")

# -----------------
# Configuration
# -----------------

COMPUTE_TOKEN_COUNTS = False  # True = compute and save; False = load from CSV
TOKEN_COUNTS_CSV = "token_counts.csv"

BASE_RESULTS_DIR = "/projectnb/vkolagrp/projects/adrd_foundation_model/results"
NIFD_PATH = Path(f"{BASE_RESULTS_DIR}/NIFD")
NACC_PATH = Path(f"{BASE_RESULTS_DIR}/NACC")
ADNI_PATH = Path(f"{BASE_RESULTS_DIR}/ADNI")
PPMI_PATH = Path(f"{BASE_RESULTS_DIR}/PPMI")
BRAINLAT_PATH = Path(f"{BASE_RESULTS_DIR}/brainlat")

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
    "NACC-3B": "LUNAR-OS-SCe",
    "NACC-3B-OS": "LUNAR-SCe",
    "NACC-3B-SCE": "LUNAR-OS",
    "NACC-3B-OS-SCE": "LUNAR",
}

model_order = [
    "Q3B",
    "LUNAR-OS-SCe",
    "LUNAR-SCe",
    "LUNAR-OS",
    "LUNAR",
    "Q7B",
]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")


# -----------------
# Data loading / token counting
# -----------------

def load_answers(dir_path: Path, dataset_name: str) -> pd.DataFrame:
    """
    Load all parquet files from a dataset directory into a DataFrame,
    keeping only a small set of columns for speed.
    """
    fpaths = list(dir_path.rglob("*.parquet"))
    dfs: list[pd.DataFrame] = []

    cols_to_read = [
        "ID",
        "ground_truth",
        "prediction",
        "generated_text",
        "finish_reason",
    ]

    for fpath in tqdm(fpaths, desc=f"Loading {dataset_name}"):
        model = fpath.parent.name.split("-", 3)[-1]
        benchmark = fpath.parent.parent.name.split("_", 1)[-1].upper()

        # Skip clinical benchmarks not used here
        if benchmark in ["MCI", "NP", "NP_MIXED", "FTLD"]:
            print(f"Skipping {fpath.parent.parent.name}")
            continue

        df = pd.read_parquet(fpath, columns=cols_to_read)
        df = df.assign(model=model, benchmark=benchmark)
        df["correct"] = (df["ground_truth"] == df["prediction"]).astype(int)
        df["dataset"] = dataset_name
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=cols_to_read + ["model", "benchmark", "correct", "dataset"])

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["model"] = df_all["model"].replace(model_name_dict)

    group_cols = ["benchmark", "model", "prediction", "ground_truth"]
    for col in group_cols:
        df_all[col] = pd.Categorical(df_all[col])

    return df_all


def count_tokens_fast(
    df: pd.DataFrame,
    text_column: str,
    batch_size: int = 1000,
) -> list[int]:
    """
    Fast token counting using the tokenizer's built-in parallelism.
    """
    texts = df[text_column].fillna("").tolist()
    token_counts: list[int] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
        batch = texts[i : i + batch_size]
        encoded = TOKENIZER(
            batch,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        token_counts.extend(len(ids) for ids in encoded["input_ids"])

    return token_counts


# -----------------
# Stats utilities
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
    """Build upper-triangular p-value matrix text for annotation."""
    n = len(matrix_content)
    row_labels = matrix_content.index.tolist()
    col_labels = matrix_content.columns.tolist()[1:]  # skip first (A)

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


def annotate_pmatrix(ax, matrix_content, xy=(0.995, 0.98), title="p-values", fontsize=8):
    matrix_text_lines = get_annotate_matrix(matrix_content)
    matrix_text_lines.insert(0, title)
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
        fontsize=fontsize - 2,
        family="monospace",
    )


def pairwise_tests_matrix(df, model_order, model_to_letter, value_col="token_count"):
    """Welch t-tests with BH correction; returns letter-labelled p-value matrix."""
    models = [str(m) for m in model_order]
    letter_labels = [model_to_letter[m] for m in models]

    pairs = [
        (models[i], models[j])
        for i in range(len(models))
        for j in range(i + 1, len(models))
    ]

    p_values: list[float] = []
    median_diffs: list[float] = []

    for m1, m2 in pairs:
        d1 = df[df["model"] == m1][value_col].values
        d2 = df[df["model"] == m2][value_col].values
        _, p_val = stats.ttest_ind(d1, d2, equal_var=False)
        p_values.append(p_val)
        median_diffs.append(np.median(d1) - np.median(d2))

    _, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

    mat = pd.DataFrame(np.nan, index=letter_labels, columns=letter_labels)
    for idx, (m1, m2) in enumerate(pairs):
        l1, l2 = model_to_letter[m1], model_to_letter[m2]
        mat.loc[l1, l2] = p_adjusted[idx]
        mat.loc[l2, l1] = p_adjusted[idx]
    np.fill_diagonal(mat.values, np.nan)

    return mat, pairs, p_adjusted, median_diffs


# -----------------
# Plotting
# -----------------

# def plot_token_distribution(
#     df: pd.DataFrame,
#     model_order: list[str],
#     value_col: str = "token_count",
#     finish_reason_filter: str | None = "stop",
#     figsize=(2.3, 2),
#     fontsize: int = 7,
#     ylim=(10, 6e4),
#     ylabel: str = "Output length (tokens)",
#     palette: str = "colorblind",
# ):
#     """
#     Half-violin + boxplot per model with a p-value matrix annotation and a
#     dedicated legend panel below the main axes.
#     """
#     _df = df.copy()
#     if finish_reason_filter is not None and "finish_reason" in _df.columns:
#         _df = _df[_df["finish_reason"] == finish_reason_filter]

#     colors = sns.color_palette(palette, n_colors=len(model_order))
#     color_map = {model: colors[i] for i, model in enumerate(model_order)}

#     letters = [chr(65 + i) for i in range(len(model_order))]
#     model_to_letter = {model: letter for model, letter in zip(model_order, letters)}

#     fig = plt.figure(figsize=figsize)
#     gs = fig.add_gridspec(2, 1, height_ratios=[20, 1], hspace=0.3)
#     ax = fig.add_subplot(gs[0])
#     ax_legend = fig.add_subplot(gs[1])
#     ax_legend.axis("off")

#     positions = np.arange(len(model_order))
#     violin_width = 0.6
#     box_width = 0.3

#     legend_handles = []
#     from matplotlib.patches import Patch

#     for i, model in enumerate(model_order):
#         model_data = _df[_df["model"] == model][value_col].values
#         color = color_map[model]

#         parts = ax.violinplot(
#             [model_data],
#             positions=[i],
#             widths=violin_width,
#             showmeans=False,
#             showextrema=False,
#             showmedians=False,
#         )
#         for pc in parts["bodies"]:
#             pc.set_facecolor(color)
#             pc.set_alpha(0.6)
#             pc.set_edgecolor("black")
#             pc.set_linewidth(0.8)
#             m = np.mean(pc.get_paths()[0].vertices[:, 0])
#             pc.get_paths()[0].vertices[:, 0] = np.clip(
#                 pc.get_paths()[0].vertices[:, 0], m, np.inf
#             )

#         ax.boxplot(
#             [model_data],
#             positions=[i - violin_width / 2 - box_width / 2 + 0.175],
#             widths=box_width,
#             patch_artist=True,
#             showfliers=False,
#             medianprops=dict(color="black", linewidth=0.8),
#             boxprops=dict(facecolor=color, edgecolor="black", alpha=0.7),
#             whiskerprops=dict(color="black", linewidth=0.8),
#             capprops=dict(color="black", linewidth=0.8),
#         )

#         legend_handles.append(
#             Patch(
#                 facecolor=color,
#                 edgecolor="black",
#                 alpha=0.8,
#                 label=f"{model} ({model_to_letter[model]})",
#             )
#         )

#     p_matrix, pairs, p_adjusted, median_diffs = pairwise_tests_matrix(
#         _df, model_order, model_to_letter, value_col=value_col
#     )
#     annotate_pmatrix(
#         ax,
#         p_matrix,
#         xy=(0.995, 0.98),
#         title="p-values               ",
#         fontsize=fontsize,
#     )

#     ax.set_xticks(positions)
#     ax.set_xticklabels([])  # legend carries labels
#     ax.set_ylabel(ylabel, fontsize=fontsize)
#     ax.set_yscale("log")
#     ax.grid(axis="y", alpha=1, linestyle="-")
#     ax.set_xlim(-0.5, len(model_order) - 0.65)
#     ax.tick_params(axis="both", labelsize=fontsize)
#     ax.set_ylim(*ylim)

#     ncol = math.ceil(len(model_order) / 2)
#     ax_legend.legend(
#         handles=legend_handles,
#         loc="center",
#         ncol=ncol,
#         frameon=False,
#         fontsize=fontsize - 2,
#         handlelength=1.5,
#         handleheight=1.5,
#         bbox_to_anchor=(0.5, 0),
#     )

#     plt.tight_layout()

#     # Optional: print stats (commented to avoid clutter)
#     # print("\nStatistical Test Results (Benjamini-Hochberg corrected):")
#     # for idx, (m1, m2) in enumerate(pairs):
#     #     sig = map_values(p_adjusted[idx])
#     #     l1, l2 = model_to_letter[m1], model_to_letter[m2]
#     #     diff = median_diffs[idx]
#     #     print(
#     #         f"{m1} ({l1}) vs {m2} ({l2}): "
#     #         f"Δ median = {diff:+.1f}, p = {p_adjusted[idx]:.4f} ({sig})"
#     #     )

#     return fig, [ax, ax_legend]


def plot_token_distribution(
    df: pd.DataFrame,
    model_order: list[str],
    value_col: str = "token_count",
    finish_reason_filter: str | None = "stop",
    figsize=(2.3, 2),
    fontsize: int = 7,
    ylim=(10, 6e4),
    ylabel: str = "Output length (tokens)",
    palette: str = "colorblind",
    linewidth: float = 0.5,
):
    """
    Half-violin + boxplot per model with a p-value matrix annotation and a
    dedicated legend panel below the main axes.
    """
    _df = df.copy()
    if finish_reason_filter is not None and "finish_reason" in _df.columns:
        _df = _df[_df["finish_reason"] == finish_reason_filter]

    colors = sns.color_palette(palette, n_colors=len(model_order))
    color_map = {model: colors[i] for i, model in enumerate(model_order)}

    letters = [chr(65 + i) for i in range(len(model_order))]
    model_to_letter = {model: letter for model, letter in zip(model_order, letters)}

    hatch_patterns = ["///", "|||", "---", "+++", "xxx", "ooo"]
    n_hatch = len(hatch_patterns)
    hatches = [hatch_patterns[i % n_hatch] for i in range(len(model_order))]
    hatch_map = {model: hatches[i] for i, model in enumerate(model_order)}

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[20, 1], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis("off")

    positions = np.arange(len(model_order))
    violin_width = 0.6
    box_width = 0.3

    legend_handles = []
    from matplotlib.patches import Patch

    for i, model in enumerate(model_order):
        model_data = _df[_df["model"] == model][value_col].values
        color = color_map[model]
        hatch = hatch_map[model]

        parts = ax.violinplot(
            [model_data],
            positions=[i],
            widths=violin_width,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.85)
            pc.set_edgecolor("black")
            pc.set_linewidth(linewidth)
            pc.set_hatch(hatch)
            m = np.mean(pc.get_paths()[0].vertices[:, 0])
            pc.get_paths()[0].vertices[:, 0] = np.clip(
                pc.get_paths()[0].vertices[:, 0], m, np.inf
            )

        ax.boxplot(
            [model_data],
            positions=[i - violin_width / 2 - box_width / 2 + 0.175],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=linewidth),
            boxprops=dict(facecolor=color, edgecolor="black", alpha=0.85, linewidth=linewidth),#, hatch=hatch),
            whiskerprops=dict(color="black", linewidth=linewidth),
            capprops=dict(color="black", linewidth=linewidth),
        )

        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor="black",
                alpha=0.85,
                linewidth=linewidth,
                hatch=hatch,
                label=f"{model} ({model_to_letter[model]})",
            )
        )

    p_matrix, pairs, p_adjusted, median_diffs = pairwise_tests_matrix(
        _df, model_order, model_to_letter, value_col=value_col
    )
    annotate_pmatrix(
        ax,
        p_matrix,
        xy=(0.995, 0.98),
        title="p-values               ",
        fontsize=fontsize,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([])  # legend carries labels
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=1, linestyle="-")
    ax.set_xlim(-0.5, len(model_order) - 0.65)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.set_ylim(*ylim)

    ncol = math.ceil(len(model_order) / 2)
    ax_legend.legend(
        handles=legend_handles,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize=fontsize - 2,
        handlelength=1.5,
        handleheight=1.5,
        bbox_to_anchor=(0.5, 0),
    )

    plt.tight_layout()

    return fig, [ax, ax_legend]


# -----------------
# Main
# -----------------

def main() -> None:
    if COMPUTE_TOKEN_COUNTS:
        print("Computing token counts from parquet files...")

        nifd = load_answers(NIFD_PATH, dataset_name="NIFD")
        adni = load_answers(ADNI_PATH, dataset_name="ADNI")
        nacc = load_answers(NACC_PATH, dataset_name="NACC")
        ppmi = load_answers(PPMI_PATH, dataset_name="PPMI")
        brainlat = load_answers(BRAINLAT_PATH, dataset_name="BrainLat")

        df = pd.concat([nifd, adni, nacc, ppmi, brainlat], ignore_index=True)
        df = df[
            [
                "dataset",
                "benchmark",
                "model",
                "ID",
                "generated_text",
                "finish_reason",
                "correct",
            ]
        ]

        df = df[df["model"].isin(model_order)].reset_index(drop=True)

        print("Counting tokens...")
        df["token_count"] = count_tokens_fast(df, "generated_text", batch_size=1000)

        print(f"Saving token counts to {TOKEN_COUNTS_CSV}...")
        df.to_csv(TOKEN_COUNTS_CSV, index=False)
    else:
        print(f"Loading token counts from {TOKEN_COUNTS_CSV}...")
        df = pd.read_csv(TOKEN_COUNTS_CSV)

    fig, _ = plot_token_distribution(
        df=df,
        model_order=model_order,
        value_col="token_count",
        finish_reason_filter="stop",
        figsize=(2.3, 2),
        fontsize=7,
        ylim=(10, 6e4),
        ylabel="Output length (tokens)",
        palette="colorblind",
    )

    out_path = "../figures/fig2_output_length_all.pdf"
    fig.savefig(out_path, dpi=200, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()