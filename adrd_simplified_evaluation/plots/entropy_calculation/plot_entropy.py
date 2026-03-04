import math
import json
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "Arial"
mpl.rcParams["hatch.linewidth"] = 0.5
sns.set_style("whitegrid")

# -----------------
# Configuration
# -----------------

DATA_DIR = "./entropies"  # directory with your JSON files

# Map filenames to display names
key_name_dict = {
    "q3b": "Q3B",
    "oversample": "LUNAR-SCe",
    "oversample_dedup": "LUNAR-OS-SCe",
    "oversample_sce_tanh": "LUNAR",
    "oversample_dedup_sce_tanh": "LUNAR-OS",
    "q7b": "Q7B",
}

key_order = [
    "Q3B",
    "LUNAR-OS-SCe",
    "LUNAR-SCe",
    "LUNAR-OS",
    "LUNAR",
    "Q7B",
]

# -----------------
# Data loading
# -----------------

def load_entropy_data(data_dir: str) -> pd.DataFrame:
    """Load all JSON entropy files into a long-form DataFrame."""
    data = {}
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if os.path.isfile(fpath):
            try:
                with open(fpath, "r") as f:
                    data[fname.split(".")[0]] = json.load(f)
            except Exception as e:
                print(f"Skipping {fname}: {e}")

    rows = []
    for key, values in data.items():
        display_name = key_name_dict.get(key, key)
        for mean_val in values["mean"]:
            rows.append({"method": display_name, "mean_entropy": mean_val})

    return pd.DataFrame(rows)


# -----------------
# Stats utilities  (identical pattern to your existing code)
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
    n = len(matrix_content)
    row_labels = matrix_content.index.tolist()
    col_labels = matrix_content.columns.tolist()[1:]

    main = []
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
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black",
                  facecolor=(1, 1, 1, 0.0), lw=0.5),
        fontsize=fontsize - 2,
        family="monospace",
    )


def pairwise_tests_matrix(df, key_order, key_to_letter, value_col="mean_entropy"):
    keys = [str(k) for k in key_order]
    letter_labels = [key_to_letter[k] for k in keys]

    pairs = [
        (keys[i], keys[j])
        for i in range(len(keys))
        for j in range(i + 1, len(keys))
    ]

    p_values, median_diffs = [], []
    for k1, k2 in pairs:
        d1 = df[df["method"] == k1][value_col].values
        d2 = df[df["method"] == k2][value_col].values
        _, p_val = stats.ttest_ind(d1, d2, equal_var=False)
        p_values.append(p_val)
        median_diffs.append(np.median(d1) - np.median(d2))

    _, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

    mat = pd.DataFrame(np.nan, index=letter_labels, columns=letter_labels)
    for idx, (k1, k2) in enumerate(pairs):
        l1, l2 = key_to_letter[k1], key_to_letter[k2]
        mat.loc[l1, l2] = p_adjusted[idx]
        mat.loc[l2, l1] = p_adjusted[idx]
    np.fill_diagonal(mat.values, np.nan)

    return mat, pairs, p_adjusted, median_diffs


# -----------------
# Plotting
# -----------------

def plot_entropy_distribution(
    df: pd.DataFrame,
    key_order: list[str],
    value_col: str = "mean_entropy",
    figsize=(2.3, 2),
    fontsize: int = 7,
    ylim=None,
    ylabel: str = "Mean Entropy",
    palette: str = "colorblind",
    linewidth: float = 0.5,
):
    colors = sns.color_palette(palette, n_colors=len(key_order))
    color_map = {k: colors[i] for i, k in enumerate(key_order)}

    letters = [chr(65 + i) for i in range(len(key_order))]
    key_to_letter = {k: l for k, l in zip(key_order, letters)}

    hatch_patterns = ["///", "|||", "---", "+++", "xxx", "ooo"]
    hatch_map = {k: hatch_patterns[i % len(hatch_patterns)] for i, k in enumerate(key_order)}

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[20, 1], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis("off")

    violin_width = 0.6
    box_width = 0.3
    legend_handles = []

    for i, key in enumerate(key_order):
        key_data = df[df["method"] == key][value_col].values
        color = color_map[key]
        hatch = hatch_map[key]

        parts = ax.violinplot(
            [key_data], positions=[i], widths=violin_width,
            showmeans=False, showextrema=False, showmedians=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.85)
            pc.set_edgecolor("black")
            pc.set_linewidth(linewidth)
            pc.set_hatch(hatch)
            # half-violin (right side only)
            m = np.mean(pc.get_paths()[0].vertices[:, 0])
            pc.get_paths()[0].vertices[:, 0] = np.clip(
                pc.get_paths()[0].vertices[:, 0], m, np.inf
            )

        ax.boxplot(
            [key_data],
            positions=[i - violin_width / 2 - box_width / 2 + 0.175],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=linewidth),
            boxprops=dict(facecolor=color, edgecolor="black", alpha=0.7, linewidth=linewidth),
            whiskerprops=dict(color="black", linewidth=linewidth),
            capprops=dict(color="black", linewidth=linewidth),
        )

        legend_handles.append(
            Patch(facecolor=color, edgecolor="black", alpha=0.85,
                  hatch=hatch, linewidth=linewidth, label=f"{key} ({key_to_letter[key]})")
        )

    p_matrix, pairs, p_adjusted, median_diffs = pairwise_tests_matrix(
        df, key_order, key_to_letter, value_col=value_col
    )
    annotate_pmatrix(ax, p_matrix, xy=(0.995, 0.98),
                     title="p-values               ", fontsize=fontsize)

    ax.set_xticks(range(len(key_order)))
    ax.set_xticklabels([])
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid(axis="y", alpha=1, linestyle="-")
    ax.set_xlim(-0.5, len(key_order) - 0.65)
    ax.tick_params(axis="both", labelsize=fontsize)
    if ylim:
        ax.set_ylim(*ylim)

    ncol = math.ceil(len(key_order) / 2)
    # ax_legend.legend(
    #     handles=legend_handles, loc="center", ncol=ncol, frameon=False,
    #     fontsize=fontsize - 2, handlelength=1.5, handleheight=1.5,
    #     bbox_to_anchor=(0.5, 0),
    # )

    plt.tight_layout()
    return fig, [ax, ax_legend]


# -----------------
# Main
# -----------------

def main():
    df = load_entropy_data(DATA_DIR)

    fig, _ = plot_entropy_distribution(
        df=df,
        key_order=key_order,
        value_col="mean_entropy",
        figsize=(2.3, 1.8),
        fontsize=7,
        ylabel="Mean Entropy",
        palette="colorblind",
        ylim=(0, 2.5)
    )

    out_path = "../../figures/fig2_entropy_distribution.pdf"
    # os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, format="pdf", bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()