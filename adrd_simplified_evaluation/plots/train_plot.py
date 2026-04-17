
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')


plt.rcParams["font.family"] = "Arial"
sns.set_theme(style="whitegrid", context="paper")


ENTROPY_CSV = "wandb/wandb_export_entropy.csv"
LENGTH_CSV = "wandb/wandb_export_mean_length.csv"
REWARD_CSV = "wandb/wandb_export_reward.csv"
REWARD_STD_CSV = "wandb/wandb_export_reward_std.csv"

ENTROPY_KEY = "train/entropy"
LENGTH_KEY = "mean_length"
REWARD_KEY = "train/reward"
REWARD_STD_KEY = "train/reward_std"


COLUMN_MAPPING = {
    "qwen2.5 3B nacc inc oversample dedup": "LUNAR-OS-SCe",
    "qwen2.5 3B nacc inc oversample dedup sce tanh": "LUNAR-OS",
    "qwen2.5 3B nacc inc oversample": "LUNAR-SCe",
    "qwen2.5 3B nacc inc oversample sce tanh cont": "LUNAR",
}

ORDER = ["LUNAR-OS-SCe", "LUNAR-OS", "LUNAR-SCe", "LUNAR"]

FONTSIZE = 7
LINEWIDTH = 1
MARKERSIZE = 4

DASHES = {
    "LUNAR": "",
    "LUNAR-SCe": (2, 2),
    "LUNAR-OS": (5, 2),
    "LUNAR-OS-SCe": (2, 1, 1, 1),
}

MARKERS = {
    "LUNAR": "o",
    "LUNAR-SCe": "s",
    "LUNAR-OS": "^",
    "LUNAR-OS-SCe": "D",
}


def load_and_clean_metric_csv(path: str, metric_key: str) -> pd.DataFrame:
    """
    Load a W&B export CSV, keep only step + metric columns, unify variants, and rename columns.
    """
    df = pd.read_csv(path)

    df["train/global_step"] = pd.to_numeric(df["train/global_step"], errors="coerce")

    metric_cols = [
        c for c in df.columns if metric_key in c and "__MIN" not in c and "__MAX" not in c
    ]
    if not metric_cols:
        raise ValueError(f"No columns found for metric '{metric_key}' in {path}")

    df = df[["train/global_step"] + metric_cols]

    rename_map = {}
    for c in metric_cols:
        run_name = c.split("-")[0].strip()
        rename_map[c] = run_name
    df = df.rename(columns=rename_map)

    cont_col = "qwen2.5 3B nacc inc oversample sce tanh cont"
    base_col = "qwen2.5 3B nacc inc oversample sce tanh"

    df[cont_col] = df[cont_col].combine_first(df[base_col])
    df = df.drop(columns=[base_col])
    df = df.rename(columns=COLUMN_MAPPING)

    return df


def to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    return df.melt(
        id_vars="train/global_step",
        value_vars=ORDER,
        var_name="Variant",
        value_name=value_name,
    )


def smooth_long_ema_stop_at_end(
    df_long: pd.DataFrame,
    value_col: str,
    step_col: str = "train/global_step",
    alpha: float = 0.0005,
) -> pd.DataFrame:
    df_long = df_long.sort_values(["Variant", step_col]).copy()

    def ema_one_variant(g: pd.DataFrame) -> pd.Series:
        values = g[value_col].to_numpy()
        out = np.full(len(values), np.nan, dtype=float)

        start = np.argmax(~np.isnan(values)) if np.any(~np.isnan(values)) else None
        if start is None:
            return pd.Series(out, index=g.index)

        ema = values[start]
        out[start] = ema

        for i in range(start + 1, len(values)):
            if np.isnan(values[i]):
                out[i] = np.nan
                continue
            ema = (1.0-alpha) * ema + alpha * values[i]
            out[i] = ema

        return pd.Series(out, index=g.index)

    df_long[value_col] = df_long.groupby("Variant", group_keys=False).apply(ema_one_variant)
    return df_long


def load_all_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_entropy = load_and_clean_metric_csv(ENTROPY_CSV, metric_key=ENTROPY_KEY)
    df_length = load_and_clean_metric_csv(LENGTH_CSV, metric_key=LENGTH_KEY)
    df_reward = load_and_clean_metric_csv(REWARD_CSV, metric_key=REWARD_KEY)
    df_reward_std = load_and_clean_metric_csv(REWARD_STD_CSV, metric_key=REWARD_STD_KEY)
    return df_entropy, df_length, df_reward, df_reward_std


def prepare_long_dfs(
    df_entropy: pd.DataFrame,
    df_length: pd.DataFrame,
    df_reward: pd.DataFrame,
    df_reward_std: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entropy_long = to_long(df_entropy, "Entropy")
    length_long = to_long(df_length, "Mean response length")
    reward_long = to_long(df_reward, "Group rewards")
    reward_std_long = to_long(df_reward_std, "Group rewards std")

    entropy_long = smooth_long_ema_stop_at_end(
        entropy_long,
        value_col="Entropy",
        alpha=0.01,
    )
    length_long = smooth_long_ema_stop_at_end(
        length_long,
        value_col="Mean response length",
        alpha=0.01,
    )
    reward_long = smooth_long_ema_stop_at_end(
        reward_long,
        value_col="Group rewards",
        alpha=0.01,
    )
    reward_std_long = smooth_long_ema_stop_at_end(
        reward_std_long,
        value_col="Group rewards std",
        alpha=0.01,
    )

    return entropy_long, length_long, reward_long, reward_std_long


def make_figure(
    entropy_long: pd.DataFrame,
    length_long: pd.DataFrame,
    reward_long: pd.DataFrame,
    reward_std_long: pd.DataFrame,
    figsize: tuple,
) -> plt.Figure:
    fig, axes = plt.subplots(
        2, 2, figsize=figsize, sharex=True, constrained_layout=False
    )

    sns.lineplot(
        data=reward_long,
        x="train/global_step",
        y="Group rewards",
        hue="Variant",
        hue_order=ORDER,
        palette="colorblind",
        style="Variant",
        style_order=ORDER,
        dashes=DASHES,
        markers=MARKERS,
        markevery=100,
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        ax=axes[0, 0],
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if axes[0, 0].legend_ is not None:
        axes[0, 0].legend_.remove()
    axes[0, 0].set_ylabel("Mean group\nrewards", fontsize=FONTSIZE)

    sns.lineplot(
        data=reward_std_long,
        x="train/global_step",
        y="Group rewards std",
        hue="Variant",
        hue_order=ORDER,
        palette="colorblind",
        style="Variant",
        style_order=ORDER,
        dashes=DASHES,
        markers=MARKERS,
        markevery=100,
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        ax=axes[0, 1],
        legend=False,
    )
    axes[0, 1].set_ylabel("Group rewards\nstd", fontsize=FONTSIZE)

    sns.lineplot(
        data=entropy_long,
        x="train/global_step",
        y="Entropy",
        hue="Variant",
        hue_order=ORDER,
        palette="colorblind",
        style="Variant",
        style_order=ORDER,
        dashes=DASHES,
        markers=MARKERS,
        markevery=100,
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        ax=axes[1, 0],
        legend=False,
    )
    axes[1, 0].set_ylabel("Mean token\nentropy", fontsize=FONTSIZE)

    sns.lineplot(
        data=length_long,
        x="train/global_step",
        y="Mean response length",
        hue="Variant",
        hue_order=ORDER,
        palette="colorblind",
        style="Variant",
        style_order=ORDER,
        dashes=DASHES,
        markers=MARKERS,
        markevery=100,
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        ax=axes[1, 1],
        legend=False,
    )
    axes[1, 1].set_ylabel("Mean response\nlength (tokens)", fontsize=FONTSIZE)

    axes[1, 0].set_xlabel("Training steps", fontsize=FONTSIZE)
    axes[1, 1].set_xlabel("Training steps", fontsize=FONTSIZE)

    for ax in axes.ravel():
        ax.tick_params(axis="both", labelsize=FONTSIZE)

    label_to_handle = {lab: h for h, lab in zip(handles, labels)}
    legend_handles = [label_to_handle[l] for l in ORDER if l in label_to_handle]
    legend_labels = [l for l in ORDER if l in label_to_handle]

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        ncol=len(legend_labels),
        frameon=False,
        fontsize=FONTSIZE,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def main(output_path: str = "../figures/fig2_train_plot.pdf", figsize=(4.3, 2.3)) -> None:
    print("Loading data")
    df_entropy, df_length, df_reward, df_reward_std = load_all_metrics()
    entropy_long, length_long, reward_long, reward_std_long = prepare_long_dfs(
        df_entropy, df_length, df_reward, df_reward_std
    )
    print("Making figure")
    fig = make_figure(entropy_long, length_long, reward_long, reward_std_long, figsize)
    fig.savefig(output_path, dpi=200, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main(output_path = "../figures/fig2_train_plot_bigger.pdf", figsize=(4.3, 2.7))
