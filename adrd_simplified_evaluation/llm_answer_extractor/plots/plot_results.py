# plots/plot_results.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_comparison(df_scores, out_path="np.pdf", benchmark="Neuropath"):
    """
    Creates and saves a grouped barplot for pass@1 and cons@k.
    """
    df_long = df_scores.melt(id_vars='metric', var_name='model', value_name='score')
    
    # Sort models by pass@1 score
    pass_values = df_long[df_long["metric"] == "pass@1"].set_index("model")["score"]
    sorted_models = pass_values.sort_values().index.tolist()

    # Reconstruct sorted long-form dataframe
    df_sorted = pd.concat([
        pd.concat([
            df_long[(df_long["model"] == model) & (df_long["metric"] == "pass@1")],
            df_long[(df_long["model"] == model) & (df_long["metric"] == "cons@k")]
        ])
        for model in sorted_models
    ], ignore_index=True)

    # Palette setup
    # key_models = ['qwen3b', 'qwen7b', 'qwen14b', 'qwen32b', 'qwen72b'] 
    key_models = [model for model in df_sorted['model'].unique() if "drgrpo" not in model]
    palette = {}
    yellow_shades = sns.color_palette("Reds", len(key_models))
    for model, color in zip(key_models, yellow_shades):
        palette[model] = color

    other_models = [m for m in df_sorted['model'].unique() if m not in key_models and m != 'clinician']
    blue_shades = sns.color_palette("Blues", len(other_models))
    for model, color in zip(other_models, blue_shades):
        palette[model] = color

    # Plot
    plt.figure(figsize=(20, 10))
    ax = sns.barplot(data=df_sorted[df_sorted['model'] != 'clinician'],
                     x='metric', y='score', hue='model', palette=palette)

    if 'neuropath' in benchmark.lower():
        # Clinician line
        clinician_df = df_sorted[df_sorted['model'] == 'clinician']
        for metric in clinician_df['metric'].unique():
            clinician_score = clinician_df[clinician_df['metric'] == metric]['score'].values[0]
            ax.axhline(y=clinician_score, linestyle='--', color='red', alpha=0.7, label='Clinician')
            break

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', label_type='edge', fontsize=9)

    plt.ylim(0, 1.05)
    plt.ylabel("Score", fontsize=14)
    plt.title(f"Model Comparison on {benchmark}", fontsize=16)
    plt.xlabel("Metric", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=12)
    plt.legend(title='Model', loc='upper left', bbox_to_anchor=(-0.4, 1.15), frameon=False, fontsize=12, title_fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', bbox_inches='tight', dpi=300)
