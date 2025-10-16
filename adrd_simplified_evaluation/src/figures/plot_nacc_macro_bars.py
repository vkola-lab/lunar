import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  

from load_metrics import load_metrics, get_cat_order

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

results_dir = sys.argv[1]
output_file = sys.argv[2]      

# Load data
tall = load_metrics(results_dir)
cat_order = get_cat_order()

nacc = tall[tall["benchmark_name"].isin(["test_cog", "test_np", "test_etpr","test_np_one",'test_np_mixed','test_pet'])]

nacc = nacc[nacc["metric"].isin(["recall_macro", "precision_macro"])]

if len(nacc) > 0:

    g = sns.catplot(
        nacc,
        x="value",
        y="model",
        col="benchmark_name",
        row="metric",
        hue="model",
        # col_wrap=3,
        height=3,
        sharex='col',
        sharey=True,
        width=0.95,
        kind="bar",
        order=cat_order,
        hue_order=cat_order,
    )

    # g.set_titles(template="{col_name}", size=8)
    g.set_titles("")

    g.set_ylabels("")
    g.set_xlabels("")

    for ax, title in zip(g.axes[0], g.col_names):
        ax.set_title(title.split('_')[-1].upper(),size=10)

    for ax, label in zip(g.axes[:,0], g.row_names):
        ax.set_ylabel(label.replace('_',' ').title(),size=10)
        # ax.yaxis.set_label_position('right')


    for ax in g.axes.flat:
        for p in ax.patches:
            ax.text(
                p.get_x() + 0.02,  # left edge of the bar
                p.get_y() + p.get_height() / 2.0,  # vertical center of the bar
                f"{p.get_width():.2f}",  # width is the bar value
                ha="left",
                va="center",
                color="white",
                size=8,
            )
        
    plt.tight_layout()

    g.savefig(output_file)
