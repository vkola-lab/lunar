import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  
import textwrap

from load_metrics import get_cat_order, load_class_metrics
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

results_dir = sys.argv[1]
figures_dir = Path(sys.argv[2])

# Load data
tall = load_class_metrics(results_dir)
cat_order = get_cat_order()

for benchmark_name in tqdm(tall['benchmark_name'].unique()):
    g = sns.catplot(
        tall[tall["benchmark_name"] == benchmark_name],
        x="value",
        y="model",
        col="class",
        row="metric",
        hue="model",
        # col_wrap=3,
        height=4,
        sharex=True,
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
        ax.set_title(textwrap.fill(title,width=20),size=10)

    for ax, label in zip(g.axes[:,0], g.row_names):
        ax.set_ylabel(label.title(),size=10)
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
        
    # plt.tight_layout()

    g.savefig(figures_dir/f'class_metrics_{benchmark_name}.pdf')
