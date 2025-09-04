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

# Filter for the figure
mcq_df = tall[
    (tall["metric"] == "accuracy") &
    (tall["benchmark_name"].isin(["medqa_test","medexpqa","professional_medicine",'clinical_knowledge']))
]

# Plot
g = sns.catplot(
    mcq_df,
    x="value",
    col="benchmark_name",
    y="model",
    hue="model",
    col_wrap=4,
    height=2.5,
    width=0.95,
    kind="bar",
    order=cat_order,
    hue_order=cat_order,
    sharex=True,
    sharey=True,
)

g.set_titles(col_template="{col_name}")
g.set_ylabels("")
g.set_xlabels("Accuracy")
g.set(xticks=[0, 0.2, 0.4, 0.6, 0.8])

for ax in g.axes:
    for p in ax.patches:
        ax.text(
            p.get_x() + 0.02,
            p.get_y() + p.get_height()/2.0,
            f"{p.get_width():.2f}",
            ha="left", va="center",
            color="white", size=10
        )

plt.tight_layout()

g.savefig(output_file)
