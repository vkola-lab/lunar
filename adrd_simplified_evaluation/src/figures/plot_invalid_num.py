
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

g = sns.catplot(
    tall[tall["metric"] == "invalid_num"],
    x="value",
    y="model",
    col="benchmark_name",
    hue="model",
    col_wrap=5,
    height=2,
    sharex=False,
    sharey=True,
    order=cat_order,
    hue_order=cat_order,
)

g.set_titles(col_template="{col_name}")

g.set_ylabels("")
g.set_xlabels("Invalid Num.")

plt.tight_layout()

g.savefig(output_file)