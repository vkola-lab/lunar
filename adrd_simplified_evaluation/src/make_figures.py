import json
from pathlib import Path
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
import sys

results_dir = sys.argv[1]

data = []
for f in Path(results_dir).rglob("metrics.json"):
    with open(f) as file:
        data.append(json.load(file))

summary = (
    pd.concat([pd.DataFrame(f) for f in data])
    .set_index(["benchmark_name", "model"])
    .sort_index()
)

summary.columns.name = "metric"

tall = summary.stack()
tall.name = "value"

tall = tall.to_frame().reset_index()

# tall = tall[~tall['metric'].isin(['invalid_percent','balanced_accuracy','precision_macro','recall_macro','f1_macro'])]

tall = tall[
    ~tall["benchmark_name"].isin(
        [
            "test_mci",
            "USMLE_ethics",
            "anatomy",
            "clinical_knowledge",
            # "medexpqa",
            # "medmcqa",
            "professional_medicine",
        ]
    )
]

cat_order = [
    "Qwen3-0.6B",
    "Llama-3.2-3B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-3B-DrGRPO",
    "Qwen3-4B",
    "Qwen2.5-7B-Instruct",
    "HuatuoGPT-o1-8B",
]


# Accuracy

g = sns.catplot(
    tall[tall['metric'] == 'accuracy'],
    x="value",
    col="benchmark_name",
    # col="metric",
    y='model',
    hue="model",
    col_wrap=3,
    height=2.5,
    width=0.99, # 
    kind="bar",
    order=cat_order,
    hue_order=cat_order,
    sharex=True,
    sharey=True,
)

g.set_titles(col_template="{col_name}")

# g.set(xlim=(0,1))

g.set(xticks=[0,0.2,0.4,0.6])

g.set_ylabels("")
g.set_xlabels("Accuracy")

for ax in g.axes:
    for p in ax.patches:
        ax.text(
            p.get_x() + 0.02,  # left edge of the bar
            p.get_y() + p.get_height() / 2.0,  # vertical center of the bar
            f"{p.get_width():.2f}",  # width is the bar value
            ha="left",
            va="center",
            color="white",
        )

g.savefig("figures/accuracy.pdf")


# Invalid num

g = sns.catplot(
    tall[tall['metric'] == 'invalid_num'],
    x = 'value',
    y = 'model',
    col = 'benchmark_name',
    hue='model',
    col_wrap=3,
    height=2,
    sharex=False,
    sharey=True,
    order=cat_order,
    hue_order=cat_order,
)

g.set_titles(col_template="{col_name}")

g.set_ylabels("")
g.set_xlabels("Invalid Num.")

g.savefig('figures/invalid_num.pdf')

# Invalid frac

g = sns.catplot(
    tall[tall['metric'] == 'invalid_frac'],
    x = 'value',
    y = 'model',
    col = 'benchmark_name',
    hue='model',
    col_wrap=3,
    height=2,
    sharex=False,
    sharey=True,
    order=cat_order,
    hue_order=cat_order,
)

g.set_titles(col_template="{col_name}")

g.set_ylabels("")
g.set_xlabels("Invalid Frac.")

g.savefig('figures/invalid_frac.pdf')