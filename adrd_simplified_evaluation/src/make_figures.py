import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# Remove benchmarks we don't want
tall = tall[
    ~tall["benchmark_name"].isin(
        [
            "test_mci",
            "USMLE_ethics",
            "anatomy",
            # "clinical_knowledge",
            # "professional_medicine",
        ]
    )
]

# Remove irrelevant models
tall = tall[~tall["model"].isin(["Qwen3-0.6B"])]

cat_order = [
    # "Qwen3-0.6B",
    # "Llama-3.2-3B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-3B-DrGRPO-subset",
    "Qwen2.5-3B-DrGRPO-Stratified",
    "Qwen2.5-3B-DrGRPO-Strat-NACC",
    "Qwen2.5-3B-DrGRPO-Strat-MedQA-NACC",
    "Qwen2.5-3B-DrGRPO-Strat-MedQA-NACC-filtered",
    "Qwen2.5-3B-DrGRPO-Strat-MedQA-NACC-sce",
    "Qwen2.5-3B-DrGRPO-Strat-MedQA-NACC-sce-scaled",
    "Qwen3-4B",
    "Qwen2.5-7B-Instruct",
    "HuatuoGPT-o1-8B",
]

# Accuracy

usmle_df = tall[
    # (tall["metric"] == "accuracy") & (tall["benchmark_name"].str.contains("USMLE"))
    (tall["metric"] == "accuracy") & (tall["benchmark_name"].isin(["medqa_test","medexpqa","professional_medicine",'clinical_knowledge' ]))
]


g = sns.catplot(
    usmle_df,
    x="value",
    col="benchmark_name",
    # col="metric",
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

plt.tight_layout()

g.set_titles(col_template="{col_name}")

# g.set(xlim=(0,1))

# g.set(xticks=[])
g.set(xticks=[0, 0.2, 0.4, 0.6, 0.8])

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
            size=10,
        )

g.savefig("figures/accuracy_mcq.pdf")


# Invalid num

g = sns.catplot(
    tall[tall["metric"] == "invalid_num"],
    x="value",
    y="model",
    col="benchmark_name",
    hue="model",
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

g.savefig("figures/invalid_num.pdf")

# Invalid frac

g = sns.catplot(
    tall[tall["metric"] == "invalid_frac"],
    x="value",
    y="model",
    col="benchmark_name",
    hue="model",
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

g.savefig("figures/invalid_frac.pdf")

# make plot of macro metrics for NACC test_ benchmarks

nacc = tall[tall["benchmark_name"].isin(["test_cog", "test_np", "test_etpr","test_np_one",'test_np_mixed','test_pet'])]

nacc = nacc[~nacc["metric"].isin(["invalid_num", "invalid_frac"])]

nacc = nacc[nacc["metric"].isin(["recall_macro", "precision_macro"])]

g = sns.catplot(
    nacc,
    x="value",
    y="model",
    col="benchmark_name",
    row="metric",
    hue="model",
    # col_wrap=3,
    height=2,
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

g.savefig("figures/nacc_macro_bars.pdf")

# pointplot to focus on small differences

g = sns.catplot(
    nacc,
    x="value",
    y="model",
    row="benchmark_name",
    col="metric",
    hue="model",
    # col_wrap=3,
    height=2,
    aspect=2,
    sharex=False,
    sharey=True,
    kind="point",
    order=cat_order,
    hue_order=cat_order,
)

g.set_titles(template="{col_name}\n{row_name}", size=10)

g.set_ylabels("")
g.set_xlabels("")
    
plt.tight_layout()

g.savefig("figures/nacc_macro_points.pdf")

# make plot of weighted metrics for NACC test_ benchmarks

nacc = tall[tall["benchmark_name"].isin(["test_cog", "test_np", "test_etpr", 'test_np_one','test_np_mixed','test_pet'])]

nacc = nacc[~nacc["metric"].isin(["invalid_num", "invalid_frac"])]

nacc = nacc[nacc["metric"].isin(["accuracy", "precision_weighted"])]

g = sns.catplot(
    nacc,
    x="value",
    y="model",
    col="benchmark_name",
    row="metric",
    hue="model",
    # col_wrap=3,
    height=2,
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

g.savefig("figures/nacc_weighted_bars.pdf")

# pointplot to focus on small differences and error bars

g = sns.catplot(
    nacc,
    x="value",
    y="model",
    row="benchmark_name",
    col="metric",
    hue="model",
    # col_wrap=3,
    height=2,
    aspect=2,
    sharex=False,
    sharey=True,
    kind="point",
    order=cat_order,
    hue_order=cat_order,
)

g.set_titles(template="{col_name}\n{row_name}", size=10)
# g.set_titles("")

g.set_ylabels("")
g.set_xlabels("")

# for ax, title in zip(g.axes[0], g.col_names):
    # ax.set_title(label.replace('_',' ').title(),size=10)

# for ax, label in zip(g.axes[:,0], g.row_names):
    # ax.set_ylabel(title.split('_')[-1].upper(),size=10)
    
plt.tight_layout()

g.savefig("figures/nacc_weighted_points.pdf")