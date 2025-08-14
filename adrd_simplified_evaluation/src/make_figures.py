import json
from pathlib import Path
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import sys

results_dir = sys.argv[1]

data = []
for f in Path(results_dir).rglob('metrics.json'):
    with open(f) as file:
        data.append(json.load(file))

summary = pd.concat([pd.DataFrame(f) for f in data]).set_index(['benchmark_name','model']).sort_index()

summary.columns.name = 'metric'

tall = summary.stack()
tall.name = 'value'

tall = tall.to_frame().reset_index()

g = sns.catplot(
    tall,
    y="benchmark_name",
    x="value",
    col="metric",
    hue="model",
    sharex=False,
    col_wrap=4,
    height=3,
    kind='bar'
)

g.set_titles(col_template="{col_name}")

# g.set(xlim=(0,1))

g.set_ylabels('')
g.set_xlabels('')

for ax in g.axes:
    for p in ax.patches:
        ax.text(
            p.get_x() + 0.02,                       # left edge of the bar
            p.get_y() + p.get_height() / 2., # vertical center of the bar
            f"{p.get_width():.2f}",          # width is the bar value
            ha="left", va="center",color='white'
        )

g.savefig('figures/bench_comparison.pdf')