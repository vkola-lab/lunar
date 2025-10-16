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
tall = load_metrics(results_dir,readable_names=False)
# cat_order = get_cat_order()

nacc = tall[tall["benchmark_name"].isin(["test_cog", "test_np", "test_etpr","test_np_one",'test_np_mixed','test_pet'])]

def extract_step(s):
    if 'Instruct' in s:
        return 0
    else:
        return int(s.split('-')[-1])


def extract_name(s):
    if 'Qwen2.5-3B' in s:
        return 'Base-3B'
    elif 'Qwen2.5-7B' in s:
        return 'Base-7B'
    elif 'NACC-7B' in s:
        return 'Ours-7B'
    elif 'sce' in s:
        return 'Ours-3BSC'
    else:
        return 'Ours-3B'
        

nacc = nacc[nacc["metric"].isin(["recall_macro", "precision_macro"])]

nacc['step'] = nacc['model'].apply(extract_step)
nacc['basename'] = nacc['model'].apply(extract_name)

if len(nacc) > 0:

    g = sns.catplot(
        nacc[nacc['basename'].str.contains('Ours')].rename(columns={'basename':'Model'}),
        # nacc,
        x="step",
        y="value",
        row="benchmark_name",
        col="metric",
        hue="Model",
        # col_wrap=3,
        height=2,
        aspect=1.5,
        sharex=True,
        sharey=False,
        # width=0.95,
        kind="point",
        native_scale=True,
        errorbar=('pi',50),
        dodge=False,
    )

    sns.move_legend(g,loc='upper left',bbox_to_anchor=(1, 1))


    g.set_titles(template="{col_name}", size=8)
    g.set_titles("")

    g.set_ylabels("")
    # g.set_xlabels("")

    for ax in g.axes.flat:
        ax.set_xlim(left=0)

    for ax, label in zip(g.axes[0], g.col_names):
        ax.set_title(label.split('_')[0].title() + ' (Macro)')

    for ax, label in zip(g.axes[:,0], g.row_names):
        ax.set_ylabel(label.split('_')[-1].upper(),size=10)
        ax.yaxis.set_label_position('left')


    for (row_name, col_name), ax in g.axes_dict.items():
       val = nacc[
        (nacc['benchmark_name'] == row_name) &
        (nacc['metric'] == col_name) &
        (nacc['basename'] == 'Base-3B')
       ]['value'].mean() 

       ax.text(1500,val-0.005,'3B')

       ax.axhline(val,linestyle='--',color='black')

    for (row_name, col_name), ax in g.axes_dict.items():
       val = nacc[
        (nacc['benchmark_name'] == row_name) &
        (nacc['metric'] == col_name) &
        (nacc['basename'] == 'Base-7B')
       ]['value'].mean() 

       ax.text(1500,val-0.005,'7B')

       ax.axhline(val,linestyle='--',color='black')
        
    plt.tight_layout()

    g.savefig(output_file)
