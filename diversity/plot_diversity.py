"""plot_diversity.py plots pre-computed diversity indices and scores."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def diversity_line_plot(df, metric, ylabel, yticks, ax):
    """Plots a lineplot for the specified diversity score.

    Args:
        df: The dataframe containing the diversity metrics.
        metric: The name of the metric to plot.
        ylabel: The y-axis title.
        yticks: A list of tickmarks to use on the y-axis.
        ax: The matplotlib axis to plot on.

    Returns:
        None
    """
    # rename categories for plotting
    df["downsampling_method"] = pd.Categorical(df["downsampling_method"]).rename_categories({
        'randomsplits': 'Random', 
        'random': 'Random', 
        'celltype_reweighted': 'Cell Type Reweighted',
        'cluster_adaptive': 'Cell Type Reweighted',
        'geometric_sketch': "Geometric Sketching",
        'geometric_sketching': "Geometric Sketching",
        'spikein10pct': "Spike-in (10%)",
        'spikein_10': "Spike-in (10%)",
        'spikein50pct': "Spike-in (50%)",
        'spikein_50': "Spike-in (50%)",
        })
    #plotting
    sns.set_style("ticks")
    sns.lineplot(x="percentage", y = metric, data=df, hue="downsampling_method", ax = ax)
    ax.grid(False)
    ax.set_xlabel("Percentage of Full Training Dataset")
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, 100])
    ax.set_ylim(ymin=0)
    ax.set_yticks(yticks)
    ax.get_legend().remove()




diversity_df = pd.read_csv("diversity.csv")

# sort alphabatically
diversity_df = diversity_df.sort_values(by=['downsampling_method'], ascending=True)

vendi_df = pd.read_csv('vendi_score.csv')
vendi_df["downsampling_method"] = vendi_df["method"]

# sort alphabatically
vendi_df = vendi_df.sort_values(by=['downsampling_method'], ascending=True)

# subset to remove spikeins
vendi_df = vendi_df[vendi_df.downsampling_method != "spikein10pct"]
vendi_df = vendi_df[vendi_df.downsampling_method != "spikein50pct"]




plt.rcParams['font.size'] = 14

fig, axes = plt.subplots(1, 3, figsize=(17,5))



diversity_line_plot(diversity_df,
                    "shannon_index",
                    "Shannon Index",
                    np.linspace(0, 8, 5),
                    axes[0])

diversity_line_plot(diversity_df,
                    "gini_simpson_index",
                    "Gini-Simpson Index",
                    np.linspace(0, 1, 6),
                    axes[1])

diversity_line_plot(vendi_df,
                    "vendi_score",
                    "Vendi Score",
                    np.linspace(0, 270, 4),
                    axes[2])

# remove bounding whitespace
fig.tight_layout()

# grab legend labels from Vendi subplot that has all 5
handles, labels = axes[2].get_legend_handles_labels()

# add legend blow plots
fig.subplots_adjust(bottom=0.3)
fig.legend(handles,
           labels,
           loc='lower center',
           bbox_to_anchor=(0.5, 0),
           title='Downsampling Method', ncol=5)



plt.savefig("diversity.png", dpi=300)
