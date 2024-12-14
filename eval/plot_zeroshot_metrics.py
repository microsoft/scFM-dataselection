import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def metrics_line_plots(evaluation, model, dataset, columns_to_plot, ylabels, csv_file):
    print(evaluation, model, dataset)
    metrics_df = pd.read_csv(csv_file)
    #metrics_df = pd.read_csv("tmp.csv", index_col=0)

    # rename categories for plotting
    metrics_df["downsampling_method"] = pd.Categorical(metrics_df["downsampling_method"]).rename_categories({'randomsplits': 'Random', 'celltype_reweighted': 'Cell Type Reweighted', 'geometric_sketch': "Geometric Sketching"})

    # plotting

    sns.set_style("ticks")
    fig, axes = plt.subplots(1, len(columns_to_plot), figsize=(22, 4), sharey=False)

    for i, var in enumerate(columns_to_plot):
        sns.lineplot(x='percentage', y=var, hue='downsampling_method', data=metrics_df, errorbar='se', ax = axes[i])
        #sns.boxplot(x='percentage', y=var, hue='downsampling_method', data=metrics_df, ax = axes[i])
        # add line for hvgs
        #axes[i].axhline(y=hvg_scib_metrics[var], color='r', linestyle='dotted')
        # add line for scVI
        #plt.axhline(y=scvi_scib_metrics[var], color='r', linestyle='dashed')
        axes[i].grid(False)
        axes[i].set_xlabel("Percentage of Full Training Dataset")
        axes[i].set_ylabel(ylabels[i])
        axes[i].legend(title='Downsampling Method')
        axes[i].set_xlim([0, 100])
        axes[i].set_ylim([0.0, 1.0])

    plt.tight_layout()

    plot_file = f"figures/{model}_{dataset}_{evaluation}_zero_shot.png"
    print("Saving:", plot_file)
    plt.savefig(plot_file)


def zero_shot_integration_plot(model, dataset):
    columns_to_plot = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label", "ASW_batch", "avg_bio"]
    ylabels = ["NMI (cluster/label)", "ARI (cluster/label)", "ASW (label)", "ASW (batch)", "AVG BIO"]
    metrics_line_plots("integration", model, dataset, columns_to_plot, ylabels)


def zero_shot_classification_plot(model, dataset, csv_file):
    columns_to_plot = ["accuracy", "precision", "recall", "micro_f1", "macro_f1"]
    ylabels = ["Accuracy", "Precision", "Recall", "Micro F1 Score", "Macro F1 Score"]
    metrics_line_plots("classification", model, dataset, columns_to_plot, ylabels, csv_file)



def main():

    """
    zero_shot_integration_plot("Geneformer", "pbmc")
    zero_shot_integration_plot("scVI", "pbmc")
    zero_shot_integration_plot("SSL", "pbmc")
    """
    import sys
    method = sys.argv[1]
    dataname = sys.argv[2]
    eval_results = sys.argv[3]
    eval_task = sys.argv[4]

    if eval_task == 'classification':
        zero_shot_classification_plot(method, dataname, eval_results)
    else:
        zero_shot_integration_plot(method, dataname, eval_results)




if __name__ == "__main__":
    main()