import os
import pickle
from random import shuffle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import anndata as ad


def get_indices_for_cell_types_below_count(df, count):
    cell_type_counts = df.cell_type.value_counts()
    # pandas series with the counts of cell types that don't have sufficient cells to sample
    keep_all_cells = cell_type_counts < count
    # just the cell type names of cell types that don't have sufficient cells to sample
    keep_all_cells = cell_type_counts[keep_all_cells].index
    small_df = df[df.cell_type.isin(keep_all_cells)]
    return small_df.soma_joinid


def plot_cell_type_barplot(df, file_name="barplot.pdf"):
    cell_type_counts = df.cell_type.value_counts()
    hist = cell_type_counts.plot(kind='barh', logx=True, figsize=(24, 24))
    for container in hist.containers:
        hist.bar_label(container)
    hist_fig = hist.get_figure()
    hist_fig.savefig(file_name)
    plt.close('all')


def cluster_reweighted_sampling(adata, sampling_percent, seed):
    df = adata.obs

    sampling_num_cells = int(adata.n_obs * sampling_percent)

    unsatisfied = True
    soma_joinids = pd.Series()

    i = 0
    while unsatisfied:
        i += 1
        print("iteration")
        print(i)
        cell_types = df.cell_type
        num_cell_types = cell_types.nunique()
        cell_type_counts = cell_types.value_counts()
        cells_per_cell_type = int(sampling_num_cells / num_cell_types)
        cell_types_below_count = cell_type_counts[cell_type_counts <
                                                  cells_per_cell_type].index
        if len(cell_types_below_count) == 0:
            break
        num_cells_from_cell_types_below_count = cell_type_counts[cell_types_below_count].sum(
        )
        soma_joinids = pd.concat(
            [soma_joinids, get_indices_for_cell_types_below_count(df, cells_per_cell_type)])
        print("len soma_joinids")
        print(len(soma_joinids))
        # subtract off the number of cells we have so far from the cells we need to sample
        sampling_num_cells = sampling_num_cells - num_cells_from_cell_types_below_count
        downsample_cell_types = cell_type_counts > cells_per_cell_type
        downsample_cell_types = cell_type_counts[downsample_cell_types].index
        df = df[df.cell_type.isin(downsample_cell_types)]
        df.cell_type = df.cell_type.cat.remove_unused_categories()

    label = "cell_type"
    g = df.groupby(label, group_keys=False)
    balanced_df = pd.DataFrame(g.apply(lambda x: x.sample(
        int(sampling_num_cells / num_cell_types), random_state=seed)))

    # cell type counts are all the same for the remaining cell types
    balanced_df.cell_type.value_counts()

    soma_joinids = pd.concat([soma_joinids, balanced_df.soma_joinid])

    print("desired percent of cells")
    print(int(adata.n_obs * sampling_percent))
    print(int(adata.n_obs * sampling_percent) / adata.n_obs)

    print("actual percent of cells")
    print(len(soma_joinids))
    print(len(soma_joinids) / adata.n_obs)

    return soma_joinids


def main():
    import sys
    merged_h5ad = sys.argv[1]
    output_dir = sys.argv[2]
    
    adata = ad.read_h5ad(merged_h5ad, backed='r')

    seeds = list(range(25))
    subsampling_percentages = np.repeat([0.01, 0.1, 0.25, 0.5, 0.75], 5)


    for i, sampling_percent in enumerate(subsampling_percentages):
        seed = seeds[i]

        print(i, f"sampling_percent={sampling_percent}", f"seed={seed}")

        soma_joinids = cluster_reweighted_sampling(
            adata, sampling_percent, seed=seed)

        print(soma_joinids)

        soma_joinids = soma_joinids.to_list()

        ind_size = len(soma_joinids)
        train_frac = 0.8
        val_frac = 0.1

        train_split = int(ind_size*train_frac)
        val_split = int(ind_size*val_frac)

        # shuffle data so that we can take consecutive values
        shuffle(soma_joinids)
        train = soma_joinids[:train_split]
        val = soma_joinids[train_split:train_split+val_split]
        test = soma_joinids[train_split+val_split:]

        print(len(train) / ind_size)
        print(len(test) / ind_size)
        print(len(val) / ind_size)

        prefix = f"idx_{sampling_percent}pct_seed{seed}"

        output_path = output_dir + prefix + "/"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_train = output_path + prefix + '_TRAIN.pkl'
        output_val = output_path + prefix + '_VAL.pkl'
        output_test = output_path + prefix + '_TEST.pkl'

        with open(output_train, "wb") as pickle_file:
            pickle.dump(train, pickle_file)

        with open(output_val, "wb") as pickle_file:
            pickle.dump(val, pickle_file)

        with open(output_test, "wb") as pickle_file:
            pickle.dump(test, pickle_file)


if __name__ == "__main__":
    main()
