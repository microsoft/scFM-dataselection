import os
import pickle

import pandas as pd
import numpy as np

import anndata as ad
import scanpy as sc
import sys
import time
from random import shuffle
from geosketch import gs
import matplotlib.pyplot as plt


class Downsample():
    def __init__(self):
        return

    def load_subsample_index(self, pickle_filename):
        with open(pickle_filename, 'rb') as pickle_file:
            sketch_index = pickle.load(pickle_file)
        return sketch_index

    def train_test_val_split(self, ind, ind_output_path, random_seed):
        """
        Given subsampled indices, randomly divide them into train, validation, and test groups.

        Args:
            ind (list[int]): List of subsampled indices (soma join ids)
            output_path (str): Path to where the split indices will be stored
            random_seed (int): Random seed used for shuffling the indices
            train_frac (float): Fraction of data used for training
            val_drac (float): Fraction of data used for validation (the rest will be test data)

        Returns:
            None
        """
        # set a random seed for reproducability and shuffles indices
        np.random.seed(random_seed)
        np.random.shuffle(ind)
        ind_size = len(ind)
        train_frac = 0.8
        val_frac = 0.1

        # get train, test, and val sizes
        train_split = int(ind_size*train_frac)
        val_split = int(ind_size*val_frac)

        # get train test and val soma join indices
        train_idx = sorted(ind[:train_split])
        val_idx = sorted(ind[train_split:train_split+val_split])
        test_idx = sorted(ind[train_split+val_split:])

        output_train = ind_output_path+'_TRAIN.pkl'
        output_val = ind_output_path+'_VAL.pkl'
        output_test = ind_output_path+'_TEST.pkl'

        with open(output_train, "wb") as pickle_file:
            pickle.dump(train_idx, pickle_file)

        with open(output_val, "wb") as pickle_file:
            pickle.dump(val_idx, pickle_file)

        with open(output_test, "wb") as pickle_file:
            pickle.dump(test_idx, pickle_file)

    def downsample_adata(self, adata, index_file, output_file):

        index = self.load_subsample_index(index_file)
        index = pd.Series(index, name="soma_joinid")

        chunk_size = int(adata.n_obs*0.01)  # approx 1% of dataset
        num_chunks = int(adata.shape[0] / chunk_size) + 1

        for chunk_idx in range(num_chunks):
            print("Chunk:", chunk_idx)

            chunk_start = chunk_size * chunk_idx
            chunk_end = chunk_size * (chunk_idx + 1)

            small_adata = adata[chunk_start:chunk_end, :]
            small_adata = small_adata.to_memory()
            small_adata = small_adata.copy()

            indices_to_keep = small_adata.obs.soma_joinid.isin(index)

            downsampled_small_adata = small_adata[indices_to_keep, :]
            downsampled_small_adata = downsampled_small_adata.copy()

            # cast indptr to int64 for merging
            downsampled_small_adata.X.indptr = np.array(
                downsampled_small_adata.X.indptr, dtype=np.int64)
            downsampled_small_adata.write_h5ad(f"tmp_{chunk_idx}.h5ad")

        tmp_files = list(map(lambda x: f"tmp_{x}.h5ad", range(num_chunks)))
        print("Concatenating (on disk) tmp files")
        ad.experimental.concat_on_disk(tmp_files, output_file)

        print("deleting tmp files")
        for tmp_file in tmp_files:
            os.remove(tmp_file)


class RandomDownsample(Downsample):
    def generate_indices(self, soma_index, seed, frac, output_path):
        ind = soma_index.soma_joinid.sample(
            frac=frac, random_state=seed).values
        self.train_test_val_split(ind, output_path, seed)

class CelltypeReweighting(Downsample):
    def get_indices_for_cell_types_below_count(self, df, count):
        cell_type_counts = df.cell_type.value_counts()
        # pandas series with the counts of cell types that don't have sufficient cells to sample
        keep_all_cells = cell_type_counts < count
        # just the cell type names of cell types that don't have sufficient cells to sample
        keep_all_cells = cell_type_counts[keep_all_cells].index
        small_df = df[df.cell_type.isin(keep_all_cells)]
        return small_df.soma_joinid


    def plot_cell_type_barplot(self, df, file_name="barplot.pdf"):
        cell_type_counts = df.cell_type.value_counts()
        hist = cell_type_counts.plot(kind='barh', logx=True, figsize=(24, 24))
        for container in hist.containers:
            hist.bar_label(container)
        hist_fig = hist.get_figure()
        hist_fig.savefig(file_name)
        plt.close('all')


    def generate_indices(self, adata, sampling_percent, seed, ind_output_path):
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
                [soma_joinids, self.get_indices_for_cell_types_below_count(df, cells_per_cell_type)])
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
        self.train_test_val_split(soma_joinids, ind_output_path, seed)

        return soma_joinids


class GeometricSketching(Downsample):

    def generate_indices(self, adata, sampling_percent, seed, ind_output_path, n_comps=50):
        """
        Loads an AnnData object from an hdf5 file, computes a geometric sketch
        (Hie et al. 2019) of the data, and saves an pickle file containing all of
        the indices of the cells contained in the sketch.

        Args:
            h5ad_file (str): The hdf5 file containing the AnnData object of interest.
            sketch_size (int): The size of the geometric sketch
            output_path (str): The pickle file to save the geometric sketch indices to.
            n_comps (int): The number of PCs to use as input to the geometric sketching algorithm.

        Returns:
            None
        """
        print('sampling percent', sampling_percent)
        print('seed', seed)

        # derived params
        sketch_size = int(adata.n_obs * sampling_percent / 100.0)
        # 1% of n_obs to (hopefully) be faster than using n_obs covering boxes (default when replace=False)
        num_covering_boxes = int(0.01 * adata.n_obs)


        print("Starting geometric sketching")

        start = time.time()

        sketch_index = gs(adata.obsm['X_pca'], sketch_size, seed=seed, k=num_covering_boxes, replace=False)

        end = time.time()

        print("Geometric sketching complete")

        print("Time Taken")
        print(end - start)


        # soma joinids pd.Series
        soma_joinids = adata.obs.soma_joinid.iloc[sketch_index]

        soma_joinids = soma_joinids.to_list()


        print("Creating train/val/test split")
        # params for train, val, test split


        # randomly shuffle data so that we can take consecutive values
        self.train_test_val_split(soma_joinids, ind_output_path, seed)

