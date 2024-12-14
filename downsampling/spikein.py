import os
import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd

import anndata as ad


# creates an empty anndata object that has the genes we need
# directory should be a directory containing adata_var.csv (the gene names) and
# idx_1pct_seed0_TEST.h5ad (any small h5ad file the the scTab genes)
def get_empty_anndata(formatted_h5ad_file, var_file):
    """Creates an AnnData object with no cells that contains the genes from the scTab dataset.

    Args:
        formatted_h5ad_file: An h5ad file for an AnnData that contains the gene from the
        scTab dataset.
        var_file: A file containing the variable names for the scTab AnnDatas.

    Returns:
        An AnnData with no cells that contains the gene from the scTab dataset.
    """
    adata = ad.read_h5ad(formatted_h5ad_file)

    var_df = pd.read_csv(var_file, index_col=0)
    var_df.index = var_df.index.map(str)

    empty_adata = adata[0:0, :]
    empty_adata.var = var_df
    empty_adata.var_names = empty_adata.var.feature_name

    return empty_adata


# https://discourse.scverse.org/t/help-with-concat/676/2
# https://discourse.scverse.org/t/loosing-anndata-var-layer-when-using-sc-concat/1605
def prep_for_evaluation(adata, formatted_h5ad_file, var_file):
    """Creates an AnnData object with no cells that contains the genes from the scTab dataset.

    Args:
        adata: An AnnData whose variable names are gene symbols.
        formatted_h5ad_file: An h5ad file for an AnnData that contains the gene from the
        scTab dataset.
        var_file: A file containing the variable names for the scTab AnnDatas.

    Returns:
        The original AnnData object with only the genes from the scTab dataset.
    """
    empty_adata = get_empty_anndata(formatted_h5ad_file, var_file)
    return ad.concat([empty_adata, adata], join="outer")[:, empty_adata.var_names]


def load_soma_join_ids_csv(soma_id_directory, dataset_name, split_name):
    csv_file = f"{dataset_name}_{split_name}_somajoinID.csv"
    filename = soma_id_directory / f"{dataset_name}_sctab" / csv_file
    return pd.read_csv(filename, index_col=0)["0"].to_list()


def replace_with_perturbation_data_low_memory(training_adata,
                                              perturbation_adata,
                                              num_perturbation_cells_to_include,
                                              output_h5ad_file,
                                              seed):
    print('replacing cells')

    random.seed(seed)


    # sample perturbation cells
    perturbation_sample_indices = random.sample(
        range(perturbation_adata.n_obs), k=num_perturbation_cells_to_include)
    perturbation_sample_indices = sorted(perturbation_sample_indices)
    perturbation_sample = perturbation_adata[perturbation_sample_indices]


    # sample training indices to replace
    indices_to_keep = random.sample(range(training_adata.n_obs), k=(
        training_adata.n_obs - num_perturbation_cells_to_include))
    indices_to_keep = sorted(indices_to_keep)
    training_adata_kept = training_adata[indices_to_keep]


    tmp_training_file = f"seed{seed}_tmp_training.h5ad"
    tmp_perturb_file = f"seed{seed}_tmp_perturb.h5ad"
    tmp_output_file = f"seed{seed}_tmp_output.h5ad"
    print("Writing temporary training sample h5ad file")
    training_adata_kept.to_memory().copy().write_h5ad(tmp_training_file)

    print("Writing temporary perturbation sample h5ad file")
    perturbation_sample.obs["is_primary_data"] = True
    perturbation_sample.write_h5ad(tmp_perturb_file)

    print("creating combined adata")
    # create new adata with perturbation data spiked in
    ad.experimental.concat_on_disk(
        [tmp_training_file, tmp_perturb_file], tmp_output_file)

    new_adata = ad.read_h5ad(tmp_output_file, backed="r+")


    # anndata doesn't like when there are nulls in a column in the obs df,
    # so we convert the offending column
    new_adata.obs["is_primary_data"] = new_adata.obs["is_primary_data"].astype(
        str).astype("category")


    print("writing new h5ad")
    # save h5ad of adata with perturbation data spiked in
    new_adata.write_h5ad(output_h5ad_file)

    print("removing temporary files")
    os.remove(tmp_training_file)
    os.remove(tmp_perturb_file)
    os.remove(tmp_output_file)



def get_training_adata(base_dir, downsampling_method, percentage, seed, var_file):
    print("Loading training adata")
    base_dir = Path(base_dir)
    h5ad_path = base_dir / downsampling_method / \
        f"idx_{percentage}pct_seed{seed}/idx_{percentage}pct_seed{seed}_TRAIN.h5ad"
    adata = ad.read_h5ad(h5ad_path, backed='r+')

    print("Loading gene names")
    var_df = pd.read_csv(var_file, index_col=0)
    var_df.index = var_df.index.map(str)

    print("Adding gene names to adata")
    adata.var_names = var_df.feature_name

    return adata


def main():
    parser = argparse.ArgumentParser(
        description='Replaces some cells in an adata object with cells from another adata object.')
    parser.add_argument('-p', '--perturbation_h5ad')
    parser.add_argument('-o', '--output_prefix')
    parser.add_argument('-s', '--spikein_percentage')
    parser.add_argument('-t', '--training_dir')
    parser.add_argument('-f', '--formatted_h5ad_file')
    parser.add_argument('-v', '--var_file')

    args = parser.parse_args()

    var_file = args.var_file
    formatted_h5ad_file = args.formatted_h5ad_file()

    # Replogle has 2.5 million cells
    # 3 sets: 310,385; 1,989,578; 247,914
    # 2 million cells in
    # "/users/adenadel/data/adenadel/scratch/scPerturb/data/individual_datasets/ReplogleWeissman2022_K562_gwps.h5ad"
    print("loading perturbation adata")
    perturbation_adata = ad.read_h5ad(args.perturbation_h5ad)

    # the concatenated adata was made by concatenating anndata objects with dense matrices and
    # missing values were populated with nans so we replace them with zeroes here
    print("replacing nans with zeroes")
    perturbation_adata.X = np.nan_to_num(perturbation_adata.X)

    print("aligning perturbation adata genes with training data genes")
    perturbation_adata = prep_for_evaluation(perturbation_adata, formatted_h5ad_file, var_file)

    downsampling_methods = ["random"]
    seeds = list(range(0, 30))
    percentages = np.repeat([1, 10, 25, 50, 75, 100], 5)

    spikein_percentage = float(args.spikein_percentage)

    for downsampling_method in downsampling_methods:
        for i, percentage in enumerate(percentages):
            seed = seeds[i]
            print(downsampling_method, seed, percentage)

            training_adata = get_training_adata(
                args.training_dir, downsampling_method, percentage, seed, var_file)
            output_h5ad_file = args.output_prefix + \
                f"idx_{percentage}pct_seed{seed}_TRAIN.h5ad"

            num_perturbation_cells_to_include = int(
                spikein_percentage * training_adata.n_obs)

            replace_with_perturbation_data_low_memory(training_adata,
                                                      perturbation_adata,
                                                      num_perturbation_cells_to_include,
                                                      output_h5ad_file,
                                                      seed)


if __name__ == "__main__":
    main()
