"""train_scvi.py contains a set of functions for pre-training scVI on scTab."""
import argparse
from pathlib import Path
import gzip

import pandas as pd
import numpy as np

import dask.distributed as dd
import h5py

import anndata as ad
import scanpy as sc



# todo documentation
def train_pca(h5ad_file,
              var_file,
              output_directory,
              prefix=""):
    """Trains a 'PCA foundation model'.

    Args:
        h5ad_file: The h5ad file containing the training data.
        var_file: A file containing the scTab variable names.
        output_directory: The directory to write the trained model.
        prefix=The model name prefix.
        n_latent: The size of the autoencoder latent space.

    Returns:
        The AnnData after being processed.
    """
    print("Loading train h5ad file")
    train_adata = ad.read_h5ad(h5ad_file)

    print("Aligning gene names")
    var_df = pd.read_csv(var_file, index_col=0)
    var_df.index = var_df.index.map(str)

    train_adata.var = var_df
    train_adata.var_names = train_adata.var.feature_name

    print("Pre-processing counts data for PCA")
    sc.pp.normalize_total(train_adata, target_sum=1e4)
    sc.pp.log1p(train_adata)
    sc.pp.highly_variable_genes(train_adata)

    print("Computing PCA")
    sc.pp.pca(train_adata, return_info=True)

    print("Saving model")
    filename = Path(output_directory) / f"{prefix}.npy"

    Path(output_directory).mkdir(parents=True, exist_ok=True)
    np.save(filename, train_adata.varm["PCs"])

    print("Done!")



def main():
    """Sets up command line arguments and trains an scVI model.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description='Train an scVI model with specified    .')
    parser.add_argument('-a', '--h5ad')
    parser.add_argument('--var_file')

    parser.add_argument('-o', '--output')

    args = parser.parse_args()

    h5ad_file = args.h5ad          # merged.h5ad"
    var_file = args.var_file       # adata_var.csv
    output_directory = args.output

    prefix = Path(h5ad_file).stem

    print("Calling train_scvi with args:")
    print("h5ad_file:", h5ad_file)
    print("output_directory:", output_directory)
    print("prefix:", prefix)

    train_pca(h5ad_file, var_file, output_directory, prefix=prefix)


if __name__ == "__main__":
    main()

