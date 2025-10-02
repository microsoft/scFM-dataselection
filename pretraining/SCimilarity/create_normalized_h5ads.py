import os
from pathlib import Path
from colorist import blue

import numpy as np
import pandas as pd

import anndata as ad
import scanpy as sc

import scimilarity.utils


def process_h5ad(h5ad, var_file, output_path):
    blue(f"Processing {h5ad} and saving to {output_path}")

    adata = ad.read_h5ad(h5ad)


    # train_adata.var.index is 0,..., 19330 it needs to be gene symbols, use adata var file
    var_df = pd.read_csv(var_file, index_col=0)
    var_df.index = var_df.index.map(str)

    # set var names for sctab datasets
    adata.var = var_df
    adata.var_names = adata.var.feature_name

    adata.obs["datasetID"] = adata.obs.dataset_id
    adata.obs["sampleID"] = adata.obs.donor_id
    adata.obs["cellTypeOntologyID"] = adata.obs.cell_type
    adata.obs["tissue"] = adata.obs.tissue
    adata.obs["disease"] = adata.obs.disease

    # workaround for scanpy/scipy fixes bug
    # https://github.com/scverse/scanpy/issues/3331
    adata.X.indptr = adata.X.indptr.astype(np.int64)
    adata.X.indices = adata.X.indices.astype(np.int64)

    # get columns for n_genes_by_counts, total_counts, total_counts_mt, pct_counts_mt from scanpy
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # manually set predicted doublets to 0
    adata.obs["predicted_doublets"] = 0

    adata.layers["counts"] = adata.X  # Move the matrix to where SCimilarity/CellArr expects it

    adata = scimilarity.utils.lognorm_counts(adata)

    adata.write_h5ad(output_path)


def process_h5ads(sctab_dir, downsampling_strategy, percentage, seed, var_file, output_sctab_dir):
    base_dir = Path(sctab_dir) / downsampling_strategy / f"idx_{percentage}pct_seed{seed}"

    train_file = base_dir / f"idx_{percentage}pct_seed{seed}_TRAIN.h5ad"
    val_file = base_dir / f"idx_{percentage}pct_seed{seed}_VAL.h5ad"
    test_file = base_dir / f"idx_{percentage}pct_seed{seed}_TEST.h5ad"

    output_dir = Path(output_sctab_dir) / downsampling_strategy / f"idx_{percentage}pct_seed{seed}"

    os.makedirs(output_dir, exist_ok=True)

    output_train_file = output_dir / f"idx_{percentage}pct_seed{seed}_TRAIN.h5ad"
    output_val_file = output_dir / f"idx_{percentage}pct_seed{seed}_VAL.h5ad"
    output_test_file = output_dir / f"idx_{percentage}pct_seed{seed}_TEST.h5ad"

    process_h5ad(train_file, var_file, output_train_file)
    process_h5ad(val_file, var_file, output_val_file)
    process_h5ad(test_file, var_file, output_test_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create normalized h5ad files for SCimilarity pretraining.")
    parser.add_argument("--sctab_dir", type=str, required=True, help="Path to the scTab directory containing the downsampling strategy directories.")
    parser.add_argument("--output_sctab_dir", type=str, help="Path to the output directory where normalized h5ad files will be saved.")
    parser.add_argument("--var_file", type=str, required=True, help="Path to the var file containing gene symbols.")

    
    args = parser.parse_args()

    downsampling_strategies = ["random", "cluster_adaptive",  "geometric_sketching"]
    percentages = np.repeat([1, 10, 25, 50, 75, 100], 5)
    seeds = list(range(30))

    for downsampling_strategy in downsampling_strategies:
        for percentage, seed, in zip(percentages, seeds):

            # 100pcts are only generated for random downsampling
            if downsampling_strategy != "random" and percentage == 100:
                continue

            process_h5ads(args.sctab_dir, downsampling_strategy, percentage, seed, args.var_file, args.output_sctab_dir)