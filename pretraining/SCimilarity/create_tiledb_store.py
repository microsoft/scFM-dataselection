import sys
import os
import pickle
import shutil
from colorist import red, green
import multiprocessing as mp
import json


import numpy as np
import pandas as pd

import anndata as ad
import scanpy as sc

from cellarr import build_cellarrdataset, CellArrDataset, MatrixOptions
import tiledb

from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl

from scimilarity.tiledb_data_models import CellMultisetDataModule
from scimilarity.training_models import MetricLearning


# todo remove this
#from tiledb_data_models import CellMultisetDataModule


from scimilarity.ontologies import *


def convert_cell_types_to_cell_ontologies(adata, cell_type_col):
    graph = import_cell_ontology()
    id_mapper = get_id_mapper(graph)
    # id_mapper maps cell ontology ids to cell types and we need the opposite here.
    id_mapper = {value: key for key, value in id_mapper.items()}
    # map each cell type name to cell ontology id
    adata.obs[cell_type_col] = adata.obs[cell_type_col].replace(id_mapper)
    return adata


# train_adata.var.index is 0,..., 19330
# it needs to be gene symbols
# use adata var file
def process_adata(train_h5ad, val_h5ad, var_file, tiledb_base_path):
    red("Reading anndata from disk")
    train_adata = ad.read_h5ad(train_h5ad)
    val_adata = ad.read_h5ad(val_h5ad)

    red("Reading var file index from disk")
    var_df = pd.read_csv(var_file, index_col=0)
    var_df.index = var_df.index.map(str)

    # set var names for sctab datasets
    train_adata.var = var_df
    train_adata.var_names = train_adata.var.feature_name

    val_adata.var = var_df
    val_adata.var_names = val_adata.var.feature_name

    # prepend train and val to dataset names because they have datasets in common
    # due to data splitting strategy not taking study into account
    train_adata.obs['dataset_id'] = 'train_' + train_adata.obs['dataset_id'].astype(str)
    val_adata.obs['dataset_id'] = 'val_' + val_adata.obs['dataset_id'].astype(str)

    # set up train and val in tiledb studies
    val_studies = list(set(val_adata.obs['dataset_id']))
    train_studies = list(set(train_adata.obs['dataset_id']))

    red('Num train studies:' + str(len(train_studies)))
    red('Num val studies:' + str(len(val_studies)))

    # convert cell type labels to cell ontology ids
    train_adata = convert_cell_types_to_cell_ontologies(train_adata, "cell_type")
    val_adata = convert_cell_types_to_cell_ontologies(val_adata, "cell_type")


    # concatenate train and val anndatas
    adata = ad.concat([train_adata, val_adata])

    red("Creating required columns")

    adata.obs["datasetID"] = adata.obs.dataset_id
    red("Creating required columns 2")

    adata.obs["sampleID"] = adata.obs.donor_id
    red("Creating required columns 3")
    adata.obs["cellTypeOntologyID"] = adata.obs.cell_type
    red("Creating required columns 4")
    adata.obs["tissue"] = adata.obs.tissue
    red("Creating required columns 5")
    adata.obs["disease"] = adata.obs.disease

    # get columns for n_genes_by_counts, total_counts, total_counts_mt, pct_counts_mt from scanpy
    red("Creating required columns 6")
    # workaround for scanpy/scipy fixes bug
    # https://github.com/scverse/scanpy/issues/3331
    adata.X.indptr = adata.X.indptr.astype(np.int64)
    adata.X.indices = adata.X.indices.astype(np.int64)

    red("columns before qc")
    print(adata.obs.columns) # todo remove
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    red("columns after qc")
    print(adata.obs.columns) # todo remove

    # manually set predicted doublets to 0
    red("Creating required columns 7")
    adata.obs["predicted_doublets"] = 0

    red("Moving counts matrix to a layer where cellarr expects it")
    adata.layers["counts"] = adata.X  # Move the matrix to where CellArr expects it

    red("columns after updates")
    print(adata.obs.columns) # todo remove



    # clean up columns
    # taken from clean_obs funciton in:
    # https://genentech.github.io/scimilarity/notebooks/training_tutorial.html
    obs = adata.obs.copy()

    columns = [
        "datasetID", "sampleID", "cellTypeOntologyID", "tissue", "disease",
        "n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt",
        "predicted_doublets",
    ]
    obs = obs[columns].copy()

    convert_dict = {
        "datasetID": str,
        "sampleID": str,
        "cellTypeOntologyID": str,
        "tissue": str,
        "disease": str,
        "n_genes_by_counts": int,
        "total_counts": int,
        "total_counts_mt": int,
        "pct_counts_mt": float,
        "predicted_doublets": int,
    }
    adata.obs = obs.astype(convert_dict)


    red("Creating cellarr collection directory")
    # delete and recreate directory if it exists already
    output_dir = tiledb_base_path 
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.mkdir(output_dir)

    adatas = [adata] # todo add val adata?

    # create gene annotation
    gene_annotation = sorted(adata.var.index.tolist())
    gene_annotation = pd.DataFrame({"cellarr_gene_index": gene_annotation})

    # create metadata
    # todo drop columns
    #columns_to_keep = []
    #metadata_df = adata.obs[columns_to_keep]
    metadata_df = adata.obs
    metadata_df = metadata_df.reset_index(drop=True)


    red("metadata_df")
    print(metadata_df)

    red("Creating cellarr dataset")
    # Build the dataset - this is where the magic happens!
    dataset = build_cellarrdataset(
        gene_annotation= gene_annotation,
        cell_metadata= metadata_df,
        output_path=tiledb_base_path,
        files=adatas,
        matrix_options=MatrixOptions(matrix_name="counts", dtype=np.int16),
        num_threads=4,  # Adjust based on your CPU - more threads = more speed (usually)
    )

    return dataset, val_studies
    


def build_cellarrdataset_from_sctab(train_h5ad, val_h5ad, tiledb_base_path):
    var_file = "/users/adenadel/data/adenadel/scratch/new_eval/adata_var.csv"
    red("Processing adata")
    dataset = process_adata(train_h5ad,val_h5ad, var_file, tiledb_base_path)
    return dataset

"""
from train_SCimilarity import *
dataset = build_cellarrdataset_from_sctab()
"""

def main():
    train_h5ad = sys.argv[1]
    val_h5ad = sys.argv[2]
    tiledb_base_path = sys.argv[3]
    val_studies_pickle_file = sys.argv[4]

    dataset, val_studies = build_cellarrdataset_from_sctab(train_h5ad, val_h5ad, tiledb_base_path)

    with open(val_studies_pickle_file, "wb") as file:
        pickle.dump(val_studies, file)





if __name__ == "__main__":
    main()
