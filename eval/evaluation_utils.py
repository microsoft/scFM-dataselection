"""evaluation_utils.py contains a set of utility functions for use
in the model evaluation scripts."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics

import anndata as ad
import scanpy as sc

from geneformer import TranscriptomeTokenizer


def scanpy_workflow(adata):
    """Processes an AnnData object containing raw counts (normalization,
    highly variable genes, PCA, UMAP, and clustering).

    Args:
        adata: An AnnData object that needs to be processed.

    Returns:
        The AnnData after being processed.
    """
    adata = adata.copy()  # don't impact original adata
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, n_iterations=2)
    return adata


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


def load_soma_join_ids(soma_id_directory, dataset_name):
    soma_id_directory = Path(soma_id_directory)
    test = load_soma_join_ids_csv(soma_id_directory, dataset_name, "TEST")
    val = load_soma_join_ids_csv(soma_id_directory, dataset_name, "VAL")
    train = load_soma_join_ids_csv(soma_id_directory, dataset_name, "TRAIN")
    return train, test, val



# todo import these two functions from tokenize_geneformer.py
def add_ensembl_ids_to_adata(adata, var_file):
    """Adds a column called ensembl_id to an AnnData object's variable metadata.

    Args:
        adat: An AnnData object.

    Returns:
        The original AnnData with an additional variable metadata column called ensembl_id.
    """
    var_df = pd.read_csv(var_file, index_col=0)
    var_df.index = var_df.index.map(str)
    adata.var = var_df
    adata.var_names = adata.var.feature_name
    adata.var["ensembl_id"] = adata.var.feature_id
    return adata


def add_n_counts_to_adata(adata):
    """Adds a column called n_counts to an AnnData object's metadata.

    Args:
        adat: An AnnData object.

    Returns:
        The original AnnData with an additional metadata column called n_counts.
    """
    # add n counts
    counts_per_cell = np.sum(adata.X, axis=1)
    adata.obs["n_counts"] = counts_per_cell
    return adata

def tokenize_adata(adata, var_file, dataset_name, cell_type_col=None):
    adata = add_ensembl_ids_to_adata(adata, var_file)
    adata = add_n_counts_to_adata(adata)
    tmp_dir = Path(f"tmp_adata_{dataset_name}")
    os.makedirs(tmp_dir)
    tmp_h5ad = tmp_dir / "tmp.h5ad"
    adata.write(tmp_h5ad)
    output_directory = Path(f"tmp_tokenized_data_{dataset_name}")
    output_prefix_chunk = ""
    if cell_type_col:
        attrs = {cell_type_col: cell_type_col}
    else:
        attrs = None
    tokenizer = TranscriptomeTokenizer(attrs, nproc=8)
    tokenizer.tokenize_data(tmp_dir, output_directory,
                     output_prefix_chunk, file_format="h5ad")
    return str(output_directory) + ".dataset"

def eval_classification_metrics(y_true, y_pred):
    """Computes metrics for cell type classificaiton given the true labels and the
    predicted labels.

    Args:
        y_true: The true cell type labels.
        y_pred: The predicted cell type labels.

    Returns:
        A dictionary containing the accuracy, precision, recall, micro F1 score,
        and marco F1 score.
    """
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(
        y_true, y_pred, average="macro")
    recall = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    micro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    macro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")

    classification_metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }

    return classification_metrics_dict


def eval_expression_reconstruction_mse(true, pred):
    mse = (np.square(true - pred)).mean(axis=None) # None gives grand average over full matrix
    return mse

