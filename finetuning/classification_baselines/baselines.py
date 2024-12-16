"""baselines.py contains a set of functions for training a logistic classifier on an AnnData."""
import sys
from pathlib import Path
import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

import scanpy as sc
import anndata as ad

from evaluation_utils import prep_for_evaluation


sys.path.append(os.path.abspath('/app/sc-SFM/eval'))
print(sys.path)




def map_celltypes(train_h5ad_file,
                  val_h5ad_file,
                  cell_type_column,
                  var_file,
                  sctab_format,
                  celltype_pkl):
                    
    train_adata = ad.read_h5ad(train_h5ad_file)
    val_adata = ad.read_h5ad(val_h5ad_file)

    train_adata = prep_for_evaluation(train_adata, sctab_format, var_file)
    val_adata = prep_for_evaluation(val_adata, sctab_format, var_file)

    print('mapping cell types')
    train_adata.obs[cell_type_column] = train_adata.obs[cell_type_column].astype(
        'category')
    cell_type_dict = dict(
        enumerate(train_adata.obs[cell_type_column].cat.categories))
    cell_type_dict = {v: k for k, v in cell_type_dict.items()}

    with open(celltype_pkl, 'wb') as file_handle:
        pickle.dump(cell_type_dict, file_handle)

    train_adata.obs[cell_type_column] = train_adata.obs[cell_type_column].apply(
        lambda x: cell_type_dict[x])
    val_adata.obs[cell_type_column] = val_adata.obs[cell_type_column].apply(
        lambda x: cell_type_dict[x])
    return train_adata, val_adata


def train_logistic_classifier(train_adata,
                              val_adata,
                              cell_type_column):
    print("preprocess train adata")
    sc.pp.normalize_per_cell(train_adata, counts_per_cell_after=1e4)
    sc.pp.log1p(train_adata)
    sc.pp.highly_variable_genes(train_adata)
    variable_genes_adata = train_adata[:, train_adata.var.highly_variable]
    celltypes = train_adata.obs[cell_type_column]
    # run LR
    variable_genes = variable_genes_adata.X
    print("train logistic classifier")
    # should we cross validate like rebecca?
    # https://github.com/clinicalml/sc-foundation-eval/blob/main/scGPT/scGPT_baselines_LR.py
    seed = 1
    c = 1.0  # default
    solver = "saga"  # faster for larger datasets
    n_jobs = -1  # all available
    var_genes_logistic_classifier = LogisticRegression(
        penalty="l1", C=c, solver=solver, random_state=seed, n_jobs=n_jobs)
    var_genes_logistic_classifier.fit(variable_genes, celltypes)
    return var_genes_logistic_classifier, train_adata.var.highly_variable


def main():
    dataset_name = sys.argv[1]
    train_h5ad_file = sys.argv[2]
    val_h5ad_file = sys.argv[3]
    cell_type_column = sys.argv[4]
    var_file = sys.argv[5]
    out_dir = sys.argv[6]
    sctab_format = sys.argv[7]

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    celltype_pkl_savepath = Path(out_dir) / f"{dataset_name}_cell_type_map.pkl"

    print("loading anndatas")
    train_adata, val_adata = map_celltypes(train_h5ad_file,
                                           val_h5ad_file,
                                           cell_type_column,
                                           var_file,
                                           sctab_format,
                                           celltype_pkl_savepath)

    variable_gene_classifier, variable_genes = train_logistic_classifier(train_adata,
                                                                         val_adata,
                                                                         cell_type_column)

    print("Writing pickle file with classifier and variable_genes")

    with open(Path(out_dir) / f"{dataset_name}_logistic_classifier.pkl", 'wb') as f:
        pickle.dump({
            "variable_gene_classifier": variable_gene_classifier,
            "variable_genes": variable_genes
        }, f)


if __name__ == "__main__":
    main()
