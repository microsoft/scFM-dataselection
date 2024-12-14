import sys
import os
from collections import defaultdict
import shutil
import random
import string
from pathlib import Path

import anndata as ad
import scanpy as sc
import pandas as pd

from model_loaders import load_scvi_model, load_ssl_model
from model_loaders import get_ssl_checkpoint_file, load_geneformer_model

from zero_shot_model_evaluators import VariableGeneZeroShotEvaluator
from zero_shot_model_evaluators import PrincipalComponentsZeroShotEvaluator

from zero_shot_model_evaluators import SSLZeroShotEvaluator, SCVIZeroShotEvaluator
from zero_shot_model_evaluators import GeneformerZeroShotEvaluator

from evaluation_utils import prep_for_evaluation


def get_scib_metrics_df(adata,
                        dataset_name,
                        downsampling_method,
                        seed,
                        percentage,
                        batch_col,
                        label_col,
                        model_directory,
                        var_file,
                        method,
                        dict_dir,
                        formatted_h5ad_file):
    metrics_dict = defaultdict(list)

    print(downsampling_method, percentage, seed)

    # set up model evaluator for each specific model
    if method == "variable_genes":
        zero_shot_evaluator = VariableGeneZeroShotEvaluator()
    elif method == "PCA":
        zero_shot_evaluator = PrincipalComponentsZeroShotEvaluator()
    elif method == "SSL":
        ssl_checkpoint_file = get_ssl_checkpoint_file(
            downsampling_method, percentage, seed, ssl_directory=model_directory)
        ssl_model = load_ssl_model(ssl_checkpoint_file)
        zero_shot_evaluator = SSLZeroShotEvaluator(model=ssl_model)
    elif method == "scVI":
        scvi_model = load_scvi_model(
            downsampling_method, percentage, seed, scvi_directory=model_directory, h5ad_file=formatted_h5ad_file)
        zero_shot_evaluator = SCVIZeroShotEvaluator(model=scvi_model)
    elif method == "Geneformer":
        geneformer_model = load_geneformer_model(
            downsampling_method, percentage, seed, model_directory)
        random_string = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=10))
        tmp_output_dir = Path(
            f"tmp_zero_shot_integration_geneformer_{random_string}")
        zero_shot_evaluator = GeneformerZeroShotEvaluator(
            geneformer_model, var_file, dict_dir, tmp_output_dir)

    scib_metrics = zero_shot_evaluator.evaluate_integration(
        adata, batch_col=batch_col, label_col=label_col)



    scib_metrics["percentage"] = percentage
    scib_metrics["seed"] = seed
    scib_metrics["downsampling_method"] = downsampling_method
    scib_metrics["dataset"] = dataset_name

    for key in scib_metrics:
        metrics_dict[key].append(scib_metrics[key])

    metrics_df = pd.DataFrame.from_dict(metrics_dict)

    # clean up tmp files
    if method == "Geneformer":
        del zero_shot_evaluator
        shutil.rmtree(tmp_output_dir)

    return metrics_df


def main():
    method = sys.argv[1]
    h5ad_file = sys.argv[2]
    label_col = sys.argv[3]
    batch_col = sys.argv[4]

    downsampling_method = sys.argv[5]
    percentage = int(sys.argv[6])
    seed = int(sys.argv[7])

    var_file = sys.argv[8]
    formatted_h5ad_file = sys.argv[9]
    model_directory = sys.argv[10]

    dict_dir = sys.argv[10]

    dataset_name = h5ad_file.strip(".h5ad")
    metrics_csv = f"zero_shot_integration_metrics_{method}_{dataset_name}_downsamplingmethod_{downsampling_method}_percentage_{percentage}_seed_{seed}.csv"

    print("loading anndata")
    adata = ad.read_h5ad(h5ad_file)

    if h5ad_file == "covid_for_publish.h5ad":
        print("removing slashes from column name (incompatible with loom format)")
        # can't have slashes in loom files
        adata.obs.rename(
            columns={"last_author/PI": "last_author_PI"}, inplace=True)

    if 'kim_lung' in h5ad_file:
        batch_cols = ["sample"]
        label_col = "cell_type"
        # drop nans, some cell types don't have labels for kim lung dataset
        adata = adata[adata.obs['cell_type'].notna()]

    if h5ad_file == "periodontitis.h5ad":
        print("removing slashes from column name (incompatible with loom format)")
        # can't have slashes in loom files
        adata.obs.rename(columns={"Tooth #/Region": "Tooth #_Region"}, inplace=True)

    if "scvi_pbmc" in h5ad_file:
        # make gene symbols the var index for scvi pbmc
        adata.var.index = adata.var.gene_symbols

    new_adata = prep_for_evaluation(adata, formatted_h5ad_file, var_file)

    if method == "SSL":
        print("processing anndata")
        sc.pp.normalize_per_cell(new_adata, counts_per_cell_after=1e4)
        sc.pp.log1p(new_adata)

    if method == "variable_genes" or method == "PCA":
        sc.pp.normalize_per_cell(new_adata, counts_per_cell_after=1e4)
        sc.pp.log1p(new_adata)
        sc.pp.highly_variable_genes(new_adata)
        variable_genes_adata = new_adata[:, new_adata.var.highly_variable]
        sc.pp.scale(new_adata)
        sc.tl.pca(new_adata)

    print(new_adata)

    out_dir = "integration_results"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    metrics_df = get_scib_metrics_df(new_adata,
                                     dataset_name,
                                     downsampling_method,
                                     seed,
                                     percentage,
                                     batch_col,
                                     label_col,
                                     model_directory,
                                     var_file,
                                     method,
                                     dict_dir,
                                     formatted_h5ad_file)
    print("Writing:", metrics_csv)
    metrics_df.to_csv(out_dir + "/" + metrics_csv)


if __name__ == "__main__":
    main()
