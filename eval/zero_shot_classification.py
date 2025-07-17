"""zero_shot_classification.py evaluates the performance of a pre-trained model
at classifying an unseen dataset without fine-tuning."""
import string
import random
from collections import defaultdict
from pathlib import Path
import sys
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import scanpy as sc
import anndata as ad
from evaluation_utils import prep_for_evaluation


from zero_shot_model_evaluators import VariableGeneZeroShotEvaluator
from zero_shot_model_evaluators import PrincipalComponentsZeroShotEvaluator

from zero_shot_model_evaluators import SCVIZeroShotEvaluator
from zero_shot_model_evaluators import SSLZeroShotEvaluator
from zero_shot_model_evaluators import GeneformerZeroShotEvaluator
from zero_shot_model_evaluators import PretrainedPrincipalComponentsZeroShotEvaluator
from zero_shot_model_evaluators import SCimilarityZeroShotEvaluator


from model_loaders import load_scvi_model, load_ssl_model, load_pca_model, load_SCimilarity_model
from model_loaders import load_geneformer_model, get_ssl_checkpoint_file


default_n_threads = 64
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"


def get_classification_metrics_df(train_adata,
                                  test_adata,
                                  dataset_name,
                                  downsampling_method,
                                  seed,
                                  percentage,
                                  cell_type_col,
                                  model_directory,
                                  var_file,
                                  method,
                                  dict_dir,
                                  formatted_h5ad_file):
    metrics_dict = defaultdict(list)

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
            f"tmp_zero_shot_classification_geneformer_{random_string}")
        zero_shot_evaluator = GeneformerZeroShotEvaluator(
            geneformer_model, var_file, dict_dir, tmp_output_dir)
    elif method == "PretrainedPCA": # todo test this
        pca_model = load_pca_model(downsampling_method, percentage, seed, model_directory)
        zero_shot_evaluator = PretrainedPrincipalComponentsZeroShotEvaluator(pca_model)
    elif method == "SCimilarity":
        scimilarity_model = load_SCimilarity_model(downsampling_method, percentage, seed, model_directory)
        zero_shot_evaluator = SCimilarityZeroShotEvaluator(scimilarity_model)

    classification_metrics = zero_shot_evaluator.evaluate_classification(
        train_adata, test_adata, cell_type_col)
    print("classification_metrics")
    print(classification_metrics)

    classification_metrics["percentage"] = percentage
    classification_metrics["seed"] = seed
    classification_metrics["downsampling_method"] = downsampling_method
    classification_metrics["dataset"] = dataset_name

    for key in classification_metrics:
        metrics_dict[key].append(classification_metrics[key])

    metrics_df = pd.DataFrame.from_dict(metrics_dict)

    return metrics_df


def main():
    method = sys.argv[1]
    h5ad_file = sys.argv[2]
    label_col = sys.argv[3]

    downsampling_method = sys.argv[4]
    percentage = int(sys.argv[5])
    seed = int(sys.argv[6])

    var_file = sys.argv[7]
    formatted_h5ad_file = sys.argv[8]
    model_directory = sys.argv[9]
    dict_dir = sys.argv[10]

    dataset_name = h5ad_file.replace(".h5ad", "")

    print("loading anndata")
    adata = ad.read_h5ad(h5ad_file)

    if dataset_name in ["pbmc", "tabula", "hlca"]:
        # sctab datasets need to have their var names added
        var_df = pd.read_csv(var_file, index_col=0)
        var_df.index = var_df.index.map(str)
        adata.var = var_df
        adata.var_names = adata.var.feature_name
        print(adata.var_names)

    if h5ad_file == "covid_for_publish.h5ad":
        print("removing slashes from column name (incompatible with loom format)")
        # can't have slashes in loom files
        adata.obs.rename(
            columns={"last_author/PI": "last_author_PI"}, inplace=True)
      
    if h5ad_file == "periodontitis.h5ad":
        print("removing slashes from column name (incompatible with loom format)")
        # can't have slashes in loom files
        adata.obs.rename(
            columns={"Tooth #/Region": "Tooth #_Region"}, inplace=True)

    if 'kim_lung' in h5ad_file:
        batch_cols = ["sample"]
        label_col = "cell_type"
        # drop nans, some cell types don't have labels for kim lung dataset
        adata = adata[adata.obs['cell_type'].notna()]

    new_adata = prep_for_evaluation(adata, formatted_h5ad_file, var_file)

    if method == "SSL" or method == "PretrainedPCA" or method == "SCimilarity":
        print("processing anndata")
        sc.pp.normalize_per_cell(new_adata, counts_per_cell_after=1e4)
        sc.pp.log1p(new_adata)

    if method == "variable_genes" or method == "PCA":
        sc.pp.normalize_per_cell(new_adata, counts_per_cell_after=1e4)
        sc.pp.log1p(new_adata)
        sc.pp.highly_variable_genes(new_adata)
        new_adata = new_adata[:, new_adata.var.highly_variable]
        sc.pp.scale(new_adata)
        sc.tl.pca(new_adata)

    train_indices, test_indices = train_test_split(
        list(range(new_adata.n_obs)), train_size=0.5, test_size=0.5)
    train_adata = new_adata[train_indices, :]
    test_adata = new_adata[test_indices, :]

    metrics_df = get_classification_metrics_df(train_adata,
                                               test_adata,
                                               dataset_name,
                                               downsampling_method,
                                               seed,
                                               percentage,
                                               label_col,
                                               model_directory,
                                               var_file,
                                               method,
                                               dict_dir,
                                               formatted_h5ad_file)

    metrics_csv = f"zero_shot_classification_metrics_{method}_{dataset_name}_downsamplingmethod_{downsampling_method}_percentage_{percentage}_seed_{seed}.csv"

    out_dir = "classification_results"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    metrics_df.to_csv(out_dir + "/" + metrics_csv)


if __name__ == "__main__":
    main()
