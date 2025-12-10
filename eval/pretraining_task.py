"""pretraining_task.py evaluates the performance of a pre-trained model
on an unseen dataset on its pre-training task."""
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
from zero_shot_model_evaluators import SSLZeroShotEvaluator, SCVIZeroShotEvaluator
from zero_shot_model_evaluators import GeneformerZeroShotEvaluator
from zero_shot_model_evaluators import VariableGeneZeroShotEvaluator
from zero_shot_model_evaluators import PrincipalComponentsZeroShotEvaluator
from model_loaders import load_scvi_model, load_ssl_model
from model_loaders import load_geneformer_model, get_ssl_checkpoint_file


default_n_threads = 64
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"


def get_pretraining_task_metrics_df(adata,
                                    dataset_name,
                                    downsampling_method,
                                    seed,
                                    percentage,
                                    model_directory,
                                    var_file,
                                    method,
                                    dict_dir,
                                    formatted_h5ad_file):
    metrics_dict = defaultdict(list)

    # set up model evaluator for each specific model
    if method == "PCA":
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
            f"tmp_zero_shot_pretraining_task_geneformer_{random_string}")
        zero_shot_evaluator = GeneformerZeroShotEvaluator(
            geneformer_model, var_file, dict_dir, tmp_output_dir)

    pretraining_task_metrics = zero_shot_evaluator.evaluate_pretraining_task(adata)
    print("pretraining_task_metrics")
    print(pretraining_task_metrics)

    pretraining_task_metrics["percentage"] = percentage
    pretraining_task_metrics["seed"] = seed
    pretraining_task_metrics["downsampling_method"] = downsampling_method
    pretraining_task_metrics["dataset"] = dataset_name

    for key in pretraining_task_metrics:
        metrics_dict[key].append(pretraining_task_metrics[key])

    metrics_df = pd.DataFrame.from_dict(metrics_dict)

    return metrics_df


def main():
    method = sys.argv[1]
    h5ad_file = sys.argv[2]
    dataset_name = sys.argv[3]

    downsampling_method = sys.argv[4]
    percentage = int(sys.argv[5])
    seed = int(sys.argv[6])

    var_file = sys.argv[7]
    formatted_h5ad_file = sys.argv[8]
    model_directory = sys.argv[9]
    dict_dir = sys.argv[10]

    print("loading anndata")
    adata = ad.read_h5ad(h5ad_file)

    # downsample to 10k cells for speed
    sc.pp.subsample(adata, n_obs=10000)
    adata.to_memory()


    if dataset_name in ["sctab", "pbmc", "tabula", "hlca"]:
        # sctab datasets need to have their var names added
        var_df = pd.read_csv(var_file, index_col=0)
        var_df.index = var_df.index.map(str)
        adata.var = var_df
        adata.var_names = adata.var.feature_name
        print(adata.var_names)


    new_adata = prep_for_evaluation(adata, formatted_h5ad_file, var_file)



    if method == "SSL":
        print("processing anndata")
        sc.pp.normalize_per_cell(new_adata, counts_per_cell_after=1e4)
        sc.pp.log1p(new_adata)



    metrics_df = get_pretraining_task_metrics_df(new_adata,
                                               dataset_name,
                                               downsampling_method,
                                               seed,
                                               percentage,
                                               model_directory,
                                               var_file,
                                               method,
                                               dict_dir,
                                               formatted_h5ad_file)

    metrics_csv = f"pretraining_task_metrics_{method}_{dataset_name}_downsamplingmethod_{downsampling_method}_percentage_{percentage}_seed_{seed}.csv"

    out_dir = "pretraining_task_results"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    metrics_df.to_csv(out_dir + "/" + metrics_csv)


if __name__ == "__main__":
    main()
