import pickle
from collections import defaultdict
from pathlib import Path
import os

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad

from evaluation_utils import prep_for_evaluation

from finetune_model_evaluators import LogisticRegressionFineTuneEvaluator
from finetune_model_evaluators import PretrainedPCAFinetuneEvaluator
from finetune_model_evaluators import SCVIFinetuneEvaluator
from finetune_model_evaluators import SSLFinetuneEvaluator
from finetune_model_evaluators import GeneformerFinetuneEvaluator


from model_loaders import load_scvi_model, get_ssl_checkpoint_file, load_ssl_classifier
from model_loaders import load_finetuned_geneformer_model, load_fine_tuned_scvi_model
from model_loaders import load_pca_model, load_fine_tuned_pretrainedpca_model


DEFAULT_N_THREADS = 64
os.environ['OPENBLAS_NUM_THREADS'] = f"{DEFAULT_N_THREADS}"
os.environ['MKL_NUM_THREADS'] = f"{DEFAULT_N_THREADS}"
os.environ['OMP_NUM_THREADS'] = f"{DEFAULT_N_THREADS}"


def get_classification_metrics_df(adata,
                                  downsampling_method,
                                  seed,
                                  percentage,
                                  cell_type_col,
                                  directory,
                                  method,
                                  dataset_name,
                                  type_dim,
                                  class_weights,
                                  child_matrix,
                                  geneformer_dict_dir,
                                  var_dir,
                                  cell_type_dict,
                                  pretrained_model_directory,
                                  scvi_train_format_h5ad):

    metrics_dict = defaultdict(list)

    if method == "logistic_regression":
        logistic_regression_pickle_file = Path(
            directory) / f"{dataset_name}_logistic_classifier.pkl"
        finetune_evaluator = LogisticRegressionFineTuneEvaluator(
            logistic_regression_pickle_file, cell_type_col, cell_type_dict)
    elif method == "SSL":
        ssl_checkpoint_file = get_ssl_checkpoint_file(
            downsampling_method, percentage, seed, directory, classifier=True)
        ssl_model = load_ssl_classifier(ssl_checkpoint_file,
                                        type_dim=type_dim,
                                        class_weights=class_weights,
                                        child_matrix=child_matrix,
                                        batch_size=16384)
        finetune_evaluator = SSLFinetuneEvaluator(
            model=ssl_model, celltype_dict=cell_type_dict)
    elif method == "PretrainedPCA":
        num_classes = len(cell_type_dict)

        fine_tuned_model = load_fine_tuned_pretrainedpca_model(
            downsampling_method,
            percentage,
            seed,
            pretrained_model_directory,
            directory,
            dataset_name,
            cell_type_col,
            num_classes)

        pretrained_pca_model = load_pca_model(
            downsampling_method,
            percentage,
            seed,
            pretrained_model_directory)

        finetune_evaluator = PretrainedPCAFinetuneEvaluator(fine_tuned_model, pretrained_pca_model, cell_type_col, cell_type_dict)


    elif method == "scVI":
        num_classes = len(cell_type_dict)
        fine_tuned_model = load_fine_tuned_scvi_model(
            downsampling_method,
            percentage,
            seed,
            pretrained_model_directory,
            scvi_train_format_h5ad,
            directory,
            dataset_name,
            cell_type_col,
            num_classes)

        pretrained_model = load_scvi_model(
            downsampling_method,
            percentage,
            seed,
            pretrained_model_directory,
            scvi_train_format_h5ad)

        finetune_evaluator = SCVIFinetuneEvaluator(
            fine_tuned_model,
            pretrained_model,
            cell_type_col,
            cell_type_dict)
    elif method == "Geneformer":
        geneformer_directory = directory
        num_classes = len(cell_type_dict)
        geneformer_model = load_finetuned_geneformer_model(
            downsampling_method, percentage, seed, dataset_name, geneformer_directory)

        finetune_evaluator = GeneformerFinetuneEvaluator(
            geneformer_model,
            num_classes,
            var_dir,
            geneformer_dict_dir,
            cell_type_col,
            cell_type_dict)

    classification_metrics = finetune_evaluator.evaluate_classification(adata)
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


def evaluate_anndata(model_directory,
                     adata,
                     downsampling_method,
                     seed,
                     percentage,
                     method,
                     dataset_name,
                     cell_type_col,
                     sctab_format,
                     var_dir,
                     out_dir,
                     ssl_metadata_path,
                     geneformer_dict_dir,
                     cell_type_dict,
                     pretrained_model_directory,
                     scvi_train_format_h5ad):
    if dataset_name in ["pbmc", "tabula", "hlca"]:
        # sctab datasets need to have their var names added
        var_df = pd.read_csv(var_dir, index_col=0)
        var_df.index = var_df.index.map(str)
        adata.var = var_df
        adata.var_names = adata.var.feature_name

    new_adata = prep_for_evaluation(adata, sctab_format, var_dir)

    # Model specific params get set for each model
    type_dim = None
    class_weights = None
    child_matrix = None

    if method == "SSL":
        cell_type_dict = pd.read_parquet(
            ssl_metadata_path+'/categorical_lookup/cell_type.parquet')
        type_dim = len(cell_type_dict)
        class_weights = np.load(ssl_metadata_path+'/class_weights.npy')
        child_matrix = np.load(
            ssl_metadata_path+'/cell_type_hierarchy/child_matrix.npy')

        print("processing anndata")

        sc.pp.normalize_total(new_adata, target_sum=1e4)
        sc.pp.log1p(new_adata)

    metrics_df = get_classification_metrics_df(new_adata,
                                               downsampling_method,
                                               seed,
                                               percentage,
                                               cell_type_col,
                                               model_directory,
                                               method,
                                               dataset_name,
                                               type_dim,
                                               class_weights,
                                               child_matrix,
                                               geneformer_dict_dir,
                                               var_dir,
                                               cell_type_dict,
                                               pretrained_model_directory,
                                               scvi_train_format_h5ad)

    metrics_csv = f"finetune_classification_metrics_{method}_{dataset_name}_downsamplingmethod_{downsampling_method}_percentage_{percentage}_seed_{seed}.csv"

    metrics_df.to_csv(out_dir+'/'+metrics_csv)


def main():

    import sys
    model_directory = sys.argv[1]
    dataset_name = sys.argv[2]
    h5ad_file = sys.argv[3]
    cell_type_col = sys.argv[4]
    var_dir = sys.argv[5]
    out_dir = sys.argv[6]
    method = sys.argv[7]
    downsampling_method = sys.argv[8]
    seed = sys.argv[9]
    percentage = int(sys.argv[10])
    sctab_format = sys.argv[11]
    ssl_metadata_path = sys.argv[12]
    cell_type_pickle_file = sys.argv[13]
    geneformer_dict_dir = sys.argv[14]
    pretrained_model_directory = sys.argv[15] # for pretrained PCA and scVI
    scvi_train_format_h5ad = sys.argv[16]

    cell_type_dict = None
    if method in ['Geneformer', "scVI", "PretrainedPCA", "logistic_regression"]:
        with open(cell_type_pickle_file, 'rb') as fh:
            cell_type_dict = pickle.load(fh)

    print('Loading adata from', h5ad_file)
    adata = ad.read_h5ad(h5ad_file)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    evaluate_anndata(model_directory,
                     adata,
                     downsampling_method,
                     seed,
                     percentage,
                     method,
                     dataset_name,
                     cell_type_col,
                     sctab_format,
                     var_dir,
                     out_dir,
                     ssl_metadata_path,
                     geneformer_dict_dir,
                     cell_type_dict,
                     pretrained_model_directory,
                     scvi_train_format_h5ad)


if __name__ == "__main__":
    main()
