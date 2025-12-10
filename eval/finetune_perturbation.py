import pickle
from collections import defaultdict
from pathlib import Path
import os
import random
import string
import sys

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad

from evaluation_utils import prep_for_evaluation

from model_loaders import load_scvi_model, get_ssl_checkpoint_file, load_ssl_model
from model_loaders import load_perturbation_model
from model_loaders import load_SCimilarity_model
from model_loaders import load_pca_model
from model_loaders import load_geneformer_model

from zero_shot_model_evaluators import GeneformerZeroShotEvaluator


from finetune_model_evaluators import NoPredictionPerturbationEvaluator
from finetune_model_evaluators import AveragePerturbationEvaluator
from finetune_model_evaluators import SCVIPerturbationEvaluator
from finetune_model_evaluators import SSLPerturbationEvaluator
from finetune_model_evaluators import SCimilarityPerturbationEvaluator
from finetune_model_evaluators import PretrainedPCAPerturbationEvaluator
from finetune_model_evaluators import GeneformerPerturbationEvaluator



DEFAULT_N_THREADS = 64
os.environ['OPENBLAS_NUM_THREADS'] = f"{DEFAULT_N_THREADS}"
os.environ['MKL_NUM_THREADS'] = f"{DEFAULT_N_THREADS}"
os.environ['OMP_NUM_THREADS'] = f"{DEFAULT_N_THREADS}"



def get_perturbation_metrics_df(adata,
                                finetuned_model_directory,
                                pretrained_model_directory,
                                downsampling_method,
                                seed,
                                percentage,
                                method,
                                dataset_name,
                                type_dim,
                                class_weights,
                                child_matrix,
                                geneformer_dict_dir,
                                var_dir,
                                scvi_train_format_h5ad,
                                perturbation_column,
                                perturbation_name,
                                cell_line_column):
    

    # we use perturbation_name.capitalize() to match the perturbation name in the adata.obs, which has the first letter capitalized
    if perturbation_name == perturbation_name.lower(): # i.e. dinaciclib, homoharringtonine need to be capitalized
        match_string = perturbation_name.capitalize()
    else: # i.e. ph797804 vs PH-797804
        match_string = perturbation_name
    perturbed_adata = adata[adata.obs[perturbation_column] == match_string].copy()
    unperturbed_adata = adata[adata.obs[perturbation_column] != match_string].copy()


    metrics_dict = defaultdict(list)

    embedding_sizes = {"scVI": 10,
                       "PretrainedPCA": 50,
                       "SSL": 64,
                       "Geneformer": 256, 
                       "SCimilarity": 128}

    if method == "NoPrediction":
        perturbation_evaluator = NoPredictionPerturbationEvaluator(cell_line_column)
    elif method == "Average":
        perturbation_evaluator = AveragePerturbationEvaluator(perturbed_adata, cell_line_column)
    elif method == "PretrainedPCA":
        fine_tuned_model = load_perturbation_model("PretrainedPCA",
                            dataset_name,
                            downsampling_method,
                            percentage,
                            seed,
                            embedding_sizes["PretrainedPCA"],
                            finetuned_model_directory)

        pretrained_model = load_pca_model(downsampling_method, percentage, seed, pretrained_model_directory)

        perturbation_evaluator = PretrainedPCAPerturbationEvaluator(fine_tuned_model, pretrained_model)
    elif method == "scVI":
        fine_tuned_model = load_perturbation_model("scVI",
                            dataset_name,
                            downsampling_method,
                            percentage,
                            seed,
                            embedding_sizes["scVI"],
                            finetuned_model_directory)

        pretrained_model = load_scvi_model(
            downsampling_method,
            percentage,
            seed,
            pretrained_model_directory,
            scvi_train_format_h5ad)

        perturbation_evaluator = SCVIPerturbationEvaluator(fine_tuned_model, pretrained_model)

    elif method == "SSL":
        fine_tuned_model = load_perturbation_model("SSL",
                            dataset_name,
                            downsampling_method,
                            percentage,
                            seed,
                            embedding_sizes["SSL"],
                            finetuned_model_directory)
        ssl_checkpoint_file = get_ssl_checkpoint_file(
            downsampling_method, percentage, seed, pretrained_model_directory, classifier=False)
        ssl_model = load_ssl_model(ssl_checkpoint_file)
        perturbation_evaluator = SSLPerturbationEvaluator(fine_tuned_model, ssl_model)
    elif method == "SCimilarity":
        fine_tuned_model = load_perturbation_model("SCimilarity",
                            dataset_name,
                            downsampling_method,
                            percentage,
                            seed,
                            embedding_sizes["SCimilarity"],
                            finetuned_model_directory)
        scimilarity_model = load_SCimilarity_model(downsampling_method, percentage, seed, pretrained_model_directory)
        perturbation_evaluator = SCimilarityPerturbationEvaluator(fine_tuned_model, scimilarity_model)

    elif method == "Geneformer":
        fine_tuned_model = load_perturbation_model("Geneformer",
                            dataset_name,
                            downsampling_method,
                            percentage,
                            seed,
                            embedding_sizes["Geneformer"],
                            finetuned_model_directory)
        # todo implement this
        geneformer_model = load_geneformer_model(downsampling_method, percentage, seed, pretrained_model_directory)

        random_string = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=10))
        tmp_output_dir = Path(
            f"tmp_zero_shot_integration_geneformer_{random_string}")
        zero_shot_evaluator = GeneformerZeroShotEvaluator(
            geneformer_model, var_dir, geneformer_dict_dir, tmp_output_dir)
        
        pretrained_model = zero_shot_evaluator

        perturbation_evaluator = GeneformerPerturbationEvaluator(
            fine_tuned_model,
            pretrained_model
        )



    labels_column = "cell_name" # todo this should be a parameter, not currently used in the eval
    perturbation_metrics = perturbation_evaluator.evaluate_perturbation(unperturbed_adata, perturbed_adata, labels_column)
    print("perturbation_metrics")
    print(perturbation_metrics)

    perturbation_metrics["percentage"] = percentage
    perturbation_metrics["seed"] = seed
    perturbation_metrics["downsampling_method"] = downsampling_method
    perturbation_metrics["dataset"] = dataset_name

    for key in perturbation_metrics:
        metrics_dict[key].append(perturbation_metrics[key])

    metrics_df = pd.DataFrame.from_dict(metrics_dict)

    return metrics_df






def evaluate_anndata(finetuned_model_directory,
                     pretrained_model_directory,
                     adata,
                     downsampling_method,
                     seed,
                     percentage,    
                     method,
                     dataset_name,
                     sctab_format,
                     var_dir,
                     out_dir,
                     metadata_path,
                     geneformer_dict_dir,
                     scvi_train_format_h5ad,
                     perturbation_column,
                     perturbation_name,
                     cell_line_column):
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

    """
    if method == "SSL":
        sc.pp.normalize_total(new_adata, target_sum=1e4)
        sc.pp.log1p(new_adata)
    """

    metrics_df = get_perturbation_metrics_df(new_adata,
                                             finetuned_model_directory,
                                            pretrained_model_directory,
                                            downsampling_method,
                                            seed,
                                            percentage,
                                            method,
                                            dataset_name,
                                            type_dim,
                                            class_weights,
                                            child_matrix,
                                            geneformer_dict_dir,
                                            var_dir,
                                            scvi_train_format_h5ad,
                                            perturbation_column,
                                            perturbation_name,
                                            cell_line_column)

    metrics_csv = f"finetune_perturbation_metrics_{method}_{dataset_name}_downsamplingmethod_{downsampling_method}_percentage_{percentage}_seed_{seed}.csv"

    metrics_df.to_csv(out_dir+'/'+metrics_csv)


def main():

    # example bash command (define variables before running in all caps):
    # FINETUNED_MODEL_DIRECTORY="/path/to/finetuned/model"
    # PRETRAINED_MODEL_DIRECTORY="/path/to/pretrained/model"
    # DATASET_NAME="dataset_name"
    # H5AD_FILE="/path/to/h5ad/file"
    # VAR_DIR="/path/to/var/file"
    # OUT_DIR="/path/to/output/directory"
    # METHOD="SSL"  # or "Average", "NoPrediction", "scVI", "Geneformer"
    # DOWNSAMPLING_METHOD="na"
    # SEED=na
    # PERCENTAGE=na
    # SCTAB_FORMAT="na"
    # METADATA_PATH="na"
    # GENEFORMER_DICT_DIR="na"
    # SCVI_TRAIN_FORMAT_H5AD="na"
    # PERTURBATION_COLUMN="drug"
    # PERTURBATION_NAME="Homoharringtonine"

    # python finetune_perturbation.py \
    #     $FINETUNED_MODEL_DIRECTORY \
    #     $PRETRAINED_MODEL_DIRECTORY \
    #     $DATASET_NAME \
    #     $H5AD_FILE \
    #     $VAR_DIR \
    #     $OUT_DIR \
    #     $METHOD \
    #     $DOWNSAMPLING_METHOD \
    #     $SEED \
    #     $PERCENTAGE \
    #     $SCTAB_FORMAT \
    #     $METADATA_PATH \
    #     $GENEFORMER_DICT_DIR \
    #     $SCVI_TRAIN_FORMAT_H5AD \
    #     $PERTURBATION_COLUMN \
    #     $PERTURBATION_NAME

    import sys
    finetuned_model_directory = sys.argv[1]
    pretrained_model_directory = sys.argv[2]
    dataset_name = sys.argv[3]
    h5ad_file = sys.argv[4]
    var_dir = sys.argv[5]
    out_dir = sys.argv[6]
    method = sys.argv[7]
    downsampling_method = sys.argv[8]
    seed = sys.argv[9]
    percentage = sys.argv[10]
    sctab_format = sys.argv[11]
    metadata_path = sys.argv[12]
    geneformer_dict_dir = sys.argv[13]
    scvi_train_format_h5ad = sys.argv[14]
    perturbation_column = sys.argv[15]
    perturbation_name = sys.argv[16]
    cell_line_column = sys.argv[17]


    adata = ad.read_h5ad(h5ad_file)
    print('read in data')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    evaluate_anndata(finetuned_model_directory,
                     pretrained_model_directory,
                     adata,
                     downsampling_method,
                     seed,
                     percentage,
                     method,
                     dataset_name,
                     sctab_format,
                     var_dir,
                     out_dir,
                     metadata_path,
                     geneformer_dict_dir,
                     scvi_train_format_h5ad,
                     perturbation_column,
                     perturbation_name,
                     cell_line_column)



if __name__ == "__main__":
    main()
