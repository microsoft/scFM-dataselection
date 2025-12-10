import sys
import random
import string
from pathlib import Path
import torch

from model_loaders import load_scvi_model, load_ssl_model, load_pca_model, load_SCimilarity_model
from model_loaders import load_geneformer_model, get_ssl_checkpoint_file

from zero_shot_model_evaluators import SCVIZeroShotEvaluator
from zero_shot_model_evaluators import SSLZeroShotEvaluator
from zero_shot_model_evaluators import GeneformerZeroShotEvaluator
from zero_shot_model_evaluators import PretrainedPrincipalComponentsZeroShotEvaluator
from zero_shot_model_evaluators import SCimilarityZeroShotEvaluator

def get_evaluator(method, model_directory, downsampling_method, percentage, seed, formatted_h5ad_file, var_file, dict_dir):
    if method == "SSL":
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
    return zero_shot_evaluator


def main():
    scvi_model_dir = sys.argv[1]
    ssl_model_dir = sys.argv[2]
    geneformer_model_dir = sys.argv[3]
    scimilarity_model_dir = sys.argv[4]

    sctab_formatted_file = sys.argv[5]
    var_file = sys.argv[6]
    dict_dir = sys.argv[7]

    scvi_evaluator = get_evaluator(
        method="scVI",
        model_directory=scvi_model_dir,
        downsampling_method="random",
        percentage=1,
        seed=0,
        formatted_h5ad_file=sctab_formatted_file,
        var_file=None,
        dict_dir=None)
    
    ssl_evaluator = get_evaluator(
        method="SSL",
        model_directory=ssl_model_dir,
        downsampling_method="randomsplits",
        percentage=1,
        seed=0,
        formatted_h5ad_file=None,
        var_file=None,
        dict_dir=None)
    
    geneformer_evaluator = get_evaluator(
        method="Geneformer",
        model_directory=geneformer_model_dir,
        downsampling_method="random",
        percentage=1,
        seed=0,
        formatted_h5ad_file=None,
        var_file=var_file,
        dict_dir=dict_dir)
    
    scimilarity_evaluator = get_evaluator(
        method="SCimilarity",
        model_directory=scimilarity_model_dir,
        downsampling_method="random",
        percentage=1,
        seed=0,
        formatted_h5ad_file=None,
        var_file=None,
        dict_dir=None)

    scvi_num_params = sum(p.numel() for p in scvi_evaluator.model.module.parameters())
    # 12,454,434

    ssl_num_params = sum(p.numel() for p in ssl_evaluator.model.parameters())
    # 20,767,683

    geneformer_num_params = sum(p.numel() for p in geneformer_evaluator.geneform.model.parameters())
    # 10,288,722

    scimilarity_num_params = sum(p.numel() for p in scimilarity_evaluator.ce.model.parameters())
    # 22,032,515 # just encoder, double this for full model
