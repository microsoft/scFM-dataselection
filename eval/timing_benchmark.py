import sys
import string
import time
from pathlib import Path
import random

import pandas as pd
print(pd.__version__)

import anndata as ad

from model_loaders import load_scvi_model, load_ssl_model, load_pca_model
from model_loaders import load_geneformer_model, get_ssl_checkpoint_file

from zero_shot_model_evaluators import SCVIZeroShotEvaluator
from zero_shot_model_evaluators import SSLZeroShotEvaluator
from zero_shot_model_evaluators import GeneformerZeroShotEvaluator
from zero_shot_model_evaluators import PretrainedPrincipalComponentsZeroShotEvaluator

def benchmark_timing(h5ad_file,
                     num_cells,
                     downsampling_method,
                     seed,
                     percentage,
                     var_file,
                     dict_dir,
                     formatted_h5ad_file,
                     models,
                     model_directory_dict):

    print("Loading data from", h5ad_file)
    adata = ad.read_h5ad(h5ad_file)
    
    print("Subsampling adata to the first {} cells".format(num_cells))
    adata = adata[:num_cells, :].copy()

    metrics_dict = {
        "model": [],
        "downsampling_method": [],
        "percentage": [],
        "seed": [],
        "elapsed_time": [],
        "num_cells": [],
    }

    for model in models:
        model_directory = model_directory_dict[model]
        print(f"Benchmarking {model}")
        # load model evaluator for each model
        if model == "SSL":
            ssl_checkpoint_file = get_ssl_checkpoint_file(
                downsampling_method, percentage, seed, ssl_directory=model_directory)
            ssl_model = load_ssl_model(ssl_checkpoint_file)
            zero_shot_evaluator = SSLZeroShotEvaluator(model=ssl_model)
        elif model == "scVI":
            scvi_model = load_scvi_model(
                downsampling_method, percentage, seed, scvi_directory=model_directory, h5ad_file=h5ad_file)
            zero_shot_evaluator = SCVIZeroShotEvaluator(model=scvi_model)
        elif model == "Geneformer":
            geneformer_model = load_geneformer_model(
                downsampling_method, percentage, seed, model_directory)
            random_string = ''.join(random.choices(
                string.ascii_uppercase + string.digits, k=10))
            tmp_output_dir = Path(
                f"tmp_zero_shot_classification_geneformer_{random_string}")
            zero_shot_evaluator = GeneformerZeroShotEvaluator(
                geneformer_model, var_file, dict_dir, tmp_output_dir)
        elif model == "PretrainedPCA": # todo test this
            pca_model = load_pca_model(downsampling_method, percentage, seed, model_directory)
            zero_shot_evaluator = PretrainedPrincipalComponentsZeroShotEvaluator(pca_model)

        
        # benchmark the time taken to get embeddings

        # for geneformer we need to measure the function both tokenizes and embeds the 
        # data, so we need to measure the time taken for the embedding step inside
        # the get_embeddings function
        # for the other models we just measure the time taken for the get_embeddings function
        # to run, since they do not require tokenization
        if model == "Geneformer": 
            start_time = time.perf_counter()
            embeddings = zero_shot_evaluator.get_embeddings(adata, return_time_taken=True)
            print("embddings")
            print(embeddings)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
        else:
            start_time = time.perf_counter()
            embeddings = zero_shot_evaluator.get_embeddings(adata)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

        print(f"Time taken for {model} to get embeddings: {elapsed_time:.2f} seconds")

        # store the metrics
        if model == "Geneformer":
            time_taken_with_tokenization = elapsed_time
            # create row in table for Geneformer+tokenization
            metrics_dict["model"].append("Geneformer+tokenization")
            metrics_dict["downsampling_method"].append(downsampling_method)
            metrics_dict["percentage"].append(percentage)
            metrics_dict["seed"].append(seed)
            metrics_dict["elapsed_time"].append(time_taken_with_tokenization)
            metrics_dict["num_cells"].append(adata.n_obs)
            # store the time taken for the embedding step
            elapsed_time = embeddings[1]
            embeddings = embeddings[0] # we don't actually use the embeddings, but this contains them like the other models now


        metrics_dict["model"].append(model)
        metrics_dict["downsampling_method"].append(downsampling_method)
        metrics_dict["percentage"].append(percentage)
        metrics_dict["seed"].append(seed)
        metrics_dict["elapsed_time"].append(elapsed_time)
        metrics_dict["num_cells"].append(adata.n_obs)


    timing_df = pd.DataFrame.from_dict(metrics_dict)
    print(timing_df)

    return timing_df

def main():
    num_cells = int(sys.argv[1])
    h5ad_file = sys.argv[2]

    downsampling_method = sys.argv[3]
    percentage = int(sys.argv[4])
    seed = int(sys.argv[5])

    var_file = sys.argv[6]
    formatted_h5ad_file = sys.argv[7]
    dict_dir = sys.argv[8]
    scvi_model_directory = sys.argv[9]
    ssl_model_directory = sys.argv[10]
    geneformer_model_directory = sys.argv[11]
    pretrained_pca_model_directory = sys.argv[12]
    outdir = sys.argv[13]



    timing_df = benchmark_timing(
        h5ad_file=h5ad_file,
        num_cells=num_cells,
        downsampling_method=downsampling_method,
        seed=seed,
        percentage=percentage,
        var_file=var_file,
        dict_dir=dict_dir,
        formatted_h5ad_file=formatted_h5ad_file,
        models=["SSL", "scVI", "Geneformer", "PretrainedPCA"],
        model_directory_dict={
            "SSL": ssl_model_directory,
            "scVI": scvi_model_directory,
            "Geneformer": geneformer_model_directory,
            "PretrainedPCA": pretrained_pca_model_directory
            }
        )

    import os

    timing_df.to_csv(os.path.join(outdir, f'{num_cells}cells_seed{seed}_timing_benchmark_results.csv'), index=False)
if __name__ == "__main__":
    main()