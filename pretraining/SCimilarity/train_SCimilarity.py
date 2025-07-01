import sys
import os
from colorist import red
import json
import multiprocessing as mp
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import anndata as ad
import scanpy as sc

from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from scimilarity.training_models import MetricLearning
from scimilarity.anndata_data_models import MetricLearningDataModule



# Adapted from example training script
# https://github.com/Genentech/scimilarity/blob/main/scripts/train.py
def train_SCimilarity(train_h5ad,
                      val_h5ad,
                      hidden_dim,
                      latent_dim,
                      margin,
                      negative_selection,
                      triplet_loss_weight,
                      lr,
                      batch_size,
                      n_batches,
                      max_epochs,
                      cosine_annealing_tmax,
                      prefix,
                      model_folder,
                      log_folder,
                      result_folder,
                      num_workers,
                      dropout,
                      input_dropout,
                      l1,
                      l2):

    if cosine_annealing_tmax == 0:
        cosine_annealing_tmax = max_epochs

    model_name = (
        f"SCimilarity_{prefix}_{batch_size}_{margin}_{latent_dim}_{len(hidden_dim)}_{triplet_loss_weight}"
    )

    red("Making output directories")
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    result_folder = os.path.join(result_folder, model_name)
    os.makedirs(result_folder, exist_ok=True)
 
    red(model_name)
    if os.path.isdir(os.path.join(model_folder, model_name)):
        sys.exit(0)

    # change this/see if it is necessary
    #filter_condition = f"cellTypeOntologyID!='nan' and total_counts>1000 and n_genes_by_counts>500 and pct_counts_mt<20 and predicted_doublets==0 and cellTypeOntologyID!='CL:0009010'"
    #filter_condition = "cellTypeOntologyID!='nan'"
    # this was used in the original script the relied on cellarr/tiledb


    red("creating datamodule")
    datamodule = MetricLearningDataModule(
        train_path = train_h5ad,
        val_path = val_h5ad,
        label_column="cell_type",
        study_column="datasetID",
        gene_order_file = None,
        batch_size = batch_size,
        num_workers=num_workers,
        sparse=False,
        remove_singleton_classes=True,
        pin_memory=False,
        persistent_workers=False,
        multiprocessing_context="fork")

    red("Creating Model Object")
    model = MetricLearning(
        datamodule.n_genes,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        input_dropout=input_dropout,
        margin=margin,
        negative_selection=negative_selection,
        sample_across_studies=True,       # check these
        perturb_labels=False,            # check these
        #perturb_labels_fraction=args.perturb_fraction, # check these
        lr=lr,
        triplet_loss_weight=triplet_loss_weight,
        l1=l1,
        l2=l2,
        max_epochs=max_epochs,
        cosine_annealing_tmax=cosine_annealing_tmax,
        #track_triplets=result_folder, # uncomment this to track triplet compositions per step
    )    

    red("Setting up logger")
    logger = TensorBoardLogger(
        log_folder,
        name=model_name,
        default_hp_metric=False,
        flush_secs=1,
        version=prefix,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    params = {
        "max_epochs": max_epochs,
        #"logger": True,
        "logger": logger,
        "accelerator": "gpu",
        "callbacks": [lr_monitor],
        "log_every_n_steps": 1,
        "limit_train_batches": n_batches,
        "limit_val_batches": 10,
        "limit_test_batches": 10,
    }

    red("Creating lightning trainer")
    trainer = pl.Trainer(**params)

    red("mp startmethod before right training")
    print(mp.get_start_method())

    ckpt_path = os.path.join(log_folder, model_name, prefix, "checkpoints")
    if os.path.isdir(ckpt_path): # resume training if checkpoints exist
        ckpt_files = sorted(
            [x for x in os.listdir(ckpt_path) if x.endswith(".ckpt")],
            key=lambda x: int(x.replace(".ckpt", "").split("=")[-1]),
        )
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=os.path.join(ckpt_path, ckpt_files[-1]),
        )
    else:
        trainer.fit(model, datamodule=datamodule)


    model.save_all(model_path=os.path.join(model_folder, model_name))

    if result_folder is not None:
        test_results = trainer.test(model, datamodule=datamodule)
        if test_results:
            with open(os.path.join(result_folder, f"{model_name}.test.json"), "w+") as fh:
                fh.write(json.dumps(test_results[0]))
    red(model_name)


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
    red("mp startmethod before")
    print(mp.get_start_method())
    # Running the script without this triggered a warning from TileDB
    # https://github.com/TileDB-Inc/TileDB-Py/issues/197
    # must be inside __main__ or a function apparently
    #https://github.com/TileDB-Inc/TileDB-Py/blob/main/examples/parallel_csv_ingestion.py
    if mp.get_start_method() is None:
        mp.set_start_method('spawn')

    # Setting start method to 'spawn' is required to
    # avoid problems with process global state when spawning via fork.
    # NOTE: *must be inside __main__* or a function.
    if mp.get_start_method(True) != "spawn":
        mp.set_start_method("spawn", True)

    red("mp startmethod after")
    print(mp.get_start_method())

    # todo check if anndata needs to be normalized etc.

    parser = argparse.ArgumentParser(
        description='Train a SCimilarity model with specified data.')
    parser.add_argument('--sctab')
    parser.add_argument('--percent')
    parser.add_argument('--downsample')
    parser.add_argument('--seed')

    ## bash script to run this:
    # python train_SCimilarity.py --sctab /users/adenadel/data/sctab --percent 1 --downsample random --seed 0

    args = parser.parse_args()
    sctab_dir = args.sctab
    percent = args.percent
    downsample = args.downsample
    seed = args.seed

    train_h5ad = Path(sctab_dir) / downsample / f"idx_{percent}pct_seed{seed}/idx_{percent}pct_seed{seed}_TRAIN.h5ad"
    val_h5ad = Path(sctab_dir) / downsample / f"idx_{percent}pct_seed{seed}/idx_{percent}pct_seed{seed}_VAL.h5ad"

    prefix = f"SCimilarity_{downsample}_{percent}pct_seed{seed}"
    

    red("Setting up model params")
    hidden_dim            = [1024, 1024, 1024]
    latent_dim            = 128
    margin                = 0.05 # triplet loss margin
    negative_selection    = "semihard" # negative selection type: [semihard, random, hardest]
    triplet_loss_weight   = 0.001
    lr                    = 0.005
    batch_size            = 1000 # todo check this 1000 was default
    n_batches             = 100 # todo check this 100 was default
    max_epochs            = 500 # todo
    cosine_annealing_tmax = 0
    prefix                = prefix
    model_folder          = "models_SCimilarity" 
    log_folder            = "logs_SCimilarity"   
    result_folder         = "results_SCimilarity" 
    num_workers           = 8
    dropout               = 0.5
    input_dropout         = 0.4
    l1                    = 1e-4
    l2                    = 0.01

    red("Training SCimilarity")
    train_SCimilarity(train_h5ad,
                      val_h5ad,
                      hidden_dim,
                      latent_dim,
                      margin,
                      negative_selection,
                      triplet_loss_weight,
                      lr,
                      batch_size,
                      n_batches,
                      max_epochs,
                      cosine_annealing_tmax,
                      prefix,
                      model_folder,
                      log_folder,
                      result_folder,
                      num_workers,
                      dropout,
                      input_dropout,
                      l1,
                      l2)


if __name__ == "__main__":
    main()

