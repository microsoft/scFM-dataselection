import sys
import os
import pickle
import shutil
from colorist import red, green
import multiprocessing as mp


import numpy as np
import pandas as pd

import anndata as ad
import scanpy as sc

from cellarr import build_cellarrdataset, CellArrDataset, MatrixOptions
import tiledb

from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl

from scimilarity.tiledb_data_models import CellMultisetDataModule
from scimilarity.training_models import MetricLearning


# todo remove this
#from tiledb_data_models import CellMultisetDataModule


from scimilarity.ontologies import *


def convert_cell_types_to_cell_ontologies(adata, cell_type_col):
    graph = import_cell_ontology()
    id_mapper = get_id_mapper(graph)
    # id_mapper maps cell ontology ids to cell types and we need the opposite here
    id_mapper = {value: key for key, value in id_mapper.items()}
    # map each cell type name to cell ontology id
    adata.obs[cell_type_col] = adata.obs[cell_type_col].replace(id_mapper)
    return adata


# train_adata.var.index is 0,..., 19330
# it needs to be gene symbols
# use adata var file
def process_adata(train_h5ad, val_h5ad, var_file, tiledb_base_path):
    red("Reading anndata from disk")
    train_adata = ad.read_h5ad(train_h5ad)
    val_adata = ad.read_h5ad(val_h5ad)

    red("Reading var file index from disk")
    var_df = pd.read_csv(var_file, index_col=0)
    var_df.index = var_df.index.map(str)

    # set var names for sctab datasets
    train_adata.var = var_df
    train_adata.var_names = train_adata.var.feature_name

    val_adata.var = var_df
    val_adata.var_names = val_adata.var.feature_name

    # prepend train and val to dataset names because they have datasets in common
    # due to data splitting strategy not taking study into account
    train_adata.obs['dataset_id'] = 'train_' + train_adata.obs['dataset_id'].astype(str)
    val_adata.obs['dataset_id'] = 'val_' + val_adata.obs['dataset_id'].astype(str)

    # set up train and val in tiledb studies
    val_studies = list(set(val_adata.obs['dataset_id']))
    train_studies = list(set(train_adata.obs['dataset_id']))

    red('Num train studies:' + str(len(train_studies)))
    red('Num val studies:' + str(len(val_studies)))

    # convert cell type labels to cell ontology ids
    train_adata = convert_cell_types_to_cell_ontologies(train_adata, "cell_type")
    val_adata = convert_cell_types_to_cell_ontologies(val_adata, "cell_type")


    # concatenate train and val anndatas
    adata = ad.concat([train_adata, val_adata])

    red("Creating required columns")

    adata.obs["datasetID"] = adata.obs.dataset_id
    red("Creating required columns 2")

    adata.obs["sampleID"] = adata.obs.donor_id
    red("Creating required columns 3")
    adata.obs["cellTypeOntologyID"] = adata.obs.cell_type
    red("Creating required columns 4")
    adata.obs["tissue"] = adata.obs.tissue
    red("Creating required columns 5")
    adata.obs["disease"] = adata.obs.disease

    # get columns for n_genes_by_counts, total_counts, total_counts_mt, pct_counts_mt from scanpy
    red("Creating required columns 6")
    # workaround for scanpy/scipy fixes bug
    # https://github.com/scverse/scanpy/issues/3331
    adata.X.indptr = adata.X.indptr.astype(np.int64)
    adata.X.indices = adata.X.indices.astype(np.int64)

    red("columns before qc")
    print(adata.obs.columns) # todo remove
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    red("columns after qc")
    print(adata.obs.columns) # todo remove

    # manually set predicted doublets to 0
    red("Creating required columns 7")
    adata.obs["predicted_doublets"] = 0

    red("Moving counts matrix to a layer where cellarr expects it")
    adata.layers["counts"] = adata.X  # Move the matrix to where CellArr expects it

    red("columns after updates")
    print(adata.obs.columns) # todo remove


    # clean up columns
    # taken from clean_obs funciton in:
    # https://genentech.github.io/scimilarity/notebooks/training_tutorial.html
    obs = adata.obs.copy()

    columns = [
        "datasetID", "sampleID", "cellTypeOntologyID", "tissue", "disease",
        "n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt",
        "predicted_doublets",
    ]
    obs = obs[columns].copy()

    convert_dict = {
        "datasetID": str,
        "sampleID": str,
        "cellTypeOntologyID": str,
        "tissue": str,
        "disease": str,
        "n_genes_by_counts": int,
        "total_counts": int,
        "total_counts_mt": int,
        "pct_counts_mt": float,
        "predicted_doublets": int,
    }
    adata.obs = obs.astype(convert_dict)


    red("Creating cellarr collection directory")
    # delete and recreate directory if it exists already
    output_dir = tiledb_base_path 
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.mkdir(output_dir)

    adatas = [adata] # todo add val adata?

    # create gene annotation
    gene_annotation = sorted(adata.var.index.tolist())
    gene_annotation = pd.DataFrame({"cellarr_gene_index": gene_annotation})

    # create metadata
    # todo drop columns
    #columns_to_keep = []
    #metadata_df = adata.obs[columns_to_keep]
    metadata_df = adata.obs
    metadata_df = metadata_df.reset_index(drop=True)


    red("metadata_df")
    print(metadata_df)

    red("Creating cellarr dataset")
    # Build the dataset - this is where the magic happens!
    dataset = build_cellarrdataset(
        gene_annotation= gene_annotation,
        cell_metadata= metadata_df,
        output_path=tiledb_base_path,
        files=adatas,
        matrix_options=MatrixOptions(matrix_name="counts", dtype=np.int16),
        num_threads=4,  # Adjust based on your CPU - more threads = more speed (usually)
    )

    return dataset, val_studies




# create required columns
# https://genentech.github.io/scimilarity/notebooks/training_tutorial.html


# Adapted from example training script
# https://github.com/Genentech/scimilarity/blob/main/scripts/train.py
def train_SCimilarity(tiledb_base_path,
                      val_studies,
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
                      suffix,
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

    # todo modify this
    model_name = (
        f"model_{batch_size}_{margin}_{latent_dim}_{len(hidden_dim)}_{triplet_loss_weight}_{suffix}"
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
    filter_condition = "cellTypeOntologyID!='nan'"

    red("Creating Cell Multiset Data Module")
    datamodule = CellMultisetDataModule(
        dataset_path=tiledb_base_path,
        gene_order=None, # genes are in the tiledb 
        val_studies=val_studies,
        exclude_studies=None,
        exclude_samples=None,
        counts_uri = "assays/counts",
        label_id_column="cellTypeOntologyID",
        study_column="datasetID",
        sample_column="sampleID",
        filter_condition=filter_condition,
        batch_size=batch_size,
        n_batches=n_batches,
        num_workers=num_workers,
        sparse=False,                         # check these
        remove_singleton_classes=True,        # check these
        persistent_workers=True,              # check these
    )
    red(f"Training data size: {datamodule.train_df.shape}")
    #red(f"Validation data size: {datamodule.val_df.shape}") # todo add val datasets

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

    # Use tensorboard to log training. Modify this based on your preferred logger.
    from pytorch_lightning.loggers import TensorBoardLogger

    red("Setting up logger")
    logger = TensorBoardLogger(
        log_folder,
        name=model_name,
        default_hp_metric=False,
        flush_secs=1,
        version=suffix,
    )

    #gpu_idx = args.g # todo check this, not currently used anywhere

    lr_monitor = LearningRateMonitor(logging_interval="step")

    params = {
        "max_epochs": max_epochs,
        "logger": True,
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

    # todo remove this comment
    #return trainer, datamodule


    red("mp startmethod before right training")
    print(mp.get_start_method())

    # todo haven't checked anything below this 
    ckpt_path = os.path.join(log_folder, model_name, suffix, "checkpoints")
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




    # todo take these as args
    train_h5ad = "/users/adenadel/data/sctab/random/idx_1pct_seed0/idx_1pct_seed0_TEST.h5ad"
    val_h5ad = "/users/adenadel/data/sctab/random/idx_1pct_seed0/idx_1pct_seed0_VAL.h5ad"
    tiledb_base_path ="./my_collection.tdb"

    """
    dataset, val_studies = build_cellarrdataset_from_sctab(train_h5ad, val_h5ad, tiledb_base_path)
    # todo we need to uncomment this when we are building more tiledbs
    # todo save val_studies to a file

    with open("val_studies.pickle", "wb") as file:
        pickle.dump(val_studies, file)
    """

    with open("val_studies.pickle", "rb") as file:
        val_studies = pickle.load(file)

    red("Setting up model params")
    hidden_dim            = [1024, 1024, 1024]
    latent_dim            = 128
    margin                = 0.05 # triplet loss margin
    negative_selection    = "semihard" # negative selection type: [semihard, random, hardest]
    triplet_loss_weight   = 0.001
    lr                    = 0.005
    batch_size            = 256 # todo check this 1000 was default
    n_batches             = 100 # todo check this 100 was default
    max_epochs            = 500
    cosine_annealing_tmax = 0
    suffix                = "tmp_suffix"  # change this
    model_folder          = "tmp_models"  # change this
    log_folder            = "tmp_logs"    # change this
    result_folder         = "tmp_results" # change this
    num_workers           = 8
    dropout               = 0.5
    input_dropout         = 0.4
    l1                    = 1e-4
    l2                    = 0.01

    red("Training SCimilarity")
    train_SCimilarity(tiledb_base_path,
                      val_studies,
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
                      suffix,
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
