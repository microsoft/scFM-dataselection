import sys
from pathlib import Path
import os
import string
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import anndata as ad
import scanpy as sc

import logging
import scvi
scvi.settings.verbosity = logging.ERROR

from evaluation_utils import prep_for_evaluation
from model_loaders import load_scvi_model, load_ssl_model, load_geneformer_model, get_ssl_checkpoint_file
from model_loaders import load_pca_model, load_SCimilarity_model
from zero_shot_model_evaluators import SCVIZeroShotEvaluator, SSLZeroShotEvaluator, GeneformerZeroShotEvaluator
from zero_shot_model_evaluators import SCimilarityZeroShotEvaluator, PretrainedPrincipalComponentsZeroShotEvaluator

# attempt to fix too many open files error
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class Tahoe100MPerturbationDataset(Dataset):
    def __init__(self, embeddor, adata, cell_line_col, var_file, sctab_format, num_control_cells):
        self.embeddor = embeddor

        # do this after subsetting so that we don't have to load big h5ads into memory
        adata = prep_for_evaluation(adata, sctab_format, var_file) 
        
        # then subset to perturbed and unperturbed cells
        # DMSO_TF is the control condition, so we use it to get unperturbed cells
        self.perturbed_adata = adata[adata.obs.drug != "DMSO_TF"]
        self.unperturbed_adata = adata[adata.obs.drug == "DMSO_TF"]

        print("number of unperturbed cells:", self.unperturbed_adata.n_obs)
        print(self.unperturbed_adata.obs.drug.value_counts())


        # subset to 20000 unperturbed cells since there are far more DMSO_TF cells than perturbed cells
        # this function has been deprecated in scanpy, but we are using an older version 
        self.unperturbed_adata = sc.pp.subsample(self.unperturbed_adata, n_obs=num_control_cells, random_state=0, copy=True)

        cell_lines = list(adata.obs.cell_name.cat.categories)

        self.cell_line_to_perturbed_cells = {}

        for cell_line in cell_lines:
            cell_line_perturbed_cells = self.perturbed_adata[self.perturbed_adata.obs[cell_line_col] == cell_line]
            self.cell_line_to_perturbed_cells[cell_line] = cell_line_perturbed_cells

        self.cell_embeddings = self.embeddor.get_embeddings(self.unperturbed_adata)
        

    def __len__(self):
        return self.unperturbed_adata.n_obs

    def __getitem__(self, idx):
        # get model embedding for the unperturbed cell
        cell = self.unperturbed_adata[idx]
        cell_embedding = self.embeddor.get_embeddings(cell)

        print('cell embedding')
        print(cell_embedding)
        print(cell_embedding.shape)

        # get the cell line name for the unperturbed cell
        cell_line = cell.obs.cell_name.iloc[0] # it's a single cell, so we can just take the first element

        # get the number of perturbed cells for this cell line
        num_perturbed_cells = len(self.cell_line_to_perturbed_cells[cell_line])

        # choose a random perturbed cell to pair with the unperturbed cell
        perturbed_cell_index = np.random.randint(0, num_perturbed_cells)

        # get the perturbed cell expression
        perturbed_cell_expression = torch.from_numpy(self.cell_line_to_perturbed_cells[cell_line].X[perturbed_cell_index].todense())

        """
        print('unperturbed cell index:', idx)
        print('perturbed cell index:', perturbed_cell_index)

        print("cell_embedding shape:", cell_embedding.shape)
        print("perturbed_cell_expression shape:", perturbed_cell_expression.shape)
        """
        
        # if cell embedding is a numpy array, convert it to a tensor
        if isinstance(cell_embedding, np.ndarray):
            cell_embedding = torch.from_numpy(cell_embedding)
            cell_embedding = cell_embedding.float() # convert to float32


        return cell_embedding, perturbed_cell_expression
    

def get_embedding_extractor(model_dir, model_name, downsampling_method, percentage, seed, sctab_format=None, scvi_h5ad_file=None, var_file=None, dict_dir=None):
    """
    Get an embedding extractor for the specified model.

    Args:
        model_dir (str): Directory containing the pre-trained models.
        model_name (str): Name of the pre-trained model architecture.
        downsampling_method (str): Downsampling method to use for loading pre-trained model.
        percentage (float): Data percentage to use for loading pre-trained model.
        seed (int): Seed to use for loading pre-trained model.

    Returns:
        ModelEmbeddingsMixin: An instance of the embedding extractor.
    """

    if model_name == "scVI":
        scvi_model = load_scvi_model(downsampling_method, percentage, seed, model_dir, scvi_h5ad_file)
        embedding_extractor = SCVIZeroShotEvaluator(model=scvi_model)
    elif model_name == "PretrainedPCA":
        model = load_pca_model(downsampling_method, percentage, seed, model_dir)
        embedding_extractor = PretrainedPrincipalComponentsZeroShotEvaluator(model)
    elif model_name == "SSL":
        ssl_checkpoint_file = get_ssl_checkpoint_file(
            downsampling_method, percentage, seed, ssl_directory=model_dir)
        ssl_model = load_ssl_model(ssl_checkpoint_file)
        embedding_extractor = SSLZeroShotEvaluator(model=ssl_model)
    elif model_name == "scVI":
        scvi_model = load_scvi_model(
            downsampling_method, percentage, seed, scvi_directory=model_dir, h5ad_file=sctab_format)
        embedding_extractor = SCVIZeroShotEvaluator(model=scvi_model)
    elif model_name == "Geneformer":
        geneformer_model = load_geneformer_model(
            downsampling_method, percentage, seed, model_dir)
        random_string = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=10))
        tmp_output_dir = Path(
            f"tmp_zero_shot_integration_geneformer_{random_string}")
        embedding_extractor = GeneformerZeroShotEvaluator(
            geneformer_model, var_file, dict_dir, tmp_output_dir)
    elif model_name == "SCimilarity":
        model = load_SCimilarity_model(downsampling_method, percentage, seed, model_dir)
        embedding_extractor = SCimilarityZeroShotEvaluator(model)
    else:
        raise ValueError(f"Model name {model_name} is not supported.")
 
    return embedding_extractor

def fine_tune_perturbation(model_dir,
                           model_name,
                           downsampling_method,
                           percentage,
                           seed,
                           train_adata,
                           val_adata,
                           sctab_format,
                           var_file,
                           scvi_format_h5ad_file,
                           geneformer_dict_dir):
    """
    Fine-tune a perturbation model using the provided parameters.

    Args:
        model_dir (str): Directory containing the pre-trained models.
        model_name (str): Name of the pre-trained model architecture.
        downsampling_method (str): Downsampling method to use for loading pre-trained model.
        percentage (float): Data percentage to use for loading pre-trained model.
        seed (int): Seed to use for loading pre-trained model.
        train_adata: Training AnnData object.
        val_adata: Validation AnnData object.

    Returns:
        perturbation_model: The fine-tuned perturbation model.
    """
    # embedding extractor
    print("Loading embedding extractor")
    embedding_extractor = get_embedding_extractor(model_dir, model_name, downsampling_method, percentage, seed, 
                                                  sctab_format=sctab_format,
                                                  scvi_h5ad_file=scvi_format_h5ad_file,
                                                  var_file=var_file,
                                                  dict_dir=geneformer_dict_dir)

    
    print("Creating Tahoe100MPerturbationDataset")
    val_adata= sc.pp.subsample(val_adata, n_obs=1000, random_state=0, copy=True)
    val_adata = prep_for_evaluation(val_adata, sctab_format, var_file) 
        
    embeddings = embedding_extractor.get_embeddings(val_adata)

    print("embeddings.shape")
    print(embeddings.shape)

    print("embeddings[0]")
    print(embeddings[0])

    return embeddings


def main():
    model_dir = sys.argv[1]
    model_name = sys.argv[2]
    dataset_name = sys.argv[3]
    train_h5ad_file = sys.argv[4]
    var_file = sys.argv[5]
    out_dir = sys.argv[6]
    downsampling_method = sys.argv[7]
    seed = sys.argv[8]
    percentage = int(sys.argv[9])
    sctab_format = sys.argv[10]
    val_h5ad_file = sys.argv[11]
    scvi_format_h5ad_file = sys.argv[12]
    geneformer_dict_dir = sys.argv[13]

    # Load training data
    print("Loading training and validation data")
    train_adata = None
    val_adata = ad.read_h5ad(val_h5ad_file)




    # train MLP perturbation model (that uses model embeddings as input)
    print("Fine-tuning MLP perturbation model")
    perturbation_model = fine_tune_perturbation(model_dir,
                                                model_name,
                                                downsampling_method,
                                                percentage,
                                                seed,
                                                train_adata,
                                                val_adata,
                                                sctab_format,
                                                var_file,
                                                scvi_format_h5ad_file,
                                                geneformer_dict_dir)





if __name__ == "__main__":
    main()
