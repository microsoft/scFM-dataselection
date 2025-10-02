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

# same as MLPClassifier used for finetuning scvi in ../scvi/scvi_mlp.py
# but without the CrossEntropyLoss and instead uses MSELoss
class MLPPerturbationModel(L.LightningModule):
    def __init__(self, n_input, n_output, n_hidden=128, dropout_rate=0.2):
        # call LightningModule constructor
        super(MLPPerturbationModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        # n_input 10=is the default for scVI
        self.layer1 = nn.Linear(n_input, n_hidden)
        # n_hidden=128 is the default scVI's VAE
        self.layer2 = nn.Linear(n_hidden, n_output)
        self.mse = nn.MSELoss()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        # class labels are ints, convert to floats
        # initializing a new tensor so it needs to be sent to the GPU if we have one
        y = y.type(torch.FloatTensor).to(self.device)
        # compare predicted expression with perturbed expression
        y_hat = self.forward(x)

        loss = self.mse(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # class labels are ints, convert to floats
        # initializing a new tensor so it needs to be sent to the GPU if we have one
        y = y.type(torch.FloatTensor).to(self.device)
        # compare predicted expression with perturbed expression
        y_hat = self.forward(x)

        loss = self.mse(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


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
        cell = self.unperturbed_adata[idx] # use this for metadata
        cell_embedding = self.cell_embeddings[idx] # use precomputed cell embeddings (recomputing them is slow with geneformer)

        #print('cell embedding')
        #print(cell_embedding)
        #print(cell_embedding.shape)

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
    cell_line_col = "cell_name" # this is the column in the tahoe 100M dataset that contains the cell line names
    train_control_cells = 20000 
    train_dataset = Tahoe100MPerturbationDataset(embedding_extractor, train_adata, cell_line_col, var_file, sctab_format, train_control_cells)
    val_control_cells = 2000 
    val_dataset = Tahoe100MPerturbationDataset(embedding_extractor, val_adata, cell_line_col, var_file, sctab_format, val_control_cells)

    # set up DataLoaders
    if model_name == "Geneformer":
        numworkers = 0 # Geneformer tokenization is not thread-safe, so we set num_workers to 0 to use the main process
    else:
        numworkers = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=numworkers)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=numworkers)

    print(embedding_extractor)
    print(train_dataset)
    print(val_dataset)

    embedding_sizes = {"scVI": 10,
                       "PretrainedPCA": 50,
                       "SSL": 64,
                       "Geneformer": 256, 
                       "SCimilarity": 128}
    
    sctab_ngenes = 19331

    perturbation_model = MLPPerturbationModel(n_input=embedding_sizes[model_name],
                                               n_output=sctab_ngenes)

    # CPU trainer
    trainer = L.Trainer(max_epochs=500, callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
    # GPU trainer
    #trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=500, callbacks=[
    #                    EarlyStopping(monitor="val_loss", mode="min")])

    trainer.fit(perturbation_model, train_dataloader, val_dataloader)

    return perturbation_model


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

    print("CUDA available:", torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    # Load training data
    print("Loading training and validation data")
    train_adata = ad.read_h5ad(train_h5ad_file)
    val_adata = ad.read_h5ad(val_h5ad_file)

    """
    # the adata.var doesn't have ensembl_ids, so we need to add them
    train_adata.var["feature_name"] = train_adata.var.index # the feature names are in the index

    df = pd.read_csv(var_file, index_col=0)
    df.index = df.index.map(str) # anndata expects string indices
    new_var = pd.merge(train_adata.var, df, on="feature_name", how="left") # this loses the gene name index

    # set index to gene names
    new_var = new_var.set_index("feature_name", drop=False)

    train_adata.var = new_var

    # repeat for validation data
    val_adata.var["feature_name"] = val_adata.var.index # the feature names are in the index
    new_var = pd.merge(val_adata.var, df, on="feature_name", how="left") # this loses the gene name index
    val_adata.var = new_var
    """





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

    print("Saving MLP perturbation model")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{out_dir}/{model_name}_{dataset_name}_perturbation_finetuned_model_{downsampling_method}_{percentage}pct_seed{seed}.pt"
    torch.save(perturbation_model.state_dict(), save_path)



if __name__ == "__main__":
    main()
