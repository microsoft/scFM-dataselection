import sys
from pathlib import Path
import os
import pickle

import pandas as pd

import torch.optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import anndata as ad

from evaluation_utils import prep_for_evaluation
from model_loaders import load_pca_model

sys.path.append(os.path.abspath('eval'))
print(sys.path)


class MLPClassifier(L.LightningModule):
    def __init__(self, n_classes, cell_type_column, n_input=50, n_hidden=128, dropout_rate=0.2):
        # call LightningModule constructor
        super(MLPClassifier, self).__init__()
        self.dropout_rate = dropout_rate
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer1 = nn.Linear(n_input, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_classes)
        self.ce = nn.CrossEntropyLoss()

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
        # compare predicted cell types with actual
        y_hat = self.forward(x)

        loss = self.ce(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # class labels are ints, convert to floats
        # initializing a new tensor so it needs to be sent to the GPU if we have one
        y = y.type(torch.FloatTensor).to(self.device)
        # compare predicted cell types with actual
        y_hat = self.forward(x)

        loss = self.ce(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class PcaCellTypeAnnotationDataset(Dataset):
    def __init__(self, adata, pca_model, cell_type_column):
        self.data adata.X @ pca_model
        self.cell_types = one_hot(torch.tensor(
            adata.obs[cell_type_column]).to(torch.int64)).numpy()

    def __len__(self):
        return len(self.cell_types)

    def __getitem__(self, idx):
        latent_representation = self.data[idx,]
        cell_type = self.cell_types[idx]
        return latent_representation, cell_type


def update_genes(fine_tune_adata, var_file, sctab_format):
    fine_tune_adata.var_names = fine_tune_adata.var.feature_name
    new_adata = prep_for_evaluation(fine_tune_adata, sctab_format, var_file)
    return new_adata


def map_celltypes(train_h5ad_file,
                  val_h5ad_file,
                  cell_type_column,
                  var_file,
                  sctab_format,
                  celltype_pkl):
    train_adata = ad.read_h5ad(train_h5ad_file)
    val_adata = ad.read_h5ad(val_h5ad_file)

    train_adata = update_genes(train_adata, var_file, sctab_format)
    val_adata = update_genes(val_adata, var_file, sctab_format)

    train_adata.obs[cell_type_column] = train_adata.obs[cell_type_column].astype(
        'category')
    cell_type_dict = dict(
        enumerate(train_adata.obs[cell_type_column].cat.categories))
    cell_type_dict = {v: k for k, v in cell_type_dict.items()}

    with open(celltype_pkl, 'wb') as f:
        pickle.dump(cell_type_dict, f)

    train_adata.obs[cell_type_column] = train_adata.obs[cell_type_column].apply(
        lambda x: cell_type_dict[x])
    val_adata.obs[cell_type_column] = val_adata.obs[cell_type_column].apply(
        lambda x: cell_type_dict[x])
    return train_adata, val_adata


def fine_tune_pca(pca_model_dir,
                  method,
                  percentage,
                  seed,
                  train_adata,
                  val_adata,
                  cell_type_column):
    n_classes = train_adata.obs[cell_type_column].nunique()
    pretrained_model = load_pca_model(method,
                                      percentage,
                                      seed,
                                      pca_directory)

    latent_dims = pretrained_model._module_kwargs['n_latent']

    train_dataset = PcaCellTypeAnnotationDataset(
        train_adata, pretrained_model, cell_type_column)
    val_dataset = PcaCellTypeAnnotationDataset(
        val_adata, pretrained_model, cell_type_column)

    # set up DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)

    classifier = MLPClassifier(n_classes=n_classes,
                               n_input=latent_dims,
                               cell_type_column=cell_type_column)

    # CPU trainer
    # trainer = L.Trainer(max_epochs=500, callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
    # GPU trainer
    trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=500, callbacks=[
                        EarlyStopping(monitor="val_loss", mode="min")])

    trainer.fit(classifier, train_dataloader, val_dataloader)

    return classifier


def main():
    scvi_model_dir = sys.argv[1]
    scvi_train_h5ad_format = sys.argv[2]
    dataset_name = sys.argv[3]
    train_h5ad_file = sys.argv[4]
    cell_type_column = sys.argv[5]
    var_file = sys.argv[6]
    out_dir = sys.argv[7]
    method = sys.argv[8]
    seed = sys.argv[9]
    percentage = sys.argv[10]
    sctab_format = sys.argv[11]
    val_h5ad_file = sys.argv[12]
    celltype_pkl_savepath = sys.argv[13]

    train_adata, val_adata = map_celltypes(
        train_h5ad_file,
        val_h5ad_file,
        cell_type_column,
        var_file,
        sctab_format,
        celltype_pkl_savepath)

    # train MLP classifier (that uses scVI embeddings)
    classifier = fine_tune_scvi(scvi_model_dir,
                                scvi_train_h5ad_format,
                                method,
                                percentage,
                                seed,
                                train_adata,
                                val_adata,
                                cell_type_column)

    # save MLP classifier (that uses scVI embeddings)
    print("Saving MLP classifier")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{out_dir}/scvi_finetuned_model_{method}_{percentage}pct_seed{seed}.pt"
    torch.save(classifier.state_dict(), save_path)


if __name__ == "__main__":
    main()
