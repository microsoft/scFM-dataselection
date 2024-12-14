import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import torch.optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import scvi
import anndata as ad

from model_loaders import load_scvi_model
from evaluation_utils import scanpy_workflow, get_empty_anndata, prep_for_evaluation

from finetune_model_evaluators import SCVIFinetuneEvaluator

 

# todo this file is just for easy import of this class, we should properly import this from the finetune-scripts/scvi equivalent
class MLPClassifier(L.LightningModule):
    def __init__(self, n_classes, cell_type_column, n_input = 10, n_hidden = 128, dropout_rate = 0.2):
        super(MLPClassifier, self).__init__() # call LightningModule constructor 
        self.dropout_rate = dropout_rate
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer1 = nn.Linear(n_input, n_hidden) # n_iput 10=is the default for scVI
        self.layer2 = nn.Linear(n_hidden, n_classes) # n_hidden=128 is the default scVI's VAE
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

