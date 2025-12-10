import os
import abc
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from pathlib import Path
import shutil
import pickle


import random
import string
import matplotlib.pyplot as plt
import seaborn as sns
import umap

import numpy as np
import pandas as pd
from scipy import stats

import sklearn

from scipy.sparse import issparse

from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import ExplainedVariance, MeanSquaredError, MetricCollection
import lightning.pytorch as pl

import anndata as ad
from anndata import AnnData
import scanpy as sc
import scvi

import scib

from evaluation_utils import tokenize_adata, eval_classification_metrics
import sys

from scimilarity import CellEmbedding


#sys.path.append('./Geneformer/')
from geneformer import perturber_utils as pu
from geneformer import classifier_utils as cu
from geneformer.evaluation_utils import classifier_predict
from sc_foundation_evals import geneformer_forward as gf
from sc_foundation_evals import data
from datasets import load_from_disk


class PerturbationEvaluator:
    def predict_perturbation(self, adata):
        raise NotImplementedError
    def evaluate_perturbation(self, unperturbed_adata, perturbed_adata, labels_column):
        predicted_perturbation = self.predict_perturbation(unperturbed_adata)

        print(predicted_perturbation[0:10, 0:20])

        """How scgen does it:
        stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
        ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
        x_diff = numpy.average(ctrl_diff.X, axis=0)
        y_diff = numpy.average(stim_diff.X, axis=0)
        m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(x_diff, y_diff)

        # todo implement this and global MSE
        """

        # todo incorporate labels column into the evaluation

        #predicted_perturbed_means =  np.array(np.average(predicted_perturbation.X, axis=0))
        #real_perturbed_means =  np.array(np.average(perturbed_adata.X, axis=0))

        if issparse(predicted_perturbation):
            predicted_perturbation = predicted_perturbation.toarray()

        if issparse(perturbed_adata.X):
            perturbed_adata.X = perturbed_adata.X.toarray()

        # currently these are a scalar rather than a P dimensional vector
        predicted_perturbed_means =  np.array(np.average(predicted_perturbation, axis=0))
        real_perturbed_means =  np.array(np.average(perturbed_adata.X, axis=0))

        predicted_perturbed_means = predicted_perturbed_means.squeeze()
        real_perturbed_means = real_perturbed_means.squeeze()

        print("predicted_perturbed_means.shape:")
        print(predicted_perturbed_means.shape)
        print("predicted_perturbed_means.shape:")
        print(real_perturbed_means.shape)

        print("predicted_perturbed_means:")
        print(predicted_perturbed_means)
        print("predicted_perturbed_means:")
        print(real_perturbed_means)

        # todo do this on a per cell line basis

        # flatten because one has shape (,19331) and the other has (1, 19331)
        m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(predicted_perturbed_means, real_perturbed_means)

        print("real_perturbed_means")
        print(real_perturbed_means)
        print("predicted_perturbed_means")
        print(predicted_perturbed_means)

        # Calculate the Mean Squared Error (MSE)
        mse = sklearn.metrics.mean_squared_error(real_perturbed_means, predicted_perturbed_means)

        metrics_dict = {
            "mse": mse,
            "R2": r_value_diff
        }
        return metrics_dict # todo test this whole function
   
class NoPredictionPerturbationEvaluator(PerturbationEvaluator):
    def __init__(self, cell_line_column):
        self.cell_line_column = cell_line_column
    def predict_perturbation(self, adata):
        """
        This evaluator does not predict perturbations, it simply returns the input data.
        """
        adata.X = adata.X.toarray()
        return adata.X
    

class AveragePerturbationEvaluator(PerturbationEvaluator):
    def __init__(self, perturbed_adata, cell_line_column):
        self.perturbed_adata = perturbed_adata
        self.cell_line_column = cell_line_column

    def predict_perturbation(self, adata):
        """
        This evaluator predicts perturbations by averaging the input data.
        """
        # Calculate the average expression for each gene across all cells
        # todo double check if this is correct
        average_expression = np.mean(self.perturbed_adata.X.toarray(), axis=0)

        # average expression is a 1D array
        # we need to repeat it once for each cell

        average_expression = np.repeat(average_expression[np.newaxis, :], adata.n_obs, axis=0)

        print("average_expression")
        print(average_expression.shape)
       
        # Create a new AnnData object with the average expression
        # todo double check if this is correct
        average_adata = ad.AnnData(X=average_expression, var=adata.var)
       
        return average_adata.X
    
class PretrainedPCAPerturbationEvaluator(PerturbationEvaluator):
    def __init__(self, fine_tuned_model, pretrained_model):
        self.fine_tuned_model = fine_tuned_model
        self.pretrained_model = pretrained_model
    def predict_perturbation(self, adata):
        latent_representation = adata.X @ self.pretrained_model
        prediction = self.fine_tuned_model.forward(torch.tensor(latent_representation).to(torch.float32)).detach().numpy()
        return prediction

class SCVIPerturbationEvaluator(PerturbationEvaluator):
    def __init__(self, fine_tuned_model, pretrained_model):
        self.fine_tuned_model = fine_tuned_model
        self.pretrained_model = pretrained_model

    def predict_perturbation(self, adata):
        with torch.no_grad():
            latent_represetation = self.pretrained_model.get_latent_representation(adata)
            prediction = self.fine_tuned_model.forward(torch.tensor(latent_represetation)).numpy()

        return prediction


class SSLPerturbationEvaluator(PerturbationEvaluator):
    def __init__(self, fine_tuned_model, pretrained_model):
        self.fine_tuned_model = fine_tuned_model
        self.pretrained_model = pretrained_model

    def predict_perturbation(self, adata):
        input_tensor = adata.X.todense()
        with torch.no_grad():
            latent_represetation = self.pretrained_model.encoder(torch.tensor(input_tensor)).numpy()
            prediction = self.fine_tuned_model.forward(torch.tensor(latent_represetation)).detach().numpy()

        return prediction
    

class SCimilarityPerturbationEvaluator(PerturbationEvaluator):
    def __init__(self, fine_tuned_model, pretrained_model):
        self.fine_tuned_model = fine_tuned_model
        self.pretrained_model = pretrained_model

        self.ce = CellEmbedding(self.pretrained_model)

    def predict_perturbation(self, adata):
        latent_representation = self.ce.get_embeddings(adata.X)
        prediction = self.fine_tuned_model.forward(torch.tensor(latent_representation)).detach().numpy()

        return prediction
    
class GeneformerPerturbationEvaluator(PerturbationEvaluator):
    def __init__(self, finetuned_model, pretrained_model):
        self.finetuned_model = finetuned_model
        self.pretrained_model = pretrained_model

    def predict_perturbation(self, adata):
        latent_representation = self.pretrained_model.get_embeddings(adata)
        prediction = self.finetuned_model.forward(torch.tensor(latent_representation)).detach().numpy()

        return prediction


class FinetuneEvaluator:
    def get_labels(self, adata):
        raise NotImplementedError
    def evaluate_classification(self, adata):
        test_predictions, test_true = self.get_labels(adata)
        classification_metrics = eval_classification_metrics(test_true, test_predictions)
        return classification_metrics


class LogisticRegressionFineTuneEvaluator(FinetuneEvaluator):
    def __init__(self, logistic_regression_pickle_file, cell_type_col, cell_type_dict):
        self.cell_type_dict = cell_type_dict
        self.cell_type_col = cell_type_col

        logistic_regression_dict = pickle.load(open(logistic_regression_pickle_file, 'rb'))

        self.classifier = logistic_regression_dict["variable_gene_classifier"]
        self.variable_genes = logistic_regression_dict["variable_genes"]

    def get_labels(self, adata):
        variable_genes_adata = adata[:, self.variable_genes]
        variable_genes = variable_genes_adata.X

        print(self.cell_type_dict)
        print(adata.obs[self.cell_type_col])
        y_pred =  self.classifier.predict(variable_genes)
        y_true = adata.obs[self.cell_type_col].cat.rename_categories(self.cell_type_dict).to_numpy()

        return y_pred, y_true

 

class SSLFinetuneEvaluator(FinetuneEvaluator):
    def __init__(self, model, celltype_dict):
        self.model = model
        self.celltype_dict = celltype_dict
        
         
    def get_labels(self, adata):
        input_tensor = adata.X.todense()
        celltype_mapping = self.celltype_dict['label'].to_dict()
        celltype_mapping_rev = {v: k for k, v in celltype_mapping.items()}
        y_true = adata.obs['cell_type'].apply(lambda x: celltype_mapping_rev[x])
        with torch.no_grad():
            logits = self.model(torch.tensor(input_tensor))
            y_pred = torch.argmax(logits, dim=1)
        return y_pred, y_true


class SCVIFinetuneEvaluator(FinetuneEvaluator):
    def __init__(self, mlp_model, pretrained_model, cell_type_col, cell_type_dict):
        self.model = mlp_model
        self.pretrained_model = pretrained_model
        self.cell_type_col = cell_type_col
        self.cell_type_dict = cell_type_dict
    def get_labels(self, adata):
        latent_represetation = self.pretrained_model.get_latent_representation(adata)
        y_hat = self.model.forward(torch.tensor(latent_represetation))
        y_pred = y_hat.argmax(dim=1)
        y_true = adata.obs[self.cell_type_col].cat.rename_categories(self.cell_type_dict).to_numpy()
        return y_pred.numpy(), y_true
    

class PretrainedPCAFinetuneEvaluator(FinetuneEvaluator):
    def __init__(self, mlp_model, pretrained_pca_model, cell_type_col, cell_type_dict):
        self.model = mlp_model
        self.pretrained_pca_model = pretrained_pca_model
        self.cell_type_col = cell_type_col
        self.cell_type_dict = cell_type_dict
    def get_labels(self, adata):
        pca_embeddings =  adata.X @ self.pretrained_pca_model
        y_hat = self.model.forward(torch.tensor(pca_embeddings, dtype=torch.float32))
        y_pred = y_hat.argmax(dim=1)
        y_true = adata.obs[self.cell_type_col].cat.rename_categories(self.cell_type_dict).to_numpy()
        return y_pred.numpy(), y_true



class GeneformerFinetuneEvaluator(FinetuneEvaluator):
    def __init__(self, model, num_classes, var_file, dict_dir, cell_type_col, cell_type_dict):
        # geneformer instance for tokenizing
        self.var_file = var_file
        num_workers = -1 # all available
        self.output_dir = "tmp_output"
        self.geneform = gf.Geneformer_instance(save_dir = self.output_dir, 
                                          saved_model_path = model,
                                          explicit_save_dir = True,
                                          num_workers = num_workers)
        self.geneform.load_pretrained_model()
        self.geneform.load_vocab(dict_dir)
        self.cell_type_col = cell_type_col
        self.cell_type_dict = {v: k for k, v in cell_type_dict.items()} # reverse cell type dict map from string to int

        # load classifier
        print("Loading geneformer model:", model)
        gf_model = pu.load_model(model_type="CellClassifier",
                                 num_classes=num_classes,
                                 model_directory=model, # path to the model
                                 mode="eval",
                                 )
        self.model = gf_model 
    def get_labels(self, adata, dataset_name="tmp"):

        adata.obs[self.cell_type_col] = adata.obs[self.cell_type_col].cat.rename_categories(self.cell_type_dict)
        input_data = tokenize_adata(adata, self.var_file, dataset_name, cell_type_col=self.cell_type_col)
        input_data = load_from_disk(input_data)
        input_data = cu.rename_cols(input_data, self.cell_type_col)

        print("Getting labels")
        torch.cuda.empty_cache()
        y_pred, y_true, logits_list = classifier_predict(model=self.model, classifier_type='cell', evalset=input_data, forward_batch_size=100)

        print("Removing tokenized data:")
        shutil.rmtree(self.output_dir, ignore_errors=True)
        shutil.rmtree(f"tmp_adata_{dataset_name}", ignore_errors=True)
        shutil.rmtree(f"tmp_tokenized_data_{dataset_name}", ignore_errors=True) # todo this one isn't working?

        return y_pred, y_true
