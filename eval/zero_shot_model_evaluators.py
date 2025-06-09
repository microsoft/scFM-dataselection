import os
import abc
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from pathlib import Path
import shutil

import random
import string

import matplotlib.pyplot as plt
import seaborn as sns
import umap

import numpy as np
import pandas as pd

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

# https://github.com/microsoft/zero-shot-scfoundation/blob/main/sc_foundation_evals/utils.py
# MODIFIED wrapper for all scib metrics from 
# https://github.com/bowang-lab/scGPT/blob/5a69912232e214cda1998f78e5b4a7b5ef09fe06/scgpt/utils/util.py#L267
def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "str_batch",
    label_key: str = "cell_type",
    embedding_key: str = "X_scGPT"
) -> Dict:
    
    # if adata.uns["neighbors"] exists, remove it to make sure the optimal 
    # clustering is calculated for the correct embedding
    # print a warning for the user
    if "neighbors" in adata.uns:
        """
        log.warning(f"neighbors in adata.uns found \n {adata.uns['neighbors']} "
                    "\nto make sure the optimal clustering is calculated for the "
                    "correct embedding, removing neighbors from adata.uns."
                    "\nOverwriting calculation of neighbors with "
                    f"sc.pp.neighbors(adata, use_rep={embedding_key}).")
        """
        adata.uns.pop("neighbors", None)
        sc.pp.neighbors(adata, use_rep=embedding_key)
        #log.info("neighbors in adata.uns removed, new neighbors calculated: "
        #         f"{adata.uns['neighbors']}")
    # in case just one batch scib.metrics.metrics doesn't work 
    # call them separately
    results_dict = dict()
    res_max, nmi_max, nmi_all = scib.metrics.clustering.opt_louvain(
            adata,
            label_key=label_key,
            cluster_key="cluster",
            use_rep=embedding_key,
            function=scib.metrics.nmi,
            plot=False,
            verbose=False,
            inplace=True,
            force=True,
    )
    results_dict["NMI_cluster/label"] = scib.metrics.nmi(
        adata, 
        "cluster",
        label_key,
        "arithmetic",
        nmi_dir=None
    )
    results_dict["ARI_cluster/label"] = scib.metrics.ari(
        adata, 
        "cluster", 
        label_key
    )
    results_dict["ASW_label"] = scib.metrics.silhouette(
        adata, 
        label_key, 
        embedding_key, 
        "euclidean"
    )   
    results_dict["graph_conn"] = scib.metrics.graph_connectivity(
        adata,
        label_key=label_key
    )
    
    # Calculate this only if there are multiple batches
    if len(adata.obs[batch_key].unique()) > 1:
        results_dict["ASW_batch"] = scib.metrics.silhouette(
            adata,
            batch_key,
            embedding_key,
            "euclidean"
        )
        results_dict["ASW_label/batch"] = scib.metrics.silhouette_batch(
            adata, 
            batch_key,
            label_key, 
            embed=embedding_key, 
            metric="euclidean",
            return_all=False,
            verbose=False
        )
        results_dict["PCR_batch"] = scib.metrics.pcr(
            adata,
            covariate=batch_key,
            embed=embedding_key,
            recompute_pca=True,
            n_comps=50,
            verbose=False
        )
    results_dict["avg_bio"] = np.mean(
        [
            results_dict["NMI_cluster/label"],
            results_dict["ARI_cluster/label"],
            results_dict["ASW_label"],
        ]
    )
    # remove nan value in result_dict
    results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}
    return results_dict




class ModelEmbeddingsMixin:
    def get_embeddings(self, adata):
        raise NotImplementedError

# done (not tested)
# todo implement eval_classification_metrics
# todo rename evaluate to evaluate_integration
# todo implement evalute_classification

# real todos
# todo test evaluate_integration
# todo evaluate_classification
class ZeroShotEvaluator(ModelEmbeddingsMixin):
    def evaluate_integration(self, adata, batch_col, label_col):
        latent_embeddings = self.get_embeddings(adata)
        adata.obsm[self.embedding_name] = latent_embeddings
        scib_metrics = eval_scib_metrics(adata, 
                                         batch_key=batch_col, 
                                         label_key=label_col,
                                         embedding_key=self.embedding_name)
        return scib_metrics

    def evaluate_classification(self, adata_train, adata_test, cell_type_col, n_neighbors=5):
        print('getting embeddings')
        train_latent_embeddings = self.get_embeddings(adata_train)
        test_latent_embeddings = self.get_embeddings(adata_test)

        print('KNN CLASSIFICATION')
        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_model.fit(train_latent_embeddings, adata_train.obs[cell_type_col])

        print('predict')
        test_predictions = knn_model.predict(test_latent_embeddings)

        print('classification metrics')
        classification_metrics = eval_classification_metrics(adata_test.obs[cell_type_col], test_predictions)

        print('done')
        return classification_metrics


class VariableGeneZeroShotEvaluator(ZeroShotEvaluator):
    def __init__(self):
        self.embedding_name = "X_variable_genes"
    def get_embeddings(self, adata):
        variable_genes_adata = adata[:, adata.var.highly_variable]
        return np.asarray(variable_genes_adata.X)

class PrincipalComponentsZeroShotEvaluator(ZeroShotEvaluator):
    def __init__(self):
        self.embedding_name = "X_PCA"
    def get_embeddings(self, adata):
        return np.asarray(adata.obsm['X_pca'])


class SSLZeroShotEvaluator(ZeroShotEvaluator):
    def __init__(self, model):
        self.model = model
        self.embedding_name = "X_SSL"
    def get_embeddings(self, adata):
        input_tensor = adata.X.todense()
        with torch.no_grad():
            latent_embeddings = self.model.encoder(torch.tensor(input_tensor))
        return latent_embeddings.numpy()


class SCVIZeroShotEvaluator(ZeroShotEvaluator):
    def __init__(self, model):
        self.model = model
        self.embedding_name = "X_scVI_foundation"
    def get_embeddings(self, adata):
        latent_embeddings = self.model.get_latent_representation(adata)
        return latent_embeddings








from pathlib import Path

import os
import logging
import warnings

#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=FutureWarning)

from sc_foundation_evals import geneformer_forward as gf
from sc_foundation_evals import data, cell_embeddings, model_output
#from sc_foundation_evals.helpers.custom_logging import log
from model_loaders import load_geneformer_model
#log.setLevel(logging.INFO)



class GeneformerZeroShotEvaluator(ZeroShotEvaluator):
    def __init__(self, model, var_file, dict_dir, tmp_output_dir):
        self.model = model # path to the model
        self.embedding_name = "X_geneformer"
        # output_dir is the path to which the results should be saved
        self.output_dir = Path(tmp_output_dir)

        self.var_file = var_file

        num_workers = -1 # all available

        self.geneform = gf.Geneformer_instance(save_dir = self.output_dir, 
                                          saved_model_path = model,
                                          explicit_save_dir = True,
                                          num_workers = num_workers)
        self.geneform.load_pretrained_model()
        self.geneform.load_vocab(dict_dir)

    # adapted from https://github.com/microsoft/zero-shot-scfoundation/blob/main/notebooks/Geneformer_zero_shot.ipynb
    def get_embeddings(self, adata):
        random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

        tokenized_data = tokenize_adata(adata, self.var_file, random_string)

        dataset_name = "tmp"
        preprocessed_path = self.output_dir / f"prepocessed/{dataset_name}/"

        # batch_size depends on available GPU memory
        batch_size = 24
        # path to where we will store the embeddings and other evaluation outputs
        model_out = os.path.join(self.output_dir, "model_outputs")
        # if you can use multithreading specify num_workers, -1 means use all available
        num_workers = -1

        in_dataset_path = self.output_dir / f"prepocessed/{dataset_name}.h5ad"

        # create the preprocessed path if it does not exist
        os.makedirs(preprocessed_path, exist_ok=True)
        # in which column in adata.obs are gene names stored? if they are in index, the index will be copied to a column with this name
        gene_col = "gene_symbols"

        adata.write_h5ad(filename=in_dataset_path)

        input_data = data.InputData(adata_dataset_path = in_dataset_path)

        # Gene names not found in var columns. Using index instead. todo add gene names to anndata
        input_data.preprocess_data(gene_col = gene_col,
                                   model_type = "geneformer",
                                   save_ext = "loom",
                                   gene_name_id_dict = self.geneform.gene_name_id,
                                   preprocessed_path = preprocessed_path)

        # not actually used, but necessary for Kasia's code to work
        label_col = "celltype" #"str_labels"

        self.geneform.tokenize_data(adata_path = os.path.join(preprocessed_path, f"{dataset_name}.loom"),
                                    dataset_path = preprocessed_path,
                                    cell_type_col = label_col)

        self.geneform.load_tokenized_dataset(os.path.join(preprocessed_path, f"{dataset_name}.dataset"))
        input_data = data.InputData(adata_dataset_path = os.path.join(preprocessed_path, f"{dataset_name}.loom"))

        self.geneform.extract_embeddings(data = input_data,
                                    batch_size = batch_size, 
                                    layer = -2)

        # todo clean up files written in this process
        print("Removing tokenized data:", tokenized_data)
        shutil.rmtree(tokenized_data, ignore_errors=True)

        print("Removing tmp adata dir:", f"tmp_adata_{random_string}")
        shutil.rmtree(f"tmp_adata_{random_string}", ignore_errors=True)

        return self.geneform.cell_embeddings

