from pathlib import Path
import argparse
import os
import pickle

import numpy as np
import torch
import pytorch_lightning as pl

import anndata as ad
import scvi
from scimilarity.training_models import MetricLearning
from transformers import BertConfig, BertForMaskedLM, TrainingArguments
from scimilarity.anndata_data_models import MetricLearningDataModule
from geneformer import GeneformerPretrainer




def untrained_pca(n_inputs, n_latent, output_name, seed):
    np.random.seed(seed)

    untrained_projection_matrix = np.random.normal(size=(n_inputs, n_latent))
    np.save(output_name, untrained_projection_matrix)


def untrained_scvi(adata, n_latent, n_hidden, output_name, seed):
    scvi.settings.seed = seed

    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata, n_latent=n_latent, n_hidden=n_hidden, gene_likelihood="nb")
    model.save(output_name)

def untrained_ssl(output_name):
    pass

def untrained_geneformer(output_dir, seed, num_genes, genecorpus_lengths_file):
    torch.manual_seed(seed)
    
    token = "token_dictionary.pkl"
    with open(token, "rb") as fp:
        token_dictionary = pickle.load(fp)
    model_type = "bert"
    max_input_size = 2048
    num_layers = 6
    num_attn_heads = 4
    num_embed_dim = 256
    intermed_size = num_embed_dim * 2
    activ_fn = "relu"
    initializer_range = 0.02
    layer_norm_eps = 1e-12
    attention_probs_dropout_prob = 0.02
    hidden_dropout_prob = 0.02

    config = BertConfig(
        hidden_size=num_embed_dim,
        num_hidden_layers=num_layers,
        initializer_range=initializer_range,
        layer_norm_eps=layer_norm_eps,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        hidden_dropout_prob=hidden_dropout_prob,
        intermediate_size=intermed_size,
        hidden_act=activ_fn,
        max_position_embeddings=max_input_size,
        model_type=model_type,
        num_attention_heads=num_attn_heads,
        pad_token_id=0, # taken from the trained models
        vocab_size=25426 # taken directly from the trained models
)

    model = BertForMaskedLM(config)

    max_lr = 1e-3
    lr_schedule_fn = "linear"
    warmup_steps = 10000
    optimizer = "adamw"
    weight_decay = 0.001
    geneformer_batch_size = 18

    training_args = TrainingArguments(
        num_train_epochs=1,
        learning_rate=max_lr,
        do_train=False,
        do_eval=False,
        group_by_length=True,
        length_column_name="length",
        disable_tqdm=False,
        lr_scheduler_type=lr_schedule_fn,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        per_device_train_batch_size=geneformer_batch_size,
        save_strategy="steps",
        save_steps=50,
        logging_steps=50,
        output_dir=output_dir,
        logging_dir="untrainded_geneformer_logs",
        fp16=True,
    )
    trainer = GeneformerPretrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        example_lengths_file=genecorpus_lengths_file,
        token_dictionary=token_dictionary,
    )

    # skip training and save model
    trainer.save_model(Path(output_dir) / "Geneformer" / "random" / f"idx_0pct_seed{seed}")


def untrained_scimilarity(output_dir, seed, sctab_ngenes, sctab_format_h5ad):
    torch.manual_seed(seed)

    downsample = "random"
    percent = 0
    prefix = f"SCimilarity_{downsample}_{percent}pct_seed{seed}"
     
    hidden_dim            = [1024, 1024, 1024]
    latent_dim            = 128
    margin                = 0.05 # triplet loss margin
    negative_selection    = "semihard" # negative selection type: [semihard, random, hardest]
    triplet_loss_weight   = 0.001
    lr                    = 0.005
    batch_size            = 1000
    n_batches             = 100 
    max_epochs            = 0
    cosine_annealing_tmax = 0
    prefix                = prefix
    model_folder          = Path(output_dir) / "models_SCimilarity" 
    log_folder            = Path(output_dir) / "logs_SCimilarity"   
    result_folder         = Path(output_dir) / "results_SCimilarity" 
    num_workers           = 8
    dropout               = 0.5
    input_dropout         = 0.4
    l1                    = 1e-4
    l2                    = 0.01
    
    model_name = (
        f"SCimilarity_{prefix}_{batch_size}_{margin}_{latent_dim}_{len(hidden_dim)}_{triplet_loss_weight}"
    )

    print("Creating scimilarity datamodule:")
    datamodule = MetricLearningDataModule(
        train_path = sctab_format_h5ad,
        val_path = sctab_format_h5ad,
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
    
    print("Creating scimilarity model:")
    model = MetricLearning(
    sctab_ngenes,
    latent_dim=latent_dim,
    hidden_dim=hidden_dim,
    dropout=dropout,
    input_dropout=input_dropout,
    margin=margin,
    negative_selection=negative_selection,
    sample_across_studies=True,       
    perturb_labels=False,            
    lr=lr,
    triplet_loss_weight=triplet_loss_weight,
    l1=l1,
    l2=l2,
    max_epochs=max_epochs,
    cosine_annealing_tmax=cosine_annealing_tmax,
    )

    trainer = pl.Trainer(accelerator="cpu", max_epochs=max_epochs)

    # attach the datamodule to the trainer without calling fit()
    trainer.datamodule = datamodule
    # attach a trainer to the model without calling fit()
    model._trainer = trainer
    
    os.makedirs(model_folder, exist_ok=True)
    model.save_all(model_path=os.path.join(model_folder, model_name))


def main():
    parser = argparse.ArgumentParser(
        description='Save untrained model files for PCA, scVI, SSL, and Geneformer.')
    parser.add_argument('--h5ad')

    parser.add_argument('-o', '--output')
    parser.add_argument('--n_latent', type=int, default=10)
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--trained_geneformer_dir', type=str, default=None)
    parser.add_argument('--trained_scimilarity_dir', type=str, default=None)
    parser.add_argument('--sctab_format_h5ad', type=str, default=None)
    parser.add_argument('--genecorpus_lengths_file', type=str, default=None)

    args = parser.parse_args()

    untrained_models_base_dir = Path(args.output)

    formatted_h5ad = args.h5ad
    #adata = ad.read_h5ad(formatted_h5ad)

    seeds = range(30,35)

    n_inputs = 19331

    pca_latent = 50

    scvi_latent = 10
    scvi_n_hidden = 128

    for seed in seeds:
        print(seed)
        """
        pca_output_name = untrained_models_base_dir / "pca" / f"idx_0pct_seed{seed}"
        untrained_pca(n_inputs, pca_latent, pca_output_name, seed)

        scvi_output_name = untrained_models_base_dir / "scvi" / f"idx_0pct_seed{seed}"
        untrained_scvi(adata, scvi_latent, scvi_n_hidden, scvi_output_name, seed)
        """
        
        untrained_geneformer(untrained_models_base_dir, seed, n_inputs, args.genecorpus_lengths_file)
        #untrained_scimilarity(untrained_models_base_dir / "SCimilarity", seed, n_inputs ,args.sctab_format_h5ad)

if __name__ == "__main__":
    main()

