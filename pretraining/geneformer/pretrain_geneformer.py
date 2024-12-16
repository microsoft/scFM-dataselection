#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import os
import pickle
import random
import numpy as np
import torch
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, TrainingArguments
import wandb
from geneformer import GeneformerPretrainer
import pytz
from huggingface_hub import hf_hub_download


# Set up environment variables
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

# Parse command line arguments for dynamic parameters
parser = argparse.ArgumentParser(description="Pretrain Geneformer with specified gene, cluster, and seed.")
parser.add_argument("--local_rank", type=int, required=False, help="Local rank for pretraining")
parser.add_argument("--gene", type=str, required=True, help="Gene identifier to process.")
parser.add_argument("--cluster", type=str, required=True, help="Cluster identifier to process.")
parser.add_argument("--seed", type=int, required=True, help="Seed number for normal generation.")
parser.add_argument("--pct", type=int, required=True, help="percent of train data.")
parser.add_argument("--out", type=str, required=True, help="out dir.")
parser.add_argument("--datapath", type=str, required=True, help="datapath.")
parser.add_argument("--lengths", type=str, required=True, help="length gile.")
parser.add_argument("--epochs", type=int, required=True, help="epochs to train")

args = parser.parse_args()

# Set local timezone
timezone = pytz.timezone("US/Eastern")
if os.environ.get('rank') == '0':
    wandb.init(
        project="text-geneformer",
        tags=["pretraining", "deepspeed"],
        reinit=True
    )
    wandb.save("pretrain.py")

# Seed setting for reproducibility
seed_num = args.seed
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Model configurations
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


# Set training parameters
num_examples = 1700000
num_gpus = 8
geneformer_batch_size = 18
max_lr = 1e-3
lr_schedule_fn = "linear"
warmup_steps = 10000
# epochs = 3
optimizer = "adamw"
weight_decay = 0.001

# Directories based on command line arguments
rootdir = f"{args.out}"
print(rootdir)
current_date = datetime.datetime.now(timezone)
datestamp = f"{current_date.strftime('%y%m%d_%H%M%S')}"

run_name = f"{datestamp}_geneformer_30M_L{num_layers}_emb{num_embed_dim}_SL{max_input_size}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_O{optimizer}_DS{num_gpus}_rank_gene_{args.gene}_cluster_{args.cluster}"

# Create directories
training_output_dir = os.path.join(rootdir, "models", run_name)
logging_dir = os.path.join(rootdir, "runs", run_name)
model_output_dir = os.path.join(training_output_dir, "models")
os.makedirs(model_output_dir, exist_ok=True)

# Ensure not overwriting existing model
model_output_file = os.path.join(model_output_dir, "pytorch_model.bin")
if os.path.isfile(model_output_file):
    raise Exception("Model already saved to this directory.")

# Load gene_ensembl_id:token dictionary
token = "/home/aiscuser/Geneformer/geneformer/token_dictionary.pkl"
with open(token, "rb") as fp:
    token_dictionary = pickle.load(fp)

# Configure the model
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
    pad_token_id=token_dictionary.get("<pad>"),
    vocab_size=len(token_dictionary)  # genes+2 for <mask> and <pad> tokens
)

# Set training arguments
training_args = TrainingArguments(
    num_train_epochs=args.epochs,
    learning_rate=max_lr,
    do_train=True,
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
    output_dir=training_output_dir,
    logging_dir=logging_dir,
    fp16=True,
    report_to=["wandb"]
)

dataset_directory = f"{args.datapath}"
lengths_file = f"{args.lengths}"

print("Creating lengths pickle file")
corpus = load_from_disk(dataset_directory)
lengths = corpus["length"]

with open(lengths_file, 'wb') as fh:
    pickle.dump(lengths, fh)

print("Starting training.")
#config = BertConfig(**config)
model = BertForMaskedLM(config)

model = model.train()


# Define the trainer
trainer = GeneformerPretrainer(
    model=model,
    args=training_args,
    train_dataset=corpus,
    example_lengths_file=lengths_file,
    token_dictionary=token_dictionary,
)

# Train
trainer.train()

# Save model
trainer.save_model(model_output_dir)