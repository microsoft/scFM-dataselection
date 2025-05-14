# Pre-Training


This directory contains directories for pre-training `PCA`, `scVI`, `SSL`, `Geneformer`. We give instructions running the pre-training scripts for each of these below:

## PCA

```bash
SAMPLING=geometric_sketching
PCT=75
SEED=23

TRAIN_FILE=/path/to/sctab/$SAMPLING/idx_"$PCT"pct_seed"$SEED"/idx_"$PCT"pct_seed"$SEED"_TRAIN.h5ad
OUTPUT_DIR=./pca_models/$SAMPLING/idx_"$PCT"pct_seed"$SEED"


python -u train_pca.py -a $TRAIN_FILE -o $OUTPUT_DIR --var_file adata_var.csv
```

## scVI

```bash
SAMPLING=random
PCT=25
SEED=12

TRAIN_FILE=/path/to/sctab/$SAMPLING/idx_"$PCT"pct_seed"$SEED"/idx_"$PCT"pct_seed"$SEED"_TRAIN.h5ad
VAL_FILE=/path/to/sctab/$SAMPLING/idx_"$PCT"pct_seed"$SEED"/idx_"$PCT"pct_seed"$SEED"_VAL.h5ad

VAR_FILE=eval/adata_var.csv

OUTPUT_DIR=./scvi_models/$SAMPLING/idx_"$PCT"pct_seed"$SEED"

python -u train_scvi.py -a $TRAIN_FILE -v $VAR_FILE -o $OUTPUT_DIR --pct $PCT --var_file $VAR_FILE
```

## SSL

To pretrain SSL, run `pretrain_ssl.sh` as follows:

```bash

SEED=0
PCT=1
METHOD=randomsplits
DATAPATH=/path/to/sctab/$METHOD/idx_"$PCT"pct_seed"$SEED"
OUTPUT=./ssl_pretrained_models/$SAMPLING/idx_"$PCT"pct_seed"$SEED"
. train_scripts/ssl/pretrain_ssl.sh --datapath $DATAPATH --model-path $OUTPUT --pct $PCT --seed $SEED --method $METHOD --max-steps 117000 --early_stopping True

```

## Geneformer
Before pretraining Geneformer we must tokenize the training data. To tokenize the data and pretrain the model, run the following commands. 

```bash
#tokenize training data 
DATANAME=idx_1pct_seed0_TRAIN
DATA_DIR=./sctab/randomsplits/idx_1pct_seed0/train/
OUTPUT_DIR=./tokenized/sctab/randomsplits/idx_1pct_seed0
VAR_FILE=eval/adata_var.csv
#change this to 'other' if tokenizing data that is not sctab
DATA_TYPE=sctab

python -u train_scripts/geneformer/tokenize_data.py --datasetname $DATANAME --data_dir $DATA_DIR --var_file $VAR_FILE --output_dir $OUTPUT_DIR --data_type $DATA_TYPE

#pretrain geneformer 
SEED=0
PCT=1
OUTPUT=./geneformer_pretrained_models/$SAMPLING/idx_"$PCT"pct_seed"$SEED"
TOKENIZED_DATAPATH=./tokenized/sctab/randomsplits/idx_1pct_seed0_TRAIN/idx_1pct_seed0_TRAIN.dataset
LENGTHS_FILE=./tokenized/idx_1pct_seed0_TRAIN_lengths.pkl
EPOCHS=3

deepspeed --num_gpus=8 --num_nodes=1 train_scripts/geneformer/pretrain_geneformer.py --gene normal --cluster normal --seed $SEED --pct $PCT --out $OUTPUT --datapath $TOKENIZED_DATAPATH --lengths $LENGTHS_FILE --epochs $EPOCHS
```
