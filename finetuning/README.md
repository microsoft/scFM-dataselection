# Fine-Tuning

This directory contains directories for fine tuning `scVI`, `SSL`, `Geneformer`, and baselines. We give instructions running the fine tuning scripts for each of these below:

## scVI

```bash
SCVI_MODEL_DIR=/path/to/scvi_models/
SCVI_FORMAT_H5AD=idx_1pct_seed0_TEST.h5ad

DATASET_NAME=intestine
TRAIN_H5AD_FILE=evaluation_data/intestine/intestine_TRAIN.h5ad
VAL_H5AD_FILE=evaluation_data/intestine/intestine_VAL.h5ad
CELL_TYPE_COL=cell_type
SCVI_CELLTYPE_PKL=scvi_cell_type.pkl

VAR_FILE=eval/adata_var.csv
OUT_DIR=scvi_finetuning_output
SCTAB_FORMAT=eval/sctab_format.h5ad

METHOD=random
SEED=0
PERCENTAGE=1
python -u scvi/scvi_mlp.py $SCVI_MODEL_DIR $SCVI_FORMAT_H5AD $DATASET_NAME $TRAIN_H5AD_FILE $CELL_TYPE_COL $VAR_FILE $OUT_DIR $METHOD $SEED $PERCENTAGE $SCTAB_FORMAT $VAL_H5AD_FILE $SCVI_CELLTYPE_PKL
```

## SSL

```bash
PRETRAINED_DIR=path/to/ssl/pretrained/model
DATAPATH=evaluation_data/intestine
MODELPATH=./ssl_finetuned_model/
. finetune-scripts/ssl/finetune_ssl.sh --pretrained-dir $PRETRAINED_DIR --datapath $DATAPATH --model-path $MODELPATH

```

## Geneformer
Before finetuning Geneformer models, the test/train/val splits of the eval data must be tokenized. See eval/ReadME.md for instructions on how to tokenize the data


Before finetuning Geneformer models, the test/train/val splits of the eval data must be tokenized. See `../data/preprocess/ReadME.md` for instructions on how to tokenize the data. Once the data is tokenized, you can finetune:

```bash
GENEFORMER_MODEL_TYPE = model-prefix
OUTPUT_DIR = geneformer_finetuning_output
CELL_TYPE_COLUMN = cell_type
INPUT_DATA_FILE_TRAIN = /path/to/{dataset}_TRAIN_tokenized.dataset  # Path to directory containing .dataset input
INPUT_DATA_FILE_TEST = /path/to/{dataset}_TEST_tokenized.dataset  # Path to directory containing .dataset input
PRETRAINED_MODEL = /path/to/pretrained/geneformer_models
INPUT_DATA_FILE_VAL = /path/to/{dataset}_VAL_tokenized.dataset


python -u finetune-scripts/geneformer/fine_tune_geneformer_cell_classification.py $GENEFORMER_MODEL_TYPE $OUTPUT_DIR $CELL_TYPE_COLUMN $INPUT_DATA_FILE_TRAIN $INPUT_DATA_FILE_TEST $PRETRAINED_MODEL $INPUT_DATA_FILE_VA
```

## Baseline

```bash
DATASET_NAME=intestine
TRAIN_H5AD_FILE=evaluation_data/intestine/intestine_TRAIN.h5ad
VAL_H5AD_FILE=evaluation_data/intestine/intestine_VAL.h5ad
CELL_TYPE_COL=cell_type

VAR_FILE=eval/adata_var.csv
OUT_DIR=baseline_finetuning_output
SCTAB_FORMAT=eval/sctab_format.h5ad

python -u classification_baselines/baselines.py $DATASET_NAME $TRAIN_H5AD_FILE $VAL_H5AD_FILE $CELL_TYPE_COL $VAR_FILE $OUT_DIR $SCTAB_FORMAT $CELLTYPE_PKL
```


