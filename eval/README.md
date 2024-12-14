# Evaluation

## Fine-tuning

Models must be fine-tuned on each of the evaluation datasets described in the `data` section of this repo in order to perform the fine-tuning evaluations. Instructions for processing the data are in the `data` directory and instructions for finetuning the models are in the `finetune-scripts` directory.

## Evaluating Models

These files define classes and functions for evaluating the various models:

```bash
evaluation_utils.py
model_loaders.py
finetune_model_evaluators.py
zero_shot_model_evaluators.py
```

These files import the classes and functions defined in the files above and use them to actually evaluate the models:
```bash
zero_shot_model_classification.py
finetune_model_classification.py
zero_shot_model_integration.py
```

These files contain auxiliary data necessary for performing evaluation (they contain the genes used in the pre-training dataset and how they are ordered in the h5ad files):
```bash
adata_var.csv
sctab_format.h5ad
```
We now demonstrate how to evaluate model performance.

### Zero-shot classification

To evaluate a model's zero-shot performance on cell type classification run `zero_shot_classification.py` as follows

```bash
# define variables for script arguments
H5AD=pbmc.h5ad
LABEL_COL=celltype

DOWNSAMPLING_METHOD=geometric_sketching

VAR_DIR=adata_var.csv
FORMATTED_H5AD=sctab_format.h5ad

PERCENTAGE=100
SEED=25

METHOD=scVI
MODEL_BASE_DIR=/path/to/scvi/models/

GENEFORMER_DICT_DIR=/path/to/geneformer_repo/Geneformer/geneformer # only needed for evaluating Geneformer, ignored for other models (can use na)

# run evaluation script
python -u zero_shot_classification.py $METHOD $H5AD $LABEL_COL $DOWNSAMPLING_METHOD $PERCENTAGE $SEED $VAR_DIR $FORMATTED_H5AD $MODEL_BASE_DIR $GENEFORMER_DICT_DIR
```

### Fine tuned classification

To evaluate a model's fine tuned performance on cell type classification run `finetune_classification.py` as follows

```bash
# define variables for script arguments
DATASET_NAME=pbmc
H5AD_FILE=pbmc.h5ad
CELL_TYPE_COL=celltype

VAR_FILE=adata_var.csv
OUT_DIR=finetune_output

DOWNSAMPLING_METHOD=random
SEED=15
PERCENTAGE=50

SCTAB_FORMAT=sctab_format.h5ad
METADATA_PATH=na

METHOD=scVI
MODEL_BASE_DIR=/path/to/scvi/models/

CELL_TYPE_PICKLE=cm_classifier_id_class_dict_train.pkl

GENEFORMER_DICT_DIR=/path/to/geneformer_repo/Geneformer/geneformer # only needed for evaluating Geneformer, ignored for other models (can use na)

PRETRAINED_SCVI_DIRECTORY=/path/to/pretrained/scvi_models/ # only needed for evaluating scVI, ignored for other models (can use na)
SCVI_TRAIN_FORMAT_H5AD=/path/to/sctab/downsampled_TRAIN.h5ad # only needed for evaluating scVI, ignored for other models (can use na)

# run evaluation script
python -u finetune_classification.py $MODEL_BASE_DIR $DATASET_NAME $H5AD_FILE $CELL_TYPE_COL $VAR_FILE $OUT_DIR $METHOD $DOWNSAMPLING_METHOD $SEED $PERCENTAGE $SCTAB_FORMAT $METADATA_PATH $CELL_TYPE_PICKLE $GENEFORMER_DICT_DIR $PRETRAINED_SCVI_DIRECTORY $SCVI_TRAIN_FORMAT_H5AD
```
### Zero-shot integration

To evaluate a model's zero-shot performance on batch integration run `zero_shot_integration.py` as follows

```bash
# define variables for script arguments
H5AD=pancreas_scib.h5ad
LABEL_COL=celltype
BATCH_COL=batch

DOWNSAMPLING_METHOD=geometric_sketching

VAR_DIR=adata_var.csv
FORMATTED_H5AD=sctab_format.h5ad

PERCENTAGE=100
SEED=25

METHOD=Geneformer
MODEL_BASE_DIR=/path/to/geneformer/models/

GENEFORMER_DICT_DIR=/path/to/geneformer_repo/Geneformer/geneformer # only needed for evaluating Geneformer, ignored for other models (can use na)


# run evaluation script
python -u zero_shot_integration.py $METHOD $H5AD $LABEL_COL $BATCH_COL $DOWNSAMPLING_METHOD $PERCENTAGE $SEED $VAR_DIR $FORMATTED_H5AD $GENEFORMER_DIR $GENEFORMER_DICT_DIR
```





#### Datasets
- Clonal Hematopoiesis ([publication](https://ashpublications.org/bloodadvances/article/8/14/3665/515374/Multiomic-profiling-of-human-clonal-hematopoiesis), [CELLxGENE](https://cellxgene.cziscience.com/collections/0aab20b3-c30c-4606-bd2e-d20dae739c45))
- Placenta ([publication](https://www.cell.com/cell-systems/fulltext/S2405-4712(24)00117-0), [CELLxGENE](https://cellxgene.cziscience.com/collections/5f80428b-222d-450b-a7de-a408186ceb86))
- Intestine-on-chip ([publication](https://www.cell.com/cell-reports/fulltext/S2211-1247(24)00575-8), [CELLxGENE](https://cellxgene.cziscience.com/collections/f30e0bf7-7b03-44b0-b312-cda799ef0240))
- Periodontitis ([publication](https://www.nature.com/articles/s41467-024-49037-y), [CELLxGENE](https://cellxgene.cziscience.com/collections/71f4bccf-53d4-4c12-9e80-e73bfb89e398))
- [Lung (Kim et al)](https://www.weizmann.ac.il/sites/3CA/lung)



