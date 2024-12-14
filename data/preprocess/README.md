# Evaluation

## Evaluating Models
This file uses a modified python script version of https://github.com/theislab/scTab/blob/devel/notebooks/store_creation/01_download_data.ipynb obtained from https://github.com/theislab/scTab/blob/devel/notebooks/store_creation/01_download_data.ipynb to download the sctab data. We also use the features.parquet file from this repo, as it is required to run this script:

```bash
download_sctab.py 
features.parquet
```

These files are used to download and concatenate perturbation data from Replogle et al (2022) for spike-in experiments
```bash
download_spikein.sh
concatenate_replogle.py
```

These files define classes and functions for preprocessing the training and evaluation datasets:

```bash
preprocess_class.py
```

These files import the classes and functions defined in the files above and use them to downsample and process the data:
```bash
preprocess_sctab.py
preprocess_spikein.py
preprocess_finetune_data.py
```

This file is for calculating class weights of a dataset, which is necessary for running SSL finetuning.
```bash
calc_class_weights.py
```

We now demonstrate how to download, downsample, and preprocess the data.
## Download and concatenate sctab
To download the full sctab dataset and concatenate it, run `download_sctab.py` as follows. Next, to dowsample scTab, please see downsampling/README.md

```bash
python -u download_sctab.py
```

### Pre-process scTab

This script read in scTab and normalizes and log transforms the counts in a memory-efficient manner. Finally, it converts the data into the parquet format so that it is compatible with the SSL dataloader. This script saves three versions of the downsampled data: raw h5ad, processed h5ad, and processed parquets. This example only process one dataset at a time (one downsampling method, one seed, one percent). METHOD should be one of the following values: random, celltype_reweighted, geometric_sketching. For 1% data splits, the seeds range from 0-4, for 10% seeds range from 5-9, for 25% seeds range from 10-14, for 50% seeds range from 15-19, for 75% seeds range from 20-24, and for 100% seeds range from 25-29. To preprocess one downsampled set of the scTab data, run `preprocess_sctab.py` as follows:

```bash
DATAPATH=sctab/idx_1pct_seed0/
OUTDIR=sctab/downsampled
SEED=0
PCT=1
METHOD=random

python -u preprocess_sctab.py --datapath $DATAPATH --outdir $OUTDIR --seed $SEED --pct $PCT --method $method
```

To create the combined Replogle `h5ad` file, run

```bash
python -u concatenate_replogle.py
```

### Pre-process Spike-ins
To download the perturbation data that we will use in the spikein experiments, run `download_spikein.sh` as follows.

```bash
. download_spikein.sh
```
To concatenate the spikein data, run `concatenate_replogle.py` as follows. Then, to generate the spikein data, please see downsampling/README.md

```bash
K562_GWPS=spikein/k562_gwps.h5ad
K562_ESSENTIAL=spikein/k562_essential_raw.h5ad
RPE1=spikein/rpe1_essential_raw.h5ad

python concatenate_replogle.py $K562_GWPS $K562_ESSENTIAL $RPE1
```
This script takes in one spike-in dataset at a time to normalize and log transform the counts, and convert it into the parquet format so that it is compatible with the SSL dataloader, just as we have done for the sctab data. The spikein data will use the same validation and test splits as their sctab counterparts, so we will copy those over to the spikein directory as well. To preprocess a spike-in dataset, run `preprocess_spikein.py` as follows. 

```bash

DATAPATH=spikein/spikein_1pct_seed0.h5ad
SEED=0
PCT=1
#VAL_TEST_PATHS refers to the path of the test and validation sets of the randomly downsampled scTab data that correspond to the percent and seed being processed
VAL_TEST_PATHS=sctab/downsampled/random
OUTDIR=$DATAPATH

PCT_SEED_LABEL=idx_${PCT}pct_seed${SEED}
# run preprocess script
cp -r ${VAL_TEST_PATHS}/${PCT_SEED_LABEL}/val ${OUTDIR}/${PCT_SEED_LABEL}
cp -r ${VAL_TEST_PATHS}/${PCT_SEED_LABEL}/test ${OUTDIR}/${PCT_SEED_LABEL}
python -u preprocess_spikein.py --datapath $DATAPATH --seed $SEED --pct $PCT --val_test_paths $VAL_TEST_PATHS --outdir $OUTDIR
```

### Download Evaluation Datasets

This script downloads the datasets that we use in our zero-shot and finetuning evaluations. The covid data does not have a Linux download command, so you can download it from here: https://drive.google.com/drive/folders/1HzweYLr6whj7XWVv8I8pS9Ax-HR7Yc-F. To download the other datasets run `download_finetune_data.sh` as follows:

```bash

. download_finetune_data.sh

```

### Pre-process Evaluation Datasets

This script processes the evaluation datasets just as we have done for scTab and the spike-in data. For SSL, we must zero-pad the genes that are in scTab but not in the evaluation data, and calculate cell type class weights. To process each of the evaluation datasets run `preprocess_finetune_datasets.py` and `calc_class_weights.py` as follows:

```bash
DATAPATH=evaluation_data/intestine 
DATANAME=intestine 
ZEROPAD=True
SCTAB_VARFILE=eval/adata_var.csv 
VAR_TYPE=symbol
CELLTYPE_COL=cell_type
SCTAB_FORMAT=eval/sctab_format.h5ad
TRAIN_SPLIT=intestine_TRAIN.h5ad

python process-finetune-data.py --datapath $DATAPATH --dataname $DATANAME --zero_pad $ZERO_PAD --sctab_var_file $SCTAB_VARFILE --var_type $VAR_TYPE --celltype_col $CELLTYPE_COL
python calc_class_weights.py $TRAIN_SPLIT $DATAPATH

```
