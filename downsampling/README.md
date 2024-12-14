# Downsampling
Below are the instructions for downsampling the scTab data and generating the spike-in datasets. 

## Random downsampling
This file is a csv of the scTab soma join indices, used for generating downsampled indices
```bash
soma_index.csv
```
To produce the randomly downsampled indices, run `random_downsampling.py` as follows.
```bash 
SCTAB_DATAPATH=full_sctab_dataset_RAW.h5ad
OUTPUTPATH=random_downsample_indices
SOMA_INDEX_FILE=soma_index.csv

python random_downsampling.py $SCTAB_DATAPATH $OUTPUTPATH $SOMA_INDEX_FILE
```

## Cell Type Reweighting

To compute the cell type reweighted downsampled indices, run `celltype_reweighted_downsampling.py` as follows:
```bash
SCTAB_DATAPATH=full_sctab_dataset_RAW.h5ad
OUTPUTPATH=celltype_reweighted_indices

python celltype_reweighted_downsampling.py $SCTAB_DATAPATH $OUTPUPATH
```

## Geometric Sketching

First, to reduce the need for extremely high memory to load the scTab dataset, we compute
HVGs on a 10pct subset of the data and restrict the total dataset to these HVGs.

```bash
TEN_PCT_H5AD_FILE=idx_10pct_seed5_TRAIN.h5ad
MERGED_H5AD_FILE=full_sctab_dataset_RAW.h5ad

OUTPUT_FILE=merged_hvg_only.h5ad

python subsample_merged_hvg_only.py $TEN_PCT_H5AD_FILE $MERGED_H5AD_FILE $OUTPUT_FILE
```

Then, we compute the principal components of the full dataset restricted to the HVGs.
```bash
H5AD_FILE=merged_hvg_only.h5ad
OUTPUT_FILE=merged_hvg_PCA.h5ad

python pca_from_hvg.py $H5AD_FILE $OUTPUT_FILE
```

Finally, we perform geometric sketching for each of the percentage, seed pairs.
```bash
H5AD_FILE=merged_hvg_PCA
sampling_percent=1
seed=0
output_dir=geometric_sketching_indices_dir

python geometric_sketch_anndata_with_PCA.py
```
## Downsample ScTab

To use the generated indices to downsample the scTab dataset for one of the downsampling percentages (i.e. 1, 10, 25, 50 or 75) in a memory-efficient way, run `downsample_in_chunks.py` as follows:

```bash
OUTPUTPATH=downsampled_sctab
INDEX_DIR=random_downsample_indices
H5AD_FILE=full_sctab_dataset_RAW.h5ad
SAMPLING_PERCENT=1

python downsample_in_chunks.py $OUTPUTPATH $INDEX_DIR $H5AD_FILE $SAMPLING_PERCENT
```


## 10% Spike-In
```bash
TRAINING_DIR=/path/to/sctab/
OUTPUT_PREFIX=perturb_spikein10pct_replogle_
PERTURBATION_H5AD=ReplogleWeissman2022_combined.h5ad

SCTAB_FORMATTED_H5AD=/path/to/idx_1pct_seed0_TEST.h5ad # for example
VAR_FILE=/path/to/adata_var.csv

python spikein.py -t $TRAINING_DIR -s 0.1 -o $OUTPUT_PREFIX -p $PERTURBATION_H5AD -f $SCTAB_FORMATTED_H5AD -v $VAR_FILE
```

## 50% Spike-In

```bash
TRAINING_DIR=/path/to/sctab/
OUTPUT_PREFIX=perturb_spikein50pct_replogle_
PERTURBATION_H5AD=ReplogleWeissman2022_combined.h5ad

SCTAB_FORMATTED_H5AD=/path/to/idx_1pct_seed0_TEST.h5ad # for example
VAR_FILE=/path/to/adata_var.csv


To generate the 10% and 50% spikein datasets using the perturbation dats, run `spikein.py` as follows: 

```bash
TRAINING_DIR=/path/to/sctab/
OUTPUT_PREFIX=perturb_spikein10pct_replogle_
PERTURBATION_H5AD=ReplogleWeissman2022_combined.h5ad

python spikein.py -t $TRAINING_DIR -s 0.1 -o $OUTPUT_PREFIX -p $PERTURBATION_H5AD -f $SCTAB_FORMATTED_H5AD -v $VAR_FILE
```

