#!/bin/bash

train_data_dir=$1
test_data_dir=$2
val_data_dir=$3
PCA_OUTPUT_PATH=$4 #make sure its .npy
SKETCH_SIZE=$5 #number of cells not fraction 
PICKLE_FILE_PATH=$6 #make sure its .pkl


conda activate ssl-env


python -c "from sketch_parquet import concat_parquet_files; concat_parquet_files('$train_data_dir', '$test_data_dir', '$val_data_dir')"

python -c "from sktech_parquet import save_pca_of_parquet_files; save_pca_of_parquet_files('$PCA_OUTPUT_PATH', n_comps=50)"

python -c "from sktech_parquet import sketch_parquet_file; sketch_parquet_file('$PCA_OUTPUT_PATH', '$SKETCH_SIZE', '$PICKLE_FILE_PATH', gs_seed=None)"

