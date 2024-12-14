import os
import sys

import numpy as np
import pandas as pd

import anndata as ad

import geneformer
from geneformer import TranscriptomeTokenizer

# Confirm the path of the imported module
print(geneformer.__file__)

import gc
from datasets import load_from_disk, concatenate_datasets
import shutil



def add_ensembl_ids_to_adata(adata, var_file, data_type):
    if data_type == 'sctab' or data_type == 'spikein':
        var_df = pd.read_csv(var_file, index_col=0)
        var_df.index = var_df.index.map(str)
        adata.var = var_df
        adata.var_names = adata.var["feature_name"].values
    adata.var["ensembl_id"] = adata.var["feature_id"]
    adata.var.index = adata.var['ensembl_id']
    return adata

def add_n_counts_to_adata(adata, geneformer_adata):
    print("Original adata.var columns:", adata.var.columns)
    
    # Save adata.var to ensure all metadata is preserved
    var_metadata = adata.var.copy()

    chunk_size = int(adata.n_obs*0.01)  # approx 1% of dataset
    num_chunks = int(adata.shape[0] / chunk_size) + 1
    print('NUM CHUNKS ', num_chunks)

    for chunk_idx in range(num_chunks):
        print("Chunk:", chunk_idx)

        chunk_start = chunk_size * chunk_idx
        chunk_end = chunk_size * (chunk_idx + 1)

        small_adata = adata[chunk_start:chunk_end, :]
        small_adata = small_adata.to_memory()
        # not sure why this is necessary, scanpy normalize total
        # wasn't normalizing the counts without this
        small_adata = small_adata.copy()

        # add n counts
        counts_per_cell = np.sum(small_adata.X, axis=1)
        small_adata.obs["n_counts"] = counts_per_cell

        # cast indptr to int64 for merging
        small_adata.X.indptr = np.array(small_adata.X.indptr, dtype=np.int64)
        small_adata.write_h5ad(f"tmp_{chunk_idx}.h5ad")

        del small_adata
        gc.collect()

    tmp_files = list(map(lambda x: f"tmp_{x}.h5ad", range(num_chunks)))
    print("Concatenating (on disk) tmp files")
    # write to new location for h5ad file
    ad.experimental.concat_on_disk(tmp_files, geneformer_adata)

    # Re-assign the saved var metadata to the concatenated adata file
    geneformer_adata_full = ad.read_h5ad(geneformer_adata)
    geneformer_adata_full.var = var_metadata  # Restore full var metadata
    geneformer_adata_full.write_h5ad(geneformer_adata)  # Save with updated var columns

    print("deleting tmp files")
    for tmp_file in tmp_files:
        os.remove(tmp_file)

    print("Final geneformer_adata.var columns:", geneformer_adata_full.var.columns)

    print("adata.var columns:", adata.var.columns)
    print("adata.obs columns after add_n_counts:", adata.obs.columns)
    return adata
    

### geneformer requires a column in the anndata object called "ensembl_id" that contains the ensembl ids
def add_ensembl_ids(geneformer_adata, var_file, data_type):
    print("Adding ensembl ids to " + geneformer_adata)
    adata = ad.read_h5ad(geneformer_adata, backed='r')
    adata = add_ensembl_ids_to_adata(adata, var_file, data_type)
    print("adata.var columns:", adata.var.columns)
    print("adata.obs columns after add_n_counts:", adata.obs.columns)
    adata.write_h5ad(geneformer_adata) ## overwrite original h5ad file

def add_n_counts(h5ad_file, geneformer_adata):
    print("Adding n_counts ids to " + h5ad_file)
    adata = ad.read_h5ad(h5ad_file, backed='r')

    adata = add_n_counts_to_adata(adata, geneformer_adata)


def get_unique_integers_not_in_set(excluded_set, length):
    """
    Generate a list of unique integers within a specified range that are not in the excluded set.

    :param excluded_set: Set of integers to exclude.
    :param range_start: Start of the range (inclusive) to generate integers.
    :param range_end: End of the range (exclusive) to generate integers.
    :return: List of unique integers not in the excluded set.
    """
    # Create a list of all integers in the specified range
    all_integers = set(range(0, int(1e7)))

    # Subtract the excluded set to get integers not in the excluded set
    unique_ints = list(all_integers - set(excluded_set.values))

    if len(unique_ints) < length:
        raise ValueError("Not enough valid integers to generate the required length.")

    random_ints = unique_ints[0:length]

    return random_ints

def tokenize(datasetname,
             geneformer_adata,
             output_dir, obs_train,
             nprocesses=32):

    output_directory = os.path.join(output_dir,f"{datasetname}")
    outfile_fulldata = os.path.join(output_directory,f"{datasetname}_tokenized.dataset")

    adata = ad.read_h5ad(geneformer_adata, backed='r')
    print("number of obs before new obs", adata.n_obs)

    n_counts = adata.obs['n_counts']
    adata.obs = obs_train
    adata.obs['n_counts'] = n_counts
    
    print("number of obs after new obs", adata.n_obs)
    chunk_size = int(adata.n_obs*0.01) ##approx 1% of dataset

    print('chunk size: ', chunk_size)
    num_chunks = int(adata.shape[0] / chunk_size) + 1
    print('NUM CHUNKS ', num_chunks)


    tk = TranscriptomeTokenizer(custom_attr_name_dict={col: col for col in adata.obs.columns}, nproc=nprocesses)

    for chunk_idx in range(num_chunks):
    # Check if the file already exists and skip if it does
        output_prefix_chunk = f"tmp_chunk{chunk_idx}"
        print(output_directory + f"{output_prefix_chunk}.dataset")
        if os.path.exists(output_directory + f"{output_prefix_chunk}.dataset"):
            print("Skipping " + str(chunk_idx) + " as it already exists.")
            continue

        chunk_start = chunk_size * chunk_idx
        chunk_end = chunk_size * (chunk_idx + 1)

        print('on chunk tokenization: ', chunk_idx)
        output_prefix_chunk = f"tmp_chunk{chunk_idx}"
        small_adata = adata[chunk_start:chunk_end, :]
        small_adata = small_adata.to_memory()
        small_adata = small_adata.copy()


        tk.tokenize_data(small_adata, output_directory, output_prefix_chunk, file_format="h5ad")
        
        del small_adata
        gc.collect()


    data_list = []
    tmp_files = list(map(lambda x: os.path.join(output_directory, "tmp_chunk" + str(x) + ".dataset"), range(num_chunks)))
    for datapath in tmp_files:
        print(datapath)
        tmp_data = load_from_disk(datapath)
        data_list.append(tmp_data)

    del tmp_data 
    gc.collect()

    full_data = concatenate_datasets(data_list)
    full_data.save_to_disk(outfile_fulldata)

    for tmp_file in tmp_files:
        shutil.rmtree(tmp_file)


def tokenize_training_data(datasetname, data_dir, output_dir, var_file, run_add_n_counts, data_type):
    
    h5ad_directory = os.path.join(data_dir, "original",f"{datasetname}")
    h5ad_file = os.path.join(h5ad_directory,f"{datasetname}.h5ad")
    geneformer_adata_path = os.path.join(data_dir, f"tokenized")
    geneformer_adata = os.path.join(geneformer_adata_path, f"{datasetname}",f"{datasetname}.h5ad")

    if "hematopoiesis" in datasetname:
        # First H5AD file
        adata = ad.read_h5ad(h5ad_file, backed='r')
        columns_to_drop = ["HTO_maxID", "HTO_secondID", "HTO_margin", "HTO_classification.global", "HTOID"]
        existing_columns = [col for col in columns_to_drop if col in adata.obs.columns]
        if existing_columns:  # Only drop if there are columns to drop
            adata.obs = adata.obs.drop(columns=existing_columns)
            adata.write_h5ad(h5ad_file)

        # Geneformer H5AD file
        adata = ad.read_h5ad(geneformer_adata, backed='r')
        existing_columns = [col for col in columns_to_drop if col in adata.obs.columns]
        if existing_columns:  # Only drop if there are columns to drop
            adata.obs = adata.obs.drop(columns=existing_columns)
            adata.write_h5ad(geneformer_adata)

    obs_train = ad.read_h5ad(h5ad_file, backed='r').obs
    if data_type == 'spikein':

        for col in obs_train.columns:
            if col == 'is_primary_data':
                obs_train[col] = obs_train[col].replace({'True': True, 'False': False, "<NA>": True})
                obs_train[col] = obs_train[col].astype(bool)
            else:
                if type(obs_train[col][0]) == np.float64:
                    if obs_train[col].isna().any():
                        if col == 'soma_joinid':
                            current_ids = obs_train[obs_train[col].notna()][col]

                            nans = obs_train[col].isna()
                            num_nans = nans.sum()

                            unique_integers = get_unique_integers_not_in_set(current_ids, num_nans)
                            unique_int_series = pd.Series(unique_integers, index=obs_train.index[nans])
                            obs_train.loc[nans, col] = unique_int_series
                elif type(obs_train[col][0]) != np.float32:
                    if obs_train[col].isna().any():
                        obs_train[col] = obs_train[col].cat.add_categories("spikein")
                        obs_train[col].fillna("spikein", inplace=True)
                        obs_train[col] = obs_train[col].astype('category')

    if run_add_n_counts:
        print(run_add_n_counts)
        print(f"Adding n_counts to h5ad: {datasetname}")
        add_n_counts(h5ad_file, geneformer_adata)

        print(f"Adding ensembl ids to h5ad: {datasetname}")
        add_ensembl_ids(geneformer_adata, var_file, data_type)

    print(run_add_n_counts)
    adata = ad.read_h5ad(geneformer_adata, backed='r')  # Load the adata to check columns
    print("adata.obs columns:", adata.obs.columns)
    print("adata.var columns:", adata.var.columns)

    print(f"Tokenizing: {datasetname} ")
    tokenize(datasetname=datasetname, geneformer_adata=geneformer_adata, output_dir=output_dir, obs_train=obs_train)



def main():
    import argparse
    import shutil
    parser = argparse.ArgumentParser(
        description='Train an scVI model with specified    .')
    parser.add_argument('--datasetname', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--var_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--run_add_n_counts', type=bool, default=True)
    parser.add_argument('--data_type', type=str, default='sctab')

    args = parser.parse_args()

    data_file = os.path.join(args.data_dir, args.datasetname+'.h5ad')
    dest_file = os.path.join(args.data_dir, 'original', args.datasetname, args.datasetname+'.h5ad')
    original = os.path.join(args.data_dir, 'original', args.datasetname)
    tokenized = os.path.join(args.data_dir, 'tokenized', args.datasetname)
    os.makedirs(original, exist_ok=True)
    os.makedirs(tokenized, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy(data_file, dest_file)

    tokenize_training_data(args.datasetname, args.data_dir, args.output_dir, args.var_file, args.run_add_n_counts, args.data_type)


if __name__ == "__main__":
    main()