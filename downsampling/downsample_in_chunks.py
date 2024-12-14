# downsample and in chunks

import os
import pickle

import numpy as np
import pandas as pd

import anndata as ad


def load_subsample_index(pickle_filename):
    """
    Loads the indices for a geometric sketch that was previously computed using
    sketch_anndata() and saved to a pickle file.
 
    Args:
        pickle_filename (str): The binary pickle file containing the sketch index.

    Returns:
        A list containing the indices of the cells found in the sketch.
    """
    with open(pickle_filename, 'rb') as pickle_file:
        sketch_index = pickle.load(pickle_file)
    return sketch_index


def downsample(h5ad_file, index_file, output_dir, output_file):
    adata = ad.read_h5ad(h5ad_file, backed='r')

    index = load_subsample_index(index_file)
    index = pd.Series(index, name="soma_joinid")

    chunk_size = int(221907) #approx 1% of dataset
    num_chunks = int(adata.shape[0] / chunk_size) + 1

    print("Changing working directory to:", output_dir)
    if os.path.exists(output_dir):
        os.chdir(output_dir)
    else:
        os.mkdir(output_dir)
        os.chdir(output_dir)

    for chunk_idx in range(num_chunks):
        print("Chunk:", chunk_idx)

        chunk_start = chunk_size * chunk_idx
        chunk_end = chunk_size * (chunk_idx + 1)

        small_adata = adata[chunk_start:chunk_end, :]
        small_adata = small_adata.to_memory()
        # not sure why this is necessary, scanpy normalize total wasn't normalizing
        # the counts without this
        small_adata = small_adata.copy()

        indices_to_keep = small_adata.obs.soma_joinid.isin(index)

        downsampled_small_adata = small_adata[indices_to_keep, :]

        # not sure why this is necessary, casting the indptr wasn't working without it
        downsampled_small_adata = downsampled_small_adata.copy()

        # cast indptr to int64 for merging
        downsampled_small_adata.X.indptr = np.array(downsampled_small_adata.X.indptr,
                                                    dtype=np.int64)

        downsampled_small_adata.write_h5ad(f"tmp_{chunk_idx}.h5ad")

    tmp_files = list(map(lambda x: f"tmp_{x}.h5ad", range(num_chunks)))
    print("Concatenating (on disk) tmp files")
    ad.experimental.concat_on_disk(tmp_files, output_dir + output_file)

    print("deleting tmp files")
    for tmp_file in tmp_files:
        os.remove(tmp_file)




def main():
    import sys

    out = sys.argv[1]
    index_dir = sys.argv[2]
    h5ad_file = sys.argv[3]
    sampling_percent = sys.argv[4]

    seed_dict = {1 : [0,1,2,3,4], 10: [5,6,7,8,9], 25: [10,11,12,13,14], 50: [15,16,17,18,19], 75: [20,21,22,23,24]}
    seeds = seed_dict[sampling_percent]

    for seed in seeds:
        prefix = f"idx_{sampling_percent}pct_seed{seed}"
        output_dir = os.path.join(out, prefix)

        os.makedirs(output_dir, exist_ok=True)

        print("Train")
        index_file = os.path.join(index_dir, prefix, prefix + "_TRAIN.pkl")
        output_file = prefix + "_TRAIN.h5ad"
        downsample(h5ad_file, index_file, output_dir, output_file)

        print("Val")
        index_file = os.path.join(index_dir, prefix, prefix + "_VAL.pkl")
        output_file = prefix + "_VAL.h5ad"
        downsample(h5ad_file, index_file, output_dir, output_file)

        print("Test")
        index_file = os.path.join(index_dir, prefix, prefix + "_TEST.pkl")
        output_file = prefix + "_TEST.h5ad"
        downsample(h5ad_file, index_file, output_dir, output_file)


if __name__ == "__main__":
    main()
