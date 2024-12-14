import os
import argparse

import anndata as ad
import numpy as np
import pandas as pd

from preprocess_class import Preprocess



parser = argparse.ArgumentParser(description="preprocess the sctab dataset")
parser.add_argument("--datapath", type=str, required=True,
    help="path to h5ad dataset to be preprocessed")
parser.add_argument("--seed", type=int, required=True, help="Seed number for normal generation.")
parser.add_argument("--pct", type=int, required=True, help="percent of train data.")
parser.add_argument("--sctab_paths", type=str, required=True,
    help="path to the corresponding val and test datasets from sctab processing")
parser.add_argument("--parquet_size", type=float, default=0.01,
    help="pct of total datasize to include in each parquet file")
parser.add_argument("--outdir", type=str, required=True, help="Cluster identifier to process.")

args = parser.parse_args()

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

preprocess = Preprocess()
pct_seed_label = "idx_"+str(args.pct)+"pct_seed"+str(args.seed)
obs_dict = {}

outfile = "spikein_"+pct_seed_label+"_TRAIN_PREPROCESSED.h5ad"
raw_data = args.datapath+'/'+pct_seed_label+'/'+"perturb_spikein10pct_replogle_"+pct_seed_label+"_TRAIN.h5ad"
preprocessed_data = args.datapath+'/'+pct_seed_label+'/'+outfile

preprocess.norm(datapath=raw_data, output_file=preprocessed_data)

val_path = args.sctab_paths+'/'+pct_seed_label+"/"+pct_seed_label+"_VAL_PREPROCESSED.h5ad"
test_path = args.sctab_paths+'/'+pct_seed_label+"/"+pct_seed_label+"_TEST_PREPROCESSED.h5ad"

obs_test = ad.read_h5ad(test_path, backed='r').obs
obs_val = ad.read_h5ad(val_path, backed='r').obs
obs_train = ad.read_h5ad(preprocessed_data, backed='r').obs

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


obs_train, obs_test, obs_val, cols_train = preprocess.convert_categorical_var(obs_train=obs_train,
                                                                              obs_test=obs_test,
                                                                              obs_val=obs_val,
                                                                              output_dir=args.datapath+'/'+pct_seed_label+'/')
obs_dict["TRAIN"] = obs_train
obs_dict["TEST"] = obs_test
obs_dict["VAL"] = obs_val


for split in ["TRAIN"]:
    preprocessed_data = args.datapath+'/'+pct_seed_label+'/spikein_'+pct_seed_label+"_"+split+"_PREPROCESSED.h5ad"
    adata = ad.read_h5ad(preprocessed_data, backed='r')
    parquet_out = os.path.join(args.outdir, pct_seed_label, split.lower())
    os.makedirs(parquet_out, exist_ok=True)
    preprocess.convert_to_dask(adata = adata,
        output_path=args.datapath+"/"+pct_seed_label+'/'+split.lower(),
        split=split,
        obs=obs_dict[split],
        cols_train=cols_train,
        parquet_size=args.parquet_size)
