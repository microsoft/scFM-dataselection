import pickle
import sys
import subprocess

import numpy as np
import anndata as ad
import pandas as pd

def split_data(ind, random_seed, train_frac, val_frac):
    """
    Given subsampled indices, randomly divide them into train, validation, and test groups.
 
    Args:
        ind (list[int]): List of subsampled indices (soma join ids)
        output_path (str): Path to where the split indices will be stored
        random_seed (int): Random seed used for shuffling the indices
        train_frac (float): Fraction of data used for training
        val_drac (float): Fraction of data used for validation (the rest will be test data)

    Returns:
        None
    """
    # set a random seed for reproducability and shuffles indices
    np.random.seed(random_seed)
    np.random.shuffle(ind)
    ind_size = len(ind)

    # get train, test, and val sizes
    train_split = int(ind_size*train_frac)
    val_split = int(ind_size*val_frac)

    # get train test and val soma join indices
    train = sorted(ind[:train_split])
    val = sorted(ind[train_split:train_split+val_split])
    test = sorted(ind[train_split+val_split:])

    # print first 10 sorted train, test, val indicies for sanity check
    print('train len', len(train))
    print('train inds', train[0:10])
    print('val_len', len(val))
    print('val inds', val[0:10])
    print('test len', len(test))
    print('test inds', test[0:10])
    print('\n')

    return train, test, val

# #training split fraction
# train_frac = float(sys.argv[1])
# #val split fraction
# val_frac = float(sys.argv[2])
# # random seed
# seed = int(sys.argv[3])
# #path to data
# datapath = sys.argv[4]
# #output path
# outdir = sys.argv[5]
# predict_var = sys.argv[6]

# #these are the subsample fractions we used for random sampling
# subsample_fracs = [0.01, 0.1, 0.25, 0.5, 0.75]

# #read in data and get soma join ids for subsampling
# adata = ad.read_h5ad(datapath, backed='r')
# obs = adata.obs
# soma_index = pd.DataFrame(adata.obs.soma_joinid)
# print('Soma join ids:')
# print(soma_index)

# #randomly subsample different fractions of the data, 5 different random seeds each to
# # get a total of of 25 randomly downsample subsets of the data
# for frac in subsample_fracs:
#     for i in np.arange(5):
#         frac_str = str(int(frac*100))
#         ind_output_dir = outdir+'/idx_'+frac_str+'pct'+'_seed'+str(seed)
#         print('making directory ', ind_output_dir)
#         subprocess.run(['mkdir', '-p', ind_output_dir], check=True)
#         ind_output_path = ind_output_dir+'/idx_'+frac_str+'pct'+'_seed'+str(seed)
#         ind = soma_index.soma_joinid.sample(frac=frac, random_state=seed).values
#         train_idx, test_idx, val_idx = split_data(ind, seed, train_frac, val_frac)

#         obs_train = obs[obs.soma_joinid.isin(train_idx)]
#         obs_val = obs[obs.soma_joinid.isin(val_idx)]
#         obs_test = obs[obs.soma_joinid.isin(test_idx)]

#         train_cats = set(obs_train[predict_var].cat.categories)
#         obs_val = obs_val[obs_val[predict_var].isin(train_cats)]
#         obs_test = obs_test[obs_test[predict_var].isin(train_cats)]

#         print(len(test_idx))
#         train_idx = [x for x in train_idx if x in obs_train.soma_joinid.values]
#         test_idx = [x for x in test_idx if x in obs_test.soma_joinid.values]
#         val_idx = [x for x in val_idx if x in obs_val.soma_joinid.values]
#         print(len(test_idx))

#         output_train = ind_output_path+'_TRAIN.pkl'
#         output_val = ind_output_path+'_VAL.pkl'
#         output_test = ind_output_path+'_TEST.pkl'

#         with open(output_train, "wb") as pickle_file:
#             pickle.dump(train_idx, pickle_file)

#         with open(output_val, "wb") as pickle_file:
#             pickle.dump(val_idx, pickle_file)

#         with open(output_test, "wb") as pickle_file:
#             pickle.dump(test_idx, pickle_file)

#         seed += 1
