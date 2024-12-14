import pickle
import numpy as np
import sys
import dask.dataframe as dd
import numpy as np
import subprocess
from train_val_test_split import split_data
import anndata as ad
import pandas as pd


sctab_data = sys.argv[1]
output_path_soma_index = sys.argv[2]
soma_index = sys.argv[3]

soma_index = pd.read_csv(soma_index)
adata = ad.read_h5ad(sctab_data, backed='r')
obs = adata.obs
#### SOMA JOIN INDS

train_frac = 0.8
val_frac= 0.1
seed = 0
subsample_fracs = [0.01, 0.1, 0.25, 0.5, 0.75]

for frac in subsample_fracs:
    for i in np.arange(5):
        frac_str = str(int(frac*100))
        ind_output_dir = output_path_soma_index+'/idx_'+frac_str+'pct'+'_seed'+str(seed)
        print('making directory ', ind_output_dir)
        subprocess.run(['mkdir', '-p', ind_output_dir], check=True)
        ind_output_path = ind_output_dir+'/idx_'+frac_str+'pct'+'_seed'+str(seed)
        ind = soma_index.soma_joinid.sample(frac=frac, random_state=seed).values

        train_idx, test_idx, val_idx = split_data(ind, seed, train_frac, val_frac)

        obs_train = obs[obs.soma_joinid.isin(train_idx)]
        obs_val = obs[obs.soma_joinid.isin(val_idx)]
        obs_test = obs[obs.soma_joinid.isin(test_idx)]

        train_idx = [x for x in train_idx if x in obs_train.soma_joinid.values]
        test_idx = [x for x in test_idx if x in obs_test.soma_joinid.values]
        val_idx = [x for x in val_idx if x in obs_val.soma_joinid.values]

        output_train = ind_output_path+'_TRAIN.pkl'
        output_val = ind_output_path+'_VAL.pkl'
        output_test = ind_output_path+'_TEST.pkl'

        with open(output_train, "wb") as pickle_file:
            pickle.dump(train_idx, pickle_file)

        with open(output_val, "wb") as pickle_file:
            pickle.dump(val_idx, pickle_file)

        with open(output_test, "wb") as pickle_file:
            pickle.dump(test_idx, pickle_file)

        seed += 1

seed = 25
frac_str='100'

for i in np.arange(5):
    ind_output_dir = output_path_soma_index+'/idx_'+frac_str+'pct'+'_seed'+str(seed)
    print('making directory ', ind_output_dir)
    subprocess.run(['mkdir', '-p', ind_output_dir], check=True)
    ind_output_path = ind_output_dir+'/idx_'+frac_str+'pct'+'_seed'+str(seed)
    ind = soma_index.soma_joinid.values
    train_idx, test_idx, val_idx = split_data(ind, seed, train_frac, val_frac)

    obs_train = obs[obs.soma_joinid.isin(train_idx)]
    obs_val = obs[obs.soma_joinid.isin(val_idx)]
    obs_test = obs[obs.soma_joinid.isin(test_idx)]

    train_idx = [x for x in train_idx if x in obs_train.soma_joinid.values]
    test_idx = [x for x in test_idx if x in obs_test.soma_joinid.values]
    val_idx = [x for x in val_idx if x in obs_val.soma_joinid.values]

    output_train = ind_output_path+'_TRAIN.pkl'
    output_val = ind_output_path+'_VAL.pkl'
    output_test = ind_output_path+'_TEST.pkl'

    with open(output_train, "wb") as pickle_file:
        pickle.dump(train_idx, pickle_file)

    with open(output_val, "wb") as pickle_file:
        pickle.dump(val_idx, pickle_file)

    with open(output_test, "wb") as pickle_file:
        pickle.dump(test_idx, pickle_file)

    seed += 1