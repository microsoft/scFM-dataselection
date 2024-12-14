"""geometric_sketch_anndata_with_PCA.py performs geometric sketching on an 
AnnData object and saves the resulting test, train, and validation indices.
The AnnData object must have principal components previously computed and 
stored in adata.obsm['X_pca']."""
import sys
import os
import time
import pickle
from random import shuffle

import anndata as ad

from geosketch import gs



H5AD_FILE = sys.argv[1]
sampling_percent = int(sys.argv[2])
seed = int(sys.argv[3])
output_dir = sys.argv[4]


print('sampling percent', sampling_percent)
print('seed', seed)

print("Loading H5AD_FILE")
adata = ad.read_h5ad(H5AD_FILE, backed='r')



# derived params
sketch_size = int(adata.n_obs * sampling_percent / 100.0)
# 1% of n_obs to (hopefully) be faster than using n_obs covering boxes (default when replace=False)
num_covering_boxes = int(0.01 * adata.n_obs)


print("Starting geometric sketching")

start = time.time()

sketch_index = gs(adata.obsm['X_pca'], sketch_size, seed=seed, k=num_covering_boxes, replace=False)

end = time.time()

print("Geometric sketching complete")

print("Time Taken")
print(end - start)


# soma joinids pd.Series
soma_joinids = adata.obs.soma_joinid.iloc[sketch_index]

soma_joinids = soma_joinids.to_list()


print("Creating train/val/test split")
# params for train, val, test split
ind_size = len(soma_joinids)
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

train_split = int(ind_size * TRAIN_FRAC)
val_split = int(ind_size * VAL_FRAC)

# randomly shuffle data so that we can take consecutive values
shuffle(soma_joinids)
train = soma_joinids[:train_split]
val = soma_joinids[train_split:train_split+val_split]
test = soma_joinids[train_split+val_split:]


prefix = f"idx_{sampling_percent}pct_seed{seed}"

output_path = output_dir + "/" + prefix + "/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

output_train = output_path + prefix + '_TRAIN.pkl'
output_val = output_path + prefix + '_VAL.pkl'
output_test = output_path + prefix + '_TEST.pkl'

print("Writing index files")

with open(output_train, "wb") as pickle_file:
    pickle.dump(train, pickle_file)

with open(output_val, "wb") as pickle_file:
    pickle.dump(val, pickle_file)

with open(output_test, "wb") as pickle_file:
    pickle.dump(test, pickle_file)
