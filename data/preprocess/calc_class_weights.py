import sys

from sklearn.utils.class_weight import compute_class_weight
import anndata as ad
import numpy as np
import os

DATAPATH = sys.argv[1]
OUTPATH = sys.argv[2]

finetune_adata = ad.read_h5ad(DATAPATH)

# calculate and save class weights
class_weights = compute_class_weight('balanced',
                                     classes=np.unique(finetune_adata.obs['cell_type']),
                                     y=finetune_adata.obs['cell_type'])

with open(OUTPATH+'/class_weights.npy','wb') as f:
    np.save(f, class_weights)

identity_mtx = np.eye(len(np.unique(finetune_adata.obs['cell_type'])))

directory = os.path.dirname(OUTPATH + '/cell_type_hierarchy/')
os.makedirs(directory, exist_ok=True)

with open(OUTPATH+'/cell_type_hierarchy/child_matrix.npy', 'wb') as f:
    np.save(f, identity_mtx)
