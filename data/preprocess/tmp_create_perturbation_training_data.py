import numpy as np
import anndata as ad

def train_test_val_split(adata, train_proportion=0.8, test_proportion=0.1, val_proportion=0.1, random_state=0):
        np.random.seed(random_state)

        ind = list(range(adata.n_obs))

        np.random.shuffle(ind)
        ind_size = len(ind)
        train_frac = 0.8
        val_frac = 0.1

        # get train, test, and val sizes
        train_split = int(ind_size * train_proportion)
        val_split = int(ind_size * val_proportion)

        # get train test and val indices
        train_idx = sorted(ind[:train_split])
        val_idx = sorted(ind[train_split:train_split+val_split])
        test_idx = sorted(ind[train_split+val_split:])

        train = adata[train_idx].copy()
        val = adata[val_idx].copy()
        test = adata[test_idx].copy()
        return train, val, test

h5ads = ["dinaciclib_subsetted.h5ad", "homoharringtonine_subsetted.h5ad", "ph797804_subsetted.h5ad", "tak901_subsetted.h5ad"]
prefixes = ["dinaciclib", "homoharringtonine", "ph797804", "tak901"]

for h5ad, prefix in zip(h5ads, prefixes):
    adata = ad.read_h5ad(h5ad)
    train, val, test = train_test_val_split(adata)

    train.write_h5ad(f"{prefix}_train.h5ad")
    val.write_h5ad(f"{prefix}_val.h5ad")
    test.write_h5ad(f"{prefix}_test.h5ad")