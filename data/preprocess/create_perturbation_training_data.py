import numpy as np
import anndata as ad

def train_test_val_split(adata, test_set, train_proportion=0.9, random_state=0):
        np.random.seed(random_state)

        # put test_set cell line in a test_data and put the rest of the cell linse in adata
        test_adata = adata[adata.obs["cell_name"] == test_set].copy()
        adata = adata[adata.obs["cell_name"] != test_set].copy()

        ind = list(range(adata.n_obs))

        np.random.shuffle(ind)
        ind_size = len(ind)

        # get train, test, and val sizes
        train_split = int(ind_size * train_proportion)
        val_split = adata.n_obs - train_split

        # get train test and val indices
        train_idx = sorted(ind[:train_split])
        val_idx = sorted(ind[train_split:train_split+val_split])

        train = adata[train_idx].copy()
        val = adata[val_idx].copy()
        return train, val, test_adata

h5ads = ["dinaciclib_subsetted.h5ad", "homoharringtonine_subsetted.h5ad", "ph797804_subsetted.h5ad", "tak901_subsetted.h5ad"]
prefixes = ["dinaciclib", "homoharringtonine", "ph797804", "tak901"]

test_set = {
    "dinaciclib": "NCI-H2030",
    "homoharringtonine": "LoVo",
    "ph797804": "BT-474",
    "tak901": "PANC-1"
}

for h5ad, prefix in zip(h5ads, prefixes):
    adata = ad.read_h5ad(h5ad)
    train, val, test = train_test_val_split(adata, test_set[prefix])

    train.write_h5ad(f"{prefix}_train.h5ad")
    val.write_h5ad(f"{prefix}_val.h5ad")
    test.write_h5ad(f"{prefix}_test.h5ad")