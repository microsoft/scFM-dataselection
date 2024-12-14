"""diversity.py computes several diversity indices that are based
on metadata on all subsets of the scTab corpus."""
from collections import defaultdict
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

import anndata as ad


def shannon_entropy(proportions):
    """Computes the Shannon Entropy for a list of probabilities/proportions.

    Args:
        proportions: A list of proportions/probabilities.

    Returns:
        The base 2 Shannon Entropy of proportions when considered as a list of probabilities.
    """
    return entropy(proportions, base=2)


def inverse_simpson_index(proportions):
    """Computes the Inverse Simpson Index for list of probabilities/proportions.

    Args:
        proportions: A list of proportions/probabilities.

    Returns:
        The Inverse Simpson Index of proportions when considered as a list of probabilities.
    """
    return 1 / (sum(p**2 for p in proportions))


def gini_simpson_index(proportions):
    """Computes the Gini-Simpson Index for list of probabilities/proportions.

    Args:
        proportions: A list of proportions/probabilities.

    Returns:
        The Gini-Index of proportions when considered as a list of probabilities.
    """
    return 1 - sum(p**2 for p in proportions)


def get_cell_type_proportions(adata, cell_type_column="cell_type"):
    """Returns the cell type proportions for an AnnData object.

    Args:
        adata: An AnnData object.
        cell_type_column: The metadata column containing the cell type labels.

    Returns:
        The cell type proportions for an AnnData object.
    """
    proportions = adata.obs[cell_type_column].value_counts() / adata.n_obs
    return proportions.to_list()

def get_h5ad(sctab_dir, method, seed, percentage):
    """Returns the h5ad file path for a specified downsampled scTab corpus.

    Args:
        sctab_dir: The main directory containing all of the scTab downsampled corpuses.
        method: The downsampling method (random, celltype_reweighted, geometric_sketching).
        seed: The random seed used (0 to 30)
        percentage: The percentage downsampled (1, 10, 25, 50, 75, 100).

    Returns:
        A string containing the h5ad file path.
    """
    h5ad_path = f"{method}/idx_{percentage}pct_seed{seed}/idx_{percentage}pct_seed{seed}_TRAIN.h5ad"
    return sctab_dir / h5ad_path



def main():
    """Loops over all of the scTab downsampled corpuses, computes diversity metrics, 
    and saves them to a CSV file.

    Returns:
        None
    """
    sctab_dir = Path(sys.argv[1])

    methods = ["random", "cluster_adaptive", "geometric_sketching"]
    seeds = list(range(30))
    subsampling_percentages = np.repeat([1, 10, 25, 50, 75, 100], 5)

    data_dict = defaultdict(list)
    for method in methods:
        for i, percentage in enumerate(subsampling_percentages):

            if method != "random" and percentage == 100:
                continue # only random has 100pcts

            seed = seeds[i]

            print("Processing:", method, percentage, seed)

            h5ad_path = get_h5ad(sctab_dir, method, seed, percentage)

            adata = ad.read_h5ad(h5ad_path, backed='r')

            proportions = get_cell_type_proportions(adata)

            data_dict["downsampling_method"].append(method)
            data_dict["percentage"].append(percentage)
            data_dict["seed"].append(seed)


            data_dict["shannon_index"].append(shannon_entropy(proportions))
            data_dict["inverse_simpson_index"].append(inverse_simpson_index(proportions))
            data_dict["gini_simpson_index"].append(gini_simpson_index(proportions))

    diversity_df = pd.DataFrame.from_dict(data_dict)
    diversity_df.to_csv("diversity.csv")

if __name__ == "__main__":
    main()
