"""compute_vendi_score.py computes the Vendi Score on all subsets
of the scTab corpus."""
import os.path
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import anndata as ad
import scanpy as sc

from vendi_score import vendi


def process_h5ad_file(h5ad_file):
    """Loads an h5ad file, performs standard pre-processing, and computes
    the Vendi Score on the highly variable genes.

    Args:
        h5ad_file: The h5ad file to compute the Vendi Score on.

    Returns:
        The Vendi Score.
    """
    if not os.path.isfile(h5ad_file):
        print(h5ad_file, "does not exist. Skipping.")
        return None

    print("loading h5ad file", h5ad_file)
    adata = ad.read_h5ad(h5ad_file)

    print("normalizing data")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    print("finding highly variable genes and restricting analysis to them")
    sc.pp.highly_variable_genes(adata)
    adata = adata[:, adata.var.highly_variable]

    print("scaling data")
    sc.pp.scale(adata)

    print("computing vendi score")

    return vendi.score_dual(np.asarray(adata.X.todense()))


def main():
    """Loops over all of the scTab downsampled corpuses, computes the Vendi Scores, 
    and saves them to a CSV file.

    Returns:
        None
    """
    sctab_dir = Path(sys.argv[1])
    spikein_dir = Path(sys.argv[2])


    output_dict = defaultdict(list)

    downsampling_methods = [
        "random", "geometric_sketching", "cluster_adaptive"]
    seeds = list(range(30))
    percentages = np.repeat([1, 10, 25, 50, 75, 100], 5)

    for method in downsampling_methods:
        for i, percentage in enumerate(percentages):
            seed = seeds[i]

            print("Processing:", method, percentage, seed)

            sctab_random_dir = sctab_dir / "random"
            h5ad_file = f"idx_{percentage}pct_seed{seed}/idx_{percentage}pct_seed{seed}_TRAIN.h5ad"

            h5ad_file = sctab_random_dir / h5ad_file

            file_vendi_score = process_h5ad_file(h5ad_file)

            output_dict["method"].append(method)
            output_dict["percentage"].append(percentage)
            output_dict["seed"].append(seed)
            output_dict["vendi_score"].append(file_vendi_score)

    downsampling_methods = ["spikein10pct", "spikein50pct"]
    # seeds = list(range(30))
    # percentages = np.repeat([1, 10, 25, 50, 75, 100], 5)

    seeds = list(range(10))
    percentages = np.repeat([1, 10], 5)

    for method in downsampling_methods:
        for i, percentage in enumerate(percentages):
            seed = seeds[i]

            print("Processing:", method, percentage, seed)

            h5ad_file = f"perturb_{method}_replogle_idx_{percentage}pct_seed{seed}_TRAIN.h5ad"

            h5ad_file = spikein_dir / h5ad_file

            file_vendi_score = process_h5ad_file(h5ad_file)

            output_dict["method"].append(method)
            output_dict["percentage"].append(percentage)
            output_dict["seed"].append(seed)
            output_dict["vendi_score"].append(file_vendi_score)

    df = pd.DataFrame.from_dict(output_dict)

    df.to_csv("vendi_score.csv")


if __name__ == "__main__":
    main()
