import sys

import matplotlib.pyplot as plt

import anndata as ad
import scanpy as sc

sctab_h5ad = sys.argv[1]

sctab = ad.read_h5ad(sctab_h5ad)

sc.pp.normalize_total(sctab)
# Logarithmize the data
sc.pp.log1p(sctab)

sc.pp.highly_variable_genes(sctab, n_top_genes=2000)
sc.tl.pca(sctab)
sc.pp.neighbors(sctab)
sc.tl.umap(sctab)

nine_most_common_tissues = sctab.obs.tissue_general.value_counts().nlargest(9).index.tolist()

sctab = sctab[sctab.obs.tissue_general.isin(nine_most_common_tissues)]


sc.pl.umap(
    sctab,
    color="tissue_general",
    #cmap="Pastel1",
    palette="Pastel1",
    # Setting a smaller point size to get prevent overlap
    size=2,
    title="",
    frameon=False,
    save="sctab_umap.pdf",
    legend_loc=None
)