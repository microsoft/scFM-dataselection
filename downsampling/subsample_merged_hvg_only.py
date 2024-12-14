"""pca_from_hvg.py performs principal component analysis on an 
AnnData object and saves the resulting AnnData object to an h5ad file
with the  principal components stored in adata.obsm['X_pca']."""
import sys

import anndata as ad
import scanpy as sc


TEN_PCT_H5AD_FILE = sys.argv[1] # idx_10pct_seed5_TRAIN.h5ad
MERGED_H5AD_FILE = sys.argv[1] # merged.h5ad
OUTPUT_FILE = sys.argv[2] # merged_hvg_only.h5ad



print("Reading 10pct h5ad")
adata = ad.read_h5ad(TEN_PCT_H5AD_FILE)


print("Processing 10pct h5ad")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)


hvg = adata.var.highly_variable


print("Reading merged h5ad")

adata = ad.read_h5ad(MERGED_H5AD_FILE)

adata = adata[:, hvg]

print("Saving merged_hvg_only h5ad")
adata.write_h5ad(filename=OUTPUT_FILE)
