"""pca_from_hvg.py performs principal component analysis on an 
AnnData object and saves the resulting AnnData object to an h5ad file
with the  principal components stored in adata.obsm['X_pca']."""
import sys

import anndata as ad
import scanpy as sc


H5AD_FILE = sys.argv[1] # merged_hvg_only.h5ad
OUTPUT_FILE = sys.argv[2] # merged_hvg_PCA.h5ad

print("Reading h5ad")
adata = ad.read_h5ad(H5AD_FILE)


# scanpy preprocessing as in tutorial
print("Normalizing")
sc.pp.normalize_total(adata, target_sum=1e4)
print("Log1p")
sc.pp.log1p(adata)

# scale and compute PCA
print("Scaling")
sc.pp.scale(adata, max_value=10)
print("PCA")
sc.tl.pca(adata, svd_solver="arpack")


print("Saving merged_hvg_pca h5ad")
adata.write_h5ad(filename=OUTPUT_FILE)
