import anndata as ad
import sys
import os

def main():
    k562_gwps = sys.argv[1]
    k562_essential = sys.argv[2]
    rpe1 = sys.argv[3]
    outdir = sys.argv[4]
    concat_data(k562_gwps, k562_essential, rpe1, outdir)

def concat_data(k562_gwps, k562_essential, rpe1, outdir):
    print("Reading first adata")
    adata1 = ad.read_h5ad(k562_gwps)
    
    print("Reading second adata")
    adata2 = ad.read_h5ad(k562_essential)
    
    print("Reading third adata")
    adata3 = ad.read_h5ad(rpe1)


    print("Combining adata")
    adata = ad.concat([adata1, adata2, adata3], join="outer")

    os.makedirs(outdir, exist_ok=True)
    print("Writing combined adata")
    adata.write_h5ad(os.path.join(outdir,"ReplogleWeissman2022_combined.h5ad"))


if __name__ == "__main__":
    main()

