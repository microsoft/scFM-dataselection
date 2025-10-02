import anndata as ad
import itertools

cell_lines = ["LoVo", "PANC-1", "NCI-H2030", "BT-474"]
drugs = ["DMSO_TF", "PH-797804", "Dinaciclib", "Homoharringtonine", "TAK-901"]

target_pairs = list(itertools.product(cell_lines, drugs))

subset_adata_list = []

for plate in range(1,15):
    print(plate)
    adata = ad.read_h5ad(f"plate{plate}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad", backed='r')
    subset_adata = adata[adata.obs[['cell_name', 'drug']].agg(tuple, axis=1).isin(target_pairs)].to_memory()
    subset_adata_list.append(subset_adata)


subsetted_adata = ad.concat(subset_adata_list)

subsetted_adata.write_h5ad("/users/adenadel/data/adenadel/scFM_reviews/perturbation_evals/four_drug_four_cell_line.h5ad")
