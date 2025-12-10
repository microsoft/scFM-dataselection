import anndata as ad
from pathlib import Path
import re

directory = Path("/users/adenadel/data/adenadel/scFM_reviews/perturbation_evals/")

h5ad_file = directory / "four_drug_four_cell_line.h5ad"

cell_lines = ["LoVo", "PANC-1", "NCI-H2030", "BT-474"]

control = "DMSO_TF"
drugs = ["PH-797804", "Dinaciclib", "Homoharringtonine", "TAK-901"]

adata = ad.read_h5ad(h5ad_file)

for drug in drugs:
    subsetted_adata = adata[adata.obs.drug.isin([control, drug])]
    file_prefix = re.sub(r'[^a-zA-Z0-9]', '', drug).lower()
    subsetted_adata.write_h5ad(directory / f"{file_prefix}_subsetted.h5ad")







