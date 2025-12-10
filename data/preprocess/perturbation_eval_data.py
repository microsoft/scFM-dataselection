import os
import argparse

import anndata as ad

from preprocess_class import Preprocess
import itertools

parser = argparse.ArgumentParser(description="preprocess the sctab dataset")
parser.add_argument("--datapath", type=str,
    required=True, help="path to Tahoe dataset")
args = parser.parse_args()

preprocess = Preprocess()
# cell_line_drug_combo = {'PANC-1': 'Dinaciclib', 'NCI-H2030': 'PH-797804', 'BT-474': 'Homoharringtonine', 'LoVo': 'TAK-901'}
cell_lines = ['PANC-1', 'NCI-H2030', 'BT-474', 'LoVo']
drugs = ['Dinaciclib', 'PH-797804', 'Homoharringtonine', 'TAK-901', 'DMSO_TF']
cell_line_drug_adata = {}

for cell_line, drug in itertools.product(cell_lines, drugs):
    cell_line_drug_adata[cell_line+'_'+drug] = []
    cell_line_drug_adata[cell_line+'_'+'DMSO_TF'] = []

plates = os.listdir(args.datapath)
for plate in plates:
    print(plate)
    adata = ad.read_h5ad(os.path.join(args.datapath, plate), backed='r')
    for cell_line, drug in itertools.product(cell_lines, drugs):
        if cell_line in adata.obs['cell_name'].unique() and drug in adata.obs['drug'].unique():
            print(cell_line, drug)
            treated_cell_line = adata[(adata.obs['cell_name'] == cell_line) & (adata.obs['drug'] == drug) & (adata.obs['pass_filter'] == 'full')]
            treated_cell_line = treated_cell_line.to_memory()
            treated_cell_line = treated_cell_line.copy()
            # treated_cell_line = preprocess.normalize_small_adatanorm(adata=treated_cell_line)
            cell_line_drug_adata[cell_line+'_'+drug].append(treated_cell_line)
            print(treated_cell_line.n_obs, '\n')

        # if cell_line in adata.obs['cell_name'].unique() and 'DMSO_TF' in adata.obs['drug'].unique():
        #     print(cell_line, 'DMSO_TF')
        #     control_cell_line = adata[(adata.obs['cell_name'] == cell_line) & (adata.obs['drug'] == 'DMSO_TF') & (adata.obs['pass_filter'] == 'full')]
        #     control_cell_line = control_cell_line.to_memory()
        #     control_cell_line = control_cell_line.copy()
        #     # control_cell_line = preprocess.norm(adata=control_cell_line)
        #     cell_line_drug_adata[cell_line+'_'+'DMSO_TF'].append(control_cell_line)
        #     print(control_cell_line.n_obs, '\n')
    
print('\n', 'concat adatas')
for cell_label, adatas in cell_line_drug_adata.items():
    adata = ad.concat(adatas)
    print(cell_label, adata.n_obs)
    assert (adata.obs['pass_filter'] == 'full').all(), "Not all cells passed the filter"
    assert (adata.obs['cell_name'] == cell_label.split('_')[0]).all(), "Cell name does not match the label"
    assert (adata.obs['drug'] == '_'.join(cell_label.split('_')[1:])).all(), "Drug name does not match the label"
    adata.write_h5ad('{}.h5ad'.format(cell_label))