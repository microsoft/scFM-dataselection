import os

import anndata as ad
import scanpy as sc
import numpy as np
import os
from os.path import join

import anndata as ad
import pandas as pd
import numpy as np
import gc

import pandas as pd
import numpy as np
import pyarrow as pa

from os.path import join
import mygene
from scipy.sparse import hstack, csr_matrix
from scipy.io import mmread
import random
from pathlib import Path

class Preprocess():
    def __init__(self):
        return 

    def norm(self, datapath, output_file, finetune_data=False):
        """
        This function preprocesses Raw Counts stored in an AnnData file by normalizing to 10k counts and applying a log-transformation.
        In order to have this process fit into memory when working with millions of cells, we iterate over small chunks of the dataset, apply the transformation, 
        save each chunk to a temporary file, then concatenate these files on disk without loading them into memory, resulting in a preprocessed version of the whole dataset.
        **Note: ad.experimental.concat_on_disk does not preserve adata.raw or adata.var layers. Store this information elsewhere before concatenating if you need it.
        It is an experimental anndata function so some quirks like that seem to not have been worked out yet.

        Inputs:
        output_dir: str, path to directory where we want to save transformed anndata object 
        output_file: str, name of h5ad file we will save transformed anndata object to
        update_datapath: boolean, if true, will update the self.datapath field to the preprocessed anndata path

        Outputs: 
        None 
        """

        # read anndata object in backed mode so that it is not loaded into memory
        adata = ad.read_h5ad(datapath, backed='r')
    
        # determine number of chunks of the data we will iterate over
        chunk_size = int(adata.n_obs*0.01) #approx 1% of dataset
        num_chunks = int(adata.shape[0] / chunk_size) + 1
        print('NUM CHUNKS ', num_chunks)

        for chunk_idx in range(num_chunks):
            print("Preprocesing Chunk:", chunk_idx)

            # get subsetting indices for the current chunk!
            chunk_start = chunk_size * chunk_idx
            chunk_end = chunk_size * (chunk_idx + 1)

            if finetune_data:
                if chunk_end > adata.n_obs:
                    chunk_end = adata.n_obs - 1

            # subset the data to a smaller chunk 
            small_adata = adata[chunk_start:chunk_end, :]
            small_adata = small_adata.to_memory()
            small_adata = small_adata.copy()
            # preprocess the chunk 
            sc.pp.normalize_per_cell(small_adata, counts_per_cell_after=1e4)
            sc.pp.log1p(small_adata)

            # cast indptr to int64 for merging (this indptr is required for the ad.experimental.concat_on_disk function)
            if not isinstance(small_adata.X, np.ndarray):
                small_adata.X.indptr = np.array(small_adata.X.indptr, dtype=np.int64)
            # save chunk to a temporary file
            small_adata.write_h5ad(f"tmp_{chunk_idx}.h5ad")

        tmp_files = list(map(lambda x: f"tmp_{x}.h5ad", range(num_chunks)))
        print("Concatenating (on disk) tmp files")
        #concatenate temporary chunk files to generate a preprocessed version of the whole dataset and save to an output h5ad file
        ad.experimental.concat_on_disk(tmp_files, output_file)

        # delete temporary chunk files we dont need them anymore
        print("deleting tmp files")
        for tmp_file in tmp_files:
            os.remove(tmp_file)

    def convert_to_dask(self, adata, output_path, split, obs, cols_train, parquet_size):
        chunk_size = int(adata.n_obs*parquet_size)
        num_chunks = int(adata.shape[0] / chunk_size) + 1
        print('NUM CHUNKS ', num_chunks)

        adata.obs = obs

        for chunk_idx in range(num_chunks):
            if os.path.exists(output_path+'/'+f"{split}_{chunk_idx}.parquet"):
                print("Skipping "+ str(chunk_idx) +" as it already exists.")
                continue
            else:
                print("Converting Chunk ", chunk_idx, " to parquet")
                chunk_start = chunk_size * chunk_idx
                chunk_end = chunk_size * (chunk_idx + 1)
                if chunk_end > adata.n_obs:
                    chunk_end = adata.n_obs - 1

                small_adata = adata[chunk_start:chunk_end, :]
                small_adata = small_adata.to_memory()
                small_adata = small_adata.copy()


                df = pd.DataFrame({'X':small_adata.X.todense().tolist(), 'soma_joinid': small_adata.obs.soma_joinid.values, 'is_primary_data': small_adata.obs.is_primary_data.values, 'dataset_id': small_adata.obs.dataset_id.values, 'donor_id': small_adata.obs.donor_id.values, "assay": small_adata.obs.assay.values, "cell_type": small_adata.obs.cell_type.values, "development_stage": small_adata.obs.development_stage.values, "disease": small_adata.obs.disease.values, "tissue": small_adata.obs.tissue.values, "tissue_general": small_adata.obs.tissue_general.values}, index=small_adata.obs_names, dtype=np.float32)
                
                schema = pa.schema([
                    ('X', pa.list_(pa.float32())),
                    ('soma_joinid', pa.int64()),
                    ('is_primary_data', pa.bool_()),
                    ('dataset_id', pa.int64()),
                    ('donor_id', pa.int64()),
                    ('assay', pa.int64()),
                    ('cell_type', pa.int64()),
                    ('development_stage', pa.int64()),
                    ('disease', pa.int64()),
                    ('tissue', pa.int64()),
                    ('tissue_general', pa.int64()),
                ])

                df.to_parquet(output_path+'/'+f"{split}_{chunk_idx}.parquet", engine='pyarrow',schema=schema)

            gc.collect()

    def convert_categorical_var(self, obs_train, obs_val, obs_test, output_dir):
        os.makedirs(os.path.join(output_dir, 'categorical_lookup'), exist_ok=True)
        obs = pd.concat([obs_train, obs_val, obs_test])
        cols_train=obs_train.columns.values
        for col in cols_train:
            if obs[col].dtype.name == 'category':
                obs[col] = obs[col].cat.remove_unused_categories()

        for col in cols_train:
            if obs[col].dtype.name == 'category':
                categories = list(obs[col].cat.categories)
                obs_train[col] = pd.Categorical(obs_train[col], categories, ordered=False)
                obs_val[col] = pd.Categorical(obs_val[col], categories, ordered=False)
                obs_test[col] = pd.Categorical(obs_test[col], categories, ordered=False)

        for col in cols_train:
            if obs_train[col].dtype.name == 'category':
                cats_train = pd.Series(dict(enumerate(obs_train[col].cat.categories))).to_frame().rename(columns={0: 'label'})
                if '/' in col:
                    col = col.replace('/', '_')
                if col == 'cell_type':
                    out = join(output_dir, 'categorical_lookup', f'{col}.parquet')
                else: 
                    out = join(output_dir, f'{col}.parquet')
                cats_train.to_parquet(out, index=True)

        for col in cols_train:
            if obs_train[col].dtype.name == 'category':

                obs_train[col] = obs_train[col].cat.codes.astype('i8')
                obs_val[col] = obs_val[col].cat.codes.astype('i8')
                obs_test[col] = obs_test[col].cat.codes.astype('i8')
        cols_train = [x for x in cols_train if x != "n_counts"]
        return obs_train, obs_test, obs_val, cols_train

    def fill_empty_cols(self, obs):
        adata_cols = obs.columns

        sctab_cols =  {
        "soma_joinid": "int64",
        "is_primary_data": "boolean",
        "dataset_id": "int64",
        "donor_id": "int64",
        "assay": "int64",
        "cell_type": "int64",
        "development_stage": "int64",
        "disease": "int64",
        "tissue": "int64",
        "tissue_general": "int64",
        "tech_sample": "int64",
        "idx": "int64",
        }   

        fill_cols = [x for x in list(sctab_cols.keys()) if x not in adata_cols]
        drop_cols = [x for x in adata_cols if x not in list(sctab_cols.keys())]

        obs = obs.drop(columns=drop_cols)
        
        val_dict = {"int64": 1, "boolean": True}
        for col in fill_cols:
            val = val_dict[sctab_cols[col]]
            obs[col] = pd.Series(val, index=obs.index, dtype=sctab_cols[col])

        return obs

    
    def filter_genes(self, x, ensembl_list):
        keep = True
        if 'ensembl' in x:
            if isinstance(x['ensembl'], list) or x['ensembl']['gene'] not in ensembl_list:
                keep = False
        else:
            keep = False
        return keep
    
    def get_ensembl_id(self, finetune_adata, sctab_features, var_type):
        if var_type == 'ensembl':
            zero_padded_genes = list(set(sctab_features) - set(finetune_adata.var_names.values))
            finetune_adata = finetune_adata[:, [x for x in finetune_adata.var_names if x in sctab_features.keys()]]
        else:
            mg = mygene.MyGeneInfo()
            gene_names = finetune_adata.var_names
            mg_results = mg.querymany(gene_names, scopes='symbol', fields='ensembl.gene', species='human')
            filtered_mg_results = [x for x in mg_results if self.filter_genes(x, list(sctab_features.keys()))]
            finetune_genes = {x['query'] : x['ensembl']['gene'] for x in filtered_mg_results}
            zero_padded_genes = list(set(sctab_features) - set(finetune_genes.values()))
            finetune_adata = finetune_adata[:, list(finetune_genes.keys())]
            finetune_adata.var.rename(index=finetune_genes, inplace=True)
        
        return zero_padded_genes, finetune_adata


    def train_test_split(self, data, save_path, dataname):
        finetune_adata = ad.read_h5ad(data)
        np.random.seed(0)
        shuffle_inds =  np.random.permutation(finetune_adata.shape[0])
        print('finetune adata before shuffle', finetune_adata.obs)
        finetune_adata = finetune_adata[shuffle_inds]
        print('finetune adata after shuffle', finetune_adata.obs)

        train_len = int(len(finetune_adata)*0.8)
        test_val_len = int(len(finetune_adata)*0.1)
        finetune_adata_train = finetune_adata[0:train_len, :]
        finetune_adata_train.write_h5ad(save_path+'/'+dataname+'_TRAIN.h5ad')
        print(finetune_adata_train.obs)
        finetune_adata_test = finetune_adata[train_len:train_len+test_val_len, :]
        finetune_adata_test.write_h5ad(save_path+'/'+dataname+'_TEST.h5ad')
        print(finetune_adata_test.obs)
        finetune_adata_val = finetune_adata[train_len+test_val_len:, :]
        finetune_adata_val.write_h5ad(save_path+'/'+dataname+'_VAL.h5ad')
        print(finetune_adata_val.obs)
    
    def get_empty_anndata(self, formatted_h5ad_file, var_file):
        adata = ad.read_h5ad(formatted_h5ad_file)

        var_df = pd.read_csv(var_file, index_col=0)
        var_df.index = var_df.index.map(str)

        empty_adata = adata[0:0, :]
        empty_adata.var = var_df
        empty_adata.var_names = empty_adata.var.feature_name

        return empty_adata
    
    def prep_for_evaluation(self, adata, formatted_h5ad_file, var_file):
        print('ADATA VAR NAMES', adata.var_names)
        empty_adata = self.get_empty_anndata(formatted_h5ad_file, var_file)
        print('sctab var names', empty_adata.var_names)
        return ad.concat([empty_adata, adata], join="outer")[:, empty_adata.var_names]
    
    def get_raw_counts(self, h5ad_file):
        """Processes an AnnData object containing raw counts in adata.raw
        and sets up metadata for evaluation scripts.

        Returns:
            None
        """
        # load adata
        adata = ad.read_h5ad(h5ad_file)

        # set adata.X to raw counts
        adata = adata.raw.to_adata()

        # store a column of ensembl IDs
        adata.var["feature_id"] = adata.var.index

        # make gene symbols the var index
        adata.var_names = adata.var.feature_name

        # Remove the name of the index coloumn
        adata.var.index.name = None

        # create a second cell type column
        adata.obs["celltype"] = adata.obs.cell_type

        # overwrite h5ad file with raw counts in adata.X
        adata.write_h5ad(h5ad_file)
    
    def read_10x_cca(self, data_path, save_data):
        mtx_file = data_path + 'Exp_data_UMIcounts.mtx'
        cells_file = data_path + "Cells.csv"
        genes_file = data_path + 'Genes.txt'
        print(mtx_file)
        matrix = mmread(mtx_file).tocsr().transpose()  # Transpose to make cells rows and genes columns

        # Read the cells file
        cells = pd.read_csv(cells_file)
        # Read the genes file
        genes = pd.read_csv(genes_file, header=None, sep='\t')
        genes.columns = ['gene_symbols']

        # Create an AnnData object
        adata = ad.AnnData(X=matrix)
        adata.obs = cells
        adata.var = genes
        adata.var_names = genes['gene_symbols']

        adata.write_h5ad(save_data)

