from preprocess_class import Preprocess
import argparse
import anndata as ad
import os
import numpy as np
import scanpy as sc
import pandas as pd
import os

parser = argparse.ArgumentParser(description="preprocess the sctab dataset")
parser.add_argument("--datapath", type=str, required=True, help="path to h5ad dataset to be preprocessed. If processing the intestine data, please point to the top directory containing both intestine datasets")
parser.add_argument("--dataname", type=str, required=True, help="dataname")
parser.add_argument("--parquet_size", type=float, default=0.01, help="pct of total datasize to include in each parquet file")
parser.add_argument("--sctab_varfile", type=str, default=None, help="sctab var file")
parser.add_argument("--sctab_format", type=str, default=None, help="sctab format h5ad")
parser.add_argument("--var_type", type=str, default='ensembl', help="gene format. 'ensembl' or other")
parser.add_argument("--zero_pad", type=str, default="True", help="if true, will zero pad the data")
parser.add_argument("--celltype_col", type=str, default="cell_type", help="cell type column")
args = parser.parse_args()

preprocess = Preprocess()
obs_dict = {}
os.makedirs(args.datapath, exist_ok=True)
raw_full_data = args.datapath+"/"+args.dataname+".h5ad"

if args.dataname == 'intestine':
    data1 = args.datapath+'/'+'intestine_on_chip_IFN.h5ad'
    data2 = args.datapath+'/'+'intestine_on_chip_media.h5ad'
    preprocess.get_raw_counts(data1)
    preprocess.get_raw_counts(data2)
    data1 = ad.read_h5ad(data1)
    data2 = ad.read_h5ad(data2)
    intestine = ad.concat([data1, data2])
    intestine.var = data1.var 
    intestine.write_h5ad(raw_full_data)
elif args.dataname == 'kim_lung':
    preprocess.read_10x_cca(args.datapath, raw_full_data)
elif args.dataname == 'pancreas_scib':
    pancreas = ad.read_h5ad(raw_full_data)
    pancreas.X = pancreas.layers['counts']
    pancreas.write_h5ad(raw_full_data)
else:
    preprocess.get_raw_counts(raw_full_data)
    
preprocess.train_test_split(data=raw_full_data, save_path=args.datapath, dataname=args.dataname)

for split in ["TEST", "VAL", "TRAIN"]:
    raw_data = args.datapath+"/"+args.dataname+"_"+split+".h5ad"
    preprocessed_data = args.datapath+'/'+args.dataname+"_"+split+"_PREPROCESSED.h5ad"
    
    preprocess.norm(datapath=raw_data, output_file=preprocessed_data, finetune_data=True)
    
    if args.zero_pad == 'True':
        zero_padded_savepath = args.datapath+"/"+args.dataname+'_ZERO_PADDED_'+split+'_PREPROCESSED.h5ad'
        zero_padded = preprocess.prep_for_evaluation(ad.read_h5ad(preprocessed_data), args.sctab_format, args.sctab_varfile)
        zero_padded.write_h5ad(zero_padded_savepath)

    obs_dict[split] = ad.read_h5ad(preprocessed_data, backed='r').obs

obs_train, obs_test, obs_val, cols_train = preprocess.convert_categorical_var(obs_train=obs_dict["TRAIN"], obs_test=obs_dict["TEST"], obs_val=obs_dict["VAL"], output_dir=args.datapath)
obs_dict["TRAIN"] = obs_train
obs_dict["TEST"] = obs_test
obs_dict["VAL"] = obs_val

for split in ["TRAIN", "TEST", "VAL"]:
    os.makedirs(args.datapath+'/'+split.lower(), exist_ok=True)
    if args.zero_pad == 'True':
        preprocessed_data = args.datapath+"/"+args.dataname+'_ZERO_PADDED_'+split+'_PREPROCESSED.h5ad'
    else: 
        preprocessed_data = args.datapath+'/'+args.dataname+"_"+split+"_PREPROCESSED.h5ad"
    if args.celltype_col != "cell_type":
            obs_dict[split]["cell_type"] = obs_dict[split][args.celltype_col]
    obs_split = preprocess.fill_empty_cols(obs=obs_dict[split])
    adata = ad.read_h5ad(preprocessed_data, backed='r')
    var = adata.var
    var.to_parquet(path=args.datapath+'/var.parquet', engine='pyarrow', compression='snappy', index=None)
    preprocess.convert_to_dask(adata=adata, output_path=args.datapath+'/'+split.lower(), split=split, obs=obs_split, cols_train=cols_train, parquet_size=args.parquet_size)