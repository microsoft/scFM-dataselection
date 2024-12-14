import os
import argparse

import anndata as ad

from preprocess_class import Preprocess

parser = argparse.ArgumentParser(description="preprocess the sctab dataset")
parser.add_argument("--datapath", type=str,
    required=True, help="path to h5ad dataset to be preprocessed")
parser.add_argument("--outdir", type=str, required=True, help="Cluster identifier to process.")
parser.add_argument("--seed", type=int, required=True, help="Seed number for normal generation.")
parser.add_argument("--pct", type=str, required=True, help="percent of train data.")
parser.add_argument("--method", type=str, required=True, help="sampling method")
parser.add_argument("--parquet_size", type=float, default=0.01,
    help="pct of total datasize to include in each parquet file")
args = parser.parse_args()


preprocess = Preprocess()
pct_seed_label = "idx_"+str(args.pct)+"pct_seed"+str(args.seed)
obs_dict = {}
out = os.path.join(args.outdir, args.method, pct_seed_label)
os.makedirs(out, exist_ok=True)
outdir = os.path.join(args.outdir, args.method, pct_seed_label, pct_seed_label)

for split in ['TRAIN', 'TEST', 'VAL']:
    raw_data = args.datapath+'/'+pct_seed_label+'_'+split+'.h5ad'
    preprocessed_data = outdir+'_'+split+'_PREPROCESSED.h5ad'
    preprocess.norm(datapath=raw_data, output_file=preprocessed_data)
    obs_dict[split] = ad.read_h5ad(preprocessed_data, backed='r').obs


obs_train, obs_test, obs_val, cols_train = preprocess.convert_categorical_var(
    obs_train=obs_dict["TRAIN"],
    obs_test=obs_dict["TEST"],
    obs_val=obs_dict["VAL"],
    output_dir=out)

obs_dict["TRAIN"] = obs_train
obs_dict["TEST"] = obs_test
obs_dict["VAL"] = obs_val

for split in ["TRAIN", "TEST", "VAL"]:
    preprocessed_data = outdir+'_'+split+'_PREPROCESSED.h5ad'
    parquet_out = os.path.join(args.outdir, args.method, pct_seed_label, split.lower())
    adata = ad.read_h5ad(preprocessed_data, backed='r')
    os.makedirs(parquet_out, exist_ok=True)
    preprocess.convert_to_dask(adata=adata,
                               output_path=parquet_out,
                               split=split,
                               obs=obs_dict[split],
                               cols_train=cols_train,
                               parquet_size=args.parquet_size)
