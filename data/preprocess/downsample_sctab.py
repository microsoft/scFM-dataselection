import os
import argparse

import anndata as ad
import numpy as np
import pandas as pd

from downsample_class import RandomDownsample, CelltypeReweighting, GeometricSketching


parser = argparse.ArgumentParser(description="downsample the sctab dataset")
parser.add_argument("--datapath", type=str, required=True,
                    help="path to sctab concatenated h5ad")
parser.add_argument("--outdir", type=str, required=True,
                    help="where to save the downsampled sctab h5ads.")
args = parser.parse_args()

adata = ad.read_h5ad(args.datapath, backed='r')
soma_index = adata.obs.soma_joinid
soma_index = pd.DataFrame(soma_index)

seed = 0
# subsample_fracs = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
subsample_fracs = [0.01] #change THIS BACK

random_downsample = RandomDownsample()
celltype_reweighting = CelltypeReweighting()
geometric_sketching = GeometricSketching()

for frac in subsample_fracs:
    for i in np.arange(1): #CHNAGE THIS BACK
        # random subsample
        frac_str = str(int(frac*100))
        rand_output_path = os.path.join(
            args.outdir, 'random', f'idx_{frac_str}pct_seed{seed}')
        os.makedirs(rand_output_path, exist_ok=True)
        # random_downsample.generate_indices(soma_index=soma_index,
        #     seed=seed,
        #     frac=frac,
        #     output_path=os.path.join(rand_output_path, f'idx_{frac_str}pct_seed{seed}'))

        # #celltype reweighted subsample
        # cr_output_path = os.path.join(args.outdir,
        #                                'celltype_reweighting',
        #                                f'idx_{frac_str}pct_seed{seed}',
        #                                f'idx_{frac_str}pct_seed{seed}')
        # print(cr_output_path)
        # os.makedirs(cr_output_path, exist_ok=True)
        # celltype_reweighting.generate_indices(adata=adata,
        #                                       sampling_percent=frac,
        #                                       seed=seed,
        #                                       ind_output_path=os.path.join(cr_output_path,
        #                                                                    f'idx_{frac_str}pct_seed{seed}'))

        #geometric_sketching
        gs_output_path = os.path.join(args.outdir,
                                       'geometric_sketching',
                                       f'idx_{frac_str}pct_seed{seed}')
        os.makedirs(gs_output_path, exist_ok=True)
        geometric_sketching.generate_indices(adata, frac, seed, gs_output_path, n_comps=50)

        seed += 1
