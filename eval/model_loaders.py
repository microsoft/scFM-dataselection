import os
from pathlib import Path

import torch
 
import scvi
import anndata as ad
from datetime import datetime

from geneformer.perturber_utils import load_model as load_geneformer



def find_latest_run_and_extract_checkpoint_ssl(ssl_dir, classifier=False):
    # List all items in the parent directory
    if classifier:
        top_dir = ssl_dir + '/trained_models/final_models/classification/SSL_CN_MLP_50pnew_run0_None'
    else:
        top_dir = ssl_dir+'/trained_models/pretext_models/masking/CN_MLP_50p'
        
    runs = [f for f in os.listdir(top_dir+'/sc-SFM-eastus') if (os.path.isdir(os.path.join(top_dir+'/sc-SFM-eastus', f)))]
    parent_dir = top_dir+'/wandb'
    if len(runs) == 1:
        subdir_path = top_dir+'/sc-SFM-eastus/'+runs[0]
        checkpoint_file = Path(subdir_path) / 'checkpoints' / 'last_checkpoint.ckpt'
        return checkpoint_file
    else:
        run_folders = [f for f in os.listdir(parent_dir) if ((os.path.isdir(os.path.join(parent_dir, f))) & (f != 'latest-run'))]
        # Function to parse the timestamp from folder names
        def parse_run(folder):
            date_str = folder.split('-')[1][:8]  # YYYYMMDD
            time_str = folder.split('_')[1][:6]   # HHMMSS
            return datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')

        # Find the latest run folder
        if run_folders:
            latest_run_folder = max(run_folders, key=parse_run)
            parts = latest_run_folder.split('-')
            subdir_name = '-'.join(parts[2:])
            subdir_path = None
            for root, dirs, files in os.walk(top_dir):
                if subdir_name in dirs:
                    subdir_path = os.path.join(root, subdir_name)
                    break

            if not subdir_path:
                return
            
            # Check if 'last_checkpoint.pt' exists in the subdirectory
            checkpoint_file = Path(subdir_path) / 'checkpoints' / 'last_checkpoint.ckpt'
            if checkpoint_file.exists():
                # print(f"Found 'last_checkpoint.ckpt' at: {checkpoint_file}")
                return checkpoint_file  # Optionally return the path or read the file
                
        else:
            return None


def get_ssl_checkpoint_file(downsampling_method, percentage, seed, ssl_directory, classifier=False):
    if downsampling_method == "celltype_reweighted":
        percentage = int(percentage) / 100
        
    directory_prefix = f"{ssl_directory}/{downsampling_method}/idx_{percentage}pct_seed{seed}"
    
    return find_latest_run_and_extract_checkpoint_ssl(directory_prefix, classifier=classifier)
    



def load_ssl_model(ssl_checkpoint_file):
    from self_supervision.models.lightning_modules.cellnet_autoencoder import MLPAutoEncoder

    # fixed model parameters
    gene_dim = 19331
    units_encoder = [512, 512, 256, 256, 64]
    units_decoder = [256, 256, 512, 512]
    batch_size = 8192

    ssl_checkpoint = torch.load(ssl_checkpoint_file, map_location=torch.device('cpu'))

    ssl_model = MLPAutoEncoder(gene_dim=gene_dim,
                          units_encoder=units_encoder,
                          units_decoder=units_decoder,
                          batch_size=batch_size)
    
    ssl_model.load_state_dict(ssl_checkpoint['state_dict'])
    ssl_model.eval()

    return ssl_model

def load_ssl_classifier(ssl_checkpoint_file, type_dim, class_weights, child_matrix, batch_size):
    from self_supervision.models.lightning_modules.cellnet_autoencoder import MLPClassifier
    # fixed model parameters
    gene_dim = 19331
    units_encoder = [512, 512, 256, 256, 64]
    # batch_size = 8192

    ssl_checkpoint = torch.load(ssl_checkpoint_file, map_location=torch.device('cpu'))
    
    # this is to account for the two linear layers in the MLP classifier
    ssl_model = MLPClassifier(gene_dim=gene_dim, type_dim=type_dim, class_weights=class_weights, child_matrix=child_matrix, units=units_encoder, batch_size=batch_size)

    ssl_model.load_state_dict(ssl_checkpoint['state_dict'])
    ssl_model.eval()

    return ssl_model



# scvi needs an adata of the correct format so we require an h5ad file for this function
# this is just an h5ad with the same var matrix as scTab
def load_scvi_model(downsampling_method, percentage, seed, scvi_directory, h5ad_file):
    if downsampling_method == "randomsplits":
        downsampling_method = "random"
    if downsampling_method == "geometric_sketch":
        downsampling_method = "geometric_sketching"
    if downsampling_method == "celltype_reweighted":
        downsampling_method = "cluster_adaptive"
    
    adata = ad.read_h5ad(h5ad_file)
    model_dir = Path(scvi_directory)  / downsampling_method / f"idx_{percentage}pct_seed{seed}" / f"idx_{percentage}pct_seed{seed}_TRAIN"

    if "spikein_10" in downsampling_method:
        model_dir = Path(scvi_directory)  / downsampling_method / f"idx_{percentage}pct_seed{seed}" / f"perturb_spikein10pct_replogle_idx_{percentage}pct_seed{seed}_TRAIN"
    if "spikein_50" in downsampling_method:
        model_dir = Path(scvi_directory)  / downsampling_method / f"idx_{percentage}pct_seed{seed}" / f"perturb_spikein50pct_replogle_idx_{percentage}pct_seed{seed}_TRAIN"

    print(model_dir)
    model = scvi.model.SCVI.load(model_dir, adata=adata)
    return model
 
def load_fine_tuned_scvi_model(downsampling_method, percentage, seed, pretrained_scvi_directory, h5ad_file, finetuned_scvi_directory, dataset_name, cell_type_column, num_classes):
    from scvi_mlp import MLPClassifier
    pretrained_model = load_scvi_model(downsampling_method,
                                       percentage,
                                       seed,
                                       scvi_directory = pretrained_scvi_directory,
                                       h5ad_file = h5ad_file)

    save_path = f"{finetuned_scvi_directory}/scvi_finetuned_model_{downsampling_method}_{percentage}pct_seed{seed}.pt"
    latent_dims = pretrained_model._module_kwargs['n_latent']
    classifier = MLPClassifier(n_classes=num_classes,
                               n_input=latent_dims,
                               cell_type_column=cell_type_column)
    classifier.load_state_dict(torch.load(save_path, weights_only=True))
    classifier.eval()
    return classifier


def load_geneformer_model(downsampling_method, percentage, seed, geneformer_directory):
    if downsampling_method == "random":
        downsampling_method = "randomsplits"
    elif downsampling_method == "geometric_sketch":
        downsampling_method = "geometric_sketching"
    geneformer_directory = Path(geneformer_directory)

    model_dir = geneformer_directory / downsampling_method / f"idx_{percentage}pct_seed{seed}"

    return model_dir


def load_finetuned_geneformer_model(downsampling_method, percentage, seed, dataset_name, geneformer_directory):
    if downsampling_method == "random":
        downsampling_method = "randomsplits"
    elif downsampling_method == "geometric_sketch":
        downsampling_method = "geometric_sketching"
    geneformer_directory = Path(geneformer_directory)

    # todo should we add the checkpoint dir here?
    # randomsplits/pbmc/idx_1pct_seed0/checkpoint-1867
    model_dir = geneformer_directory / downsampling_method / dataset_name / f"idx_{percentage}pct_seed{seed}"

    return model_dir
