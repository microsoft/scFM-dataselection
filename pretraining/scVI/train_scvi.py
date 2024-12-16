"""train_scvi.py contains a set of functions for pre-training scVI on scTab."""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import scvi
import anndata as ad


def train_scvi(h5ad_file,
               val_h5ad_file,
               var_file,
               output_directory,
               prefix="",
               n_layers=1,
               n_latent=10,
               n_hidden=128,
               epochs=50):
    """Trains an scVI model.

    Args:
        h5ad_file: The h5ad file containing the training data.
        val_h5ad_file: The h5ad file containing the validation data.
        var_file: A file containing the scTab variable names.
        output_directory: The directory to write the trained model.
        prefix=The model name prefix.
        n_layers: The number of layers in both the encoder and decoder.
        n_latent: The size of the autoencoder latent space.
        n_hidden: The number of hidden neurons.
        epochs: The number of training epochs.

    Returns:
        The AnnData after being processed.
    """
    scvi.settings.num_threads = 100

    print("Loading train h5ad file")
    # train_adata = ad.read_h5ad(h5ad_file, backed="r+")
    train_adata = ad.read_h5ad(h5ad_file)

    print("Loading val h5ad file")
    # val_adata = ad.read_h5ad(val_h5ad_file, backed="r+")
    val_adata = ad.read_h5ad(val_h5ad_file)

    print("Aligning gene names")
    var_df = pd.read_csv(var_file, index_col=0)
    var_df.index = var_df.index.map(str)

    val_adata.var = var_df
    val_adata.var_names = val_adata.var.feature_name

    train_adata.var = var_df
    train_adata.var_names = train_adata.var.feature_name

    print("Concatenating train and val h5ad files")
    adata = ad.concat([val_adata, train_adata])

    num_train = train_adata.n_obs
    num_val = val_adata.n_obs

    total_obs = num_train + num_val

    validation_size = num_val / total_obs
    print("validation_size:")
    print(validation_size)

    print("Setting up anndata")
    # , layer="counts", batch_key="batch")
    scvi.model.SCVI.setup_anndata(adata)
    print("scvi constructor")
    model = scvi.model.SCVI(adata, n_layers=n_layers,
                            n_latent=n_latent, n_hidden=n_hidden, gene_likelihood="nb")

    print("Training scVI")
    # model.train(max_epochs=epochs, train_size=1.0, batch_size=1024)
    # use shuffle_set_split=False so that val dataset is used as val data
    # order is val, train, test as shown here:
    # https://github.com/scverse/scvi-tools/blob/d094c9b3c14e8cb3ac3a309b9cf0160aff237393/scvi/dataloaders/_data_splitting.py#136
    model.train(max_epochs=epochs, train_size=1.0 - validation_size,
                batch_size=1024, shuffle_set_split=False, early_stopping=True)

    print("Saving model")
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    model.save(Path(output_directory) / prefix)

    print("Saving losses")
    plt.plot(model.history['reconstruction_loss_train']
             ['reconstruction_loss_train'], label='train')
    reconstruction_loss_file = Path(
        output_directory) / (prefix + "_reconstruction_loss.png")
    plt.savefig(reconstruction_loss_file)


def main():
    """Sets up command line arguments and trains an scVI model.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description='Train an scVI model with specified    .')
    parser.add_argument('-a', '--h5ad')
    parser.add_argument('-v', '--val_h5ad')
    parser.add_argument('--var_file')

    parser.add_argument('-o', '--output')
    parser.add_argument('--n_latent', type=int, default=10)
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--pct', type=int)

    args = parser.parse_args()

    h5ad_file = args.h5ad          # merged.h5ad"
    val_h5ad_file = args.val_h5ad
    var_file = args.var_file       # adata_var.csv
    output_directory = args.output
    n_layers = args.n_layers       # scVI default is 1
    n_latent = args.n_latent       # scVI default is 10
    n_hidden = args.n_hidden       # scVI default is 128
    pct = args.pct

    prefix = Path(h5ad_file).stem

    print("Calling train_scvi with args:")
    print("h5ad_file:", h5ad_file)
    print("val_h5ad_file:", val_h5ad_file)
    print("output_directory:", output_directory)
    print("prefix:", prefix)
    print("n_layers:", n_layers)
    print("n_latent:", n_latent)
    print("n_hidden:", n_hidden)

    full_dataset_epochs = 40
    num_epochs = int(full_dataset_epochs * (100 / pct))

    print("pct:", pct)
    print("num_epochs:", num_epochs)

    train_scvi(h5ad_file, val_h5ad_file, var_file, output_directory, prefix=prefix,
               n_layers=n_layers, n_latent=n_latent, n_hidden=n_hidden, epochs=num_epochs)


if __name__ == "__main__":
    main()
