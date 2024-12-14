# Evaluating the role of pre-training dataset size and diversity on single-cell foundation model performance

This repository contains the code that accompanies our paper, "Evaluating the role of pre-training dataset size and diversity on single-cell foundation model performance". You can find the preprint of the paper here.

# Project Description

In this project, we assess three model architectures pre-trained to perform as foundation models in the context of single-cell RNA-seq: scVI, SSL, and Geneformer. We pre-trained these models on subsets of the scTab corpus using three different downsampling schemes (uniform random downsampling, cell type re-weighting, and geometric sketching) and evaluated these models in (1) the zero-shot regime and (2) when fine-tuned.

Our evaluation uses two main tasks: cell type classification and batch integration. In these tasks, we compare the performance of Geneformer and scGPT against simple baselines and investigate the role of pre-training dataset size and diversity on downstream performance.



# Dependencies

First install `python` package dependencies

```
pip install -r requirements.txt
```

Install our fork of `ssl_in_scg`

```
git clone https://github.com/v-mahughes/ssl_in_scg/tree/6e67bde8c949710099b137858ad6efe013ce18fe

cd ssl_in_scg
pip install -e .
```

Install our fork of `Geneformer`

```
git clone https://github.com/lcrawlab/Geneformer

cd Geneformer
pip install .
```

Install `zero-shot-scfoundation`
```
git clone https://github.com/microsoft/zero-shot-scfoundation

cd sc_foundation_evals
pip install .
```

# Reproducing results

## Downloading the scTab corpus

The instructions for downloading the scTab corpus are in the `data/preprocess` directory.

## Creating pre-training datasets

The instructions for downampling the scTab corpus to generate pre-training datasets are in the `downsampling` directory.

## Pre-training foundation models

The instructions for pre-training all models are in the `train_scripts` directory. Each model architecture has its own directory.

## Fine-tuning foundation models

The instructions for pre-training all models are in the `finetune-scripts` directory. Each model architecture has its own directory.


## Evaluating model performance

The instructions for evaluating all models are in the `eval` directory. There are scripts for both zero-shot and fine-tuned evaluations.

## Reproducing figures

Jupyter notebooks that produce each of the figures (after running all model evaluations) are in the `plotting` directory.

## Questions and Feedback

If you have any questions, or find any issues with the code, please open an issue in this repository. We also welcome any contributions to the code - be sure to checkout the Contributing section below.

If you have questions or concerns with this project and do not want to create an issue, please contact
[Alan DenAdel](mailto:alan_denadel@brown.edu), [Ava Amini](mailto:ava.amini@microsoft.com), or [Lorin Crawford](mailto:lcrawford@microsoft.com). Any feedback on the software, manuscript, and tutorials is appreciated.

## Relevant Citations
A. DenAdel, M. Hughes, A. Thoutam, A. Gupta, A.W. Navia, N. Fusi, S. Raghavan, P.S. Winter, A.P. Amini, and L. Crawford. Evaluating the role of pre-training dataset size and diversity on single-cell foundation model performance. _bioRxiv_.

## License

This project is available under the MIT License.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
