# Evaluating the role of pre-training dataset size and diversity on single-cell foundation model performance

This repository contains the code that accompanies our paper, "Evaluating the role of pre-training dataset size and diversity on single-cell foundation model performance". You can find the preprint of the paper [here](https://www.biorxiv.org/content/10.1101/2024.12.13.628448v1).

# Project Description

In this project, we assess three model architectures pre-trained to perform as foundation models in the context of single-cell RNA-seq: scVI, SSL, and Geneformer. We pre-trained these models on subsets of the scTab corpus using three different downsampling schemes (uniform random downsampling, cell type re-weighting, and geometric sketching) and evaluated these models in (1) the zero-shot regime and (2) when fine-tuned.

Our evaluation uses two main tasks: cell type classification and batch integration. In these tasks, we compare the performance of scVI, SSL, and Geneformer against simple baselines and investigate the role of pre-training dataset size and diversity on downstream performance.

![Fig. 1: Strategy to assess the effects of pre-training dataset size and diversity on scFM performance. (A) Schematic of the downsampling approaches, sizes of downsampled pre-training datasets, and data splitting strategy. (B) An example of what evaluation performance might \textit{a priori} be expected to look like as a function of pre-training dataset size and diversity.](images/fig1.png)


# Dependencies

First install `python` package dependencies (this can take 15+ minutes)

```
pip install -r requirements.txt
```

Install our fork of `ssl_in_scg` (this should only take about 2 minutes)

```
git clone https://github.com/v-mahughes/ssl_in_scg
cd ssl_in_scg
git fetch
git switch early-stopping
pip install -e .
```

Install our fork of `Geneformer` (this should only take 20 seconds)

```
git clone https://github.com/lcrawlab/Geneformer

cd Geneformer
pip install .
```

Install `zero-shot-scfoundation` (this should only take 10 seconds)
```
git clone https://github.com/microsoft/zero-shot-scfoundation

cd zero-shot-scfoundation
pip install .
```

# Reproducing results

![Fig. 3a: Schematic of analysis to find the learning saturation point. For each family of models (i.e., a downsampling strategy paired with a model) a saturation threshold of 95 percent of the maximum performance was computed, and the minimum pre-training dataset size that produced a model surpassing that threshold was identified. This dataset size was denoted the learning saturation point and is considered the point at which model performance saturated as a function of pre-training dataset size.](images/fig3a.png)


## Downloading the scTab corpus

The instructions for downloading the scTab corpus are in the `data/preprocess` directory.

## Creating pre-training datasets

The instructions for downampling the scTab corpus to generate pre-training datasets are in the `downsampling` directory.

## Pre-training foundation models

The instructions for pre-training all models are in the `pretraining` directory. Each model architecture has its own directory.

## Fine-tuning foundation models

The instructions for pre-training all models are in the `finetuning` directory. Each model architecture has its own directory.


## Evaluating model performance

The instructions for evaluating all models are in the `eval` directory. There are scripts for both zero-shot and fine-tuned evaluations.

## Reproducing figures

Jupyter notebooks that produce each of the figures (after running all model evaluations) are in the `plotting` directory.

## Questions and Feedback

If you have any questions, or find any issues with the code, please open an issue in this repository. We also welcome any contributions to the code - be sure to checkout the Contributing section below.

If you have questions or concerns with this project and do not want to create an issue, please contact
[Alan DenAdel](mailto:alan_denadel@brown.edu), [Ava Amini](mailto:ava.amini@microsoft.com), or [Lorin Crawford](mailto:lcrawford@microsoft.com). Any feedback on the software, manuscript, and tutorials is appreciated.

## Relevant Citation (BibTeX)

```
@article {DenAdel2024.12.13.628448,
	author = {DenAdel, Alan and Hughes, Madeline and Thoutam, Akshaya and Gupta, Anay and Navia, Andrew W. and Fusi, Nicolo and Raghavan, Srivatsan and Winter, Peter S. and Amini, Ava P. and Crawford, Lorin},
	title = {Evaluating the role of pre-training dataset size and diversity on single-cell foundation model performance},
	elocation-id = {2024.12.13.628448},
	year = {2024},
	doi = {10.1101/2024.12.13.628448},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/12/17/2024.12.13.628448},
	eprint = {https://www.biorxiv.org/content/early/2024/12/17/2024.12.13.628448.full.pdf},
	journal = {bioRxiv}
}
```

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
