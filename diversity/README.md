# Downsampling
Below are the instructions for computing and plotting diversity score for the downsampled scTab pre-training datasets.

## Shannon Index and Gini-Simpson Index

These scores are computed by running `diversity.py`

```
python diversity.py /path/to/sctab/downsampled/datasets
```


## Vendi Score

These scores are computed by running `compute_vendi_score.py`

```
python compute_vendi_score.py /path/to/sctab/downsampled/datasets
```

## Plotting

The line plots are generated with `plot_diversity.py`

```
python plot_diversity.py
```
