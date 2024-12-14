mkdir -p evaluation_data/kim_lung
mkdir -p evaluation_data/pancreas_scib
mkdir -p evaluation_data/intestine
mkdir -p evaluation_data/hematopoiesis
mkdir -p evaluation_data/placenta
mkdir -p evaluation_data/periodontitis

wget -O evaluation_data/kim_lung/Data_Kim2020_lung.zip https://www.dropbox.com/sh/97paub4mfkcviz1/AADXOe8pWbvlolETToAte3GYa?dl=1
unzip Sevaluation_data/kim_lung/Data_Kim2020_lung.zip

wget -O evaluation_data/pancreas_scib/pancreas_scib.h5ad https://figshare.com/ndownloader/files/24539828

wget -O evaluation_data/hematopoiesis/hematopoiesis.h5ad https://datasets.cellxgene.cziscience.com/d72106bd-d03b-45e6-a0fa-ca2a831ef092.h5ad

wget -O evaluation_data/placenta/placenta.h5ad https://datasets.cellxgene.cziscience.com/50bbe1a2-5f27-47f5-a809-046459a4ae5e.h5ad

wget -O evaluation_data/intestine/intestine_on_chip_IFN.h5ad https://datasets.cellxgene.cziscience.com/4fd4f3a1-8c41-4f73-808b-196766ed942d.h5ad

wget -O evaluation_data/intestine/intestine_on_chip_media.h5ad https://datasets.cellxgene.cziscience.com/610bb418-705a-4493-a1fb-e14b8379af4f.h5ad

wget -O evaluation_data/periodontitis/periodontitis.h5ad https://datasets.cellxgene.cziscience.com/91796603-b12b-4d62-88cf-006957399338.h5ad


