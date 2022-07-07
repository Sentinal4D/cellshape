[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Python Version](https://img.shields.io/pypi/pyversions/cellshape.svg)](https://pypi.org/project/cellshape)
[![PyPI](https://img.shields.io/pypi/v/cellshape.svg)](https://pypi.org/project/cellshape)
[![Downloads](https://pepy.tech/badge/cellshape)](https://pepy.tech/project/cellshape)
[![Wheel](https://img.shields.io/pypi/wheel/cellshape.svg)](https://pypi.org/project/cellshape)
[![Development Status](https://img.shields.io/pypi/status/cellshape.svg)](https://github.com/Sentinal4D/cellshape)
[![Tests](https://img.shields.io/github/workflow/status/Sentinal4D/cellshape/tests)](
    https://github.com/Sentinal4D/cellshape/actions)
[![Coverage Status](https://coveralls.io/repos/github/Sentinal4D/cellshape/badge.svg?branch=master)](https://coveralls.io/github/Sentinal4D/cellshape?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="https://github.com/Sentinal4D/cellshape/blob/main/img/cellshape.png" 
     alt="Cellshape logo by Matt De Vries">

# 3D single-cell shape analysis of cancer cells using geometric deep learning


This is a Python package for 3D cell shape features and classes using deep learning. Please refer to our preprint on bioRxiv [here](https://www.biorxiv.org/content/10.1101/2022.06.17.496550v1).

cellshape is the main package which imports from sub-packages:
- [cellshape-helper](https://github.com/Sentinal4D/cellshape-helper): Facilitates point cloud generation from 3D binary masks.
- [cellshape-cloud](https://github.com/Sentinal4D/cellshape-cloud): Implementations of graph-based autoencoders for shape representation learning on point cloud input data.
- [cellshape-voxel](https://github.com/Sentinal4D/cellshape-voxel): Implementations of 3D convolutional autoencoders for shape representation learning on voxel input data.
- [cellshape-cluster](https://github.com/Sentinal4D/cellshape-cluster): Implementation of deep embedded clustering to add to autoencoder models.

## Installation and requirements
### Dependencies
The software requires Python 3.7 or greater, `PyTorch`, `torchvision`, `pyntcloud`, `numpy`, `scikit-learn`, `tensorboard`, `tqdm`, `datetime`. This repo makes extensive use of [`cellshape-cloud`](https://github.com/Sentinal4D/cellshape-cloud), [`cellshape-cluster`](https://github.com/Sentinal4D/cellshape-cluster), [`cellshape-helper`](https://github.com/Sentinal4D/cellshape-helper), and [`cellshape-voxel`](https://github.com/Sentinal4D/cellshape-voxel). to reproduce our results in our paper, only [`cellshape-cloud`](https://github.com/Sentinal4D/cellshape-cloud), [`cellshape-cluster`](https://github.com/Sentinal4D/cellshape-cluster) are needed.

### To install
1. We recommend creating a new conda environment:
```bash 
conda create --name cellshape-env python=3.8
conda activate cellshape-env
pip install --upgrade pip
```
2. Install cellshape from pip
```bash
pip install cellshape
```

### Hardware requirements
We have tested this software of an Ubuntu 20.04LTS with 128Gb RAM and NVIDIA Quadro RTX 6000 GPU.

## Data structure

Our data is structured in the following way:

```
cellshapeData/
    all_data_stats.csv
    Plate1/
        stacked_pointcloud/
            Binimetinib/
                0010_0001_accelerator_20210315_bakal01_erk_main_21-03-15_12-37-27.ply
                ...
            Blebbistatin/
            ...
    Plate2/
        stacked_pointcloud/
    Plate3/
        stacked_pointcloud/
```
### Data availability
Datasets to reproduce our results in our paper are available [here](https://sandbox.zenodo.org/record/1080300#.YsX7f3XMIaz).

## Usage
The following steps assume that one already has point cloud representations of cells or nuclei. If you need to generate point clouds from 3D binary masks please go to [`cellshape-helper`](https://github.com/Sentinal4D/cellshape-helper).

The training procedure follows two steps:
1. Training the dynamic graph convolutional foldingnet (DFN) autoencoder to automatically learn shape features.
2. Adding the clustering layer to refine shape features and learn shape classes simultaneously.

Inference can be done after each step. 

For help on all command line options run:
```bash
cellshape-train -h
```
### 1. Train DFN autoencoder
```bash
cellshape-train \
--model_type "cloud" \
--train_type "pretrain" \
--cloud_dataset_path "path/to/cellshapeData/" \
--dataset_type "SingleCell" \
--dataframe_path "path/to/cellshapeData/all_data_stats.csv" \
--output_dir "path/to/output/"
--num_epochs_autoencoder 250 \
--encoder_type "dgcnn" \
--decoder_type "foldingnetbasic"
--num_features 128 \
```

This step will create an output directory `"path/to/output/"` with the subfolders: `nets`, `reports`, and `runs` which contain the model weights, logged outputs, and tensorboard runs respectively for each experiment. Each experiment is named with the following convention {encoder_type}_{decoder_type}_{num_features}_{train_type}_{xxx}, where {xxx} is a counter. For example, if this was the first experiment you have run, the trained model weights will be saved to: `path/to/output/nets/dgcnn_foldingnetbasic_128_pretrain_001.pt`.

To monitor the training using Tensorboard, run:
```bash
tensorboard --logdir "path/to/output/runs/"
```

### 2. Add clustering layer to refine shape features and learn shape classes simultaneously
```bash
cellshape-train \
--model_type "cloud" \
--train_type "DEC" \
--pretrain False \ # this was done in the previous step
--cloud_dataset_path "path/to/cellshapeData/" \
--dataset_type "SingleCell" \
--dataframe_path "path/to/cellshapeData/all_data_stats.csv" \
--output_dir "path/to/output/"
--num_epochs_clustering 250 \
--num_features 128 \
--num_clusters 5 \
--pretrained_path "path/to/output/nets/pretrained_autoencoder.pt" # path/to/output/nets/dgcnn_foldingnetbasic_128_pretrain_001.pt in our example
```

## For developers
* Fork the repository
* Clone your fork
```bash
git clone https://github.com/USERNAME/cellshape
```
* Install an editable version (`-e`) with the development requirements (`dev`)
```bash
cd cellshape
pip install -e .[dev] 
```
* To install pre-commit hooks to ensure formatting is correct:
```bash
pre-commit install
```

* To release a new version:

Firstly, update the version with bump2version (`bump2version patch`, 
`bump2version minor` or `bump2version major`). This will increment the 
package version (to a release candidate - e.g. `0.0.1rc0`) and tag the 
commit. Push this tag to GitHub to run the deployment workflow:

```bash
git push --follow-tags
```

Once the release candidate has been tested, the release version can be created with:

```bash
bump2version release
```

## References
[1] An Tao, 'Unsupervised Point Cloud Reconstruction for Classific Feature Learning', [GitHub Repo](https://github.com/AnTao97/UnsupervisedPointCloudReconstruction), 2020
