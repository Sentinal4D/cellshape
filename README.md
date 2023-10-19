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
[![Coverage Status](https://coveralls.io/repos/github/Sentinal4D/cellshape/badge.svg?branch=main)](https://coveralls.io/github/Sentinal4D/cellshape?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="https://github.com/Sentinal4D/cellshape/blob/main/img/cellshape.png" 
     alt="Cellshape logo by Matt De Vries">

# 3D single-cell shape analysis of cancer cells using geometric deep learning


This is a Python package for 3D cell shape features and classes using deep learning. Please refer to our preprint [here](https://www.biorxiv.org/content/10.1101/2022.06.17.496550v1).

cellshape is the main package which imports from sub-packages:
- [cellshape-helper](https://github.com/Sentinal4D/cellshape-helper): Facilitates point cloud generation from 3D binary masks.
- [cellshape-cloud](https://github.com/Sentinal4D/cellshape-cloud): Implementations of graph-based autoencoders for shape representation learning on point cloud input data.
- [cellshape-voxel](https://github.com/Sentinal4D/cellshape-voxel): Implementations of 3D convolutional autoencoders for shape representation learning on voxel input data.
- [cellshape-cluster](https://github.com/Sentinal4D/cellshape-cluster): Implementation of deep embedded clustering to add to autoencoder models.

## Installation and requirements
### Dependencies
The software requires Python 3.7 or greater. The following are package dependencies that are installed automatically when cellshape is installed: [`PyTorch`](https://pytorch.org/), [`pyntcloud`](https://github.com/daavoo/pyntcloud), [`numpy`](https://numpy.org/), [`scikit-learn`](https://scikit-learn.org/stable/), `tensorboard`, [`tqdm`](https://github.com/tqdm/tqdm) (The full list is shown in the [setup.py](https://github.com/Sentinal4D/cellshape/blob/main/setup.py) file). This repo makes extensive use of [`cellshape-cloud`](https://github.com/Sentinal4D/cellshape-cloud), [`cellshape-cluster`](https://github.com/Sentinal4D/cellshape-cluster), [`cellshape-helper`](https://github.com/Sentinal4D/cellshape-helper), and [`cellshape-voxel`](https://github.com/Sentinal4D/cellshape-voxel). To reproduce our results in our paper, only [`cellshape-cloud`](https://github.com/Sentinal4D/cellshape-cloud), [`cellshape-cluster`](https://github.com/Sentinal4D/cellshape-cluster) are needed.

### To install
1. We recommend creating a new conda environment. In the terminal, run:
```bash 
conda create --name cellshape-env python=3.8 -y
conda activate cellshape-env
pip install --upgrade pip
```
2. Install cellshape from pip. In the same terminal, run:
```bash
pip install cellshape
```
This should take ~5mins or less.

### Hardware requirements
We have tested this software on an Ubuntu 20.04LTS and 18.04LTS with 128Gb RAM and NVIDIA Quadro RTX 6000 GPU.

## Data availability and structure
### Data availability
Update (19/10/2023): Our sample data was originally published on Zenodo Sandbox, however, there are currently issues with this website and the link to the data is broken. We are working to put the data on a public data store and will update this page when this is done.

Old:
Datasets to reproduce our results in our paper are available [here](https://sandbox.zenodo.org/record/1080300#.YsX7f3XMIaz). 
- SamplePointCloudData.zip contains a sample dataset of a point cloud of cells in order to test our code.
- FullData.zip contains 3 plates of point cloud representations of cells for several treatments. This data can be used to reproduce our results.
- Output.zip contains trained model weights and deep learning cell geometric features extracted using these trained models.
- BinaryCellMasks.zip contains a sample set of binary masks of cells which can be used as input to [`cellshape-helper`](https://github.com/Sentinal4D/cellshape-helper) to test our point cloud generation code. 

### Data structure
We suggest testing our code on the data contained in `SamplePointCloudData.zip`. This data is structured in the following way:

```
cellshapeSamplePointCloudDatset/
    small_data.csv
    Plate1/
        stacked_pointcloud/
            Binimetinib/
                0010_0120_accelerator_20210315_bakal01_erk_main_21-03-15_12-37-27.ply
                ...
            Blebbistatin/
            ...
    Plate2/
        stacked_pointcloud/
    Plate3/
        stacked_pointcloud/
```
This data structure is only necessary if wanting to use our data. If you would like to use your own dataset, you may structure it in any way as long as the extension of the point clouds are `.ply`. If using your own data structure, please define the parameter `--dataset_type` as `"Other"`.


## Usage
The following steps assume that one already has point cloud representations of cells or nuclei. If you need to generate point clouds from 3D binary masks, please go to [`cellshape-helper`](https://github.com/Sentinal4D/cellshape-helper).

### Downloading the dataset
We suggest testing our code on the data contained in `SamplePointCloudData.zip`. Please download the data and unzip the contents into a directory of your choice. We recommend doing this in your `~Documents/` folder. This is used as parameters in the steps below, so please remember where you download the data to. Downloading and unzipping the data can be done in the terminal. You might need to first install `wget` and `unzip` with `apt-get` (e.g. `apt-get install wget`).
1. Download the data into the `~/Documents/` folder with wget
```bash
cd ~/Documents
wget https://sandbox.zenodo.org/record/1080300/files/SamplePointCloudDataset.zip
```
2. Unzip the data with unzip:
```bash 
unzip SamplePointCloudDataset.zip
```

This will create a directory called `cellshapeSamplePointCloudDatset` under your `~Documents/` folder, i.e. `/home/USER/Documents/cellshapeSamplePointCloudDatset/` (`USER` will be different for you).

### Training
The training procedure follows two steps:
1. Training the dynamic graph convolutional foldingnet (DFN) autoencoder to automatically learn shape features.
2. Adding the clustering layer to refine shape features and learn shape classes simultaneously.

Inference can be done after each step. 

Our training functions are run through a command line interface with the command ```cellshape-train```.
For help on all command line options, run the following in the terminal:
```bash
cellshape-train -h
```
#### 1. Train DFN autoencoder
The first step trains the autoencoder without the additional clustering layer. Run the following in the terminal. Remember to change the `--cloud_dataset_path`, `--dataframe_path`, and `--output_dir` parmaeters to be specific to your directories, if you have saved the data somewhere else. To test the code, we train for 5 epochs. First make sure you're in the directory where you downloaded the data to. If this is your `~/Documents/ folder, go into this:
```bash
cd ~/Documents
```
Then run the following:

```bash
cellshape-train \
--model_type "cloud" \
--pretrain "True" \
--train_type "pretrain" \
--cloud_dataset_path "./cellshapeSamplePointCloudDataset/" \
--dataset_type "SingleCell" \
--dataframe_path "./cellshapeSamplePointCloudDataset/small_data.csv" \
--output_dir "./cellshapeOutput/" \
--num_epochs_autoencoder 5 \
--encoder_type "dgcnn" \
--decoder_type "foldingnetbasic" \
--num_features 128 \
```

This step will create an output directory `/home/USER/Documents/cellshapeOutput/` with the subfolders: `nets`, `reports`, and `runs` which contain the model weights, logged outputs, and tensorboard runs, respectively, for each experiment. Each experiment is named with the following convention `{encoder_type}_{decoder_type}_{num_features}_{train_type}_{xxx}`, where {xxx} is a counter. For example, if this was the first experiment you have run, the trained model weights will be saved to: `/home/USER/Documents/cellshapeOutput/nets/dgcnn_foldingnetbasic_128_pretrained_001.pt`. This path will be used in the next step for the `--pretrained-path` parameter.


#### 2. Add clustering layer to refine shape features and learn shape classes simultaneously
The next step is to add the clustering layer to refine the model weights. As before, run the following in the terminal. Remember to change the `--cloud_dataset_path`, `--dataframe_path`, `--output_dir`, and `--pretrained-path` parmaeters to be specific to your directories. If you have followed the previous steps, then you will still be in the `~Documents/` path. In the same terminal, run:

```bash
cellshape-train \
--model_type "cloud" \
--train_type "DEC" \
--pretrain False \
--cloud_dataset_path "./cellshapeSamplePointCloudDataset/" \
--dataset_type "SingleCell" \
--dataframe_path "./cellshapeSamplePointCloudDataset/small_data.csv" \
--output_dir "./cellshapeOutput/" \
--num_features 128 \
--num_clusters 5 \
--pretrained_path "./cellshapeOutput/nets/dgcnn_foldingnetbasic_128_pretrained_001.pt" \
```

To monitor the training using [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html), in a new terminal run:
```bash
pip install tensorboard
cd ~/Documents
tensorboard --logdir "./cellshapeOutput/runs/"
```

#### Alternatively, the training steps can be run sequentially through one command line
This would be to state that you would like to `pretrain` and that you want to train `DEC`. 
```bash
cellshape-train \
--model_type "cloud" \
--train_type "DEC" \
--pretrain True \
--cloud_dataset_path "./cellshapeSamplePointCloudDataset/" \
--dataset_type "SingleCell" \
--dataframe_path "./cellshapeSamplePointCloudDataset/small_data.csv" \
--output_dir "./cellshapeOutput/" \
--num_features 128 \
--num_clusters 5 \
```


### Inference
Example inference notebooks can be found in the `docs/notebooks/` folder.

## Issues
If you have any problems, please raise an issue [here](https://github.com/Sentinal4D/cellshape/issues)

## Citation
```bibtex
@article{DeVries2022single,
	author = {Matt De Vries and Lucas Dent and Nathan Curry and Leo Rowe-Brown and Vicky Bousgouni and Adam Tyson and Christopher Dunsby and Chris Bakal},
	title = {3D single-cell shape analysis using geometric deep learning},
	elocation-id = {2022.06.17.496550},
	year = {2023},
	doi = {10.1101/2022.06.17.496550},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/03/27/2022.06.17.496550},
	eprint = {https://www.biorxiv.org/content/early/2023/03/27/2022.06.17.496550.full.pdf},
	journal = {bioRxiv}
}
```

## References
[1] An Tao, 'Unsupervised Point Cloud Reconstruction for Classific Feature Learning', [GitHub Repo](https://github.com/AnTao97/UnsupervisedPointCloudReconstruction), 2020
