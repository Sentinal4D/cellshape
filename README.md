README
================

[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)


<img src="https://github.com/Sentinal4D/cellshape/blob/main/img/cellshape.png" 
     alt="Cellshape logo by Matt De Vries">

# 3D single-cell shape analysis of cancer cells using geometric deep learning


This is a package for **automatically learning** and **clustering** cell
shapes from 3D images. Please refer to our preprint on bioRxiv [here](https://www.biorxiv.org/content/10.1101/2022.06.17.496550v1)

**cellshape** is available for everyone.

## Graph neural network
<https://github.com/Sentinal4D/cellshape-cloud> Cellshape-cloud is an
easy-to-use tool to analyse the shapes of cells using deep learning and,
in particular, graph-neural networks. The tool provides the ability to
train popular graph-based autoencoders on point cloud data of 2D and 3D
single cell masks as well as providing pre-trained networks for
inference.

## Clustering
<https://github.com/Sentinal4D/cellshape-cluster>

Cellshape-cluster is an easy-to-use tool to analyse the cluster cells by
their shape using deep learning and, in particular,
deep-embedded-clustering. The tool provides the ability to train popular
graph-based or convolutional autoencoders on point cloud or voxel data
of 3D single cell masks as well as providing pre-trained networks for
inference.

<https://github.com/Sentinal4D/cellshape-voxel>

## Convolutional neural network
Cellshape-voxel is an easy-to-use tool to analyse the shapes of cells
using deep learning and, in particular, 3D convolutional neural
networks. The tool provides the ability to train 3D convolutional
autoencoders on 3D single cell masks as well as providing pre-trained
networks for inference.  

## Point cloud generation
<https://github.com/Sentinal4D/cellshape-helper>

<figure>
<img src="img/github_cellshapes.png" style="width:100.0%" alt="Fig 1: cellshape workflow" />
</figure>

## For developers
* Fork the repository
* Clone your fork
```bash
git clone https://github.com/USERNAME/cellshape-cloud 
```
* Install an editable version (`-e`) with the development requirements (`dev`)
```bash
cd cellshape-cloud
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
