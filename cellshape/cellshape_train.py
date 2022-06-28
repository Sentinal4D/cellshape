import torch
from torch.utils.data import DataLoader

import cellshape_cloud as cscloud
import cellshape_voxel as csvoxel
import cellshape_cluster as cscluster
from cellshape_cloud.vendor.chamfer_distance import ChamferDistance
from cellshape_voxel.losses import FocalTverskyLoss


def cellshape_train(params):
    input_dir = params["input_dir"]
    batch_size = params["batch_size"]
    learning_rate_autoencoder = params["learning_rate_autoencoder"]
    num_epochs_autoencoder = params["num_epochs_autoencoder"]
    num_features = params["num_features"]
    k = params["k"]
    encoder_type = params["encoder_type"]
    decoder_type = params["decoder_type"]
    output_dir = params["output_dir"]
    num_clusters = params["num_clusters"]
    num_epochs_clustering = params["num_epochs_clustering"]
    learning_rate_clustering = params["learning_rate_clustering"]
    gamma = params["gamma"]
    alpha = params["alpha"]
    divergence_tolerance = params["divergence_tolerance"]
    update_interval = params["update_interval"]
    model_type = params["model_type"]
    # train_type = params["train_type"]

    if model_type == "cloud":

        autoencoder = cscloud.CloudAutoEncoder(
            num_features=num_features,
            k=k,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
        )

        dataset = cscloud.PointCloudDataset(input_dir)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        reconstruction_criterion = ChamferDistance()

        optimizer = torch.optim.Adam(
            autoencoder.parameters(),
            lr=learning_rate_autoencoder * 16 / batch_size,
            betas=(0.9, 0.999),
            weight_decay=1e-6,
        )

        (
            autoencoder,
            name_logging,
            name_model,
            name_writer,
            name,
        ) = cscloud.train(
            autoencoder,
            dataloader,
            num_epochs_autoencoder,
            reconstruction_criterion,
            optimizer,
            output_dir,
        )

    else:
        autoencoder = csvoxel.VoxelAutoEncoder(
            num_features=num_features,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
        )

        dataset = csvoxel.VoxelDataset(input_dir)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        reconstruction_criterion = FocalTverskyLoss()

        optimizer = torch.optim.Adam(
            autoencoder.parameters(),
            lr=learning_rate_autoencoder * 16 / batch_size,
            betas=(0.9, 0.999),
            weight_decay=1e-6,
        )

        (
            autoencoder,
            name_logging,
            name_model,
            name_writer,
            name,
        ) = csvoxel.train(
            autoencoder,
            dataloader,
            num_epochs_autoencoder,
            reconstruction_criterion,
            optimizer,
            output_dir,
        )

    model = cscluster.DeepEmbeddedClustering(
        autoencoder=autoencoder, num_clusters=num_clusters, alpha=alpha
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )  # it is very important that shuffle=False here!
    dataloader_inf = DataLoader(
        dataset, batch_size=1, shuffle=False
    )  # it is very important that batch_size=1 and shuffle=False here!

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate_clustering * 16 / batch_size,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
    )

    reconstruction_criterion = ChamferDistance()
    cluster_criterion = torch.nn.KLDivLoss(reduction="sum")

    cscluster.train(
        model,
        dataloader,
        dataloader_inf,
        num_epochs_clustering,
        optimizer,
        reconstruction_criterion,
        cluster_criterion,
        update_interval,
        gamma,
        divergence_tolerance,
        output_dir,
    )
