import torch
from torch.utils.data import DataLoader
import argparse

import cellshape_cloud as cscloud
import cellshape_voxel as csvoxel
import cellshape_cluster as cscluster
from cellshape_cloud.vendor.chamfer_distance import ChamferLoss
from cellshape_voxel.losses import FocalTverskyLoss
from cellshape_train import cellshape_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cellshape")
    parser.add_argument(
        "--model_type",
        default="cloud",
        type=str,
        choices=["cloud", "voxel"],
        help="Please provide the type of model: [cloud, voxel]",
    )
    parser.add_argument(
        "--cloud_convert",
        default=False,
        type=bool,
        help="Do you need to convert 3D images to point clouds?",
    )
    parser.add_argument(
        "--train_type",
        default="full",
        type=str,
        choices=["pretrain", "full"],
        help="Please provide the type of training mode: [pretrain, full]",
    )
    parser.add_argument(
        "--pretrain",
        default=True,
        type=bool,
        help="Please provide whether or not to pretrain " "the autoencoder",
    )
    parser.add_argument(
        "--dataset_path",
        default="./datasets/",
        type=str,
        help="Please provide the path to the "
        "dataset of 3D images or point clouds",
    )
    parser.add_argument(
        "--dataframe_path",
        default="./dataframe/",
        type=str,
        help="Please provide the path to the dataframe "
        "containing information on the dataset.",
    )
    parser.add_argument(
        "--output_path",
        default="./",
        type=str,
        help="Please provide the path for where to save output.",
    )
    parser.add_argument(
        "--num_epochs_autoencoder",
        default=250,
        type=int,
        help="Provide the number of epochs for the " "autoencoder training.",
    )
    parser.add_argument(
        "--num_epochs_clustering",
        default=250,
        type=int,
        help="Provide the number of epochs for the " "autoencoder training.",
    )
    parser.add_argument(
        "--num_features",
        default=128,
        type=int,
        help="Please provide the number of " "features to extract.",
    )
    parser.add_argument(
        "--num_clusters",
        default=3,
        type=int,
        help="Please provide the number of clusters to find.",
    )
    parser.add_argument(
        "--k", default=20, type=int, help="Please provide the value for k."
    )
    parser.add_argument(
        "--encoder_type",
        default="dgcnn",
        type=str,
        help="Please provide the type of encoder.",
    )
    parser.add_argument(
        "--decoder_type",
        default="foldingnetbasic",
        type=str,
        help="Please provide the type of decoder.",
    )
    parser.add_argument(
        "--learning_rate_autoencoder",
        default=0.0001,
        type=float,
        help="Please provide the learning rate "
        "for the autoencoder training.",
    )
    parser.add_argument(
        "--learning_rate_clustering",
        default=0.00001,
        type=float,
        help="Please provide the learning rate "
        "for the autoencoder training.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Please provide the batch size.",
    )
    parser.add_argument(
        "--proximal",
        default=0,
        type=int,
        help="Please provide the value of proximality "
        "[0 = distal, 1 = proximal, 2 = both].",
    )
    parser.add_argument(
        "--pretrained_path",
        default="/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,"
        "share=data/DBI/DUDBI/DYNCESYS/mvries/ResultsAlma/TearingNetNew/"
        "nets/dgcnn_foldingnet_128_008.pt",
        type=str,
        help="Please provide the path to a pretrained autoencoder.",
    )

    args = parser.parse_args()
    pretrain = args.pretrain
    train_type = args.train_type
    params = {"pretrain": pretrain, "train_type": train_type}

    # Cache some errors
    if train_type == "pretrain" and not pretrain:
        print("Nothing to do :(")
        exit()

    if args.cloud_convert and not (args.model_type == "cloud"):
        print(
            "Not converting to point clouds "
            "as you are not using the cloud models."
        )

    params["pretrained_path"] = args.pretrained_path
    params["dataframe_path"] = args.dataframe_path
    params["dataset_path"] = args.dataset_path
    params["output_path"] = args.output_path
    params["num_epochs_autoencoder"] = args.num_epochs_autoencoder
    params["num_features"] = args.num_features
    params["k"] = args.k
    params["encoder_type"] = args.encoder_type
    params["decoder_type"] = args.decoder_type
    params["learning_rate_autoencoder"] = args.learning_rate_autoencoder
    params["batch_size"] = args.batch_size
    params["num_clusters"] = args.num_clusters
    params["num_epochs_clustering"] = args.num_epochs_clustering
    params["learning_rate_clustering"] = args.learning_rate_clustering
    params["gamma"] = args.gamma
    params["alpha"] = args.alpha
    params["divergence_tolerance"] = args.divergence_tolerance
    params["update_interval"] = args.update_interval
    params["model_type"] = args.model_type
    params["train_type"] = args.train_type

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

        reconstruction_criterion = ChamferLoss()

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

    reconstruction_criterion = ChamferLoss()
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

    cellshape_train(params=params)
