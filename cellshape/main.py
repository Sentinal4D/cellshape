import argparse
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
    params["learning_rate_autoencoder"] = args.learning_rate
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

    if (
        args.train_type == "full"
        and (not args.cloud_convert)
        and args.pretrain
    ):
        cellshape_train(params=params)
