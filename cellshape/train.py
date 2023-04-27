import torch
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
import logging

import cellshape_cloud as cscloud
import cellshape_cluster as cscluster
from cellshape_cloud.vendor.chamfer_distance import ChamferLoss


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(description="Cellshape")
    parser.add_argument(
        "--model_type",
        default="cloud",
        type=str,
        choices=["cloud", "voxel"],
        help="Please provide the type of model: [cloud, voxel].",
    )
    parser.add_argument(
        "--cloud_convert",
        default="False",
        type=str2bool,
        help="Do you need to convert 3D images to point clouds?"
        " If you do, please go to cellshape-helper.",
    )
    parser.add_argument(
        "--num_points",
        default=2048,
        type=int,
        help="The number of points used in each point cloud.",
    )
    parser.add_argument(
        "--train_type",
        default="DEC",
        type=str,
        choices=["pretrain", "DEC"],
        help="Please provide the type of training mode: [pretrain, full]."
        " 'pretrain' is to train the autoencoder only, "
        "'DEC' is to add the clustering layer.",
    )
    parser.add_argument(
        "--pretrain",
        default="True",
        type=str2bool,
        help="Please provide whether or not to pretrain the autoencoder.",
    )
    parser.add_argument(
        "--tif_dataset_path",
        default="./TestTiff/",
        type=str,
        help="Please provide the path to the " "dataset of 3D tif images",
    )
    parser.add_argument(
        "--mesh_dataset_path",
        default="./TestMesh/",
        type=str,
        help="Please provide the path to the " "dataset of 3D meshes.",
    )
    parser.add_argument(
        "--cloud_dataset_path",
        default="./TestCloud/",
        type=str,
        help="Please provide the path to the " "dataset of the point clouds.",
    )
    parser.add_argument(
        "--dataset_type",
        default="SingleCell",
        type=str,
        choices=["SingleCell", "Other"],
        help="Please provide the type of dataset. "
        "If using the one from our paper, then choose 'SingleCell', "
        "otherwise, choose 'Other'.",
    )
    parser.add_argument(
        "--dataframe_path",
        default="./all_data_removedwrong_ori.csv",
        type=str,
        help="Please provide the path to the dataframe "
        "containing information on the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default="./Test_output/",
        type=str,
        help="Please provide the path for where to save output.",
    )
    parser.add_argument(
        "--num_epochs_autoencoder",
        default=1,
        type=int,
        help="Provide the number of epochs for the autoencoder training.",
    )
    parser.add_argument(
        "--num_epochs_clustering",
        default=3,
        type=int,
        help="Provide the number of epochs for the clustering training.",
    )
    parser.add_argument(
        "--num_features",
        default=128,
        type=int,
        help="Please provide the number of features to extract.",
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
        choices=["dgcnn", "foldingnet"],
        help="Please provide the type of encoder.",
    )
    parser.add_argument(
        "--decoder_type",
        default="foldingnetbasic",
        type=str,
        choices=["foldingnetbasic", "foldingnet"],
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
        "--update_interval",
        default=1,
        type=int,
        help="How often to update the target "
        "distribution for the kl divergence.",
    )
    parser.add_argument(
        "--gamma",
        default=1,
        type=int,
        help="Please provide the value for gamma.",
    )
    parser.add_argument(
        "--alpha",
        default=1.0,
        type=float,
        help="Please provide the value for alpha.",
    )
    parser.add_argument(
        "--divergence_tolerance",
        default=0.01,
        type=float,
        help="Please provide the divergence tolerance.",
    )
    parser.add_argument(
        "--proximal",
        default=2,
        type=int,
        help="Do you want to look at cells distal "
        "or proximal to the coverslip?"
        "[0 = distal, 1 = proximal, 2 = both].",
    )
    parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str,
        help="Please provide the path to a pretrained autoencoder.",
    )

    args = parser.parse_args()
    for arg, value in sorted(vars(args).items()):
        print(f"Argument {arg}: {value}")

    # First decide whether it is a cloud or a voxel model:
    # Lets' deal with cloud first
    if args.model_type == "cloud":
        # If no, continue.
        # Do we want to pretrain the autoencoder FIRST?
        # Yes
        if args.pretrain:
            # Do we want to pretrain (autoencoder without
            # clustering layer) ONLY?
            # Yes
            if args.train_type == "pretrain":
                cscloud.train_autoencoder(args)
                exit()

            # No, we want to train the full thing
            else:
                (
                    autoencoder,
                    name_logging_ae,
                    name_model_ae,
                    name_writer_ae,
                    name_ae,
                ) = cscloud.train_autoencoder(args)

                model = cscluster.DeepEmbeddedClustering(
                    autoencoder=autoencoder, num_clusters=args.num_clusters
                )

                (
                    name_logging,
                    name_model,
                    name_writer,
                    name,
                ) = cscluster.get_experiment_name(
                    model=model, output_dir=args.output_dir
                )

                if args.dataset_type == "SingleCell":
                    dataset = cscloud.SingleCellDataset(
                        args.dataframe_path, args.cloud_dataset_path
                    )
                else:
                    dataset = cscloud.PointCloudDataset(
                        args.cloud_dataset_path
                    )

                dataloader = DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False
                )
                dataloader_inf = DataLoader(
                    dataset, batch_size=1, shuffle=False
                )

                optimizer = torch.optim.Adam(
                    autoencoder.parameters(),
                    lr=args.learning_rate_clustering * 16 / args.batch_size,
                    betas=(0.9, 0.999),
                    weight_decay=1e-6,
                )
                reconstruction_criterion = ChamferLoss()
                cluster_criterion = torch.nn.KLDivLoss(reduction="sum")

                logging_info = name_logging, name_model, name_writer, name

                name_logging, name_model, name_writer, name = logging_info
                now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                logging.basicConfig(filename=name_logging, level=logging.INFO)
                logging.info(
                    f"Started training model {name} at {now} "
                    f"using autoencoder which is saved at {name_model_ae}."
                )
                print(
                    f"Started training model {name} at {now}."
                    f"using autoencoder which is saved at {name_model_ae}."
                )
                for arg, value in sorted(vars(args).items()):
                    logging.info(f"Argument {arg}: {value}")
                    print(f"Argument {arg}: {value}")

                cscluster.train(
                    model=model,
                    dataloader=dataloader,
                    dataloader_inf=dataloader_inf,
                    num_epochs=args.num_epochs_clustering,
                    optimizer=optimizer,
                    reconstruction_criterion=reconstruction_criterion,
                    cluster_criterion=cluster_criterion,
                    update_interval=args.update_interval,
                    gamma=args.gamma,
                    divergence_tolerance=args.divergence_tolerance,
                    logging_info=logging_info,
                )

        # No, we don't want to pretrain the autoencoder first.
        # maybe we already have one, or we want
        # to not start from pretrained features
        else:
            print(f"args pretrain == {args.pretrain}")
            autoencoder = cscloud.CloudAutoEncoder(
                num_features=args.num_features,
                k=args.k,
                encoder_type=args.encoder_type,
                decoder_type=args.decoder_type,
            )
            everything_working = True
            file_not_found = False
            wrong_architecture = False
            try:
                checkpoint = torch.load(str(args.pretrained_path))
            except FileNotFoundError:
                print(
                    "This model doesn't exist."
                    " Please check the provided path and try again."
                )
                checkpoint = {"model_state_dict": None}
                file_not_found = True
                everything_working = False
            except AttributeError:
                print("No pretrained model given.")
                checkpoint = {"model_state_dict": None}
                everything_working = False
            try:
                autoencoder.load_state_dict(checkpoint["model_state_dict"])
                print(f"The loss of the loaded model is {checkpoint['loss']}.")
            except RuntimeError:
                print(
                    "The model architecture given doesn't "
                    "match the one provided."
                )
                print("Training from scratch")
                wrong_architecture = True
                everything_working = False
            except AttributeError:
                print("Training from scratch.")
            except:
                print("Training from scratch.")

            model = cscluster.DeepEmbeddedClustering(
                autoencoder=autoencoder, num_clusters=args.num_clusters
            )

            (
                name_logging,
                name_model,
                name_writer,
                name,
            ) = cscluster.get_experiment_name(
                model=model, output_dir=args.output_dir
            )
            cluster_criterion = torch.nn.KLDivLoss(reduction="sum")

            if args.dataset_type == "SingleCell":
                dataset = cscloud.SingleCellDataset(
                    args.dataframe_path, args.cloud_dataset_path
                )
            else:
                dataset = cscloud.PointCloudDataset(args.cloud_dataset_path)

            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False
            )
            dataloader_inf = DataLoader(dataset, batch_size=1, shuffle=False)

            optimizer = torch.optim.Adam(
                autoencoder.parameters(),
                lr=args.learning_rate_clustering * 16 / args.batch_size,
                betas=(0.9, 0.999),
                weight_decay=1e-6,
            )
            reconstruction_criterion = ChamferLoss()
            cluster_criterion = torch.nn.KLDivLoss(reduction="sum")

            logging_info = name_logging, name_model, name_writer, name

            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            logging.basicConfig(filename=name_logging, level=logging.INFO)
            if everything_working:
                logging.info(
                    f"Started training cluster model {name} at {now} "
                    f"using autoencoder which is "
                    f"saved at {args.pretrained_path}."
                )
                print(
                    f"Started training model {name} at {now}."
                    f"using autoencoder which is s"
                    f"aved at {args.pretrained_path}."
                )
            if file_not_found:
                logging.info(
                    f"The autoencoder model at {args.pretrained_path}"
                    f" doesn't exist. "
                    f"If you knew this already, then don't worry. "
                    f"If not, then check the path and try again."
                )
                logging.info("Training from scratch")
                print(
                    f"The autoencoder model at "
                    f"{args.pretrained_path} doesn't exist."
                    f"if you knew this already, then don't worry. "
                    f"If not, then check the path and try again."
                )
                print("Training from scratch")

            if wrong_architecture:
                logging.info(
                    f"The autoencoder model at {args.pretrained_path} has "
                    f"a different architecture to the one provided "
                    f"If not, then check the path and try again"
                )
                logging.info("Training from scratch")
                print(
                    f"The autoencoder model at {args.pretrained_path} "
                    f"has a different architecture to the one provided "
                    f"If not, then check the path and try again"
                )
                print("Training from scratch")

            for arg, value in sorted(vars(args).items()):
                logging.info(f"Argument {arg}: {value}")

            cscluster.train(
                model=model,
                dataloader=dataloader,
                dataloader_inf=dataloader_inf,
                num_epochs=args.num_epochs_clustering,
                optimizer=optimizer,
                reconstruction_criterion=reconstruction_criterion,
                cluster_criterion=cluster_criterion,
                update_interval=args.update_interval,
                gamma=args.gamma,
                divergence_tolerance=args.divergence_tolerance,
                logging_info=logging_info,
            )


if __name__ == "__main__":
    main()
