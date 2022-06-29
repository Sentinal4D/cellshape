import torch
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
import logging

import cellshape_cloud as cscloud
import cellshape_cluster as cscluster
import cellshape_helper as cshelper
from cellshape_cloud.vendor.chamfer_distance import ChamferLoss


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
        "--num_points",
        default=2048,
        type=int,
        help="The number of points used in each point cloud.",
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
        help="Please provide whether or not to pretrain the autoencoder",
    )
    parser.add_argument(
        "--tif_dataset_path",
        default="./dataset_tif/",
        type=str,
        help="Please provide the path to the " "dataset of 3D tif images",
    )
    parser.add_argument(
        "--mesh_dataset_path",
        default="./dataset_mesh/",
        type=str,
        help="Please provide the path to the " "dataset of 3D meshes.",
    )
    parser.add_argument(
        "--cloud_dataset_path",
        default="./dataset_cloud/",
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
        help="Provide the number of epochs for the autoencoder training.",
    )
    parser.add_argument(
        "--num_epochs_clustering",
        default=250,
        type=int,
        help="Provide the number of epochs for the autoencoder training.",
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

    # First decide whether it is a cloud or a voxel model:
    # Lets' deal with cloud first
    if args.model_type == "cloud":
        # Do we need to convert the data?
        # Yes:
        if args.cloud_convert:
            cshelper.tif_to_pc_directory(
                tif_directory=args.dataset_tif,
                save_mesh=args.dataset_mesh,
                save_points=args.dataset_cloud,
                num_points=args.num_points,
            )
        # If no, continue.
        # Do we want to pretrain the autoencoder FIRST?
        # Yes
        if args.pretrain:
            # Do we want to pretrain (autoencoder without
            # clustering layer) ONLY?
            # Yes
            if args.train_type == "pretrain":
                output = cscloud.train_autoencoder(args)
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

                # todo: logging.close()
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
                    dataset = cscloud.PointCloudDataset(args.input_dir)

                dataloader = DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False
                )
                dataloader_inf = DataLoader(
                    dataset, batch_size=1, shuffle=False
                )

                optimizer = torch.optim.Adam(
                    autoencoder.parameters(),
                    lr=args.learning_rate_cluster * 16 / args.batch_size,
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
                checkpoint = torch.load(args.pretrained_path)
            except FileNotFoundError:
                print(
                    "This model doesn't exist."
                    "Please check the provided path and try again."
                )
                checkpoint = {"model_state_dict": None}
                file_not_found = True
                everything_working = False
            try:
                autoencoder.load_state_dict(checkpoint["model_state_dict"])
                print(f"The loss of the loaded model is {checkpoint['loss']}")
            except RuntimeError:
                print(
                    "The model architecture given doesn't "
                    "match the one provided."
                )
                print("Training from scratch")
                wrong_architecture = True
                everything_working = False
            except AttributeError:
                print("Training from scratch")

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
                dataset = cscloud.PointCloudDataset(args.input_dir)

            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False
            )
            dataloader_inf = DataLoader(dataset, batch_size=1, shuffle=False)

            optimizer = torch.optim.Adam(
                autoencoder.parameters(),
                lr=args.learning_rate_cluster * 16 / args.batch_size,
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
                    f" doesn't exist."
                    f"if you knew this already, then don't worry. "
                    f"If not, then check the path and try again"
                )
                logging.info("Training from scratch")
                print(
                    f"The autoencoder model at "
                    f"{args.pretrained_path} doesn't exist."
                    f"if you knew this already, then don't worry. "
                    f"If not, then check the path and try again"
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

    # ############################################################
    # # Cache some errors
    # if args.train_type == "pretrain" and not args.pretrain:
    #     print("Nothing to do :(")
    #     exit()
    #
    # if args.cloud_convert and not (args.model_type == "cloud"):
    #     print(
    #         "Not converting to point clouds "
    #         "as you are not using the cloud models."
    #     )
    #
    # if args.model_type == "cloud":
    #
    #     autoencoder = cscloud.CloudAutoEncoder(
    #         num_features=args.num_features,
    #         k=args.k,
    #         encoder_type=args.encoder_type,
    #         decoder_type=args.decoder_type,
    #     )
    #
    #     dataset = cscloud.PointCloudDataset(args.input_dir)
    #
    #     dataloader = DataLoader(dataset,
    #     batch_size=args.batch_size, shuffle=True)
    #
    #     reconstruction_criterion = ChamferLoss()
    #
    #     optimizer = torch.optim.Adam(
    #         autoencoder.parameters(),
    #         lr=args.learning_rate_autoencoder * 16 / args.batch_size,
    #         betas=(0.9, 0.999),
    #         weight_decay=1e-6,
    #     )
    #
    #     (
    #         autoencoder,
    #         name_logging,
    #         name_model,
    #         name_writer,
    #         name,
    #     ) = cscloud.train(
    #         autoencoder,
    #         dataloader,
    #         args.num_epochs_autoencoder,
    #         reconstruction_criterion,
    #         optimizer,
    #         args.output_dir,
    #     )
    #
    # else:
    #     autoencoder = csvoxel.VoxelAutoEncoder(
    #         num_features=args.num_features,
    #         encoder_type=args.encoder_type,
    #         decoder_type=args.decoder_type,
    #     )
    #
    #     dataset = csvoxel.VoxelDataset(args.input_dir)
    #
    #     dataloader = DataLoader(dataset,
    #     batch_size=args.batch_size, shuffle=True)
    #
    #     reconstruction_criterion = FocalTverskyLoss()
    #
    #     optimizer = torch.optim.Adam(
    #         autoencoder.parameters(),
    #         lr=args.learning_rate_autoencoder * 16 / args.batch_size,
    #         betas=(0.9, 0.999),
    #         weight_decay=1e-6,
    #     )
    #
    #     (
    #         autoencoder,
    #         name_logging,
    #         name_model,
    #         name_writer,
    #         name,
    #     ) = csvoxel.train(
    #         autoencoder,
    #         dataloader,
    #         args.num_epochs_autoencoder,
    #         reconstruction_criterion,
    #         optimizer,
    #         args.output_dir,
    #     )
    #
    # if args.train_type == "pretrain" and args.pretrain:
    #     print()
    #
    # model = cscluster.DeepEmbeddedClustering(
    #     autoencoder=autoencoder, num_clusters=args.num_clusters
    # )
    #
    # dataloader = DataLoader(
    #     dataset, batch_size=args.batch_size, shuffle=False
    # )  # it is very important that shuffle=False here!
    # dataloader_inf = DataLoader(
    #     dataset, batch_size=1, shuffle=False
    # )  # it is very important that batch_size=1 and shuffle=False here!
    #
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=args.learning_rate_clustering * 16 / args.batch_size,
    #     betas=(0.9, 0.999),
    #     weight_decay=1e-6,
    # )
    #
    # reconstruction_criterion = ChamferLoss()
    # cluster_criterion = torch.nn.KLDivLoss(reduction="sum")
    #
    # cscluster.train(
    #     model=model,
    #     dataloader=dataloader,
    #     dataloader_inf=dataloader_inf,
    #     num_epochs=args.num_epochs_clustering,
    #     optimizer=optimizer,
    #     reconstruction_criterion=reconstruction_criterion,
    #     cluster_criterion=cluster_criterion,
    #     update_interval=args.update_interval,
    #     gamma=args.gamma,
    #     divergence_tolerance=args.divergence_tolerance,
    #     logging_info=logging_info,
    # )
    #
    # cellshape_train(args)
