import fnmatch
import logging
import os


def get_model_name(model):
    encoder_type = model.encoder_type
    decoder_type = model.decoder_type
    num_features = model.encoder.num_features
    model_name = f"{encoder_type}_{decoder_type}_{num_features}"
    return model_name


def get_experiment_name(model, output_dir):
    model_name = get_model_name(model)
    dirs = [output_dir + "runs", output_dir + "reports", output_dir + "nets"]
    list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))
    reports_list = sorted(os.listdir(output_dir + "reports"), reverse=True)
    if reports_list:
        for file in reports_list:
            if fnmatch.fnmatch(file, model_name + "*"):
                idx = int(str(file)[-7:-4]) + 1
                break
    try:
        idx
    except NameError:
        idx = 1

    name = model_name + "_" + str(idx).zfill(3)
    name_reports = name + ".log"
    name_model = name + ".pt"
    name_logging = os.path.join(output_dir + "reports", name_reports)
    name_model = os.path.join(output_dir + "nets", name_model)
    name_writer = os.path.join(output_dir + "runs", name)

    return name_logging, name_model, name_writer, name


def print_log(string):
    logging.info(string)
    print(string)
