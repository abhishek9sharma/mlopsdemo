import os
import joblib
import json

from mlcore.dbhelper import read_data, store_data
from mlcore.train_eval_helper import get_df_from_dict
from mlcore.train_eval_helper import *
from mlcore.utils import set_logger

from tensorflow import keras


def load_model(fname, model_folder="../models/", cur_logger=None):
    """Loads the model from a disk location given the model name

    :param fname: name of the model file which needs to be loaded
    :param model_folder: folder where the models are stored/loaded from, defaults to "../models/"
    :param cur_logger: logger of the calling object, defaults to None
    :return:
    """
    if cur_logger is None:
        cur_logger = set_logger("opslogger")

    model_path = os.path.join(model_folder, fname) + ".mdl"
    model = joblib.load(model_path)
    logstr = "Loaded model stored at {}".format(model_path)
    cur_logger.info(logstr)
    return model


def save_model(model, fname, model_folder="../models/", model_type="normal"):
    """Saves the model to a disk location given the model name

    :param model: model object which needs to be stored
    :param fname: name of the model file which needs to be stored
    :param model_type: type of model classical vs dl helps to store accodingly
    :param model_folder: folder where the models are stored/loaded from, defaults to "../models/"
    """

    model_path = os.path.join(model_folder, fname) + ".mdl"
    if model_type == "SEQDL":
        modelobj = model["obj"]
        dl_model_path = model_path.replace(".mdl", ".deep_mdl")
        modelobj.save(dl_model_path)
        del model["obj"]

    joblib.dump(model, model_path)
    print("model stored at {}".format(model_path))


def get_deployed_model(
    mldbpath="../data/mldb.sqlite",
    deploy_table_name="deployed_model",
    model_folder="../models/",
    cur_logger=None,
):
    """Gets info about the latest deployed model from the model registry (database)

    :param mldbpath: model registry(database) path, defaults to "../data/mldb.sqlite"
    :param deploy_table_name: table where model info is stored, defaults to "deployed_model"
    :param model_folder: folder where the models are stored/loaded from, defaults to "../models/"
    :param cur_logger: logger of the calling object, defaults to None, defaults to None
    :return:
    """
    if cur_logger is None:
        cur_logger = set_logger("opslogger")

    deployed_model_info = None
    deployed_model_obj = None
    try:
        deployed_model_info = read_data(mldbpath, "deployed_model").iloc[0].to_dict()
        if deployed_model_info:
            deployed_model_name = deployed_model_info["final_model_name"]
            deployed_model_obj = load_model(
                deployed_model_name, model_folder=model_folder
            )
            if deployed_model_obj["type"] == "SEQDL":
                deep_model_path = os.path.join(
                    model_folder, deployed_model_name + ".deep_mdl"
                )
                actual_obj = keras.models.load_model(deep_model_path)
                deployed_model_obj["obj"] = actual_obj
            return deployed_model_name, deployed_model_obj
    except Exception as e:
        logstr = "Could not load deployed model it may not exist exception {}".format(
            str(e)
        )
        cur_logger.info(logstr)
        return None, None


def get_deployment_hist():

    depl_hist = None
    try:
        mldbpath = "../data/mldb.sqlite"
        depl_hist = read_data(mldbpath, "hist_deployed_models")
    except:
        print("Failed to fetch model dep hist")

    return depl_hist
