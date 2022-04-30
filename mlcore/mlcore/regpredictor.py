from collections import UserDict
import pandas as pd
import os, json, random

from mlcore.data_helper import load_data, load_data_from_file_path
from mlcore.utils import set_logger
from pathlib import Path
from mlcore.dbhelper import read_data
from mlcore.modelops import load_model
from mlcore.train_eval_helper_reg import (
    get_prediction_pipeline,
)
#from tensorflow import keras


class Predictor:
    def __init__(self, user_data_folder, logger):

        self.logger = logger
        self.deployed_model_info = None
        self.deployed_model_name = None
        self.deployed_model_obj = None
        self.set_deployed_model()
        logger.info(self.deployed_model_name)

    def set_deployed_model(self):
        """Sets the latest deployed model to be used for prediction"""

        try:
            (
                deployed_model_info,
                deployed_model_name,
                deployed_model_obj,
            ) = self.resolve_model()
            self.deployed_model_info = deployed_model_info
            self.deployed_model_name = deployed_model_name
            self.deployed_model_obj = deployed_model_obj
        except Exception as e:
            error = "Could not load model :: {} occured".format(e)
            self.logger.info(error)

    def init_logger(self, logger):
        if logger is None:
            return set_logger("Predictor Obj log")
        else:
            return logger

    def resolve_model(self):
        """Resolves/loads the latest model metadata
            to be used for prediction

        :return: returns the model info, model name and loaded model object
        """

        mldbpath = "data/mldbreg.sqlite"
        model_folder = "models/"
        deployed_model_obj = None
        deployed_model_info = None
        deployed_model_name = None
        error = None
        try:
            deployed_model_info = (
                read_data(mldbpath, "deployed_model").iloc[0].to_dict()
            )
            if (
                deployed_model_info is not None
                or deployed_model_info != self.deployed_model_info
            ):
                deployed_model_name = deployed_model_info["final_model_name"]
                deployed_model_obj = load_model(
                    deployed_model_name,
                    model_folder=model_folder,
                    cur_logger=self.logger,
                )
            if deployed_model_obj["type"] == "SEQDL":
                deep_model_path = os.path.join(
                    model_folder, deployed_model_name + ".deep_mdl"
                )
                actual_obj = keras.models.load_model(deep_model_path)
                deployed_model_obj["obj"] = actual_obj
        except Exception as e:
            error = "Exception occured {}".format(e)
            raise ValueError(error)
        return deployed_model_info, deployed_model_name, deployed_model_obj

    def transform_json_to_df(self, req_json):
        """Transforms a request json to a pandas dataframe

        :param req_json: Request JSON that is actually passed in by WEB API
        :return: A pandas dataframe constructed from thr input json
        """
        pred_req = [req_json]
        df = pd.DataFrame.from_records(pred_req, index="id")
        return df

    def extract_features(self, req_df):
        """Extracts some time based and user features based in input request

        :param req_df: Dataframe with extra featurs populated
        """

        req_df = extract_user_features(req_df, self.user_data)
        return req_df

    def predict(self, req_df):
        """Finds the merchants which a user is likely to click

        :param req_df: a dataframe which capture the info in the actual request
                    as well as extracted features
        :param num_of_items_req: num of of merchants which may be clikcke next, defaults to 5
        :return:
        """

        features = self.deployed_model_obj["features"]
        model_type = self.deployed_model_obj["type"]

        pred_items = []
        if model_type == "classical":
            pred_pipeline = get_prediction_pipeline(self.deployed_model_obj)
            prediction_df = req_df[features]
            preds = pred_pipeline.predict(prediction_df)
            # bug here preds len is coming in different so below is a workaround only
            if len(preds)==1:
                pred_score = str(preds[0])
            else:
                pred_score = str(preds[0][0])
      
        
        preds_dict = {"model": self.deployed_model_name, "predicted_price": pred_score}
        return preds_dict

    def predict_flow(self, pred_req, req_type="df"):
        """Runs the prediction flow on an imput ( a json request or dataframe)
        which contain info about the user for which a recoomendation has to be made

        :param pred_req: a json object or a dataframe which captures basic user information
        :param req_type: type of object being passed, defaults to "df"
        :return: A json containing merchants which may be clicked/ an error message if processing failed
        """

        try:

            if req_type == "json":
                req_df = self.transform_json_to_df(pred_req)
            elif req_type == "df":
                req_df = pred_req

            req_df.reset_index(inplace=True)
            id = req_df["id"].iloc[0]
            prediction_response = {"id": id}

            # if self.deployed_model_name is None:
            self.set_deployed_model()

            if self.deployed_model_name is None:
                return {"id": str(id), "message": "No deployed model found"}


            
            preds = self.predict(req_df)
            prediction_response = {"id": str(id), "predictioninfo": preds}
            return json.loads(json.dumps(prediction_response, indent=4))

        except Exception as e:
            error = " Exception {} occured ".format(e)
            return {"id": id, "message": error}
