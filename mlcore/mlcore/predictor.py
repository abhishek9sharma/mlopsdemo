from collections import UserDict
import pandas as pd
import os, json, random

from mlcore.feature_extractor import *
from mlcore.data_helper import load_data, load_data_from_file_path
from mlcore.utils import set_logger
from pathlib import Path
from mlcore.dbhelper import read_data
from mlcore.modelops import load_model
from mlcore.train_eval_helper import (
    get_prediction_pipeline,
    get_ordered_preds_single,
)
from mlcore.train_eval_helper_n2v import get_ordered_preds_n2v
from mlcore.train_eval_helper_seq import get_ordered_preds_seq
from tensorflow import keras


class Predictor:
    def __init__(self, user_data_folder, logger):

        self.logger = logger
        self.user_data = load_data(
            "users", data_folder=user_data_folder, logger=self.logger
        )
        self.deployed_model_info = None
        self.deployed_model_name = None
        self.deployed_model_obj = None
        self.set_deployed_model()
        self.user_merchant_hist = None
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

        mldbpath = "data/mldb.sqlite"
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
        df = pd.DataFrame.from_records(pred_req, index="req_id")
        return df

    def extract_features(self, req_df):
        """Extracts some time based and user features based in input request

        :param req_df: Dataframe with extra featurs populated
        """

        req_df = extract_user_features(req_df, self.user_data)
        extract_time_features(req_df)
        return req_df

    def predict(self, req_df, num_of_items_req=5):
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
            clf_pipeline = get_prediction_pipeline(self.deployed_model_obj)
            prediction_df = req_df[features]
            preds = get_ordered_preds_single(prediction_df, clf_pipeline)
            pred_items = preds[:num_of_items_req]

        if model_type == "N2V":
            n2vmodel = self.deployed_model_obj["obj"]
            user_id = req_df["user_id"].iloc[0]
            user_id_str = "user_" + str(req_df["user_id"].iloc[0])
            merchants2vecdict = self.deployed_model_obj["merchants2vecdict"]
            n2vkeys = n2vmodel.wv.key_to_index.keys()

            if user_id_str not in n2vkeys:
                random_merchants = random.sample(
                    list(merchants2vecdict.keys()), num_of_items_req
                )
                for merchant in random_merchants:
                    pred_items.append({"merchant": merchant, "score": "0"})
            else:
                similar_merchants = get_ordered_preds_n2v(
                    user_id, merchants2vecdict, n2vmodel, return_scores=True
                )
                for merchant in similar_merchants:
                    merchant_id = str(merchant).replace("merchant_", "")
                    score = similar_merchants[merchant]
                    pred_items.append({"merchant": merchant_id, "score": str(score)})

        if model_type == "SEQDL":

            model_metadata = self.deployed_model_obj
            user_merchant_hist_path = model_metadata["user_merchant_hist_data_path"]
            user_merchant_hist_path = user_merchant_hist_path.replace(
                "../", ""
            )  # fix if time permits
            if self.user_merchant_hist is None:
                self.user_merchant_hist = load_data_from_file_path(
                    user_merchant_hist_path, logger=self.logger
                )
            word2ix = model_metadata["train_tokenizer"].word_index
            ix2word = {w: i for i, w in word2ix.items()}
            user_id = req_df["user_id"].iloc[0]
            uids = self.user_merchant_hist.uids.tolist()
            if user_id not in uids:
                print(word2ix)
                random_merchants = random.sample(list(word2ix.keys()), num_of_items_req)
                for merchant in random_merchants:
                    pred_items.append({"merchant": merchant, "score": "0"})
            else:
                user_past_seq = self.user_merchant_hist[
                    self.user_merchant_hist.uids == user_id
                ]["merchant_click_seq"].iloc[0]
                user_past_seq = " ".join(list(user_past_seq))
                similar_merchants = get_ordered_preds_seq(
                    user_past_seq, model_metadata, ix2word, return_scores=True
                )
                for merchant in similar_merchants:
                    merchant_id = str(merchant).replace("merch", "")
                    score = similar_merchants[merchant]
                    pred_items.append({"merchant": merchant_id, "score": str(score)})

        preds_dict = {"model": self.deployed_model_name, "items": pred_items}
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
            req_id = req_df["req_id"].iloc[0]
            prediction_response = {"req_id": req_id}

            # if self.deployed_model_name is None:
            self.set_deployed_model()

            if self.deployed_model_name is None:
                return {"req_id": req_id, "message": "No deployed model found"}

            if "num_of_items_req" in req_df:
                num_of_items_req = req_df["num_of_items_req"].iloc[0]
            else:
                num_of_items_req = 5

            req_df = self.extract_features(req_df)
            preds = self.predict(req_df, num_of_items_req)
            prediction_response = {"req_id": req_id, "preds": preds}

            return json.loads(json.dumps(prediction_response, indent=4))

        except Exception as e:
            error = " Exception {} occured ".format(e)
            return {"req_id": req_id, "message": error}
