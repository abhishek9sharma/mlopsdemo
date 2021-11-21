import numpy
import pandas as pd
import os

from fastapi import APIRouter
from api.app.schemas.pred_req_schema import PredictionReq
from mlcore.predictor import Predictor
from mlcore.utils import set_logger


data_folder = os.environ["DATA_FOLDER_SVC"]
log_folder = os.environ["LOG_FOLDER_SVC"]
pred_svc_logger = set_logger("pred_svc", log_folder, True)
cached_pred_obj = Predictor(data_folder, logger=pred_svc_logger)


router = APIRouter()


@router.post("/predict")
def predict(pred_req: PredictionReq):

    pred_response = {}

    req_id = pred_req.req_id
    try:
        pred_response = cached_pred_obj.predict_flow(
            pred_req.__dict__, req_type="json"
        )
    except Exception as e:
        error = " Exception {} occured ".format(e)
        pred_response = {"req_id": req_id, "message": error}

    return pred_response
