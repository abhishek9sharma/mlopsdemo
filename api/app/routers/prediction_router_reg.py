import numpy
import pandas as pd
import os

from fastapi import APIRouter
from api.app.schemas.pred_req_schema_reg import PredictionReqReg
from mlcore.regpredictor import Predictor
from mlcore.utils import set_logger


data_folder = os.environ["DATA_FOLDER_SVC"]
log_folder = os.environ["LOG_FOLDER_SVC"]
pred_svc_logger = set_logger("pred_svc_reg", log_folder, True)
cached_pred_obj = Predictor(data_folder, logger=pred_svc_logger)


router = APIRouter()


@router.post("/predictreg")
def predict(pred_req: PredictionReqReg):

    pred_response = {}

    id = pred_req.id
    try:
        pred_response = cached_pred_obj.predict_flow(
            pred_req.__dict__, req_type="json"
        )
    except Exception as e:
        error = " Exception {} occured ".format(e)
        pred_response = {"id": id, "message": error}

    return pred_response
