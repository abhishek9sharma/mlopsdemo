import pandas as pd
from mlcore.modelops import load_model, get_deployed_model
from mlcore.utils import set_logger
import os

def test_load_model():

    mldbpath = 'test_data/test_mldb.sqlite'
    model_folder = 'test_data/'
    os.environ['LOG_FOLDER']='test_data/logs/'
    os.environ['DATA_FOLDER']='test_data/'
    
    lgr = set_logger('abc', 'test_data/logs/')
    deployed_model_obj = load_model('KNN_model_10_30_2021_07_41_18', model_folder=model_folder, 
    cur_logger=lgr)
    
    assert  deployed_model_obj['Classifier']=='KNN'
    assert  deployed_model_obj['type']=='classical'


def test_deployed_model():

    mldbpath = 'test_data/test_mldb.sqlite'
    model_folder = 'test_data/'
    os.environ['LOG_FOLDER']='test_data/logs/'
    os.environ['DATA_FOLDER']='test_data/'
    
    lgr = set_logger('abc', 'test_data/logs/')
    deployed_model_name, deployed_model_obj = get_deployed_model(mldbpath, model_folder=model_folder, 
    cur_logger=lgr)
    
    assert  deployed_model_obj['Classifier']=='KNN'
    assert  deployed_model_obj['type']=='classical'
    assert  deployed_model_name=='KNN_model_10_30_2021_07_41_18'
    


    