import numpy as np
import pandas as pd
import math
import joblib


from ast import literal_eval
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import *
from mlcore.data_helper import load_data_from_file_path



def get_df_from_dict(dictin, idxname=None, columnsin=None):
    """Converts a dictionary to a dataframe

    :param dictin: a dictionary containing some keys and values
    :param idxname: key to be used as index for the dataframe, defaults to None
    :param columnsin: column names to be assigned to dataframe, defaults to None
    :return:
    """
    df = pd.DataFrame.from_dict(dictin, orient="index")
    if columnsin:
        df.columns = columnsin
    if idxname:
        df.index.name = idxname
    df = df.reset_index()
    return df


def get_prediction_pipeline(model_metadata):
    """Extracts prediction pipeline for classical scikit learn models

    :param model_metadata: a dictionary containing metadata about a model
    :return: the model pipeline associated with the model
    """
    model_obj = model_metadata["obj"]
    preprocessor = model_metadata["preprocessor"]
    return make_pipeline(preprocessor, model_obj)


def train_models(models, train_data, target, logger, dask_client=None, randomized_search=False, scoring_metric=None):
    """Trains a set of models on the given training data/labels


    :param models: a dictionary of models which need to be trained
    :param train_data: a dataframe containing all possible features ( actual features used are specific to model) of train data
    :param target: the column which contains the actual labels for training data
    :param logger: logger object passed from the calling module
    :param dask_client: ask client to use for training, esting, defaults to None
    :param randomized_search: flag specifying to tune params by randomized search
    :param scoring_metric: scoring metric to be used for randomized hyper param search
    
    :return: a dictionary of models after fitting training params on the data
    """

    for modelkey, model_metadata in models.items():
        logger.info("Training started for " + modelkey)
        #model_metadata["MRR"] = 0

        # resolve feature data/params
        features = model_metadata["features"]
        X_train = train_data[features]
        y_train = train_data[target]
        model_params = model_metadata["param_grid"]
        model_type = model_metadata["type"]

        if model_type == "classical":
            model_pipeline = get_prediction_pipeline(model_metadata)
            if dask_client:
                with joblib.parallel_backend("dask"):
                    model_pipeline.fit(X_train, y_train)
            else:
                if randomized_search and modelkey not in ['LinReg']:
                    # randomized_search if time permits fix bugs
                    search = RandomizedSearchCV(estimator = model_pipeline,
                                                param_distributions = model_params,
                                                n_iter = 50, 
                                                cv = 5, 
                                                verbose=10, 
                                                random_state=35, 
                                                n_jobs = -1,
                                                scoring = scoring_metric

                                                )
                    try:
                        search.fit(X_train, y_train)
                        best_params = str(search.best_params_)
                        model_pipeline.set_params(**literal_eval(best_params))
                        model_pipeline.fit(X_train, y_train)
                    except Exception as e:
                        logger.info(" Exception {} while param search for {} switching to default fit".format(e, modelkey))
                        model_pipeline.fit(X_train, y_train)
                else:
                    model_pipeline.fit(X_train, y_train)
        
        if model_type == "DL":
            model_fitted_params = fit_dl_model(
                X_train,
                y_train,
                param_dict=model_params,
                logger=logger,
                scoring_metric = scoring_metric
            )

            for key in model_fitted_params:
                models[modelkey][key] = model_fitted_params[key]
                
        logger.info("Training ended for " + modelkey)

    return models


def compute_error(y_test, y_pred,error_metric):
    """ Computes error based on a metric
    :param y_test: target data
    :param y_pred: predicted data
    :param error_metric: error metric to evalualte model performance on (MAE, RMSE, etc.)
    :param error_metric: error metric to evalualte model performance on (MAE, RMSE, etc.)
    :return: the computed error
    """
    
    # if error_metric=='mean_squared_error':
    #     error = mean_squared_error(y_test, y_pred)
    # if error_metric=='mean_absolute_error':
    #     error = mean_absolute_error(y_test, y_pred)

    try:
        error = eval(error_metric)(y_test, y_pred)
        return error
    except Exception as e:
        logger.info(" Exception {} occured while computing metric {}".format(e, error_metric))
        return math.isNan()


def compute_error_model(model_metadata, X_test, y_test, target,error_metric):
    """Computes the model MRR based on test data

    :param model_metadata:  a dictionary containing metadata about a model
    :param X_test:  a dataframe containing features specfic to the model being evaluated
    :param y_test: a dataframe of target labels
    :param target: the column which contains the actual labels for training data
    :param error_metric: error metric to evalualte model performance on (MAE, RMSE, etc.)
    :return: the computed error
    """
    
    model_pipeline = get_prediction_pipeline(model_metadata)
    pred_prices = model_pipeline.predict(X_test)
    error = compute_error(y_test, pred_prices, error_metric)
    return error

def test_models(models, test_data, target, logger, dask_client=None, error_metric='RMSE'):
    """

    :param models: a dictionary of models which need to be trained
    :param test_data: a dataframe containing all possible features (actual features used are specific to model) of test data
    :param target: the column which contains the actual labels for testing data
    :param logger: logger object passed from the calling module
    :param dask_client: dask client to use for testing, defaults to None
    :param error_metric: error metric to evalualte model performance on (MAE, RMSE, etc.)
    :return:   a dictionary of models after computing eval metrics on the test data
    """
    for modelkey, model_metadata in models.items():

        model_metadata[error_metric] = 0
        features = model_metadata["features"]
        X_test = test_data[features]
        y_test = test_data[target]

        logger.info("Testing started for " + modelkey)

        model_type = model_metadata["type"]
        if model_type == "classical":
            if dask_client:
                with joblib.parallel_backend("dask"):
                    model_metadata[error_metric] = compute_error_model(
                        model_metadata, X_test, y_test, target, error_metric
                    )
            else:
                model_metadata[error_metric] = compute_error_model(
                    model_metadata, X_test, y_test, target,error_metric
                )

  
        # if model_type == "SEQDL":
        #     model_metadata["MRR"] = compute_mrr_model(
        #         model_metadata, X_test, y_test, target
        #     )
        logger.info("Testing ended for {} on metric {}".format(modelkey,error_metric))

    return models


def get_best_model(comparison_result, metric="fmeasure", reverse=False):
    """
    Returns the model row in dataframe which has the best performance on a given metric

    :param comparison_result: a dataframe containing  evaluation info about various models
    :param metric: metric on which a comparison is be made, defaults to "fmeasure"
    :return: a dataframe row corresponding to the model which has the best score on the given metric
    """
    if reverse:
        model_row = comparison_result[
            comparison_result[metric] == comparison_result[metric].min()
        ]
    else:
        model_row = comparison_result[
            comparison_result[metric] == comparison_result[metric].max()
        ]
    return model_row.iloc[0]
