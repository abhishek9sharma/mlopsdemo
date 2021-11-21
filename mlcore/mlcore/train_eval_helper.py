import numpy as np
import pandas as pd
import joblib


from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from mlcore.train_eval_helper_n2v import (
    fit_node2vec,
    get_ordered_preds_n2v,
    get_merchant_vecs,
)
from mlcore.train_eval_helper_seq import (
    fit_sequence_model,
    extract_user_merchant_hist,
    get_ordered_preds_seq,
)

from mlcore.data_helper import load_data_from_file_path


def compute_pred_probs(X, model_pipeline):
    """
    Computes the probability scores of possible labels
    Scores are generated using the scikit learn model on the
    on features passed in as dataframe

    :param X: feature instance data frame on which prediction is to be made
    :param model_pipeline: model pipeline used to make prediction
    :return: a dataframe containign probabilities of predictions
    """

    pred_arr = model_pipeline.predict_proba(X)
    pred_dict = {"label_probs": list(pred_arr)}
    pred_df = pd.DataFrame(pred_dict, columns=["label_probs"])
    pred_df.index = X.index
    return pred_df


def get_ordered_preds_single(X, model_pipeline):

    """
    Returns an ordered list of dicts {merchant_id: score} ordered by scores
    Scores are generated using the scikit learn model on on a single feature instance passed

    :param X: single feature instance data frame on which prediction is to be made
    :param model_pipeline: model pipeline used to make prediction
    :return: a list of dicts of {merchant_id: score}
    """
    probs = model_pipeline.predict_proba(X)
    best_n = np.argsort(probs)
    preds = [model_pipeline.classes_[predicted_cat] for predicted_cat in best_n]
    probs_sorted = sorted(probs[0], reverse=True)
    return [
        {"merchant": str(i[0]), "score": str(i[1])}
        for i in zip(preds[0][::-1], probs_sorted)
    ]


def get_ordered_preds(probs, model_pipeline):
    """Gets the predicted labels ordered by their probabilities

    :param probs: probabilities of class labels
    :param model_pipeline: classifier object/pipeline used to make prediction
    :return: possible labels ordered based on their probability
    """
    best_n = np.argsort(probs)
    preds = [model_pipeline.classes_[predicted_cat] for predicted_cat in best_n]
    return preds[::-1]


def get_rr(rw):
    """
    Returns the reciprocal rank given a list and expected label
    :param rw: a tuple of (ordered_labels, org_label)
    :return: reciprocal rank score based on (ordered_labels, org_label)
    """
    ordered_labels = rw[0]
    org_label = rw[1]
    if org_label in ordered_labels:
        org_found_at = ordered_labels.index(org_label) + 1
    else:
        org_found_at = len(ordered_labels)
    return 1 / org_found_at


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


def train_models(models, train_data, target, logger, dask_client=None):
    """Trains a set of models on the given training data/labels


    :param models: a dictionary of models which need to be trained
    :param train_data: a dataframe containing all possible features ( actual features used are specific to model) of train data
    :param target: the column which contains the actual labels for training data
    :param logger: logger object passed from the calling module
    :param dask_client: ask client to use for training, esting, defaults to None
    :return: a dictionary of models after fitting training params on the data
    """

    for modelkey, model_metadata in models.items():
        logger.info("Training started for " + modelkey)
        model_metadata["MRR"] = 0

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
                model_pipeline.fit(X_train, y_train)
                # support for grid  search add later if time permits
                # search.fit(X_train, y_train)
                # search = RandomizedSearchCV(model_pipeline, model_params, cv=5, n_iter=50, verbose=10)

        if model_type == "N2V":
            n2vmodel = fit_node2vec(
                train_data, features, target, param_dict=model_params
            )
            models[modelkey]["obj"] = n2vmodel
            models[modelkey]["merchants2vecdict"] = get_merchant_vecs(n2vmodel)
        logger.info("Training ended for " + modelkey)

        if model_type == "SEQDL":
            user_merchant_hist_data_path = model_metadata["user_merchant_hist_path"]
            model_fitted_params = fit_sequence_model(
                train_data,
                features,
                param_dict=model_params,
                logger=logger,
                user_merchant_hist_data_path=user_merchant_hist_data_path,
            )

            for key in model_fitted_params:
                models[modelkey][key] = model_fitted_params[key]

        logger.info("Training ended for " + modelkey)

    return models


def compute_mrr_input(model_metadata, X_test, y_test, target):
    """Computes the RR for a given data frame

    :param model_metadata: a dictionary containing metadata about a model
    :param X_test:  a dataframe containing features specfic to the model being evaluated
    :param y_test: a dataframe of target labels
    :param target: the column which contains the actual labels for training data
    :return: a dataframe containing reciprcal rank computed for all feature instance in X_test
    """

    model_type = model_metadata["type"]

    if model_type == "classical":
        model_pipeline = get_prediction_pipeline(model_metadata)
        pred_df = compute_pred_probs(X_test, model_pipeline=model_pipeline)
        pred_df["order_of_preds"] = pred_df["label_probs"].apply(
            get_ordered_preds, model_pipeline=model_pipeline
        )
        pred_df["y_org"] = y_test[target]

    if model_type == "N2V":
        n2vmodel = model_metadata["obj"]
        merchants2vecdict = model_metadata["merchants2vecdict"]
        pred_df = X_test
        pred_df["order_of_preds"] = X_test.user_id.apply(
            get_ordered_preds_n2v,
            merchants=merchants2vecdict,
            n2vmodel=n2vmodel,
        )
        pred_df["y_org"] = y_test[target]

    if model_type == "SEQDL":

        word2ix = model_metadata["train_tokenizer"].word_index
        ix2word = {w: i for i, w in word2ix.items()}
        user_merchant_hist_path = model_metadata["user_merchant_hist_data_path"]
        user_merchant_hist = load_data_from_file_path(user_merchant_hist_path)
        pred_df = X_test

        # some bug forcing target to be mergedf before training
        pred_df["y_org"] = y_test[target]
        pred_df = pd.merge(
            pred_df,
            user_merchant_hist,
            left_on="user_id",
            right_on="uids",
            how="left",
        )

        no_hist_users = pred_df[pred_df.merchant_click_seq.isna()]
        print(
            "No history found for {} users dropping them for MRR compute".format(
                no_hist_users.shape[0]
            )
        )
        pred_df.dropna(subset=["merchant_click_seq"], inplace=True)
        pred_df["merchant_click_seq"] = pred_df.merchant_click_seq.apply(
            lambda x: " ".join(list(x))
        )

        pred_df["order_of_preds_str"] = pred_df.merchant_click_seq.apply(
            get_ordered_preds_seq,
            model_dict=model_metadata,
            ix2word=ix2word,
        )

        pred_df["order_of_preds"] = pred_df.order_of_preds_str.apply(
            lambda x: [s.replace("merch", "") for s in x]
        )

    pred_df["rr"] = pred_df[["order_of_preds", "y_org"]].apply(get_rr, axis=1)
    return pred_df


def compute_mrr_model(model_metadata, X_test, y_test, target):
    """Computes the model MRR based on test data

    :param model_metadata:  a dictionary containing metadata about a model
    :param X_test:  a dataframe containing features specfic to the model being evaluated
    :param y_test: a dataframe of target labels
    :param target: the column which contains the actual labels for training data
    :return: the MRR score computed based on reciprocal rank of each feature instancd in X_test
    """
    pred_df = compute_mrr_input(model_metadata, X_test, y_test, target)
    mrr = pred_df.rr.mean()
    pred_df = None
    return mrr


def test_models(models, test_data, target, logger, dask_client=None):
    """

    :param models: a dictionary of models which need to be trained
    :param test_data: a dataframe containing all possible features (actual features used are specific to model) of test data
    :param target: the column which contains the actual labels for testing data
    :param logger: logger object passed from the calling module
    :param dask_client: dask client to use for testing, defaults to None
    :return:   a dictionary of models after computing eval metrics on the test data
    """
    for modelkey, model_metadata in models.items():

        model_metadata["MRR"] = 0
        features = model_metadata["features"]
        X_test = test_data[features]
        y_test = test_data[target]

        logger.info("Testing started for " + modelkey)
        # model_pipeline = get_prediction_pipeline(model_metadata)

        model_type = model_metadata["type"]

        if model_type == "classical":
            if dask_client:
                with joblib.parallel_backend("dask"):
                    model_metadata["MRR"] = compute_mrr_model(
                        model_metadata, X_test, y_test, target
                    )
            else:
                model_metadata["MRR"] = compute_mrr_model(
                    model_metadata, X_test, y_test, target
                )

        if model_type == "N2V":
            model_metadata["MRR"] = compute_mrr_model(
                model_metadata, X_test, y_test, target
            )

        if model_type == "SEQDL":
            model_metadata["MRR"] = compute_mrr_model(
                model_metadata, X_test, y_test, target
            )
        logger.info("Testing ended for " + modelkey)

    return models


def get_best_model(comparison_result, metric="fmeasure"):
    """
    Returns the model row in dataframe which has the best performance on a given metric

    :param comparison_result: a dataframe containing  evaluation info about various models
    :param metric: metric on which a comparison is be made, defaults to "fmeasure"
    :return: a dataframe row corresponding to the model which has the best score on the given metric
    """
    model_row = comparison_result[
        comparison_result[metric] == comparison_result[metric].max()
    ]
    return model_row.iloc[0]
