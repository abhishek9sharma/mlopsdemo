import os
import pandas as pd

from mlcore.utils import set_logger


def load_data_from_file_path(data_path, data_storage_type=".csv", logger=None):
    """Loads data given a file path

    :param data_path: file path of data file
    :param data_storage_type: file extension e.g .parquet, defaults to ".csv"
    try:
    """
    if logger is None:
        logger = set_logger("load_data_logger")
    df = None
    try:
        if data_storage_type == ".parquet":
            df = pd.read_parquet(data_path)
            logger.info(
                "Loaded data file {} in dataframe with shape {}".format(
                    data_path, df.shape
                )
            )
        if data_storage_type == ".csv":
            df = pd.read_csv(data_path,encoding= 'unicode_escape')
            logger.info(
                "Loaded data file {} in dataframe with shape {}".format(
                    data_path, df.shape
                )
            )
    except Exception as e:
        err = " Exception : {} occured".format(e)
        logger.info(err)
        raise ValueError(err)

    return df


def load_data(schema_name, data_storage_type=".csv", data_folder=None, logger=None):
    """Loads data based on schema name

    :param schema_name: Name of the schema that needs to be loaded
    :param data_storage_type: format in which the data source is present, defaults to ".parquet"
    :param data_folder: folder in which the data source is present, defaults to None
    :param logger: logger object of the calling module, defaults to None
    :return: A dataframe containing data loaded from requested resource, returns None in case of error
    """
    df = None
    try:

        if data_folder is None:
            data_folder = os.environ["DATA_FOLDER"]
        if logger is None:
            logger = set_logger("load_data_logger")
        data_path = os.path.join(data_folder, schema_name + data_storage_type)
        if data_storage_type == ".parquet":
            df = pd.read_parquet(data_path)
            logger.info(
                "Loaded schema {} in dataframe with shape {}".format(
                    schema_name, df.shape
                )
            )
        if data_storage_type == ".csv":
            df = pd.read_csv(data_path,encoding= 'unicode_escape')
            logger.info(
                "Loaded schema {} in dataframe with shape {}".format(
                    schema_name, df.shape
                )
            )
    except Exception as e:
        logger.info(" Exception : {} occured".format(e))

    return df


def save_data(
    data,
    schema_name,
    data_storage_type=".csv",
    data_folder=None,
    logger=None,
):
    """[summary]
    :param data: a pandas data frame containing data that needs to be stored
    :param schema_name: Name of the schema that needs to be loaded
    :param data_storage_type: format in which the data source is present, defaults to ".parquet"
    :param data_folder: folder in which the data needs to be stored
    :param logger: logger object of the calling module, defaults to None
    :return: path of the data file
    """
    df = None
    try:

        if data_folder is None:
            data_folder = os.environ["DATA_FOLDER"]
        if logger is None:
            logger = set_logger("save_data_logger")
        data_path = os.path.join(data_folder, schema_name + data_storage_type)
        if data_storage_type == ".parquet":
            data.to_parquet(data_path)
            logger.info(
                "Saved schema {} in dataframe with shape {} at path {}".format(
                    schema_name, data.shape, data_path
                )
            )
        if data_storage_type == ".csv":
            data.to_csv(data_path, index=False)
            logger.info(
                "Saved schema {} in dataframe with shape {} at path {}".format(
                    schema_name, data.shape, data_path
                )
            )
    except Exception as e:
        logger.info(" Exception : {} occured".format(e))

    return data_path
