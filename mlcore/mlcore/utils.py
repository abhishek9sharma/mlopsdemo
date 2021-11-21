import logging
import os
from datetime import datetime


def get_currtime_str():
    """Gets the current time in a particular format

    :return: a string representation of current timestamp
    """

    timestampformat = "%Y%m%d__%H%M%S"
    currtime_str = str(datetime.now().strftime(timestampformat))
    return currtime_str


def set_logger(identifier, logfolder=None, ignore_ts=None):
    """Creates a logger object

    :param identifier: name of the log file identifier
    :param logfolder: folder name where log file will be created
    :param ignore_ts: boolan flag to signal if timestamp should be included in log file nams
    :return: returns a logger object
    """
    if logfolder is None:
        logfolder = os.environ["LOG_FOLDER"]

    if ignore_ts:
        uniqfilename = os.path.join(logfolder, identifier + ".log")

    else:
        uniqfilename = os.path.join(
            logfolder, identifier + "_" + get_currtime_str() + ".log"
        )

    logger = logging.getLogger(str(uniqfilename))
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logfilepath = uniqfilename
    file_handler = logging.FileHandler(logfilepath)
    console_handler = logging.StreamHandler()

    logformat = logging.Formatter("%(asctime)s:%(message)s")
    file_handler.setFormatter(logformat)
    console_handler.setFormatter(logformat)

    # console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
