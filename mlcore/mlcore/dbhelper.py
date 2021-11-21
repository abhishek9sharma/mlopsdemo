import sqlite3
from sqlite3 import Error
import pandas as pd


def create_connection(db_path="../data/mldb.sqlite"):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
    except Error as e:
        raise ValueError(" error {} occured".format(str(e)))
    finally:
        if conn:
            conn.close()


def store_data(df, db_path="./data/mldb.sqlite", table_name=None):
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists="append")
    except Error as e:
        raise ValueError(" error {} occured".format(str(e)))
    finally:
        if conn:
            conn.close()


def overwrite_data(df, db_path="./data/mldb.sqlite", table_name=None):
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists="replace")
    except Error as e:
        raise ValueError(" error {} occured".format(str(e)))
    finally:
        if conn:
            conn.close()


def read_data(db_path="./data/mldb.sqlite", table_name=None, where_condition=None):
    df = None
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT * from {}".format(table_name)
        if where_condition:
            query = "SELECT * from {} where {}".format(table_name, where_condition)
            # print(query)
        df = pd.read_sql_query(query, conn)
    except Error as e:
        raise ValueError(" error {} occured".format(str(e)))
    finally:
        if conn:
            conn.close()
    return df
