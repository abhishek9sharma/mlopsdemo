import pandas as pd


def extract_time_features(df):
    """Extract some time based features based on info present in the dataframe

    :param df: pandas dataframe containing some time based information

    """
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["hour_of_day"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.day_name()
    df["month_of_year"] = df["created_at"].dt.month
    df["date"] = df["created_at"].dt.date


def extract_user_features(df, users_df):
    """Extract some time user features based on info present
    2 dataframes

    :param df: contains click information of various users
    :param users_df: contains user specific information
    :return: a dataframe containing new features and some existing information
    """

    df["created_at"] = pd.to_datetime(df["created_at"])
    df = pd.merge(df, users_df, left_on="user_id", right_on="id")
    df["click_delta_signup"] = (
        df.created_at - pd.to_datetime(df.signup_datetime)
    ) / pd.Timedelta(minutes=1)
    df["click_delta_first_purchase"] = (
        df.created_at - pd.to_datetime(df.lifetime_first_purchase_datetime)
    ) / pd.Timedelta(minutes=1)
    return df
