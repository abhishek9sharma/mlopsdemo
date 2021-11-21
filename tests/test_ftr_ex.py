import pandas as pd
from mlcore.feature_extractor import extract_time_features, extract_user_features


def test_time_ftrs_extract():

    test_df = pd.DataFrame(["2021-02-03 23:47:27"], columns=["created_at"])
    extract_time_features(test_df)
    assert test_df["hour_of_day"].iloc[0] == 23
    assert test_df["day_of_week"].iloc[0] == "Wednesday"
    assert test_df["month_of_year"].iloc[0] == 2


def test_time_ftrs_extract():

    test_df = pd.DataFrame(
        [("2021-02-03 23:47:27" ,1)], columns=["created_at", "user_id"]
    )
    user_df = pd.DataFrame(
        [("2021-02-03 23:34:10.822", "2021-02-03 23:47:27", 1)],
        columns=["signup_datetime", "lifetime_first_purchase_datetime", "id"],
    )
    ftr_data = extract_user_features(test_df, user_df)
    assert  ftr_data['click_delta_signup'].iloc[0]==13.269633333333333
    assert  ftr_data['click_delta_first_purchase'].iloc[0]==0

