from mlcore.train_eval_helper_reg import get_df_from_dict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def test_df_from_dict():

    dfdict = {
        'LinReg' : {'obj': LinearRegression(),
            'param_grid':{},
            "type": 'classical',
            "features":['f1', 'f2']

                },
    'RF' : {'obj': RandomForestRegressor(),
        'param_grid':{'max_depth': [10, 30, 60, 90]},
        "type": "classical",
        "features":['f1', 'f2','f3']

        },
        }

    df = get_df_from_dict(dfdict)
    dfdictret = df.to_dict(orient='list')
    assert dfdictret['index']==list(dfdict.keys())
    assert dfdictret['features'][0]==['f1', 'f2']
    assert dfdictret['features'][1]==['f1', 'f2', 'f3']

    assert df.shape==(2,5)