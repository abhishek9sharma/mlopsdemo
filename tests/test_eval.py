from mlcore.train_eval_helper import get_df_from_dict, get_rr


def test_rr():

    assert round(get_rr(([1,2,3],2)),1)==0.5
    assert round(get_rr(([1,2,3],3)),2)==0.33
    assert round(get_rr(([1,2,3],1)),1)==1



def test_df_from_dict():

    dfdict = {
        1:{"merchant":2},
        2:{"merchant":4}

             }
    df = get_df_from_dict(dfdict, idxname='user')
    dfdictret = df.to_dict(orient='list')
    assert dfdictret['user']==list(dfdict.keys())
    assert dfdictret['merchant']==[v['merchant'] for k,v in dfdict.items()]
    assert df.shape ==(2,2)
    