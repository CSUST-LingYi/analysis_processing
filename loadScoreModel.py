import joblib as jl
import pandas as pd


def predict_cluster(test_arr):
    test_arr = pd.DataFrame(test_arr)
    try:
        rf = jl.load('./model/score_RF_model.pkl')
        cluster = rf.predict(test_arr)
    except ValueError:
        return -1
    else:
        return cluster
