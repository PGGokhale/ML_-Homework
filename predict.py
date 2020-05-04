### YOU WRITE THIS ###
from joblib import load
from preprocess import prep_data
import pandas as pd


def predict_from_csv(path_to_csv):

    df = pd.read_csv(path_to_csv)
    
    X, y = prep_data(df)

    reg = load("dt.joblib")
    
    predictions = reg.predict(X)

    return predictions

if __name__ == "__main__":
    predictions = predict_from_csv("fish_holdout_demo.csv")
    print(predictions)
######



