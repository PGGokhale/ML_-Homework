import numpy as np
import pandas as pd
def prep_data(df):

    X= pd.get_dummies(
    df[[column for column in df.columns if (column != "Weight") ]], drop_first = True
    ).values

     
    y = df["Weight"].values

    return X, y