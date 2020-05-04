import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump
from preprocess import prep_data
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error

df = pd.read_csv("fish_participant.csv").set_index("Species")

print(df.head)
print(df.dtypes)

X, y = prep_data(df)

lr = LinearRegression()
print(cross_validate(lr, X,y, scoring ="neg_mean_squared_error"))
# lr.fit(X, y)


# df_holdout = pd.read_csv("fish_holdout_demo.csv").set_index("Species")
# X_hold, y_true = prep_data(df_holdout)

# y_predict = lr.predict(X_hold)
# print([y_true, y_predict ])

# print(mean_squared_error(y_true, y_predict))


# dump(lr, "reg.joblib")