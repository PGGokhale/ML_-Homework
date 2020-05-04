import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from joblib import dump
from preprocess import prep_data
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

df = pd.read_csv("fish_participant.csv")

print(df.head)
print(df.dtypes)

X, y = prep_data(df)

dt = DecisionTreeRegressor()
cross_validate(dt, X,y, scoring ="neg_mean_squared_error",
              cv=KFold(random_state=123, shuffle=True))['test_score'].mean()
dt.fit(X,y)
# dump(dt, 'dt.joblib') 


df_holdout = pd.read_csv("fish_holdout_demo.csv")
X_hold, y_true = prep_data(df_holdout)

y_predict = dt.predict(X_hold)
print([y_true, y_predict ])

print(mean_squared_error(y_true, y_predict))


# dump(dt, "dt.joblib")