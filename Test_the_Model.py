### WE WRITE THIS ###
from sklearn.metrics import mean_squared_error
import pandas as pd    
from predict import predict_from_csv

ho_predictions = predict_from_csv("fish.csv")
ho_truth = pd.read_csv("fish.csv")["Weight"].values
ho_mse = mean_squared_error(ho_truth, ho_predictions)
print(ho_mse)
######