import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 


FORECAST_STEP = 3

def binary_classification(prev, forecast):
    if prev >= forecast:
        return 0
    else:
        return 1

df = pd.read_csv("data/BTC-USD.csv")
btc_df = df.set_index([df["Date"]])[["Close"]]
btc_df["Forecast"] = btc_df["Close"].shift(-FORECAST_STEP)
print(btc_df.head())
