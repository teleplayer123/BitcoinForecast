import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import datetime

from utils import binary_classification

FORECAST_STEP = 3

scaler = MinMaxScaler()

df = pd.read_csv("data/btc-usd-yf.csv")
df.set_index("Date", inplace=True)
df.set_index(pd.to_datetime(df.index), inplace=True)
df.fillna(method="ffill", inplace=True)
#i = df.loc[pd.to_datetime(datetime.date(2020, 10, 12))]
#print(i)

main_df = df.copy()
main_df = main_df[["Close"]]
main_df["Forecast"] = main_df["Close"].shift(-FORECAST_STEP)
main_df["Class"] = list(map(binary_classification, main_df["Close"], main_df["Forecast"]))
scaled_data = scaler.fit_transform(main_df[["Close", "Class"]])

#TODO: train data should not include NaN values created by shift -forecast