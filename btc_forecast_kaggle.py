import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

from utils import preprocess_data, visualize_loss
from build_model import simple_rnn_model


FORECAST_STEP = 10
SEQ_LEN = 60
scaler = MinMaxScaler()

df = pd.read_csv("data/btc-kaggle.csv")
df["Date"] = pd.to_datetime(df["Timestamp"], unit="s").dt.date
group_by_date = df.groupby("Date")
prices = group_by_date["Close"].mean()
test_data = prices[len(prices)-FORECAST_STEP:]
main_df = pd.DataFrame(prices[:len(prices)-FORECAST_STEP])
main_df["Forecast"] = main_df["Close"].shift(-FORECAST_STEP)

main_df = main_df[["Close", "Forecast"]]
scaled_data = scaler.fit_transform(main_df)

X_train, X_test, y_train, y_test = preprocess_data(scaled_data, SEQ_LEN, 0.2)

