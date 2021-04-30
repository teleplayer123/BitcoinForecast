import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 

from build_model import build_model

FEATURE_COLUMNS = ["Time", "Low", "High", "Open", "Close", "Volume"]
SEQ_LEN = 60
FORECAST_STEP = 3
BATCH_SIZE = 64
EPOCHS = 10

def binary_classification(prev, forecast):
    if prev >= forecast:
        return 0
    else:
        return 1

def train_test_split(df, test_ratio):
    np.random.seed(42)
    test_size = int(len(df) * test_ratio)
    indices = np.random.permutation(len(df))
    test_indices = indices[-test_size:]
    train_indices = indices[:-test_size]
    X, y = df.iloc[train_indices], df.iloc[test_indices]
    return X, y

def seq_split(data, seq_len):
    seq_data = []
    seqs = []
    i = 0
    while i < len(data)-seq_len:
        if len(seqs) == seq_len:
            seq_data.append(seqs)
        seqs.append(data[i:i+seq_len])
        i += 1
    seq_data = sum(seq_data, [])
    return np.array(seq_data)

def preprocess_data(data, seq_len, test_ratio):
    seq_data = seq_split(data, seq_len)
    test_size = int(len(seq_data) * test_ratio)
    X_train = seq_data[:-test_size, :-1, :]
    y_train = seq_data[:-test_size, -1, :]
    X_test = seq_data[-test_size:, :-1, :]
    y_test = seq_data[-test_size:, -1, :]
    return X_train, y_train, X_test, y_test

        
df = pd.read_csv("data/BTC-USD.csv", names=FEATURE_COLUMNS)
df["Time"] = pd.to_datetime(df["Time"], unit="s").dt.date
df.set_index("Time", inplace=True)
df = df[["Close"]]
btc_df = df.copy()
btc_df["Forecast"] = btc_df["Close"].shift(-FORECAST_STEP)
btc_df["Class"] = list(map(binary_classification, btc_df["Close"], btc_df["Forecast"]))
data = btc_df[["Close", "Class"]].copy()

X_train, y_train, X_test, y_test = preprocess_data(data, SEQ_LEN, 0.2)
print(X_train.shape)
print(y_train.shape)

model = build_model(128, input_shape=(X_train.shape[1:]))
model.summary()
res = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test))
print(res.history)
