from collections import deque
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
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

def split_seq(df, test_ratio):
    np.random.seed(42)
    test_size = int(len(df) * test_ratio)
    indices = np.random.permutation(len(df))
    test_indices = indices[-test_size:]
    train_indices = indices[:-test_size]
    X, y = df.iloc[train_indices], df.iloc[test_indices]
    return X, y

def process_seq(seq):
    data = []
    X = []
    y = []
    seq = seq.drop("Forecast", 1)
    for col in seq.columns: 
        if col != "Class":
            seq[col] = seq[col].pct_change()
            seq.dropna(inplace=True)
            seq[col] = scale(seq[col].values)
    seq.dropna(inplace=True)
    seqs = deque(maxlen=SEQ_LEN)
    for val in seq.values:
        row = [v for v in val[:-1]]
        seqs.append(row)
        if len(seqs) == SEQ_LEN:
            data.append([np.array(row), val[-1]])
    for s, i in data:
        X.append(s)
        y.append(i)
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]
    return np.array(X), np.array(y)
        
df = pd.read_csv("data/BTC-USD.csv", names=FEATURE_COLUMNS)
df.set_index("Time", inplace=True)
df = df[["Close", "Volume"]]
btc_df = df.copy()
btc_df["Forecast"] = btc_df["Close"].shift(-FORECAST_STEP)
btc_df["Class"] = list(map(binary_classification, btc_df["Close"], btc_df["Forecast"]))

X, y = split_seq(btc_df, 0.2)


X_train, y_train = process_seq(X)
X_test, y_test = process_seq(y)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = build_model(128, 32, 3, in_shape=(X_train.shape[1:]))
model.summary()
res = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test))
print(res.history)
