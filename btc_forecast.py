import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 
import pickle
from time import time
import os
from tensorflow.keras.callbacks import ModelCheckpoint

from build_model import build_model, build_dense_model, build_n_layer_model

FEATURE_COLUMNS = ["Time", "Low", "High", "Open", "Close", "Volume"]
SEQ_LEN = 60
FORECAST_STEP = 10
BATCH_SIZE = 64
EPOCHS = 16
NAME = f"RNN-BTC-Model-N_Layer-SEQ-{SEQ_LEN}-PRED-{FORECAST_STEP}-Timestamp-{time()}"

if not os.path.exists("models"):
    os.mkdir("models")
if not os.path.exists("training"):
    os.mkdir("training")
if not os.path.exists("testing"):
    os.mkdir("testing")


def binary_classification(prev, forecast):
    if prev >= forecast:
        return 0
    else:
        return 1

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

def visualize_results(results):
    res = results.history
    plt.figure(figsize=(12,4))
    plt.plot(res["val_loss"])
    plt.plot(res["loss"])
    plt.title("Loss")
    plt.legend(["val_loss", "loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    plt.figure(figsize=(12,4))
    plt.plot(res["val_accuracy"])
    plt.plot(res["accuracy"])
    plt.title("Accuracy")
    plt.legend(["val_accuracy", "accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

scaler = MinMaxScaler()
        
df = pd.read_csv("data/BTC-USD.csv", names=FEATURE_COLUMNS)
df["Time"] = pd.to_datetime(df["Time"], unit="s").dt.date
df.set_index("Time", inplace=True)
df = df[["Close"]]
btc_df = df.copy()
btc_df["Forecast"] = btc_df["Close"].shift(-FORECAST_STEP)
btc_df["Class"] = list(map(binary_classification, btc_df["Close"], btc_df["Forecast"]))
data = scaler.fit_transform(btc_df[["Close", "Class"]])

X_train, y_train, X_test, y_test = preprocess_data(data, SEQ_LEN, 0.2)
print(X_train.shape)
print(y_train.shape)

X_test_path = f"testing/X-test-{NAME}.bin"
y_test_path = f"testing/y-test-{NAME}.bin"

with open(X_test_path, "wb") as fh:
    try:
        pickle.dump(X_test, fh)
    except pickle.PickleError as err:
        print("ERROR: {}".format(err))

with open(y_test_path, "wb") as fh:
    try:
        pickle.dump(y_test, fh)
    except pickle.PickleError as err:
        print("ERROR: {}".format(err))    


checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq=BATCH_SIZE*BATCH_SIZE)


model = build_n_layer_model(128, 3, input_shape=(X_train.shape[1:]))
model.summary()
model.save_weights(checkpoint_path.format(epoch=0))
res = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test), callbacks=[cp_callback])
print(res.history)
visualize_results(res)
model.save(f"models/{NAME}")
