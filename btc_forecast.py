import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 
import pickle
from time import time
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from build_model import build_model, build_dense_model, build_n_layer_model, simple_model, build_model_v2

FEATURE_COLUMNS = ["Time", "Low", "High", "Open", "Close", "Volume"]
SEQ_LEN = 60
FORECAST_STEP = 30
BATCH_SIZE = 64
EPOCHS = 20
NAME = f"RNN-BTC-Model-buildmodelv2_nodes128_lossBCE-SEQ-{SEQ_LEN}-PRED-{FORECAST_STEP}-Timestamp-{time()}"
DIR_NAME = "-".join(NAME.split("-")[:4])


if not os.path.exists("models"):
    os.mkdir("models")
if not os.path.exists("logs"):
    os.mkdir("logs")
if not os.path.exists("training"):
    os.mkdir("training")
if not os.path.exists(f"training/{DIR_NAME}"):
    os.mkdir(f"training/{DIR_NAME}")
if not os.path.exists("testing"):
    os.mkdir("testing")
if not os.path.exists(f"testing/{DIR_NAME}"):
    os.mkdir(f"testing/{DIR_NAME}")


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
    X_train = seq_data[:-test_size, :-1]
    y_train = seq_data[:-test_size, -1]
    X_test = seq_data[-test_size:, :-1]
    y_test = seq_data[-test_size:, -1]
    return X_train, X_test, y_train, y_test

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
df.set_index("Time", inplace=True)
df = df[["Close"]]
btc_df = df.copy()
btc_df["Forecast"] = btc_df["Close"].shift(-FORECAST_STEP)
btc_df["Class"] = list(map(binary_classification, btc_df["Close"], btc_df["Forecast"]))
data = scaler.fit_transform(btc_df[["Close", "Class"]])

X_train_set, X_test, y_train_set, y_test = preprocess_data(data, SEQ_LEN, 0.2)
#print(X_train_set.shape)
#print(y_train_set.shape)

X_train, X_val, y_train, y_val = train_test_split(X_train_set, y_train_set, test_size=0.2, random_state=42)
#print(X_train.shape)
#print(y_train.shape)

X_test_path = f"testing/{DIR_NAME}/X-test-{NAME}.bin"
y_test_path = f"testing/{DIR_NAME}/y-test-{NAME}.bin"

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



cp_dir = "training/{}".format(DIR_NAME)
cp_fn = "cp-{epoch:04d}.ckpt"
checkpoint_path = os.path.join(cp_dir, cp_fn)
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq=BATCH_SIZE*BATCH_SIZE)
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

model = build_model_v2(128, input_shape=(X_train.shape[1:]))
model.summary()
model.save_weights(checkpoint_path.format(epoch=0))
res = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                validation_data=(X_val, y_val), callbacks=[tensorboard, cp_callback])
print(res.history)
visualize_results(res)
model.save(f"models/{NAME}")