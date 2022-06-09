import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 
import pickle
from time import time
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from build_model import build_model, build_dense_model, build_n_layer_model, cuda_lstm_model
from utils import binary_classification, preprocess_data, visualize_results, visualize_loss, train_test_split

FEATURE_COLUMNS = ["Time", "Low", "High", "Open", "Close", "Volume"]
SEQ_LEN = 60
FORECAST_STEP = 30
BATCH_SIZE = 64
EPOCHS = 10
   
scaler = MinMaxScaler()

df = pd.read_csv("data/BTC-USD.csv", names=FEATURE_COLUMNS)
df.set_index("Time", inplace=True)
btc_df = df.copy()
btc_df["Forecast"] = btc_df["Close"].shift(-FORECAST_STEP)
btc_df["Class"] = list(map(binary_classification, btc_df["Close"], btc_df["Forecast"]))
btc_df.dropna(inplace=True)
btc_df["Close"] = scaler.fit_transform(btc_df[["Close"]])
btc_df["Volume"] = scaler.fit_transform(btc_df[["Volume"]])
btc_df.dropna(inplace=True)

X_train, y_train, X_test, y_test = preprocess_data(btc_df[["Close", "Volume", "Class"]], SEQ_LEN, 0.2)
print("X_train shape:  ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

SHAPE = f"{X_train.shape[1]}_{X_train.shape[2]}"
UNITS = 128
LAYERS = 3
MODEL_NAME = build_n_layer_model.__name__
NAME = f"Model-{MODEL_NAME}_UNITS_{UNITS}_LAYERS_{LAYERS}_SHAPE_{SHAPE}_BIDIRECTIONAL_Timestamp_{int(time())}"

if not os.path.exists("models"):
    os.mkdir("models")
if not os.path.exists("logs"):
    os.mkdir("logs")
if not os.path.exists("training"):
    os.mkdir("training")
if not os.path.exists(f"training/{NAME}"):
    os.mkdir(f"training/{NAME}")
if not os.path.exists("testing"):
    os.mkdir("testing")
if not os.path.exists(f"testing/{NAME}"):
    os.mkdir(f"testing/{NAME}")


X_test_path = f"testing/{NAME}/X-test-{int(time())}.bin"
y_test_path = f"testing/{NAME}/y-test-{int(time())}.bin"


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



cp_dir = "training/{}".format(NAME)
cp_fn = "cp-{epoch:04d}.ckpt"
checkpoint_path = os.path.join(cp_dir, cp_fn)
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq=BATCH_SIZE*BATCH_SIZE)
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

model = cuda_lstm_model(SEQ_LEN, X_train.shape[1:])
model.build((None, *X_train.shape[1:]))
model.summary()
model.save_weights(checkpoint_path.format(epoch=0))
res = model.fit(X_train, y_train, epochs=EPOCHS,
                validation_split=0.2, callbacks=[tensorboard, cp_callback])
print(res.history)
visualize_results(res)
model.save(f"models/{NAME}.h5")
