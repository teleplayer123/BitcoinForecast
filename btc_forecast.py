import numpy as np 
from sklearn.preprocessing import scale, MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 
import pickle
from time import time
import os
import inspect
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from build_model import build_model, build_dense_model, build_n_layer_model, simple_rnn_model, build_model_v2
from utils import binary_classification, preprocess_data, visualize_results, visualize_loss, train_test_split

FEATURE_COLUMNS = ["Time", "Low", "High", "Open", "Close", "Volume"]
SEQ_LEN = 60
FORECAST_STEP = 3
BATCH_SIZE = 64
EPOCHS = 20

        
df = pd.read_csv("data/BTC-USD.csv", names=FEATURE_COLUMNS)
df.set_index("Time", inplace=True)
btc_df = df.copy()
btc_df["Forecast"] = btc_df["Close"].shift(-FORECAST_STEP)
btc_df["Class"] = list(map(binary_classification, btc_df["Close"], btc_df["Forecast"]))
btc_df.dropna(inplace=True)
btc_df["Close"] = scale(btc_df["Close"].values)
btc_df["Volume"] = scale(btc_df["Volume"].values)

X_train, y_train, X_test, y_test = train_test_split(btc_df[["Close", "Class"]], SEQ_LEN, 0.2)
print("X_train shape:  ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

UNITS = 128
LAYERS = 3
MODEL_NAME = build_n_layer_model.__name__
NAME = f"RNN-BTC-Model-{MODEL_NAME}_UNITS_{UNITS}_LAYERS_{LAYERS}_SHAPE_{X_train.shape[1]}x{X_train.shape[2]}-SEQLEN-{SEQ_LEN}-FORECASTSTEP-{FORECAST_STEP}-BATCHSIZE-{BATCH_SIZE}-EPOCHS-{EPOCHS}-Timestamp-{time()}"
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

model = build_n_layer_model(UNITS, LAYERS, input_shape=(X_train.shape[1:]))
model.summary()
model.save_weights(checkpoint_path.format(epoch=0))
res = model.fit(X_train, y_train, epochs=EPOCHS,
                validation_split=0.1, callbacks=[tensorboard, cp_callback])
print(res.history)
visualize_results(res)
model.save(f"models/{NAME}")