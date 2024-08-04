import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 
import pickle
from time import time
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from build_model import build_n_layer_model
from utils import preprocess_univariate_data, visualize_results, visualize_loss

SEQ_LEN = 6
FORECAST_STEP = 3
BATCH_SIZE = 12
EPOCHS = 200
   
   
scaler = MinMaxScaler()

df = pd.read_csv("data/btc-usd-yf.csv")
df.set_index("Date", inplace=True)
btc_df = df.copy()
btc_df.dropna(inplace=True)
data = scaler.fit_transform(btc_df[["Close"]])

X_train, y_train, X_test, y_test = preprocess_univariate_data(data, SEQ_LEN, FORECAST_STEP, 0.05)
print("X_train shape:  ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

SHAPE = [None, 1]
FEATURES = 1
BIDIRECTIONAL = True
UNITS = 128
LAYERS = 6
MODEL_NAME = build_n_layer_model.__name__
TEST_NAME = f"{MODEL_NAME}-UNITS{UNITS}-LAYERS{LAYERS}-SHAPE{SHAPE[0]}_{SHAPE[1]}-FEATS{FEATURES}-Bidir{BIDIRECTIONAL}"
NAME = f"Model-{MODEL_NAME}-SEQLEN-{SEQ_LEN}-FORECAST-{FORECAST_STEP}-BATCH-{BATCH_SIZE}-EPOCHS-{EPOCHS}-Timestamp-{int(time())}"

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


X_test_path = f"testing/{NAME}/X-test-{TEST_NAME}-{int(time())}.bin"
y_test_path = f"testing/{NAME}/y-test-{TEST_NAME}-{int(time())}.bin"


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

model = build_n_layer_model(UNITS, LAYERS, input_shape=SHAPE, n_out=FEATURES, bidirectional=BIDIRECTIONAL, loss="mse", metrics=None, activation=None)
model.summary()
model.save_weights(checkpoint_path.format(epoch=0))
res = model.fit(X_train, y_train, epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[tensorboard, cp_callback])
print(res.history)
visualize_loss(res)
model.save(f"models/{NAME}.h5")