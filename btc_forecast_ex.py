import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 
import pickle
from time import time
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from build_model import cuda_lstm_model
from utils import binary_classification, preprocess_df, visualize_results

FEATURE_COLUMNS = ["Time", "Low", "High", "Open", "Close", "Volume"]
SEQ_LEN = 60
FORECAST_STEP = 3
BATCH_SIZE = 64
EPOCHS = 10
   

df = pd.read_csv("data/BTC-USD.csv", names=FEATURE_COLUMNS)
df.set_index("Time", inplace=True)
btc_df = df[["Close", "Volume"]]
btc_df["Forecast"] = btc_df["Close"].shift(-FORECAST_STEP)
btc_df["Class"] = list(map(binary_classification, btc_df["Close"], btc_df["Forecast"]))
btc_df.dropna(inplace=True)

tms = sorted(btc_df.index.values)
last_5pct = sorted(btc_df.index.values)[-int(0.05*len(tms))]

valid_df = btc_df[(btc_df.index >= last_5pct)]
btc_df = btc_df[(btc_df.index < last_5pct)]

X_train, y_train = preprocess_df(btc_df, SEQ_LEN)
X_test, y_test = preprocess_df(valid_df, SEQ_LEN)


SHAPE = f"{np.array(X_train).shape[1]}_{np.array(X_train).shape[2]}"
UNITS = 128
LAYERS = 3
MODEL_NAME = cuda_lstm_model.__name__
NAME = f"Model-{MODEL_NAME}_UNITS_{UNITS}_LAYERS_{LAYERS}_SHAPE_{SHAPE}_BIDIRECTIONAL_Timestamp_{int(time())}"

cp_dir = "training/{}".format(NAME)
cp_fn = "cp-{epoch:04d}.ckpt"
checkpoint_path = os.path.join(cp_dir, cp_fn)
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq=BATCH_SIZE*BATCH_SIZE)
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

model = cuda_lstm_model(np.array(X_train).shape[1:])

res = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_data=(X_test, y_test), callbacks=[tensorboard, cp_callback])
score = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {score[0]}")
print(f"Accuracy: {score[1]}")
visualize_results(res)
