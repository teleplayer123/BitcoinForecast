import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 
import pickle
from time import time
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from build_model import build_n_layer_model
from utils import preprocess_data, visualize_loss, visualize_results, binary_classification


SEQ_LEN = 6
FORECAST_STEP = 3
BATCH_SIZE = 12
EPOCHS = 100
MODEL = "build_n_layer_model_layers4"
DATASET = "btc_kaggle"
NAME = f"RNN-BTC-Model_{MODEL}_Dataset_{DATASET}-SEQ-{SEQ_LEN}-PRED-{FORECAST_STEP}-Timestamp-{time()}"
DIR_NAME = "-".join(NAME.split("-")[:3])


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

scaler = MinMaxScaler()
        
df = pd.read_csv("data/btc-kaggle.csv")
df.dropna(inplace=True)
df["Date"] = pd.to_datetime(df["Timestamp"], unit="s").dt.date
group_by_date = df.groupby("Date")
group_means = group_by_date["Close"].mean()
main_df = pd.DataFrame(group_means.values, columns=["Close"], index=group_means.index)

main_df["Forecast"] = main_df["Close"].shift(-FORECAST_STEP)
main_df["Class"] = list(map(binary_classification, main_df["Close"], main_df["Forecast"]))
main_df.dropna(inplace=True)
scaled_data = scaler.fit_transform(main_df[["Close", "Class"]])

X_train, X_test, y_train, y_test = preprocess_data(scaled_data, SEQ_LEN, 0.2)
#print(X_train_set.shape)
#print(y_train_set.shape)


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
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq=BATCH_SIZE**2*EPOCHS)
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

model = build_n_layer_model(128, 4, input_shape=(X_train.shape[1:]), metrics=None)
model.summary()
model.save_weights(checkpoint_path.format(epoch=0))
res = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                validation_split=0.1, callbacks=[tensorboard, cp_callback])
print(res.history)
visualize_loss(res)
model.save(f"models/{NAME}")

