import numpy as np 
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import scale

def build_model(n_nodes, n_out, n_layers, input_shape, drop_rate=0.5,
                loss="mae", opt="adam", metrics=["accuracy"], batch_normalization=True):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(LSTM(n_nodes, input_shape=input_shape, return_sequences=True))
        elif i == n_layers-1:
            model.add(LSTM(n_nodes))
        else:
            model.add(LSTM(n_nodes, return_sequences=True))
        model.add(Dropout(drop_rate))
        if batch_normalization:
            model.add(BatchNormalization())
    model.add(Dense(n_out, activation="relu"))
    model.add(Dropout(drop_rate))
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

