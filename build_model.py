import numpy as np 
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Activation
from tensorflow.keras.models import Sequential

def build_n_layer_model(n_nodes, n_out, n_layers, in_shape, drop_rate=0.2,
                loss="sparse_categorical_crossentropy", opt="adam", metrics=["accuracy"], batch_normalization=True):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(LSTM(n_nodes, input_shape=in_shape, return_sequences=True))
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
    model.compile(optimizer=opt, loss=loss)
    return model

def build_model(units):
    model = Sequential()
    model.add(LSTM(units, input_shape=(None, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(units))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    return model


def create_model(units):
    model = Sequential()
    model.add(LSTM(units, activation="sigmoid", input_shape=(None, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
