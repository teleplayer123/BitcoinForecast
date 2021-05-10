import numpy as np 
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Activation, InputLayer, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy

def build_n_layer_model(n_nodes, n_layers, input_shape, drop_rate=0.2, batch_normalization=True,
                        loss="binary_crossentropy", opt="adam", metrics=["accuracy"], activ="softmax"):
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
    model.add(Dense(2, activation=activ))
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

def build_model(units, input_shape):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(units))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_model_v2(units, input_shape):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(units))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_dense_model(n_nodes, n_hidden, input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Flatten("channels_first"))
    for _ in range(n_hidden):
        model.add(Dense(n_nodes, activation="relu"))
    model.add(Dense(1))
    opt = Adam(learning_rate=1e-6)
    model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])
    return model

def simple_model(n_nodes, input_shape):
    model = Sequential()
    model.add(LSTM(n_nodes, activation="sigmoid", input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    return model