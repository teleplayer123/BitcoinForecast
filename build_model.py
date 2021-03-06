from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import CuDNNLSTM, Dense, InputLayer, Flatten, SimpleRNN, Dropout, LSTM
import tensorflow as tf


def build_n_layer_model(n_nodes, n_layers, input_shape, n_out=2, drop_rate=0.2, batch_normalization=True,
                        loss="binary_crossentropy", opt="adam", metrics=["accuracy"],
                        activation="softmax"):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(LSTM(n_nodes, return_sequences=True, input_shape=input_shape))
        elif i == n_layers-1:
            model.add(LSTM(n_nodes))
        else:
            model.add(LSTM(n_nodes, return_sequences=True))
        model.add(Dropout(drop_rate))
        if batch_normalization:
            model.add(tf.keras.layers.BatchNormalization())
    model.add(Dense(n_out, activation=activation))
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

def cuda_lstm_model(input_shape):
    window_size = 128
    dropout = 0.2
    model = Sequential()
    model.add(CuDNNLSTM(window_size, return_sequences=True,
            input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(CuDNNLSTM(window_size, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(CuDNNLSTM(window_size, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(tf.keras.layers.BatchNormalization())
    
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(dropout))
    
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_model(units, input_shape, n_out=1, activation="softmax",
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(LSTM(units))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dense(n_out, activation=activation))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def build_model_v2(units, input_shape):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(LSTM(units))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def build_dense_model(n_nodes, n_hidden, input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Flatten("channels_first"))
    for _ in range(n_hidden):
        model.add(Dense(n_nodes, activation="relu"))
    model.add(Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate=1e-6)
    model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])
    return model

def simple_lstm_model(n_nodes, input_shape):
    model = Sequential()
    model.add(LSTM(n_nodes, activation="sigmoid", input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    return model

def simple_rnn_model(n_nodes, n_steps_out):
    model = Sequential()
    model.add(SimpleRNN(n_nodes, input_shape=[None, 1], return_sequences=True))
    model.add(SimpleRNN(n_nodes, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(Dense(n_steps_out)))
    model.compile(loss="mse", optimizer="adam")
    return model
