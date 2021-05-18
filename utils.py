from collections import deque
import numpy as np 
import matplotlib.pyplot as plt

def binary_classification(prev, forecast):
    bullish = 1  #price increase
    bearish = 0  #price decrease
    if prev >= forecast:
        return bearish
    else:
        return bullish

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

def preprocess_sequences(df, seq_len):
    sequential_data = []
    sequences = deque(maxlen=seq_len)
    for val in df.values:
        seq = [v for v in val[:-1]]
        sequences.append(seq)
        if len(sequences) == seq_len:
            sequential_data.append([np.array(sequences), val[-1]])
    #even out bulls(1) and bears(0) to avoid learning based on trend
    bullish, bearish = [], []
    for s, i in sequential_data:
        if i == 1:
            bullish.append([s, i])
        elif i == 0:
            bearish.append([s, i])
        else:
            raise ValueError(f"Error processing data by class: expected either 0 or 1, but got {i}")
    min_len = min(len(bullish), len(bearish))
    bullish = bullish[:min_len]
    bearish = bearish[:min_len]
    sequential_data = bullish + bearish
    np.random.shuffle(sequential_data)
    xs, ys = [], []
    for x, y in sequential_data:
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def seq_split_train_test(seq_data, n_steps, test_ratio):
    test_size = int(len(seq_data) * test_ratio)
    X_train = seq_data[:-test_size, :-n_steps]
    y_train = seq_data[:-test_size, -n_steps:]
    X_test = seq_data[-test_size:, :-n_steps]
    y_test = seq_data[-test_size:, -n_steps:]
    return X_train, X_test, y_train, y_test


def train_test_split(data, seq_len, test_ratio):
    test_size = int(len(data) * test_ratio)
    indices = sorted(data.index.values)
    test_split = indices[-test_size]
    test_data = data[(data.index >= test_split)]
    train_data = data[(data.index < test_split)]
    X_train, y_train = preprocess_sequences(train_data, seq_len)
    X_test, y_test = preprocess_sequences(test_data, seq_len)
    return X_train, y_train, X_test, y_test


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

def visualize_loss(results):
    res = results.history
    plt.figure(figsize=(12,4))
    plt.plot(res["val_loss"])
    plt.plot(res["loss"])
    plt.title("Loss")
    plt.legend(["val_loss", "loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()