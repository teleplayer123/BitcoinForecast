import numpy as np 
import matplotlib.pyplot as plt

def binary_classification(prev, forecast):
    if prev >= forecast:
        return 0
    else:
        return 1

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

def train_test_split(seq_data, test_ratio):
    test_size = int(len(seq_data) * test_ratio)
    X_train = seq_data[:-test_size, :-1]
    y_train = seq_data[:-test_size, -1]
    X_test = seq_data[-test_size:, :-1]
    y_test = seq_data[-test_size:, -1]
    return X_train, X_test, y_train, y_test

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
    plt.figure(figsize=(18,10))
    plt.plot(res["val_loss"])
    plt.plot(res["loss"])
    plt.title("Loss")
    plt.legend(["val_loss", "loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    plt.figure(figsize=(18,10))
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