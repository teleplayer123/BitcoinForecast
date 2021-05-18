from collections import defaultdict
import numpy as np 
import os
import pickle
from tensorflow.keras.optimizers import Adam 
import tensorflow as tf 

from build_model import build_model, build_n_layer_model, build_model_v2


UNITS = 128
SHAPE = (60, 2)
LAYERS = 3

DIRNAME = "RNN-BTC-Model-buildnlayermodel_nodes128_lossSCC"

model = build_n_layer_model(UNITS, LAYERS, input_shape=SHAPE, loss="sparse_categorical_crossentropy",
                            activation="softmax", opt=Adam(learning_rate=0.001, decay=1e-6))

test_sets = defaultdict(list)
test_dirs = []
for dn in os.listdir("testing"):
    test_dirs.append(f"testing/{dn}")
for dn in test_dirs:
    for fn in sorted(os.listdir(dn)):
        test_sets[dn].append(os.path.join(dn,fn))

X_test_path, y_test_path = test_sets.get(f"testing/{DIRNAME}")[0], test_sets.get(f"testing/{DIRNAME}")[1]


with open(X_test_path, "rb") as fh:
    X_test = pickle.loads(fh.read())
with open(y_test_path, "rb") as fh:
    y_test = pickle.loads(fh.read())

checkpoint_dir = "training/RNN-BTC-Model-buildnlayermodel_nodes128_lossSCC"
latest = tf.train.latest_checkpoint(checkpoint_dir)

#loss, acc = model.evaluate(X_test, y_test)

model.load_weights(latest)
loss, acc = model.evaluate(X_test, y_test, batch_size=64)
print("Loss: {:.4f}".format(loss))
print("Accuracy: {:5.2f}".format(100 * acc))