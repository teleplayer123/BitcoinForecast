from collections import defaultdict
import numpy as np 
import os
import pickle
from sklearn.metrics import mean_squared_error
import tensorflow as tf 

from build_model import build_n_layer_model


UNITS = 128
SHAPE = [None, 1]
LAYERS = 6
BATCH_SIZE = 12
FEATURES = 1
BIDIRECTIONAL = True
ACTIVATION = None
LOSS = "mse"
METRICS = None

DIRNAME = "Model-build_n_layer_model-SEQLEN-6-FORECAST-3-BATCH-12-EPOCHS-200-Timestamp-1621774027"

model = build_n_layer_model(UNITS, LAYERS, input_shape=SHAPE, n_out=FEATURES, bidirectional=BIDIRECTIONAL, loss=LOSS, activation=ACTIVATION, metrics=METRICS)
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

checkpoint_dir = f"training/{DIRNAME}"
latest = tf.train.latest_checkpoint(checkpoint_dir)

#loss, acc = model.evaluate(X_test, y_test)

model.load_weights(latest)
loss = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print("Loss: {:.4f}".format(loss))

print(np.shape(y_test))
print(np.shape(y_test[:, -1]))

y_pred = model.predict(X_test, batch_size=BATCH_SIZE)

mse = mean_squared_error(y_test[:, -1], y_pred)
print(f"Prediction Loss: {mse}")