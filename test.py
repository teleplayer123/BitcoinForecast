from build_model import build_model, build_n_layer_model
import os
import pickle
import tensorflow as tf 

UNITS = 128
SHAPE = (59, 2)

model = build_model(UNITS, SHAPE)

test_sets = []
test_dirs = []
for dn in os.listdir("testing"):
    test_dirs.append(f"testing/{dn}")
for dn in test_dirs:
    for fn in os.listdir(dn):
        test_sets.append(os.path.join(dn,fn))
test_sets = sorted(test_sets)
print(test_sets)
X_test_path, y_test_path = test_sets[0], test_sets[1]

with open(X_test_path, "rb") as fh:
    X_test = pickle.loads(fh.read())
with open(y_test_path, "rb") as fh:
    y_test = pickle.loads(fh.read())

checkpoint_dir = "training/RNN-BTC-Model-buildmodel"
latest = tf.train.latest_checkpoint(checkpoint_dir)

#loss, acc = model.evaluate(X_test, y_test)

model.load_weights(latest)
loss, acc = model.evaluate(X_test, y_test, batch_size=64)
print("Loss: {:.4f}".format(loss))
print("Accuracy: {:5.2f}".format(100 * acc))