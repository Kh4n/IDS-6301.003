from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers as reg
import numpy as np
import pandas as pd
import tensorflow as tf
import cust_datagen as cd
import utils
import h5py
import os, shutil

def relative_error(ytrue, ypred):
    return tf.reduce_mean((  (ytrue-ypred)/(ytrue+ypred) )**2)

logdir_name = "logs_dnn"
if os.path.isdir(logdir_name):
    shutil.rmtree(logdir_name)

# CHANGE THESE PATHS
paths = open("pathconfig.cfg","r").read().split("\n")
CSV_TRAIN = paths[0]
CSV_VALIDATION = paths[1]

H5_COMBINED = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined.hdf5"
combined_h5 = h5py.File(H5_COMBINED, 'r')

# links to combined train/val data (IF YOU GET ANY ERRORS REDOWNLOAD THE DATA):
# https://drive.google.com/file/d/11yVYZgVJE2zgGkPPzuSOO06MqTNCRDVV/view?usp=sharing
# https://drive.google.com/file/d/1ZjtGgooqZ0qRd_10MSy7Ds93aS3Z4v7p/view?usp=sharing

dstport_embedding = 15
protocol_embedding = 3
columns = list(range(3,79))
useless = [i for i,n in enumerate(combined_h5["minmaxes"]) if n[1] - n[0] == 0]
columns = [k for k in columns if k not in useless]
print(columns, useless)
dims = (len(columns))

# in_dstport = layers.Input(shape=(1))
# dstport = layers.Embedding(65535+1, dstport_embedding, input_length=1)(in_dstport)
# dstport = layers.Flatten()(dstport)

# in_protocol = layers.Input(shape=(1))
# protocol = layers.Embedding(17+1, protocol_embedding, input_length=1)(in_protocol)
# protocol = layers.Flatten()(protocol)

inputs = layers.Input(shape=dims, name='in')

# all_in = layers.Concatenate()([protocol, dstport, inputs])

x = layers.Activation("sigmoid")(inputs)
x = layers.Dense(140, kernel_regularizer=reg.l1(0.001))(x)
x = layers.Activation("relu")(x)

x = layers.Dense(35)(x)
x = layers.Activation("relu")(x)

x = layers.Dense(16)(x)
x = layers.Activation("relu")(x)

x = layers.Dense(16)(x)
x = layers.Activation("relu")(x)

x = layers.Dense(35)(x)
x = layers.Activation("relu")(x)

x = layers.Dense(dims)(x)
# outputs = layers.Activation("sigmoid")(x)
# outputs = utils.MinmaxDenorm(combined_h5["combined"][columns])(x)
outputs = x

model = keras.Model(inputs=[inputs], outputs=outputs, name='ids_model')
print(model.summary())

batch_size = 1024

# steps_per_epoch = 100
# gen = cd.IDSDataGeneratorBasic({"Benign": 0, "Malicious": 1}, CSV_TRAIN, (76), steps_per_epoch, batch_size=batch_size)
# vgen = cd.IDSDataGeneratorBasic({"Benign": 0, "Malicious": 1}, CSV_VALIDATION, (76), steps_per_epoch//10, batch_size=batch_size)

data = combined_h5["combined"][:]
gen, vgen = cd.IDSDataGeneratorUnsupervised.create_data_generators(data, combined_h5, columns, 0.2, batch_size=batch_size)

sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="mse",
            #   optimizer=keras.optimizers.RMSprop(lr=0.001),
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit_generator(
    gen, epochs=5, validation_data=vgen, shuffle=False,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir=logdir_name)]#, workers=8, use_multiprocessing=True
)

with open("OUT", "w+") as f:
    with np.printoptions(threshold=np.inf):
        f.write(str(vgen[0]))
        f.write("\n\n\n\n\n\n")
        f.write(str(model.predict(vgen, steps=1)))
print(model.predict(vgen, steps=1))