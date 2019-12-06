from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import tensorflow as tf
import cust_datagen as cd
import utils
import h5py
import os, shutil
import deprecated as dep

logdir_name = "logs_dnn_reduced"
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
# useless = [i for i,n in enumerate(combined_h5["minmaxes"]) if n[1] - n[0] == 0]
# columns = [k for k in columns if k not in useless]
columns = dep.indices_to_use
print(columns)
dims = (len(columns))

in_dstport = layers.Input(shape=(1))
dstport = layers.Embedding(65535+1, dstport_embedding, input_length=1)(in_dstport)
dstport = layers.Flatten()(dstport)

in_protocol = layers.Input(shape=(1))
protocol = layers.Embedding(17+1, protocol_embedding, input_length=1)(in_protocol)
protocol = layers.Flatten()(protocol)

inputs = layers.Input(shape=dims, name='in')

x = layers.Concatenate()([protocol, dstport, inputs])
x = layers.Dense(64)(x)
# x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(64)(x)
# x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(64)(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=[in_protocol, in_dstport, inputs], outputs=outputs, name='ids_model')
print(model.summary())

batch_size = 1024


data = combined_h5["combined"][:]
gen, vgen = cd.IDSDataGeneratorBasic.create_data_generators(data, combined_h5, columns, 0.2, batch_size=batch_size)

sgd = tf.keras.optimizers.SGD(lr=1, decay=1e-6, momentum=0.9, nesterov=True)
rmsp = tf.keras.optimizers.RMSprop(lr=0.01)
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="mse",
              optimizer=adam,
              metrics=['binary_accuracy', utils.true_positive_rate, utils.false_positive_rate])

history = model.fit_generator(
    gen, epochs=500, validation_data=vgen, shuffle=True,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir=logdir_name)], workers=8, use_multiprocessing=True
)

# out = model.predict(vgen)
# print(out)