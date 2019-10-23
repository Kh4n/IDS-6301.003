from tensorflow.keras import layers
from tensorflow.keras import regularizers as reg
import numpy as np
import pandas as pd
import tensorflow as tf
import cust_datagen as cd
import attention_models as am
import utils
import h5py
import os, shutil

logdir_name = "logs_attention"
if os.path.isdir(logdir_name):
    shutil.rmtree(logdir_name)

# CHANGE THESE PATHS
CSV_TRAIN = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined_train_seq"
CSV_VALIDATION = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined_val_seq"

# HDF5 link: https://drive.google.com/file/d/1GeCiqkdh3aqY8MUJztrvhHy-H0Ghd5Jo/view?usp=sharing
H5_COMBINED = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined.hdf5"
combined_h5 = h5py.File(H5_COMBINED, 'r')

attention_window = 8
dstport_embedding = 8
protocol_embedding = 3
columns = list(range(3,79))
useless = [i for i,n in enumerate(combined_h5["minmaxes"]) if n[1] - n[0] == 0]
columns = [k for k in columns if k not in useless]
print(columns, useless)

dims = (attention_window, len(columns))

in_dstport = layers.Input(shape=(attention_window))
dstport = layers.Embedding(65535+1, dstport_embedding, input_length=attention_window)(in_dstport)

in_protocol = layers.Input(shape=(attention_window))
protocol = layers.Embedding(17+1, protocol_embedding, input_length=attention_window)(in_protocol)

inputs = layers.Input(shape=dims)

at = layers.Concatenate()([protocol, dstport, inputs])
at = layers.TimeDistributed(layers.Dense(32), input_shape=(attention_window, len(columns)+dstport_embedding+protocol_embedding))(at)
at = layers.TimeDistributed(layers.BatchNormalization())(at)
at = layers.Activation("relu")(at)

# at = layers.TimeDistributed(layers.Dense(32))(at)
# at = layers.TimeDistributed(layers.BatchNormalization())(at)
# at = layers.Activation("relu")(at)

# at = layers.TimeDistributed(layers.Dense(32))(at)
# at = layers.TimeDistributed(layers.BatchNormalization())(at)
# at = layers.Activation("relu")(at)


at = layers.TimeDistributed(layers.Dense(1))(at)
at = layers.Activation("relu")(at)


at = am.WeightedAverageAttention()([inputs, at])


x = layers.Dense(64)(at)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(64)(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(64)(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.4)(x)


outputs = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=[in_protocol, in_dstport, inputs], outputs=outputs, name='ids_model')
print(model.summary())

batch_size = 128

# steps_per_epoch = 100
# max_val_steps = (utils.rawcount(CSV_VALIDATION) - 1)//(batch_size*dims[0])
# max_train_steps = (utils.rawcount(CSV_TRAIN) - 1)//(batch_size*dims[0])

# gen  = cd.IDSDataGeneratorAttention({"Benign": 0, "Malicious": 1}, CSV_TRAIN,      dims, max_train_steps, batch_size=batch_size)
# vgen = cd.IDSDataGeneratorAttention({"Benign": 0, "Malicious": 1}, CSV_VALIDATION, dims, max_val_steps,   batch_size=batch_size)

data = combined_h5["combined"][:]
gen, vgen = cd.IDSDataGeneratorAttentionH5.create_data_generators(data, combined_h5, attention_window, columns, 0.2, batch_size=batch_size)

sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1.)
rmsp = tf.keras.optimizers.RMSprop(lr=0.001)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['binary_accuracy', utils.true_positive_rate, utils.false_positive_rate])

history = model.fit_generator(
    gen, epochs=3000, validation_data=vgen, shuffle=False,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir=logdir_name)], workers=8, use_multiprocessing=True
)

# with open("OUT", "a+") as f:
#     with np.printoptions(threshold=np.inf):
#         f.write(str(vgen[0]))
#         f.write("\n\n\n\n\n\n")
#         f.write(str(model.predict(vgen, steps=1)))
# print(model.predict(vgen, steps=1))