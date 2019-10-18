from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import tensorflow as tf
import cust_datagen as cd
import attention_models as am
import utils
import h5py

# CHANGE THESE PATHS
CSV_TRAIN = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined_train_seq"
CSV_VALIDATION = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined_val_seq"

H5_COMBINED = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined.hdf5"
combined_h5 = h5py.File(H5_COMBINED, 'r')

# HDF5 link
# https://drive.google.com/file/d/1GeCiqkdh3aqY8MUJztrvhHy-H0Ghd5Jo/view?usp=sharing

# print("begin")
# dims = (30, 76)
# batch_size = 2048
# group_size = batch_size*dims[0]
# df = pd.read_csv(CSV_TRAIN, sep=',', skiprows=range(1, 100*group_size), nrows=group_size, converters=cd.converters)
# # for c in cd.norm_cols:
# #     df[c] = (df[c] - cd.norm_cols[c][0])/(cd.norm_cols[c][1]-cd.norm_cols[c][0])

# x = np.reshape(df.iloc[:,3:-1].values, [batch_size, *dims]) 
# y = df["Label"].apply(lambda s: 0 if s=="Benign" else 1)[dims[0]-1::dims[0]]

# print("complete")
# exit(0)

attention_window = 128
columns = list(range(3,79))
dims = (attention_window, len(columns))

inputs = layers.Input(shape=dims, name='in')

at = layers.Flatten()(inputs)

at = layers.Dense(256)(at)
at = layers.Dropout(0.4)(at)
at = layers.Activation("relu")(at)
# at = layers.BatchNormalization()(at)

at = layers.Dense(128)(at)
at = layers.Dropout(0.4)(at)
at = layers.Activation("relu")(at)
# at = layers.BatchNormalization()(at)

at = layers.Dense(128)(at)
at = layers.Dropout(0.4)(at)
at = layers.Activation("relu")(at)
# at = layers.BatchNormalization()(at)

at = layers.Dense(64)(at)
at = layers.Dropout(0.4)(at)
at = layers.Activation("relu")(at)
# at = layers.BatchNormalization()(at)

at = layers.Dense(dims[0])(at)
# at = layers.Activation("relu")(at)

at = am.WeightedAverageAttention()([inputs, at])
at = layers.Activation("relu")(at)

x = layers.Dense(64)(at)
x = layers.Dropout(0.4)(x)
x = layers.Activation("relu")(x)

x = layers.Dense(64)(x)
x = layers.Dropout(0.4)(x)
x = layers.Activation("relu")(x)

x = layers.Dense(64)(x)
x = layers.Dropout(0.4)(x)
x = layers.Activation("relu")(x)

outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='ids_model')

batch_size = 2048

# steps_per_epoch = 100
# max_val_steps = (utils.rawcount(CSV_VALIDATION) - 1)//(batch_size*dims[0])
# max_train_steps = (utils.rawcount(CSV_TRAIN) - 1)//(batch_size*dims[0])

# gen  = cd.IDSDataGeneratorAttention({"Benign": 0, "Malicious": 1}, CSV_TRAIN,      dims, max_train_steps, batch_size=batch_size)
# vgen = cd.IDSDataGeneratorAttention({"Benign": 0, "Malicious": 1}, CSV_VALIDATION, dims, max_val_steps,   batch_size=batch_size)

gen, vgen = cd.IDSDataGeneratorAttentionH5.create_data_generators(combined_h5, attention_window, columns, 0.2, batch_size=batch_size)

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=0.001),
              metrics=['binary_accuracy', utils.true_positive_rate, utils.false_positive_rate])

history = model.fit_generator(
    gen, epochs=100, validation_data=vgen#, workers=8, use_multiprocessing=True
)

# print(model.predict(vgen))