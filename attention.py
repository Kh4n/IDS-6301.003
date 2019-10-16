from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import tensorflow as tf
import cust_datagen as cd
import attention_models as am


# CHANGE THESE PATHS
CSV_TRAIN = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined_train_seq"
CSV_VALIDATION = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined_val_seq"

dims = (32, 76)

inputs = layers.Input(shape=dims, name='in')

at = layers.Flatten()(inputs)

at = layers.Dense(256)(at)
at = layers.Dropout(0.4)(at)
at = layers.Activation("sigmoid")(at)
# at = layers.BatchNormalization()(at)

at = layers.Dense(128)(at)
at = layers.Dropout(0.4)(at)
at = layers.Activation("sigmoid")(at)
# at = layers.BatchNormalization()(at)

at = layers.Dense(dims[0])(at)
# at = layers.Activation("sigmoid")(at)

at = am.WeightedAverageAttention()([inputs, at])

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

steps_per_epoch = 100
batch_size = 1024

gen = cd.IDSDataGeneratorAttention({"Benign": 0, "Malicious": 1}, CSV_TRAIN, dims, steps_per_epoch, batch_size=batch_size)
vgen = cd.IDSDataGeneratorAttention({"Benign": 0, "Malicious": 1}, CSV_VALIDATION, dims, steps_per_epoch//10, batch_size=batch_size)

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])

history = model.fit_generator(
    gen, epochs=500, validation_data=vgen#, workers=8
)