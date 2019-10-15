from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import tensorflow as tf
import cust_datagen

def cust_accuracy(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    correct = tf.math.less(diff, 0.5)
    correct = tf.cast(correct, tf.float32)
    return tf.math.reduce_mean(correct)

# CHANGE THESE PATHS
CSV_TRAIN = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined_train"
CSV_VALIDATION = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined_val"

# links to combined train/val data (IF YOU GET ANY ERRORS REDOWNLOAD THE DATA):
# https://drive.google.com/file/d/11yVYZgVJE2zgGkPPzuSOO06MqTNCRDVV/view?usp=sharing
# https://drive.google.com/file/d/1ZjtGgooqZ0qRd_10MSy7Ds93aS3Z4v7p/view?usp=sharing

inputs = layers.Input(shape=(76), name='in')
x = layers.Dense(64, activation='relu', use_bias=False)(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dense(64, activation='relu', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(2, activation='softmax')(x)


model = keras.Model(inputs=inputs, outputs=outputs, name='ids_model')

# I have set these very high to test. If you get memory errors, reduce them
steps_per_epoch = 1000
batch_size = 1024

gen = cust_datagen.IDSDataGeneratorBasic({"Benign": 0, "Malicious": 1}, CSV_TRAIN, (76), steps_per_epoch, batch_size=batch_size)
vgen = cust_datagen.IDSDataGeneratorBasic({"Benign": 0, "Malicious": 1}, CSV_VALIDATION, (76), steps_per_epoch, batch_size=batch_size)

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

history = model.fit_generator(
    gen, epochs=5, validation_data=vgen, workers=12
)