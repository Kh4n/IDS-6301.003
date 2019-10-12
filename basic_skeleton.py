from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cust_datagen

# CHANGE THESE PATHS
CSV_COMBINED = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined_train"
CSV_VALIDATION = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined_val"

# links to combined train/val data:
# https://drive.google.com/file/d/11yVYZgVJE2zgGkPPzuSOO06MqTNCRDVV/view?usp=sharing
# https://drive.google.com/file/d/1nEVzJdtVZUbNzBmgLRCm6b5E7yOoM0Kl/view?usp=sharing

inputs = keras.Input(shape=(76), name='in')
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(2, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='ids_model')

# I have set these very high to test. If you get memory errors, reduce them
steps_per_epoch = 1000
batch_size = 1024

gen = cust_datagen.IDSDataGenerator({"Benign": 0, "Malicious": 1}, CSV_COMBINED, (76), steps_per_epoch, batch_size=batch_size)
vgen = cust_datagen.IDSDataGenerator({"Benign": 0, "Malicious": 1}, CSV_VALIDATION, (76), steps_per_epoch, batch_size=batch_size)

model.compile(loss='mse',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

history = model.fit_generator(
    gen, epochs=5, validation_data=vgen, workers=12
)