from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cust_datagen

CSV_COMBINED = <ENTER PATH TO combined.out HERE>

inputs = keras.Input(shape=(76), name='in')
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(2, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='ids_model')

gen = cust_datagen.IDSDataGenerator({"Benign": 0, "Malicious": 1}, 8000000, CSV_COMBINED, batch_size=256)

model.compile(loss='mse',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

history = model.fit_generator(gen,
                    epochs=5, steps_per_epoch=10)