from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import tensorflow as tf
import cust_datagen as cd

# CHANGE THESE PATHS
CSV_TRAIN = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined_train"
CSV_VALIDATION = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined_val"

# dims = (30, 76)
# batch_size = 3
# group_size = batch_size*dims[0]
# df = pd.read_csv(CSV_TRAIN, sep=',', skiprows=range(1,1*group_size), nrows=group_size, converters=cd.converters)
# for c in cd.norm_cols:
#     df[c] = (df[c] - cd.norm_cols[c][0])/(cd.norm_cols[c][1]-cd.norm_cols[c][0])

# x = np.reshape(df.iloc[:,3:-1].values, [batch_size, *dims]) 
# y = df["Label"].apply(lambda s: 0 if s=="Benign" else 1)[dims[0]-1::dims[0]]
# y = keras.utils.to_categorical(y, num_classes=2)

# print(df["Label"])
# print(x.shape)
# print(y)
# exit(0)

# links to combined train/val data (IF YOU GET ANY ERRORS REDOWNLOAD THE DATA):
# https://drive.google.com/file/d/11yVYZgVJE2zgGkPPzuSOO06MqTNCRDVV/view?usp=sharing
# https://drive.google.com/file/d/1ZjtGgooqZ0qRd_10MSy7Ds93aS3Z4v7p/view?usp=sharing

inputs = layers.Input(shape=(76), name='in')

x = layers.Dense(64)(inputs)
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

model = keras.Model(inputs=inputs, outputs=outputs, name='ids_model')
# print(model.summary())

steps_per_epoch = 100
batch_size = 1024

gen = cd.IDSDataGeneratorBasic({"Benign": 0, "Malicious": 1}, CSV_TRAIN, (76), steps_per_epoch, batch_size=batch_size)
vgen = cd.IDSDataGeneratorBasic({"Benign": 0, "Malicious": 1}, CSV_VALIDATION, (76), steps_per_epoch//100, batch_size=batch_size)

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])

history = model.fit_generator(
    gen, epochs=500, validation_data=vgen,# workers=8
)

out = model.predict(vgen)
print(out)