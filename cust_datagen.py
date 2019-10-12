from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import psutil

class IDSDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, classes, combined_csv, input_dims, steps_per_epoch, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.classes = classes
        self.combined_csv = combined_csv
        self.dims = input_dims

        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        df = pd.read_csv(self.combined_csv, sep=',', skiprows=index*self.batch_size, nrows=self.batch_size)
        x = df.iloc[:,3:-1].values
        y = [0 if k==["Benign"] else 1 for k in df.iloc[:,-1:].values.tolist()]
        y = keras.utils.to_categorical(y, num_classes=len(self.classes))

        return x.astype(np.float), y
        