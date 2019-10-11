from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

class IDSDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, classes, num_rows, combined_csv, batch_size=32, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.classes = classes
        self.shuffle = shuffle
        self.num_rows = num_rows
        self.combined_csv = combined_csv
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_rows / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        length = len(self)

        df = pd.read_csv(self.combined_csv, sep=',', skiprows=index*self.batch_size, nrows=self.batch_size)
        x = df.iloc[:,3:-1].values
        y = [0 if k=="Benign" else 1 for k in df.iloc[:,-1:].values.tolist()]
        y = keras.utils.to_categorical(y, num_classes=len(self.classes))
        # print(x, y)
        return x.astype(np.float), y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.indexes = np.arange(len(self.list_IDs))
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)
        