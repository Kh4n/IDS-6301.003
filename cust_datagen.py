from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import pandas as pd
import psutil
import random
import sys
import utils

class IDSDataGeneratorBasic(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, combined_h5, offset, columns, steps_per_epoch, batch_size=32):
        'Initialization'
        self.combined_h5 = combined_h5
        self.offset = offset
        self.columns = columns
        self.data = data
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.mins = self.combined_h5["minmaxes"][self.columns,0]
        self.col_ranges = self.combined_h5["minmaxes"][self.columns,1] - self.mins

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        batch = self.data[self.offset + index*self.batch_size:self.offset + (index+1)*self.batch_size]
        dstport = batch[:, 0]
        protocol = batch[:, 1]
        x = (batch[:, self.columns] - self.mins)/self.col_ranges 
        y = batch[:, -1]
        # y = keras.utils.to_categorical(y, num_classes=len(self.classes))

        return [protocol, dstport, x], y
    
    @classmethod
    def create_data_generators(cls, data, combined_h5, columns, val_split, batch_size=32):
        np.random.shuffle(data)
        tot_rows = len(data)
        train_split = 1-val_split
        steps_per_epoch = int(train_split*tot_rows)//batch_size
        train_gen = cls(data, combined_h5, 0, columns, steps_per_epoch, batch_size=batch_size)
        steps_per_epoch = int(val_split*tot_rows)//batch_size
        val_gen = cls(data, combined_h5, int(train_split*tot_rows), columns, steps_per_epoch, batch_size=batch_size)
        return (train_gen, val_gen)

class IDSDataGeneratorAttentionH5(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, combined_h5, indices, attention_window, columns, steps_per_epoch=None, batch_size=32):
        'Initialization'
        self.combined_h5 = combined_h5
        self.attention_window = attention_window
        self.columns = columns
        self.data = data
        self.indices = indices

        if steps_per_epoch is None:
            self.steps_per_epoch = len(self.indices)//batch_size
        else:
            self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.mins = self.combined_h5["minmaxes"][self.columns,0]
        self.col_ranges = self.combined_h5["minmaxes"][self.columns,1] - self.mins
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        cur_indices = np.reshape(
            [np.arange(k,k+self.attention_window) for k in self.indices[index*self.batch_size:(index+1)*self.batch_size]],
            self.batch_size*self.attention_window
        )
        batch = self.data[cur_indices]
        protocol = np.reshape(batch[:,1], [self.batch_size, self.attention_window])
        dstport = np.reshape(batch[:,0], [self.batch_size, self.attention_window])
        x = np.reshape((batch[:, self.columns]-self.mins)/self.col_ranges, [self.batch_size, self.attention_window, len(self.columns)])
        y = batch[:, -1]
        y = np.reshape(y[self.attention_window-1::self.attention_window], [self.batch_size, 1])
        return [protocol, dstport, x], y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    @classmethod
    def create_data_generators(cls, data, combined_h5, attention_window, columns, val_split, steps_per_epoch=None, batch_size=32):
        # tot_rows = int(0.1*len(combined_h5["combined"]))
        tot_rows = len(data)
        train_split = 1-val_split
        indices = np.arange(0, tot_rows-attention_window)
        np.random.shuffle(indices)
        train_gen = cls(data, combined_h5, indices[0:int(train_split*len(indices))], attention_window, columns, steps_per_epoch=steps_per_epoch, batch_size=batch_size)
        steps_per_epoch = None if steps_per_epoch is None else int(steps_per_epoch*val_split)
        val_gen = cls(data, combined_h5, indices[int(train_split*len(indices)):], attention_window, columns, steps_per_epoch=steps_per_epoch, batch_size=batch_size)
        return (train_gen, val_gen)


class IDSDataGeneratorUnsupervised(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, combined_h5, offset, columns, steps_per_epoch, batch_size=32):
        'Initialization'
        self.combined_h5 = combined_h5
        self.offset = offset
        self.columns = columns
        self.data = data
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.mins = self.combined_h5["minmaxes"][self.columns,0]
        self.col_ranges = self.combined_h5["minmaxes"][self.columns,1] - self.mins

        self.e_mins = self.combined_h5["minmaxes"][0:2,0]
        self.e_col_ranges = self.combined_h5["minmaxes"][0:2,1] - self.e_mins

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        batch = self.data[self.offset + index*self.batch_size:self.offset + (index+1)*self.batch_size]
        # dstport = batch[:, 0]
        # protocol = batch[:, 1]
        x = (batch[:, self.columns] - self.mins)/self.col_ranges
        # x = batch[:, self.columns]
        # y = batch[:, -1]
        # y = keras.utils.to_categorical(y, num_classes=len(self.classes))
        # y = np.hstack([np.expand_dims(protocol, axis=-1) , x])

        return x, x
    
    @classmethod
    def create_data_generators(cls, data, combined_h5, columns, val_split, batch_size=32):
        np.random.shuffle(data)
        tot_rows = len(data)
        train_split = 1-val_split
        steps_per_epoch = int(train_split*tot_rows)//batch_size
        train_gen = cls(data, combined_h5, 0, columns, steps_per_epoch, batch_size=batch_size)
        steps_per_epoch = int(val_split*tot_rows)//batch_size
        val_gen = cls(data, combined_h5, int(train_split*tot_rows), columns, steps_per_epoch, batch_size=batch_size)
        return (train_gen, val_gen)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]