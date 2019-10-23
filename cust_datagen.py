from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import psutil
import random
import sys
import utils

norm_cols = {
    'Dst Port': [0, 65535],
    'Protocol': [0, 17],
    'Flow Duration': [-828220000000, 120000000],
    'Tot Fwd Pkts': [1, 58091],
    'Tot Bwd Pkts': [0, 123118],
    'TotLen Fwd Pkts': [0, 9789747],
    'TotLen Bwd Pkts': [0.0, 156360426],
    'Fwd Pkt Len Max': [0, 64440],
    'Fwd Pkt Len Min': [0, 1460],
    'Fwd Pkt Len Mean': [0.0, 16529.3138401559],
    'Fwd Pkt Len Std': [0.0, 18401.5827717299],
    'Bwd Pkt Len Max': [0, 65160],
    'Bwd Pkt Len Min': [0, 1460],
    'Bwd Pkt Len Mean': [0.0, 33879.28358],
    'Bwd Pkt Len Std': [0.0, 21326.2385],
    'Flow Byts/s': [0.0, 19005438.5894467],
    'Flow Pkts/s': [-0.0088953248, 2000000.0],
    'Flow IAT Mean': [-828220000000.0, 120000000.0],
    'Flow IAT Std': [0.0, 474354474600.909],
    'Flow IAT Max': [-828220000000, 968434000000],
    'Flow IAT Min': [-947405000000, 120000000.0],
    'Fwd IAT Tot': [-828220000000, 120000000.0],
    'Fwd IAT Mean': [-828220000000.0, 120000000.0],
    'Fwd IAT Std': [0.0, 474354474600.909],
    'Fwd IAT Max': [-828220000000, 968434000000],
    'Fwd IAT Min': [-947405000000, 120000000.0],
    'Bwd IAT Tot': [0.0, 120000000.0],
    'Bwd IAT Mean': [0.0, 120000000.0],
    'Bwd IAT Std': [0.0, 84800000.0],
    'Bwd IAT Max': [0.0, 120000000.0],
    'Bwd IAT Min': [0.0, 120000000.0],
    'Fwd PSH Flags': [0, 1],
    'Bwd PSH Flags': [0, 0],
    'Fwd URG Flags': [0, 1],
    'Bwd URG Flags': [0, 0],
    'Fwd Header Len': [0, 2275036],
    'Bwd Header Len': [0, 2462372],
    'Fwd Pkts/s': [0.0, 4000000.0],
    'Bwd Pkts/s': [0.0, 2000000.0],
    'Pkt Len Min': [0, 1460],
    'Pkt Len Max': [0, 65160],
    'Pkt Len Mean': [0.0, 17344.98473],
    'Pkt Len Std': [0.0, 22788.28621],
    'Pkt Len Var': [0.0, 519000000.0],
    'FIN Flag Cnt': [0, 1],
    'SYN Flag Cnt': [0, 1],
    'RST Flag Cnt': [0, 1],
    'PSH Flag Cnt': [0, 1],
    'ACK Flag Cnt': [0, 1],
    'URG Flag Cnt': [0, 1],
    'CWE Flag Count': [0, 1],
    'ECE Flag Cnt': [0, 1],
    'Down/Up Ratio': [0, 237],
    'Pkt Size Avg': [0.0, 17478.40769],
    'Fwd Seg Size Avg': [0.0, 16529.3138401559],
    'Bwd Seg Size Avg': [0.0, 33879.28358],
    'Fwd Byts/b Avg': [0, 0],
    'Fwd Pkts/b Avg': [0, 0],
    'Fwd Blk Rate Avg': [0, 0],
    'Bwd Byts/b Avg': [0, 0],
    'Bwd Pkts/b Avg': [0, 0],
    'Bwd Blk Rate Avg': [0, 0],
    'Subflow Fwd Pkts': [1, 58091],
    'Subflow Fwd Byts': [0, 9789747],
    'Subflow Bwd Pkts': [0, 123118],
    'Subflow Bwd Byts': [0, 156360426],
    'Init Fwd Win Byts': [-1, 65535],
    'Init Bwd Win Byts': [-1, 65535],
    'Fwd Act Data Pkts': [0, 18290],
    'Fwd Seg Size Min': [0, 56],
    'Active Mean': [0.0, 114000000.0],
    'Active Std': [0.0, 74900000.0],
    'Active Max': [0.0, 114000000],
    'Active Min': [0.0, 114000000],
    'Idle Mean': [0.0, 395571421052.63104],
    'Idle Std': [0.0, 262247866338.599],
    'Idle Max': [0.0, 968434000000],
    'Idle Min': [0.0, 239934000000]
}

def handle_nan_inf(s):
    if s == "Infinity":
        return 0
    elif s == "NaN":
        return 0
    else:
        return float(s)

converters = {"Flow Byts/s": handle_nan_inf, "Flow Pkts/s": handle_nan_inf}


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

def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return [a[p], b[p]], b[p]