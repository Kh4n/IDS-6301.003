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

headers = {
    "Dst Port": np.int32,
    "Protocol": np.int32,
    "Timestamp": np.int32,
    "Flow Duration": np.float32,
    "Tot Fwd Pkts": np.float32,
    "Tot Bwd Pkts": np.float32,
    "TotLen Fwd Pkts": np.float32,
    "TotLen Bwd Pkts": np.float32,
    "Fwd Pkt Len Max": np.float32,
    "Fwd Pkt Len Min": np.float32,
    "Fwd Pkt Len Mean": np.float32,
    "Fwd Pkt Len Std": np.float32,
    "Bwd Pkt Len Max": np.float32,
    "Bwd Pkt Len Min": np.float32,
    "Bwd Pkt Len Mean": np.float32,
    "Bwd Pkt Len Std": np.float32,
    "Flow Byts/s": np.float32,
    "Flow Pkts/s": np.float32,
    "Flow IAT Mean": np.float32,
    "Flow IAT Std": np.float32,
    "Flow IAT Max": np.float32,
    "Flow IAT Min": np.float32,
    "Fwd IAT Tot": np.float32,
    "Fwd IAT Mean": np.float32,
    "Fwd IAT Std": np.float32,
    "Fwd IAT Max": np.float32,
    "Fwd IAT Min": np.float32,
    "Bwd IAT Tot": np.float32,
    "Bwd IAT Mean": np.float32,
    "Bwd IAT Std": np.float32,
    "Bwd IAT Max": np.float32,
    "Bwd IAT Min": np.float32,
    "Fwd PSH Flags": np.float32,
    "Bwd PSH Flags": np.float32,
    "Fwd URG Flags": np.float32,
    "Bwd URG Flags": np.float32,
    "Fwd Header Len": np.float32,
    "Bwd Header Len": np.float32,
    "Fwd Pkts/s": np.float32,
    "Bwd Pkts/s": np.float32,
    "Pkt Len Min": np.float32,
    "Pkt Len Max": np.float32,
    "Pkt Len Mean": np.float32,
    "Pkt Len Std": np.float32,
    "Pkt Len Var": np.float32,
    "FIN Flag Cnt": np.float32,
    "SYN Flag Cnt": np.float32,
    "RST Flag Cnt": np.float32,
    "PSH Flag Cnt": np.float32,
    "ACK Flag Cnt": np.float32,
    "URG Flag Cnt": np.float32,
    "CWE Flag Count": np.float32,
    "ECE Flag Cnt": np.float32,
    "Down/Up Ratio": np.float32,
    "Pkt Size Avg": np.float32,
    "Fwd Seg Size Avg": np.float32,
    "Bwd Seg Size Avg": np.float32,
    "Fwd Byts/b Avg": np.float32,
    "Fwd Pkts/b Avg": np.float32,
    "Fwd Blk Rate Avg": np.float32,
    "Bwd Byts/b Avg": np.float32,
    "Bwd Pkts/b Avg": np.float32,
    "Bwd Blk Rate Avg": np.float32,
    "Subflow Fwd Pkts": np.float32,
    "Subflow Fwd Byts": np.float32,
    "Subflow Bwd Pkts": np.float32,
    "Subflow Bwd Byts": np.float32,
    "Init Fwd Win Byts": np.float32,
    "Init Bwd Win Byts": np.float32,
    "Fwd Act Data Pkts": np.float32,
    "Fwd Seg Size Min": np.float32,
    "Active Mean": np.float32,
    "Active Std": np.float32,
    "Active Max": np.float32,
    "Active Min": np.float32,
    "Idle Mean": np.float32,
    "Idle Std": np.float32,
    "Idle Max": np.float32,
    "Idle Min": np.float32,
    "Label": np.float32,
}

def handle_nan_inf(s):
    if s == "Infinity":
        return 0
    elif s == "NaN":
        return 0
    else:
        return float(s)

converters = {"Flow Byts/s": handle_nan_inf, "Flow Pkts/s": handle_nan_inf}

class IDSDataGeneratorAttention(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, classes, combined_csv, input_dims, steps_per_epoch, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.classes = classes
        self.combined_csv = combined_csv
        self.dims = input_dims

        self.steps_per_epoch = steps_per_epoch
        self.fsize = utils.rawcount(self.combined_csv)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        group_size = self.batch_size*self.dims[0]
        df = pd.read_csv(self.combined_csv, sep=',', skiprows=range(self.rand_offset,(index+1)*group_size), nrows=group_size, converters=converters)
        for c in norm_cols:
            if norm_cols[c][1]-norm_cols[c][0] > 0:
                df[c] = (df[c] - norm_cols[c][0])/(norm_cols[c][1]-norm_cols[c][0])

        x = np.reshape(df.iloc[:,3:-1].values, [self.batch_size, *self.dims]) 
        y = df["Label"].apply(lambda s: 0 if s=="Benign" else 1)[self.dims[0]-1::self.dims[0]]
        # y = keras.utils.to_categorical(y, num_classes=2)

        return x.astype(np.float64), y.astype(np.float64)
    
    def on_epoch_end(self):
        avail_shift = self.fsize - self.steps_per_epoch*self.batch_size*self.dims[0]
        self.rand_offset = random.randint(1, avail_shift)