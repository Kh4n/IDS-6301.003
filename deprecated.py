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