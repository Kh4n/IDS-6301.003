import numpy as np
import datetime
import h5py

# REMEMBER TO SET NROWS ON LINE 98

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

CSV_COMBINED = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined.out"
H5_COMBINED = "/mnt/CSE-CIC-2018/Processed Traffic Data for ML Algorithms/combined_2.hdf5"


combined_h5 = h5py.File(H5_COMBINED, 'x')
ncolumns = 80

# TODO: change this line to set the number of rows
nrows = None

dset = combined_h5.create_dataset("combined", shape=(nrows, ncolumns), chunks=(2048*32, ncolumns))
minmax = combined_h5.create_dataset("minmaxes", shape=(ncolumns, 2))
for k in range(0,ncolumns):
    minmax[k] = [float('inf'), float('-inf')]
rows = []

with open(CSV_COMBINED, 'r') as f:
    i = 0
    for line in f:
        row = line.strip().split(',')
        # this is a hack to allow timestamps to have accurate second representations in 32 bit floating point
        row[2] = datetime.datetime.strptime(row[2], "%d/%m/%Y %H:%M:%S").timestamp() % 100_000_000
        row[-1] = 0 if row[-1] == "Benign" else 1

        if row[16] == "NaN":
            row[16] = 0
        elif row[16] == "Infinity":
            row[16] = 0

        if row[17] == "NaN":
            row[17] = 0
        elif row[17] == "Infinity":
            row[17] = 0

        row = [float(k) for k in row]
        rows.append(row)
        i += 1
        
        if i % 100_000 == 0:
            print(f"Processed {i} items")
            dset[i-100_000:i] = rows
            for k in range(0,ncolumns):
                mm = np.vstack(([np.min(dset[i-100_000:i][:, k]), np.max(dset[i-100_000:i][:, k])], minmax[k]))
                minmax[k] = [np.min(mm[:,0]), np.max(mm[:,1])]
            rows = []
            tmp = i

print(f"Processed {i} items")
dset[tmp:i] = rows
for k in range(0,ncolumns):
    mm = np.vstack(([np.min(dset[tmp:i][:, k]), np.max(dset[tmp:i][:, k])], minmax[k]))
    minmax[k] = [np.min(mm[:,0]), np.max(mm[:,1])]

combined_h5.close()