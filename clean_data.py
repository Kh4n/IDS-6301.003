import numpy as np
import datetime
import h5py

# REMEMBER TO SET NROWS ON LINE 98

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