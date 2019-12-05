import pandas as pd
import numpy as np
import re
import h5py
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

def main():
    H5_COMBINED = 'combined.hdf5'
    combined_h5 = h5py.File(H5_COMBINED, 'r') 
    data = combined_h5["combined"][:]
    df = pd.read_csv('Friday-23-02-2018_TrafficForML_CICFlowMeter.csv')
    df[df.columns[16]] = df[df.columns[16]].convert_objects(convert_numeric=True)
    del df['Label']

  
    labels = data[:,-1]
     
    data = np.delete(data, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 0)
    print('finished data split')
    #X_train.fillna(X_train.mean(), inplace=True)
    #X_train.dropna(inplace=True)
    KNN = KNeighborsClassifier(n_neighbors=6) #label = majority class from nearest 6 points
    KNN.fit(X_train,y_train)
    y_pred = KNN.predict(X_test)
    confusionMatrix = confusion_matrix(y_test, y_pred)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision: ", precision_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    print("F1: ", f1_score(y_test, y_pred))   
    print('Confusion Matrix: ')
    print(confusionMatrix)
    #confusionMatrix = confusion_matrix(y_test, y_pred)
    #fig, ax = plt.subplots(figsize=(5,5))

    x_axis_labels = ['Benign', 'Malicious'] # labels for x-axis
    y_axis_labels = ['Benign', 'Malicious'] # labels for y-axis

    #print(f'Encoded Labels for Classes are are {list(le.classes_)}')
    print('\033[1m'+'Confusion Matrix:'+'\033[0m')
    print(confusionMatrix)
    #sn.heatmap(confusionMatrix, annot=True)
    plt.figure(figsize=(9,9))
    sn.heatmap(confusionMatrix,xticklabels=x_axis_labels, yticklabels=y_axis_labels, annot=True, fmt=".3f", linewidths=.5, square = True)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    all_sample_title = 'Accuracy: {0}%'.format(accuracy_score(y_test, y_pred)*100)
    plt.title(all_sample_title, size = 15)


    featureDict = dict()
    for i in range(len(df.columns)):
        featureDict.update({df.columns[i]:features[i]})
    print('\033[1m'+'Features of Importance:'+'\033[0m')
    #print(featureDict)
    print(sorted(featureDict.items(), key=lambda x: x[1], reverse=True))



main()
