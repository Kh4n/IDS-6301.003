import pandas as pd
import numpy as np
import re
import h5py
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
    del df['Label']

    '''df['Flow Byts/s'] = df['Flow Byts/s'].convert_objects(convert_numeric=True)
    df['Flow Pkts/s'] = df['Flow Pkts/s'].convert_objects(convert_numeric=True)
    df['Label'].replace(benign , 0,inplace=True)
    df.loc[df.Label != 0, 'Label'] = 1
    del df['Flow ID']
    del df['Src IP']
    del df['Dst IP']
    del df['Timestamp']
    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)
    labels = df['Label']
    df.replace('', np.nan, inplace = True)
    df.dropna(inplace =True)
    del df['Label']'''
    labels = data[:,-1]
     
    data = np.delete(data, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 0)
    #X_train.fillna(X_train.mean(), inplace=True)
    #X_train.dropna(inplace=True)
    DTreeClf = DecisionTreeClassifier(criterion = 'entropy')
    DTreeClf.fit(X_train, y_train)
    decisionTree = DTreeClf.tree_
    features = DTreeClf.feature_importances_
    y_pred = DTreeClf.predict(X_test)
    confusionMatrix = confusion_matrix(y_test, y_pred)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision: ", precision_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    print("F1: ", f1_score(y_test, y_pred))   
    print('Confusion Matrix: ')
    print(confusionMatrix)

    featureDict = dict()
    for i in range(len(df.columns)):
        featureDict.update({df.columns[i]:features[i]})
    print('\033[1m'+'Features of Importance:'+'\033[0m')
    #print(featureDict)
    print(sorted(featureDict.items(), key=lambda x: x[1], reverse=True))



main()
