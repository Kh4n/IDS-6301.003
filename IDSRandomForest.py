import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing

def main():
    benign = 'Benign'
    #df = pd.read_csv('Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv',dtype={20:float,21:float})
    df = pd.read_csv('Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv')
    #print(df.iloc[20])    
    df = df.astype({'Flow ID': str})
    df['Label'].replace(benign , 0,inplace=True)
    df.loc[df.Label != 0, 'Label'] = 1
    del df['Flow ID']
    del df['Src IP']
    del df['Dst IP']
    del df['Timestamp']
    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)
    labels = df['Label']
    del df['Label']
    X_train, X_test, y_test, y_train = train_test_split(df, labels, test_size = 0.2, random_state = 0)
    X_train.fillna(X_train.mean(), inplace=True)
    DTreeClf = DecisionTreeClassifier(criterion = 'entropy')
    DTreeClf.fit(X_train, y_train)
    decisionTree = DTreeClf.tree_
    features = DTreeClf.feature_importances_
    y_pred = clf.predict(X_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))
    print("F1: ", metrics.f1_score(y_test, y_pred))
    print(features)    



main()
