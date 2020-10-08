# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np

def load_credit_data():
    path = str(Path(__file__).parent.parent.parent.joinpath('Tree_Classifier\data', 'credit.txt'))
    data = pd.read_csv(path,sep=",",header=None)
    class_label = data.iloc[:,data.shape[1]-1].copy().to_numpy()
    data = data.to_numpy()
    data = np.delete(data,data.shape[1]-1,1)
    return data, class_label

def createMatrix(predicted_labels, true_labels):
    matrix = np.zeros((2,2))
    for p_value, t_value in zip(predicted_labels,true_labels):
        if p_value == 0 and t_value == 0:
            matrix[0][0]+=1
        if p_value == 0 and t_value == 1:
            matrix[1][0]+=1
        if p_value == 1 and t_value == 0:
            matrix[0][1]+=1
        if p_value == 1 and t_value == 1:
            matrix[1][1]+=1
    return matrix

def load_eclipse_data():
    path = str(Path(__file__).parent.parent.parent.joinpath('Tree_Classifier\data', 'eclipse-metrics-packages-2.0.csv'))
    data = pd.read_csv(path, sep = ";")
    data.drop(data.columns[44:], axis=1, inplace= True)
    data.drop(['plugin', 'packagename'], axis=1, inplace= True)
    class_label = data.iloc[:,1].copy().to_numpy()
    data.drop(data.columns[1], axis=1, inplace= True)
    data = data.to_numpy()
    return data, np.clip(class_label, 0, 1)

def load_eclipse_testdata():
    path = str(Path(__file__).parent.parent.parent.joinpath('Tree_Classifier\data', 'eclipse-metrics-packages-3.0.csv'))
    data = pd.read_csv(path, sep = ";")
    data.drop(data.columns[44:], axis=1, inplace= True)
    data.drop(['plugin', 'packagename'], axis=1, inplace= True)
    class_label = data.iloc[:,1].copy().to_numpy()
    data.drop(data.columns[1], axis=1, inplace= True)
    data = data.to_numpy()
    return data, np.clip(class_label, 0, 1)

def acc_pred_rec(matrix):
    accuracy = round((matrix[0][0]+matrix[1][1])/(matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1]), 2)
    precision = round((matrix[0][0])/(matrix[0][0]+matrix[0][1]), 2)
    recall = round((matrix[0][0])/(matrix[0][0]+matrix[1][0]), 2)
    return accuracy, precision, recall
    
if __name__ == "__main__":
    obs_test, label_test = load_credit_data()