# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np

def load_credit_data():
    path = str(Path(__file__).parent.parent.parent.joinpath('Tree_Classifier\data', 'pima.txt'))
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
    
if __name__ == "__main__":
    obs_test, label_test = load_credit_data()