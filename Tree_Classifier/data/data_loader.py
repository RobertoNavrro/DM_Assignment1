# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

def load_credit_data():
    path = str(Path(__file__).parent.parent.parent.joinpath('Tree_Classifier\data', 'credit.txt'))
    data = pd.read_csv(path,sep=",")
    class_label = data["class"].copy()
    data = data.drop(columns="class")
    return data, class_label

if __name__ == "__main__":
    credit_data = load_credit_data()
