# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

def load_credit_data():
    path = str(Path(__file__).parent.parent.parent.joinpath('Tree_Classifier\data', 'credit.txt'))
    data = pd.read_csv(path,sep=",")
    class_label = data["class"].copy().to_numpy()
    # class_label.resize(class_label.shape[0],1)
    data = data.drop(columns="class").to_numpy()
    return data, class_label