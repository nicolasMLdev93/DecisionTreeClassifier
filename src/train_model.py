import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv('../data/clear_dataset.csv')

def train_model(dataset):
    # Obtengo las variables independientes y dependientes (x1,x2 .. xn,y)
    X = dataset.iloc[:,1:] # X
    Y = dataset.iloc[:,0] # Y
    tree = DecisionTreeClassifier(max_depth=4)
    tree.fit(X,Y)
    return tree

model = train_model(dataset)