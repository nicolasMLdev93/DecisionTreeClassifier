import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes.Decision_Tree import Decision_Tree
import pandas as pd
from tests.test_dataframe import X_test_M,X_test_B
# Cargar dataset
dataset = pd.read_csv('../data/clear_dataset.csv')

# Separar features y target
def get_features(dataset):
    X = dataset.iloc[:, 1:]  # Variables independientes
    Y = dataset.iloc[:, 0]   # Variable dependiente
    return X, Y

x_train, y_train = get_features(dataset)

# Instanciar y entrenar modelo
classifier_instance:Decision_Tree = Decision_Tree()
classifier_instance.train_model(x_train, y_train)


# Evaluaciones
result_B = classifier_instance.predict_data(X_test_B)
result_M = classifier_instance.predict_data(X_test_M)

# Análisis con resultado benigno x
print(f"El análisis de los datos del modelo sostiene que {'si tiene patología X' if result_B[0] == 1 else 'no tiene patología X'}")

# Análisis con resultado maligno 
print(f"El análisis de los datos del modelo sostiene que {'si tiene patología X' if result_M[0] == 1 else 'no tiene patología X'}")
