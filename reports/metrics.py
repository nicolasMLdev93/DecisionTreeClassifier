import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes.Decision_Tree import Decision_Tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

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

# Predicciones
y_pred = classifier_instance.predict_data(x_train)
y_proba = classifier_instance.get_probability(x_train)

# Métricas
accuracy:float = classifier_instance.get_acc_score(x_train, y_train) * 100
precision:float = classifier_instance.get_precision_score(x_train, y_train) * 100

# Mostrar resultados
print(f"Predicción primera muestra: {y_pred[0]}")
print(f"Probabilidad de maligno primera muestra: {y_proba[0][1]*100:.2f}%")
print(f"Exactitud (Accuracy): {accuracy:.2f}% - Porcentaje de predicciones correctas")
print(f"Precisión: {precision:.2f}% - Porcentaje de positivos predichos correctamente")

# Visualización del árbol
plt.figure(figsize=(16, 16))
plot_tree(classifier_instance.model,
          class_names=['Benigno', 'Maligno'],
          filled=True, rounded=True)
plt.savefig('tree.png', dpi=300, bbox_inches='tight')
plt.show()
