from train_model import model
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Visualización del Árbol de decisión con las métricas de c/u de sus nodos

plt.figure(figsize=(16,16))
plot_tree(model, class_names=['Benigno','Maligno'],
          filled=True, rounded=True)
plt.show()