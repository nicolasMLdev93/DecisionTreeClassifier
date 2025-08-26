from train_model import model
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Visualización del Árbol de decisión con las métricas de cada nodo
plt.figure(figsize=(16,16))
plot_tree(model, class_names=['Benigno','Maligno'],
          filled=True, rounded=True)

# Guardar gráfico
plt.savefig('tree.png', dpi=300, bbox_inches='tight')

# Mostrar gráfico
plt.show()
