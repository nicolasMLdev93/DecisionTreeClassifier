import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from train_model import model

# Columnas del dataset
columns = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
           'compactness_mean','concavity_mean','concave_points_mean','symmetry_mean',
           'fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se',
           'smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se',
           'fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst',
           'smoothness_worst','compactness_worst','concavity_worst','concave_points_worst',
           'symmetry_worst','fractal_dimension_worst']

# Datos de prueba (ejemplo)
X_test = pd.DataFrame([[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776,
                        0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053,
                        8.589, 153.4, 0.0064, 0.04904, 0.05373, 0.01587,
                        0.03003, 0.00619, 25.38, 17.33, 184.6, 2019.0,
                        0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]], 
                      columns=columns)

# Etiqueta real
y_true = [1]  # 1 = maligno, 0 = benigno

# Predicciones
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad de clase 1

# Métricas (para un solo registro, no todas son definidas)
accuracy = accuracy_score(y_true, y_pred) * 100       # % de predicciones correctas
precision = precision_score(y_true, y_pred) * 100     # % de predicciones positivas correctas
recall = recall_score(y_true, y_pred) * 100           # % de positivos reales correctamente detectados
f1 = f1_score(y_true, y_pred) * 100                   # Media armónica de precision y recall

# Mostrar resultados
print(f"Predicción: {y_pred[0]}")
print(f"Probabilidad de maligno: {y_proba[0]*100:.2f}%")
print(f"Exactitud (Accuracy): {accuracy:.2f}% - Porcentaje de predicciones correctas")
print(f"Precisión: {precision:.2f}% - Porcentaje de positivos predichos correctamente")
print(f"Recall: {recall:.2f}% - Porcentaje de positivos reales detectados")
print(f"F1-score: {f1:.2f}% - Media armónica de precisión y recall")
print("AUC: No definido para un solo registro")
