import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Creamos 2 dataframes, uno maligno y otro benigno para hacer las predicciones de la Patología X

# Columnas de variables independientes
columns = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
               'compactness_mean','concavity_mean','concave_points_mean','symmetry_mean',
               'fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se',
               'smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se',
               'fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst',
               'smoothness_worst','compactness_worst','concavity_worst','concave_points_worst',
               'symmetry_worst','fractal_dimension_worst']

# Input con datos para predicción maligna
X_test_M = pd.DataFrame([[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776,
                            0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053,
                            8.589, 153.4, 0.0064, 0.04904, 0.05373, 0.01587,
                            0.03003, 0.00619, 25.38, 17.33, 184.6, 2019.0,
                            0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]], 
                          columns=columns)

# Input con datos para predicción benigna    
X_test_B = pd.DataFrame([[10.0, 10.0, 65.0, 400.0, 0.10, 0.10, 0.05, 0.02, 0.15, 0.05,
                        0.2, 0.3, 1.0, 20.0, 0.005, 0.02, 0.01, 0.005, 0.02, 0.003,
                        12.0, 12.0, 75.0, 450.0, 0.12, 0.10, 0.06, 0.02, 0.15, 0.05]],
                      columns=columns)