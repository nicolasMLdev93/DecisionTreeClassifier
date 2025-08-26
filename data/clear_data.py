import pandas as pd
import numpy as np

dataset = pd.read_csv('wdbc.csv') 

def clear_dataset(df):

    # Eliminación de columnas poco relevantes para el entrenamiento del modelo
    df.drop(columns=['id'],inplace=True)

    # Convierto a todo los features en float
    cols = [
    'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
    'compactness_mean','concavity_mean','concave_points_mean','symmetry_mean',
    'fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se',
    'smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se',
    'fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst',
    'smoothness_worst','compactness_worst','concavity_worst','concave_points_worst',
    'symmetry_worst','fractal_dimension_worst'
    ]

    # Convertir a float y redondear a 4 decimales
    df[cols] = df[cols].astype(float).round(5)

    diagnosis_index = {
        'B':0,
        'M':1
    }

    # Convierto la variable dependiente a integer ( Diagnosis (M = malignant, B = benign) )
    df['diagnosis'] = df['diagnosis'].map(diagnosis_index)

    # Eliminación de datos nulos y redundantes
    df.dropna()
    df.drop_duplicates()

    return df

clear_ds = clear_dataset(dataset)
clear_ds.to_csv('clear_dataset.csv',index=False) 
