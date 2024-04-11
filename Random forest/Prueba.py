# Prueba con random forest
# Importar librerias

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Carga de datos 
variables = pd.read_csv("Base de datos.csv", header=None)
labels_df = pd.read_csv("resultados.csv", header=None)
df = pd.concat([data_df, labels_df], axis=1)

print(df)

