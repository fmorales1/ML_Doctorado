## Importar librerias ---------------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

 
# INPUT y OUTPUT --------------------------------------------------------------
input_data = pd.read_csv("Base de datos.csv")
output_data = pd.read_csv("results")
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)


# Dividir datos en conjuntos de entrenamiento y prueba-------------------------
X_train, X_test, y_train, y_test  = train_test_split(input_data_scaled, output_data, test_size=0.1, random_state=42)


# Creacion del modelo ---------------------------------------------------------
model = tf.keras.Sequential([
    layers.Input(shape=(input_data.shape[1],)),
    layers.Dense(20,activation="relu"),
    layers.Dense(20,activation="relu"),
    layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer, loss = "mean_squared_error")

# Entrenamiento de la red -----------------------------------------------------
entrenamiento = model.fit(X_train, y_train, epochs=900000, batch_size=16, validation_split=0.1)

# Ver resultados
loss = model.evaluate(X_test, y_test)
print(f'Error cuadrático medio en datos de prueba: {loss}')
plt.xlabel("ciclos de entrenamiento")
plt.ylabel("errores")
plt.plot(entrenamiento.history["loss"])
plt.show()

# Predicciones
predictions = model.predict(X_test)
print(predictions)
