import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Cargar el archivo CSV
data = pd.read_csv('data-1883.csv')

# Crear una función para entrenar y predecir una bola individual
def entrenar_y_predecir_bola(data, bola):
    # Separar las bolas restantes y la bola objetivo
    X = data.drop(columns=[bola, 'numero', 'fecha']).values  # Otras bolas como entrada
    y = data[bola].values  # La bola específica a predecir es la salida

    # Normalizar los datos para el modelo
    X_normalized = X / 56
    y_normalized = y / 56

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

    # Redimensionar para LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Crear el modelo RNN
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], 1), return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1, activation='linear'))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), verbose=1)

    # Hacer una predicción para el siguiente valor de la bola
    prediccion = model.predict(X_test[:1])
    return prediccion[0][0] * 56  # Desnormalizar el valor

# Predecir cada bola (de bola-1 a bola-6) y la bola comodín
predicciones = {}
for i in range(1, 7):
    predicciones[f"bola-{i}"] = entrenar_y_predecir_bola(data, f"bola-{i}")

# Predecir la bola comodín
predicciones["bola-comodin"] = entrenar_y_predecir_bola(data, "bola-comodin")

# Mostrar las predicciones
print("Predicciones para el próximo sorteo:")
for bola, valor in predicciones.items():
    print(f"{bola}: {round(valor)}")
