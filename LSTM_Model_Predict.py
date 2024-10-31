import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Función para encontrar el archivo más reciente de un modelo dado el prefijo
def obtener_modelo_reciente(prefijo):
    archivos = [f for f in os.listdir('Models') if f.startswith(prefijo) and f.endswith('.keras')]
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos para el prefijo: {prefijo}")
    archivos.sort(reverse=True)  # Ordenar por nombre de forma descendente
    return os.path.join('Models', archivos[0])  # Retorna el archivo más reciente

# Cargar los datos preprocesados para obtener las últimas entradas
data = pd.read_csv('data_normalized.csv')
X = data.drop(columns=['numero', 'fecha']).values
X = X[-1:]  # Seleccionar el último registro para predecir el siguiente sorteo
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Función para cargar el modelo y hacer una predicción
def predecir_bola(prefijo):
    modelo_path = obtener_modelo_reciente(prefijo)
    model = load_model(modelo_path)
    prediccion = model.predict(X)
    return prediccion[0][0] * 56  # Desnormalizar la predicción

# Función para ajustar la predicción dentro del rango permitido
def ajustar_rango(prediccion, minimo, maximo):
    return max(min(round(prediccion), maximo), minimo)

# Función para asegurar que no haya duplicados en las predicciones
def ajustar_sin_duplicados(predicciones, minimo, maximo):
    unique_predicciones = set()
    resultado = []
    for pred in predicciones:
        while pred in unique_predicciones or pred < minimo or pred > maximo:
            pred += 1
            if pred > maximo:  # Volver al mínimo si se excede el máximo
                pred = minimo
        unique_predicciones.add(pred)
        resultado.append(pred)
    return resultado

# Realizar predicciones para cada bola
predicciones = [
    ajustar_rango(predecir_bola(f'LSTM_Model_bola-{i}_'), *rango)
    for i, rango in enumerate([(1, 42), (2, 46), (4, 52), (5, 54), (12, 55), (17, 56)], 1)
]
# Predicción para la bola comodín
predicciones.append(ajustar_rango(predecir_bola('LSTM_Model_bola-comodin_'), 1, 56))

# Asegurar que no haya números repetidos en las predicciones
predicciones = ajustar_sin_duplicados(predicciones, 1, 56)

# Imprimir las predicciones ajustadas y únicas
print("Predicciones ajustadas para el próximo sorteo:")
for i, valor in enumerate(predicciones[:-1], 1):
    print(f"bola-{i}: {valor}")
print(f"bola-comodin: {predicciones[-1]}")
