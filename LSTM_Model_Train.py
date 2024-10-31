import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

# Crear el directorio Models si no existe
os.makedirs('Models', exist_ok=True)

# Cargar el archivo CSV preprocesado
data = pd.read_csv('data_normalized.csv')

# Función para entrenar y guardar un modelo LSTM para cada bola
def entrenar_y_guardar_bola(data, bola):
    X = data.drop(columns=[bola, 'numero', 'fecha']).values
    y = data[bola].values

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Redimensionar para LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Crear el modelo
    # B1 100,50,50,16
    # B2 50,50,20,16
    # B3
    # B4
    # B5
    # B6
    # CC
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(100, return_sequences=True),  # Aumentamos a 100 unidades LSTM
        LSTM(100),  # Otra capa LSTM con 50 unidades
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping para evitar el sobreajuste
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Entrenar el modelo con early stopping
    model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test),
              callbacks=[early_stopping], verbose=1)

    # Guardar el modelo en formato .keras
    model.save(f'Models/LSTM_Model_{bola}_{datetime.now().strftime("%Y%m%d%H%M%S")}.keras')

# Entrenar y guardar modelos para cada bola
for i in range(1, 7):
    entrenar_y_guardar_bola(data, f'bola-{i}')

# Entrenar y guardar el modelo para la bola comodín
entrenar_y_guardar_bola(data, 'bola-comodin')
