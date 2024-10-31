import pandas as pd

# Cargar el archivo CSV
data = pd.read_csv('data-1883.csv')

# Normalizar las bolas entre 0 y 1 dividiendo entre 56
for i in range(1, 7):
    data[f'bola-{i}'] = data[f'bola-{i}'] / 56
data['bola-comodin'] = data['bola-comodin'] / 56

# Guardar los datos preprocesados para el modelo de entrenamiento
data.to_csv('data_normalized.csv', index=False)
