import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Cargar el dataset concentlite.csv
data = pd.read_csv("concentlite.csv")

# Extraer características y etiquetas
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=10)

# Definir la arquitectura de la red neuronal
input_size = X_train.shape[1]
hidden_sizes = [16, 8]  # Cambia estos números según tu elección de capas y neuronas
output_size = 1

# Definir la tasa de aprendizaje
learning_rate = 0.01

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Definir la derivada de la función de activación
def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights_and_biases(input_size, hidden_sizes, output_size):
    sizes = [input_size] + hidden_sizes + [output_size]
    weights = [np.random.randn(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
    biases = [np.zeros((1, sizes[i+1])) for Si in range(len(sizes)-1)]
    return weights, biases
# Inicializar pesos y sesgos
weights, biases = initialize_weights_and_biases(input_size, hidden_sizes, output_size)

# Entrenar la red neuronal usando retropropagación
num_epochs = 1000

for epoch in range(num_epochs):
    # Propagación hacia adelante
    layer_outputs = []
    layer_inputs = [X_train]
    for i in range(len(hidden_sizes)):
        layer_input = layer_inputs[-1]
        layer_output = sigmoid(np.dot(layer_input, weights[i]) + biases[i])
        layer_outputs.append(layer_output)
        layer_inputs.append(layer_output)
    
    # Calcular el error
    output_error = y_train.reshape(-1, 1) - layer_outputs[-1]
    
    # Retropropagación
    for i in range(len(hidden_sizes) - 1, -1, -1):  # Cambio aquí
        delta = output_error * sigmoid_derivative(layer_outputs[i])
        output_error = delta.dot(weights[i].T)
        weights[i] += layer_inputs[i].T.dot(delta) * learning_rate
        biases[i] += np.sum(delta, axis=0, keepdims=True) * learning_rate


# Evaluar el modelo en el conjunto de prueba
def predict(X):
    layer_input = X
    for i in range(len(hidden_sizes)):
        layer_output = sigmoid(np.dot(layer_input, weights[i]) + biases[i])
        layer_input = layer_output
    return layer_output

y_pred = predict(X_test)

plt.figure(figsize=(8, 6))

# Graficar puntos de la clase 0 (azul)
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='blue', marker='o', label='Clase 0')

# Graficar puntos de la clase 1 (rojo)
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='red', marker='x', label='Clase 1')

plt.title("Clasificación por el perceptrón multicapa")
plt.legend()
plt.show()