import numpy as np
import os
import json

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model_fn(model_dir):
    # Carga los parámetros que exportamos en el paso anterior
    data = np.load(os.path.join(model_dir, 'model_params.npz'))
    return data['w'], data['b'], data['mean'], data['std']

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return np.array(input_data['inputs'])
    raise ValueError("Se esperaba application/json")

def predict_fn(input_data, model):
    w, b, mean, std = model
    # IMPORTANTE: Normalizar la entrada igual que en el entrenamiento
    X_norm = (input_data - mean) / std
    
    # Lógica de tu notebook: z = X @ w + b
    z = np.dot(X_norm, w) + b
    prob = sigmoid(z)
    return prob.tolist()

def output_fn(prediction, content_type):
    return json.dumps({'risk_probability': prediction})