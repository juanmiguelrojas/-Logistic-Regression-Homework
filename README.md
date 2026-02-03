# Heart Disease Risk Prediction - Logistic Regression

## Descripción

Este proyecto implementa **regresión logística** para predecir el riesgo de enfermedad cardíaca usando funciones del profesor de los notebooks de referencia (`ClassificationAndLogisticRegression/`).

## Dataset

- **Fuente:** UCI Heart Disease (Kaggle)
- **Muestras:** 303 pacientes
- **Features seleccionadas:** Age, Cholesterol, BP, Max HR, ST depression, Number of vessels fluro
- **Target:** Presencia (1) vs Ausencia (0) de enfermedad cardíaca

## Contexto introductorio

La enfermedad cardíaca es la principal causa de muerte a nivel mundial. En este ejercicio se aplica regresión logística al dataset de Kaggle (303 pacientes) para demostrar paso a paso: EDA, implementación desde cero, visualización de fronteras de decisión, regularización L2 y un flujo conceptual de despliegue en Amazon SageMaker.

## Instrucciones del Homework (resumen)

- Implementar las funciones teóricas vistas en clase: `sigmoid`, costo (cross-entropy), gradientes y `gradient descent` usando NumPy.
- Usar Pandas para EDA y Matplotlib para visualización. No usar scikit-learn para el entrenamiento del modelo (solo métricas permitidas).
- Split 70/30 (preferible estratificado), normalización Z-score, seleccionar ≥6 features.
- Tunear λ (L2) en [0, 0.001, 0.01, 0.1, 1.0] y comparar métricas y `||w||`.

## Paso 5: Despliegue y evidencia (espacio para imágenes)

En esta sección se documenta el procedimiento de despliegue en SageMaker. Añade aquí capturas de pantalla o imágenes que evidencien tu trabajo (training job, endpoint, respuesta de inferencia). Puedes colocar imágenes en la carpeta `docs/images/` y referenciarlas abajo.

### Imágenes sugeridas (añadir archivos en `docs/images/` y enlazarlos):

- `docs/images/sagemaker_training.png` — Estado del entrenamiento (Console/SageMaker).
- `docs/images/sagemaker_endpoint.png` — Configuración del endpoint desplegado.
- `docs/images/inference_response.png` — Resultado de una inferencia de prueba (ej. Age=60, Chol=300 → Prob=0.68).

*Añade las imágenes anteriores reemplazando los archivos de ejemplo o actualiza los nombres según tus capturas.*

## Entregables

- Notebook: `Heart_Disease_Risk_Prediction.ipynb` (o `heart_disease_lr_analysis.ipynb`) con todas las secciones documentadas.
- `heart.csv` o enlace a Kaggle.
- README actualizado con evidencia y capturas.

## Envío y Evaluación

- Sube el repositorio a GitHub y comparte el enlace en la plataforma de entrega.
- Rubrica (100 pts): EDA (10), Implementación (35), Visualizaciones/Análisis (20), Regularización/Tuning (15), Despliegue/Repo (15), Claridad (5).

## Recomendaciones para entrega

- Incluye imágenes en `docs/images/` y referencia los archivos en este README.
- Añade un pequeño `gallery` o sección final en el notebook con las capturas y notas finales de tu experimento.
- Comprime archivos grandes (si corresponde) y verifica que el notebook sea ejecutable en orden (Restart & Run All) antes de subir.

## Estructura del Código

### 1. Carga y Preparación de Datos
- Carga del CSV
- Binarización del target
- Normalización Z-score
- Split train/test (70/30)

### 2. Funciones del Profesor (copiadas directamente)

Las siguientes funciones se implementan usando exactamente el código del profesor:

- **`sigmoid(z)`**: Función de activación
- **`compute_cost_log_reg_reg(w, b, X, y, lam)`**: Costo con L2 regularización
- **`compute_gradient_log_reg_reg(w, b, X, y, lam)`**: Gradientes
- **`gradient_descent_log_reg_reg(...)`**: Descenso de gradiente

### 3. Entrenamiento y Tuning

- Entrenamiento con diferentes valores de λ ∈ [0, 0.001, 0.01, 0.1, 1.0]
- Evaluación en conjunto de validación
- Selección del mejor λ basado en F1-score

### 4. Resultados

| λ | Train Acc | Test Acc | Test F1 | ||w|| |
|---|-----------|----------|---------|--------|
| 0.0 | 0.8571 | 0.8421 | 0.8444 | 2.347 |
| 0.001 | 0.8619 | 0.8553 | 0.8612 | 2.341 |
| **0.01** | **0.8619** | **0.8553** | **0.8612** | **2.184** |
| 0.1 | 0.8619 | 0.8553 | 0.8612 | 1.897 |
| 1.0 | 0.8333 | 0.8289 | 0.8261 | 0.412 |

**Mejor λ:** 0.01

## Ejecución

```bash
# Instalar dependencias
pip install numpy pandas matplotlib scikit-learn jupyter

# Ejecutar notebook
jupyter notebook Heart_Disease_Risk_Prediction.ipynb
```

## Archivos

- `heart.csv` - Dataset original
- `Heart_Disease_Risk_Prediction.ipynb` - Notebook simplificado
- `README.md` - Este archivo
- `ClassificationAndLogisticRegression/` - Notebooks de referencia del profesor

## Sección: Despliegue en Amazon SageMaker (guía resumida)

Esta sección describe, en alto nivel, los pasos para desplegar el modelo en SageMaker. Añade capturas en `docs/images/` para documentar tu despliegue.

1. Preparar artefactos:
	- Guardar el mejor modelo en JSON (pesos, bias, `X_mean`, `X_std`) o `joblib`/`pickle`.
	- Empaquetar código de entrenamiento (`train.py`) y de inferencia (`inference.py`).

2. `train.py` (resumen):
	- Cargar datos desde S3 o dataset local.
	- Entrenar modelo (mismo código del notebook) y guardar artefacto en `/opt/ml/model/`.

3. `inference.py` (resumen):
	- `model_fn(model_dir)`: cargar el modelo desde disco.
	- `input_fn(request_body, content_type)`: parsear JSON de entrada.
	- `predict_fn(input_data, model)`: normalizar usando `X_mean`/`X_std`, calcular `sigmoid(w·x + b)` y devolver probabilidad.
	- `output_fn(prediction, content_type)`: serializar respuesta JSON.

4. Entrenar en SageMaker:
	- Subir datos a S3 y lanzar un `Estimator` (o usar un contenedor propio).
	- Ejecutar `estimator.fit()` apuntando al S3 location.

5. Desplegar endpoint:
	- `model = estimator.create_model()` → `predictor = model.deploy(...)`.
	- Probar con una petición JSON: `{ "age":60, "cholesterol":300, "bp":140, "max_hr":110, "st_depression":1.5, "vessels":1 }`.

6. Monitoreo y limpieza:
	- Habilitar CloudWatch, DataCapture para monitorizar drift y latencia.
	- Eliminar endpoint después de las pruebas para evitar costes.

Ejemplo rápido de `predict_fn` (pseudocódigo):

```python
def predict_fn(input_data, model):
	 x = np.array([input_data['age'], input_data['cholesterol'], input_data['bp'], input_data['max_hr'], input_data['st_depression'], input_data['vessels']])
	 x_norm = (x - np.array(model['X_mean'])) / np.array(model['X_std'])
	 prob = sigmoid(np.dot(model['weights'], x_norm) + model['bias'])
	 return {'probability': float(prob)}
```

Recursos y referencia rápida:
- SageMaker docs: https://docs.aws.amazon.com/sagemaker/
- Ejemplos: https://github.com/aws/amazon-sagemaker-examples

Añade aquí las capturas y detalles de tu ejecución en SageMaker (training job id, endpoint name, tiempos y latencias observadas).

## Librerías Permitidas

- numpy
- pandas  
- matplotlib
- scikit-learn (solo para métricas: accuracy, precision, recall, f1, confusion_matrix)

## Notas

- Las funciones core están copiadas directamente de los notebooks del profesor
- No se usa scikit-learn para entrenar el modelo (solo para métricas)
- La regularización L2 controla la complejidad del modelo
- El modelo tiene ~84% de accuracy en el conjunto de test

## Evidencia de Overfitting y Efecto de Regularización

- ¿Cómo detectamos overfitting en este proyecto?:
	- Gap entre entrenamiento y prueba: cuando la precisión/F1 en entrenamiento es considerablemente mayor que en test, el modelo probablemente memorizó ruido.
	- Normas de pesos (`||w||`): un `||w||` grande sugiere mayor complejidad; al aumentar `λ` (L2) el `||w||` disminuye.
	- Visualizaciones de frontera de decisión (2D): modelos sin regularización muestran fronteras que pasan por puntos aislados; con regularización las fronteras se vuelven más suaves.

- Dónde verlo en el repositorio:
	- `Heart_Disease_Risk_Prediction.ipynb`:
		- Gráfica de convergencia de la función de costo por iteración (cost vs iter): ayuda a detectar aprendizaje inestable o divergencia.
		- Tabla comparativa y gráficos `Test F1 / Train F1` vs `λ`: muestra cómo cambia la generalización con regularización.
		- `||w||` vs `λ` plot: verifica que L2 reduce la magnitud de los pesos.
		- Decision boundary plots (feature pairs): comparación visual entre `λ=0` y el mejor `λ` para ver el efecto de suavizado.
	- Notebooks del profesor en `ClassificationAndLogisticRegression/`:
		- `APENDIX-RidgeVsGradientDescentInRegularizedLinearRegression.ipynb`: demuestra cómo L2 (Ridge) introduce weight decay y mejora generalización en ejemplos sintéticos.
		- `week2_classification_hour2_regularization_with_derivatives.ipynb`: contiene las derivaciones y ejemplos que muestran cómo la regularización modifica las gradientes y la actualización de pesos.

- Recomendaciones prácticas:
	- Use la combinación de métricas (F1, precisión, recall) y `||w||` para seleccionar `λ` en lugar de solo la exactitud.
	- Para conjuntos pequeños, usar validación estratificada o K-fold para una estimación más robusta del rendimiento.
	- Si se detecta overfitting persistente, considerar aumentar el conjunto de datos o usar técnicas como regularización más fuerte, reducción de características, o aumento de datos si aplica.
