# Modelo de Regresion Lineal con TensorFlow Serving

Implementacion didactica y despliegue del modelo de regresion lineal visto en clase para resolver la funcion `y = 2x + 1` usando Keras y TensorFlow Serving.

Sigue los principios Keep It Simple (KISS) y You Aren't Gonna Need It (YAGNI), manteniendo el codigo conciso y sin abstracciones innecesarias.

## Estructura del Directorio

```text
cw-linear-regression/
├── .gitignore          # Exclusion de entorno virtual y archivos temporales
├── client.py           # Cliente HTTP simple para consumir TensorFlow Serving
├── linear_model.py     # Definicion, entrenamiento y exportacion del modelo
├── README.md           # Documentacion e instrucciones de ejecucion
└── linear_model/       # Bundle exportado en formato SavedModel
    └── 1/
        ├── saved_model.pb
        ├── fingerprint.pb
        └── variables/
```

## Flujo del Modelo

1. Generacion de datos: Crea muestras de $X$ con NumPy y calcula las etiquetas $Y = 2X + 1$.
2. Barajado y particion: Desordena las muestras y divide en entrenamiento (80%) y validacion (20%).
3. Arquitectura: Red de una sola capa densa (`units=1`, `input_shape=[1]`) compilada con `SGD` y `MeanSquaredError`.
4. Entrenamiento y prediccion: 20 epocas de optimizacion y validacion de predicciones locales.
5. Exportacion: Guarda el bundle SavedModel en `linear_model/1/` para TensorFlow Serving.

## Instrucciones de Uso

### 1. Entrenar y exportar modelo

```bash
python3 linear_model.py
```

### 2. Desplegar en Docker (TensorFlow Serving)

```bash
docker pull tensorflow/serving
docker run -d --name serving_base tensorflow/serving
docker cp linear_model serving_base:/models/linear_model
docker commit --change "ENV MODEL_NAME linear_model" serving_base my_serving_base
docker rm -f serving_base
docker run -d --name linear_model_serving -p 8501:8501 my_serving_base
```

### 3. Probar inferencia con cliente

```bash
python3 client.py
```

## Nota sobre HTTP 400 Bad Request

La capa densa espera entradas de rango 2 con forma `(None, 1)`. Al enviar `{"instances": [0.0, 1.0, 4.0]}` se envian escalares de rango 0, provocando error 400. La estructura correcta enviada por `client.py` es `{"instances": [[0.0], [1.0], [4.0]]}`.
