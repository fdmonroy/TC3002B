# Red Neuronal desde Cero (Clasificacion GPS de Ciudades)

Implementacion didactica de una red neuronal profunda desarrollada desde cero con NumPy (sin frameworks de caja negra) para resolver un problema de clasificacion binaria sobre coordenadas geograficas (latitud y longitud).

## Estructura del Directorio

```text
cw-neural-network-scratch/
├── .gitignore                  # Exclusion de entorno virtual y archivos temporales
├── dataset.py                  # Generador de coordenadas GPS y aplicacion Streamlit
├── neural_network_numpy.py     # Algoritmo de red neuronal, entrenamiento y exportacion
├── README.md                   # Documentacion tecnica e instrucciones de ejecucion
├── template_red_neuronal.xlsx  # Plantilla de validacion matematica
├── validation_weights.xlsx     # Hoja Excel generada con pesos y evaluacion de 20 puntos
└── data/
    ├── city1.csv               # Coordenadas de Ciudad 1 (Buenos Aires, etiqueta 0)
    └── city2.csv               # Coordenadas de Ciudad 2 (El Cairo, etiqueta 1)
```

## Arquitectura de la Red

La red implementa una topologia perceptron multicapa de 3 capas densas:
- Capa de Entrada: 2 neuronas (latitud, longitud).
- Capa Oculta 1: 4 neuronas, funcion de activacion ReLU ($W_1 \in \mathbb{R}^{2 \times 4}$, $b_1 \in \mathbb{R}^{1 \times 4}$).
- Capa Oculta 2: 8 neuronas, funcion de activacion ReLU ($W_2 \in \mathbb{R}^{4 \times 8}$, $b_2 \in \mathbb{R}^{1 \times 8}$).
- Capa de Salida: 1 neurona, funcion de activacion Sigmoide ($W_3 \in \mathbb{R}^{8 \times 1}$, $b_3 \in \mathbb{R}^{1 \times 1}$).

## Algoritmo de Aprendizaje

1. Propagacion hacia adelante (Forward Pass):
   $$Z_1 = X W_1 + b_1, \quad A_1 = \max(0, Z_1)$$
   $$Z_2 = A_1 W_2 + b_2, \quad A_2 = \max(0, Z_2)$$
   $$Z_3 = A_2 W_3 + b_3, \quad A_3 = \frac{1}{1 + e^{-Z_3}}$$

2. Funcion de Costo (Mean Squared Error):
   $$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (A_3^{(i)} - Y^{(i)})^2$$

3. Retropropagacion del Error (Backpropagation):
   $$\delta_3 = (A_3 - Y) \odot A_3 (1 - A_3)$$
   $$\delta_2 = (\delta_3 W_3^T) \odot \mathbb{I}(A_2 > 0)$$
   $$\delta_1 = (\delta_2 W_2^T) \odot \mathbb{I}(A_1 > 0)$$

4. Descenso de Gradiente:
   $$W_l \leftarrow W_l - \alpha (A_{l-1}^T \delta_l), \quad b_l \leftarrow b_l - \alpha \sum \delta_l$$

## Instrucciones de Uso

### 1. Generar Datasets

Genera 300 puntos sinteticos por ciudad via linea de comandos:

```bash
python3 dataset.py --cli --points 300
```

Para visualizar las coordenadas interactivamente en un mapa web:

```bash
streamlit run dataset.py
```

### 2. Entrenar y Validar la Red

Ejecuta el entrenamiento de la red por 500 epocas, imprime las predicciones sobre 20 puntos de prueba (10 de cada ciudad) y actualiza la hoja de validacion Excel:

```bash
python3 neural_network_numpy.py
```

### 3. Validacion en Excel

El archivo `validation_weights.xlsx` contiene la evaluacion matematica exacta en celdas de Excel:
- Renglones 3 a 12: 10 coordenadas de prueba de Ciudad 1 (salida aproximada a 0.0).
- Renglones 13 a 22: 10 coordenadas de prueba de Ciudad 2 (salida aproximada a 1.0).
- Celdas `E3:H4` y `E8:H8`: Pesos $W_1$ y sesgos $b_1$.
- Celdas `T3:AA6` y `T9:AA9`: Pesos $W_2$ y sesgos $b_2$.
- Celdas `AU3:AU10` y `AU13`: Pesos $W_3$ y sesgo $b_3$.
