# Red de Hopfield con Figuras 8x5

Implementacion didactica de una red de Hopfield discreta para reconocimiento de patrones bidimensionales en cuadriculas de 8 renglones por 5 columnas (40 neuronas).

El codigo esta escrito en Python puro sin dependencias externas como NumPy o TensorFlow. Sigue los principios Keep It Simple y Do It Yourself.

## Estructura del Directorio

```text
shapes-hopfield/
├── dataset/
│   ├── digit_0.txt
│   ├── digit_1.txt
│   ├── digit_2.txt
│   ├── digit_4.txt
│   └── test/
│       ├── test_clean_0.txt
│       ├── test_noisy_1.txt
│       ├── test_noisy_2.txt
│       ├── test_noisy_4.txt
│       └── test_shifted_1.txt
├── hopfield.py
└── README.md
```

## Formato de los Archivos de Datos

Cada figura se representa como una cuadricula de texto de 8 filas y 5 columnas:
- `1` representa una celda activa o negra.
- `-1` representa una celda inactiva o blanca.

Cada patron se aplana renglon por renglon a un vector bipolar de 40 elementos.

## Algoritmo

1. Carga de patrones: Lee los archivos de texto en `dataset/` y aplana las matrices a vectores de 40 dimensiones.
2. Entrenamiento: Calcula el producto exterior de cada vector consigo mismo y suma las matrices resultantes. Se asigna cero a la diagonal principal para eliminar la auto retroalimentacion.
3. Inferencia: Recibe un patron de prueba y actualiza las neuronas de forma iterativa mediante la funcion de activacion escalon bipolar hasta alcanzar un estado estable.
4. Clasificacion: Determina el patron almacenado mas cercano utilizando el producto punto.

## Instrucciones de Uso

### Ejecutar todas las pruebas

Para entrenar con el dataset y evaluar todos los archivos de prueba en `dataset/test/`:

```bash
python3 hopfield.py
```

### Probar un archivo especifico

Para evaluar un archivo particular:

```bash
python3 hopfield.py dataset/test/test_shifted_1.txt
```
