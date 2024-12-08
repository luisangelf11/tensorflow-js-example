# Configuración Básica de TensorFlow.js 

TensorFlow.js permite desarrollar modelos de aprendizaje automático directamente en JavaScript. Este ejemplo muestra cómo configurar un modelo básico para predecir la suma de dos números, explicando el propósito de cada función y concepto utilizado.

---

## 1. Importar TensorFlow.js

La librería TensorFlow.js es necesaria para trabajar con modelos de aprendizaje automático:

```javascript
import tf from '@tensorflow/tfjs';
```

## 2. Entrenamiento de datos

```javascript
const trainingData = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [2] },
    { input: [2, 2], output: [4] },
    { input: [2, 3], output: [5] },
    { input: [3, 5], output: [8] },
];

```
- input: Representa las características de entrada (los números a sumar).

- output: Representa el resultado esperado (la suma de los números).

`¿Por qué se usa?`
El modelo necesita un conjunto de datos para aprender a hacer predicciones. Estos datos le enseñan qué salida debería asociarse con una entrada específica.

## 3. Convertir datos a tensores

TensorFlow.js utiliza tensores, estructuras de datos optimizadas, para procesar la información de manera eficiente.

```javascript
const xs = tf.tensor2d(trainingData.map(data => data.input)); // Entradas
const ys = tf.tensor2d(trainingData.map(data => data.output)); // Salidas
```
- tf.tensor2d:
Convierte los datos a un tensor bidimensional, que es la estructura que el modelo necesita para entrenar.

Propósito: Facilita operaciones matemáticas intensivas y optimiza el rendimiento computacional.

## 4. Crear modelo

Un modelo define cómo se procesan los datos de entrada para producir salidas. Este ejemplo usa un modelo secuencial, que organiza las capas de forma lineal:

```javascript
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [2], units: 8, activation: 'relu' })); // Capa oculta
model.add(tf.layers.dense({ units: 1, activation: 'linear' })); // Capa de salida
```
### Explicación de las Funciones

Modelo Secuencial (tf.sequential):
Se utiliza para apilar capas de manera ordenada. Es fácil de implementar y adecuado para la mayoría de los casos simples.

Capa Densa (tf.layers.dense):
Es una capa totalmente conectada donde cada neurona recibe entradas de todas las neuronas de la capa anterior.

#### Propiedades:

1. inputShape: [2]:
Define que la entrada tiene 2 características (dos números a sumar).

2. units: 8:
Número de neuronas en la capa, lo que permite al modelo aprender representaciones más complejas.

3. activation: 'relu':
Función de activación utilizada para introducir no linealidad.
ReLU (Rectified Linear Unit) reemplaza valores negativos con cero, ayudando al modelo a aprender patrones más complejos.

Capa de Salida:

1. units: 1:
Una sola neurona para predecir un valor continuo (el resultado de la suma).

2. activation: 'linear':
Se utiliza para problemas de regresión, ya que no altera el valor producido por la neurona.

## 5. Compilar modelo

La compilación del modelo define cómo este aprenderá y qué tan bien evaluará su rendimiento.

```javascript
model.compile({
    optimizer: tf.train.adam(0.1), // Optimizador Adam con tasa de aprendizaje
    loss: 'meanSquaredError', // Función de pérdida para regresión
});
```
### Explicación de los Parámetros

- optimizer:
Es el algoritmo que ajusta los pesos del modelo para reducir la pérdida.

- Adam: Es eficiente y ampliamente usado en problemas de machine learning.

- tf.train.adam(0.1):
Configura Adam con una tasa de aprendizaje de 0.1 para determinar cuánto ajustarán los pesos en cada paso.

- loss:
Es la métrica que evalúa qué tan lejos están las predicciones del modelo de los valores reales.

- meanSquaredError:
Mide el promedio de los cuadrados de los errores, ideal para problemas de regresión.

- categoricalCrossentropy: se utiliza en problemas de clasificación multiclase. Su objetivo es medir la diferencia entre las probabilidades predichas por el modelo y las etiquetas reales, utilizando la entropía cruzada.

## 6. Entrenar el modelo

```javascript
(async () => {
    console.log('Entrenando...');
    await model.fit(xs, ys, {
        epochs: 500, // Iteraciones de entrenamiento
        batchSize: 4, // Número de muestras por lote
        verbose: 1, // Muestra el progreso en consola
    });

    console.log('Entrenamiento completado.');
})();

```