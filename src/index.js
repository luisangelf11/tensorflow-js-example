import tf from '@tensorflow/tfjs'

// Datos de entrenamiento: suma de dos números
const trainingData = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [2] },
    { input: [2, 2], output: [4] },
    { input: [2, 3], output: [5] },
    { input: [3, 5], output: [8] },
];

// Convertir datos a tensores
const xs = tf.tensor2d(trainingData.map(data => data.input)); // Entradas
const ys = tf.tensor2d(trainingData.map(data => data.output)); // Salidas

// Crear el modelo
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [2], units: 8, activation: 'relu' })); // Capa oculta
model.add(tf.layers.dense({ units: 1, activation: 'linear' })); // Capa de salida

// Compilar el modelo
model.compile({
    optimizer: tf.train.adam(0.1), // Optimizador Adam con tasa de aprendizaje
    loss: 'meanSquaredError', // Pérdida para regresión
});

// Entrenar el modelo
(async () => {
    console.log('Entrenando...');
    await model.fit(xs, ys, {
        epochs: 500, // Más épocas para mejorar el aprendizaje
        batchSize: 4, // Tamaño de lote pequeño
        verbose: 1, // Ver progreso durante el entrenamiento
    });

    console.log('Entrenamiento completado.');

    // Realizar predicciones
    const testInput = tf.tensor2d([[1, 2], [3, 5], [4, 6], [10, 20]]); // Ejemplos de prueba
    const predictions = model.predict(testInput);
    predictions.array().then(array => {
        console.log('Predicciones:');
        array.forEach((pred, i) => {
            console.log(`Entrada: ${testInput.arraySync()[i]} => Predicción: ${pred[0].toFixed(2)}`);
        });
    });
})();