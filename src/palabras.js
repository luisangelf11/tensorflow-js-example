import tf from '@tensorflow/tfjs'

// Diccionario de palabras clave y respuestas
const greetings = {
  "hola": "¡Hola! ¿Cómo estás?",
  "buenos días": "¡Buenos días! Espero que tengas un gran día.",
  "buenas tardes": "¡Buenas tardes! ¿Cómo va tu día?",
  "buenas noches": "¡Buenas noches! Que descanses bien.",
};

// Convertir palabras a índices numéricos
const wordToIndex = {
  "hola": 0,
  "buenos días": 1,
  "buenas tardes": 2,
  "buenas noches": 3,
};

// Respuestas codificadas
const indexToResponse = [
  "¡Hola! ¿Cómo estás?",
  "¡Buenos días! Espero que tengas un gran día.",
  "¡Buenas tardes! ¿Cómo va tu día?",
  "¡Buenas noches! Que descanses bien.",
];

// Datos de entrenamiento
const trainingData = [
  { input: "hola", output: [1, 0, 0, 0] },
  { input: "buenos días", output: [0, 1, 0, 0] },
  { input: "buenas tardes", output: [0, 0, 1, 0] },
  { input: "buenas noches", output: [0, 0, 0, 1] },
];

// Convertir datos a tensores
const xs = tf.tensor2d(trainingData.map(data => [wordToIndex[data.input]])); // Entradas como índices
const ys = tf.tensor2d(trainingData.map(data => data.output)); // Salidas como one-hot

// Crear el modelo
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [1], units: 8, activation: 'relu' })); // Capa oculta
model.add(tf.layers.dense({ units: 4, activation: 'softmax' })); // Capa de salida

// Compilar el modelo
model.compile({
  optimizer: tf.train.adam(0.01),
  loss: 'categoricalCrossentropy',
});

// Entrenar el modelo
(async () => {
  console.log('Entrenando...');
  await model.fit(xs, ys, {
    epochs: 300,
    batchSize: 4,
    verbose: 1,
  });

  console.log('Entrenamiento completado.');

  // Probar el modelo con nuevos datos
  const testWords = ["hola", "buenas tardes", "buenas noches", "buenos días", "buenas tarde"];
  const testIndices = tf.tensor2d(testWords.map(word => [wordToIndex[word]]));

  const predictions = model.predict(testIndices);
  predictions.array().then(array => {
    console.log('Respuestas:');
    array.forEach((pred, i) => {
      const index = pred.indexOf(Math.max(...pred)); // Obtener la clase más probable
      console.log(`Palabra: "${testWords[i]}" => Respuesta: "${indexToResponse[index]}"`);
    });
  });
})();
