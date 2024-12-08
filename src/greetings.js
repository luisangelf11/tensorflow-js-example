import tf from '@tensorflow/tfjs'

//Dictionary of data
const greetings = {
    "hola": 0,
    "buenas tardes": 1,
    "buenas noches": 2,
    "buenos dias": 4,
    "adios": 5
}

const indexToResponse = ["hello", "good afternoon", "good evening", "good morning", "good bye"]

//Data training
const trainingData = [
    {input: "hola", output: [1,0,0,0,0]},
    {input: "buenas tardes", output: [0,1,0,0,0]},
    {input: "buenas noches", output: [0,0,1,0,0]},
    {input: "buenos dias", output: [0,0,0,1,0]},
    {input: "adios", output: [0,0,0,0,1]},
]

//Convert data to tensors
const xs = tf.tensor2d(trainingData.map(data => [greetings[data.input]]))
const ys = tf.tensor2d(trainingData.map(data => data.output))

//Create model
const model = tf.sequential()
model.add(tf.layers.dense({inputShape: [1], units: 8, activation: 'relu'})) //Hidde
model.add(tf.layers.dense({units: 5, activation: 'softmax'})) //Output

//Compiler model
model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "categoricalCrossentropy"
});

//Train model
(async()=>{
    console.log('Training....')
    await model.fit(xs, ys, {
        epochs: 300,
        batchSize: 4,
        verbose: 1
    })
    console.log('Train completed!')
    //new data test
    const testWords = ["hola", "adios", "buenos dias", "buenas tardes", "buenas noches"]
    const testIndexs = tf.tensor2d(testWords.map(word => [greetings[word]]))

    const predictions = model.predict(testIndexs)
    predictions.array().then(array =>{
        console.log('Responses:')
        array.forEach((pred, i)=>{
            const index = pred.indexOf(Math.max(...pred))
            console.log(`Word: "${testWords[i]}" => Respuesta: "${indexToResponse[index]}"`);
        })
    })
})();