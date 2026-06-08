import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = null

// Feature importance weights used when encoding products and users.
// Higher values give more influence to that feature during training.
const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
}

// Scales a numeric value to the 0–1 range using min-max normalization.
// Keeps continuous features (price, age) balanced so none dominates training.
// Formula: (value - min) / (max - min)
const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

// Builds the shared context object used across encoding, training, and inference.
// Computes normalization bounds, category/color index maps, and per-product average buyer age.
function makeContext(catalog, users) {
    const ages = users.map(u => u.age)
    const prices = catalog.map(p => p.price)

    const minAge = Math.min(...ages)
    const maxAge = Math.max(...ages)

    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)

    const colors = [...new Set(catalog.map(p => p.color))]
    const categories = [...new Set(catalog.map(p => p.category))]

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        }))

    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        }))

    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(
            p => {
                ageSums[p.name] = (ageSums[p.name] || 0) + user.age
                ageCounts[p.name] = (ageCounts[p.name] || 0) + 1
            }
        )
    });

    const productAvgAgeNorm = Object.fromEntries(
        catalog.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name]
                : midAge
            return [product.name, normalize(avg, minAge, maxAge)];
        })
    )

    return {
        catalog,
        products: catalog,
        users,
        colorsIndex,
        categoriesIndex,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        dimensions: 2 + categories.length + colors.length,
        productAvgAgeNorm,
        categoriesIndex,
        colorsIndex
    }
}

// Creates a weighted one-hot encoded tensor for categorical features (category, color).
const oneHotWeighted = (index, lenght, weight) =>
    tf.oneHot(index, lenght).cast('float32').mul(weight)

// Encodes a single product into a fixed-size numeric feature vector.
// Combines normalized price, average buyer age, weighted category, and weighted color.
function encodeProduct(product, context) {
    const price = tf.tensor1d([
        normalize(
            product.price,
            context.minPrice,
            context.maxPrice) * WEIGHTS.price
    ])

    const age = tf.tensor1d([
        (
            context.productAvgAgeNorm[product.name] ?? 0.5
        ) * WEIGHTS.age
    ])

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    )
    return tf.concat1d(
        [price, age, category, color]
    )
}

// Encodes a user into a single feature vector representing their shopping profile.
// If the user has purchases, averages the encoded purchased products.
// Otherwise, falls back to age-only encoding with zeroed product-related features.
function encodeUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(
            user.purchases.map(
                product => encodeProduct(product, context)
            )
        )
            .mean(0)
            .reshape([
                1,
                context.dimensions
            ])
    }
    return tf.concat1d([
        tf.zeros([1]),
        tf.tensor1d([
            normalize(user.age, context.minAge, context.maxAge)
            * WEIGHTS.age
        ]),
        tf.zeros([context.numCategories]),
        tf.zeros([context.numColors]),
    ]).reshape([1, context.dimensions])
}

// Generates supervised training examples from all users with purchase history.
// Each sample pairs a user vector with a product vector; label is 1 if purchased, 0 otherwise.
function createTrainingData(context) {
    const inputs = []
    const labels = []

    context.users
        .filter(u => u.purchases.length)
        .forEach(user => {
            const useVector = encodeUser(user, context).dataSync()
            context.products.forEach(product => {
                const productVector = encodeProduct(product, context)
                    .dataSync()

                const label = user.purchases.some(
                    purchase => purchase.name === product.name ?
                        1 : 0
                )
                inputs.push([...useVector, ...productVector])
                labels.push(label)
            })

        })
    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimension: context.dimensions * 2
    }

}

// Builds a sequential neural network, compiles it, and trains on the provided data.
// Architecture: 128 → 64 → 32 ReLU hidden layers, sigmoid output for binary classification.
// Streams epoch logs back to the main thread during training.
async function configureNeuralNetAndTrain(trainData) {
    const model = tf.sequential()
    model.add(
        tf.layers.dense(
            {
                inputShape: [trainData.inputDimension],
                units: 128,
                activation: 'relu',

            }
        )
    )
    model.add(
        tf.layers.dense(
            {
                units: 64,
                activation: 'relu',

            }
        )
    )
    model.add(
        tf.layers.dense(
            {
                units: 32,
                activation: 'relu',

            }
        )
    )
    model.add(
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
    )

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    })
    return model

}

// Full training pipeline: loads catalog, builds context, precomputes product vectors,
// creates training data, trains the model, and notifies the main thread when complete.
async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const catalog = await (await fetch('/data/products.json')).json()

    const context = makeContext(catalog, users)

    context.productVectors = catalog.map(product => {
        return {
            name: product.name,
            meta: { ...product },
            vector: encodeProduct(product, context).dataSync()
        }
    })


    _globalCtx = context

    const trainData = createTrainingData(context);
    _model = await configureNeuralNetAndTrain(trainData)
    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });
    postMessage({ type: workerEvents.trainingComplete });
}

// Runs inference for a given user by pairing their vector with every product vector.
// Builds the input tensor used to score how likely the user is to buy each product.
function recommend(user, ctx) {
    if (!_model) return;
    const context = _globalCtx

    const useVector = encodeUser(user, _globalCtx).dataSync()
    const inputs = context.productVectors.map(({ vector }) => {
        return [...useVector, ...vector]
    })

    const inputTensor = tf.tensor2d(inputs)
    const predictions = _model.predict(inputTensor)

    const scores = predictions.dataSync()
    const recommendations = context.productVectors.map((item, index) => {
        return {
            ...item.meta,
            name: item.name,
            score: scores[index]
        }
    })
    const sortedItems = recommendations
        .sort((a, b) => b.score - a.score)


     postMessage({
         type: workerEvents.recommend, 
         user,
         recommendations: sortedItems});

}

// Maps incoming worker messages to their corresponding handler functions.
const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

// Entry point: receives messages from the main thread and dispatches by action type.
self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
