import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
//the net 
const WEIGHTS = {
    category: 0.4,//most important
    color: 0.3,
    price: 0.2,
    age: 0.1
}

// Normalize continuous values (price, age) to 0–1 range
// Why? Keeps all features balanced so no one dominates training
// Formula: (val - min) / (max - min)
// Example: price=129.99, minPrice=39.99, maxPrice=199.99 → 0.56

const normalize = (value, min, max) => (value - min) / ((max - min) || 1)


function makeContext(catalog, users) {
    //normalize
    const ages = users.map(u => u.age)
    const prices = catalog.map(p => p.price)

    const minAge = Math.min(...ages)
    const maxAge = Math.max(...ages)

    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)

    const colors = [...new Set(catalog.map(p => p.color))]
    const categories = [...new Set(catalog.map(p => p.category))]

    //map the index
    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        }))

    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        }))

    //calculate the average of the products purchased, help to personalise
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    //forEach executes a provided function once for each array element
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
    //transform to tensors
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
        dimensions: 2 + categories.length + colors.length,//price+age+color+categories
        productAvgAgeNorm,
        categoriesIndex,
        colorsIndex
    }
}

const oneHotWeighted = (index, lenght, weight) =>
    tf.oneHot(index, lenght).cast('float32').mul(weight)

function encodeProduct(product, context) {
    //Normalizing to a range between 0 and 1 and applying weight to the normalization.
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

function encodeUser(user, context) {

    //return the shopping profile
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
}

function createTrainingData(context) {
    //walk through each of the users on the list 
    const inputs = []
    const labels = []

    context.users.forEach(user => {
        const useVector = encodeUser(user, context).dataSync()
        context.products.forEach(product => {
            const productVector = encodeProduct(product, context)
                .dataSync()

            const label = user.purchases.some(
                purchase => purchase.name === product.name ?
                    1 : 0
            )
            //combine user   + prouct 
            inputs.push([...useVector, ...productVector])
            labels.push(label)
        })

    })
    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimension: context.dimensions * 2
        //the lenght is the user vector  + productVector
    }

}

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
    debugger
    const trainData = createTrainingData(context);
    debugger
    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);


}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
