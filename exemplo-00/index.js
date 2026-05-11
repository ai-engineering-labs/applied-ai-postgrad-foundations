import tf from '@tensorflow/tfjs-node';


// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

function trainModel(inputXs, outputYs) {

    const model = tf.sequential();
    //Primeira camada da rede;
    //Entrada de 7 posicoes (idade normalizada + 3 cores + 3 localizacoes)
    //80 neuronios, mais complexidade e a rede pode aprender melhor
    //A Relu age como um filtro. Somente dados positivos passam.
    //se for zero ou negativa infomacao e descartada
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))

    //saida 3 neuronios, pois sao  3 categorias; softmax retorna a probabilidade de cada categoria.
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    //compilar o modelo, adaam é adaptive model, ajusta o peso o model de forma eficiente. Aprende com erros
    //acertos.  Loss: compara o scored de cada categoria com a resposta certa. 
    //metrics:['accuracy'] mostra a precisao do modelo.
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy', //classfiicacao de imagens, recomendacoes, etc.
        metrics: ['accuracy']
    })

    return model;
}

async function predict(model, pessoaTensorNormalizada) {
//transforma o array js para o tensor js 
    const tfInput = tf.tensor2d(pessoaTensorNormalizada)
//faz a predicao (output sera um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    console.log(predArray)
    return predArray[0].map((prob, index)=>({prob, index}))
}

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

inputXs.print();
outputYs.print();

const model = await trainModel(inputXs, outputYs); //await para esperar o modelo ser treinado

// Treinamento do modelo
await model.fit(
    inputXs,
    outputYs,
    {
        verbose: 0,
        epochs: 100,
        shuffle: true, // embaralha os dados para treinar melhor.
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch} - Loss: ${logs.loss}`);
            }
        }
    })


//Teste do modelo
const pessoa = { nome: "José", idade: 28, cor: "verde", localizacao: "São Paulo" }
//normalizando a idade usando mesmo padrao do treino
//exemplo : idade_min=25, idade_max=40, entao(28-25)/(40-25)=0.2
const idadeNormalizada = (pessoa.idade - 25) / (40 - 25);
console.log(idadeNormalizada);

const pessoaTensorNormalizada = [
    [
        idadeNormalizada, //idade normalizada
        0, //azul
        0, //vermelho
        1, //verde
        1,//sao paulo
        0,//rio
        0,//curitiba
    ]// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
];

const predictions = await predict(model, pessoaTensorNormalizada);
const results = predictions
.sort((a,b)=>b.prob-a.prob) // ordena as probabilidades de forma decrescente
.map(p=>`${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)})}`)
.join('\n') // junta as categorias com uma quebra de linha

console.log(results)










