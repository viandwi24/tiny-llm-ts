import fs from 'fs'
import path from 'path'
import { resolve } from 'path'
import { existsSync } from 'fs'
import { loadConfig, resolveTokenizerConfig, resolvePretrainTokenizeConfig, resolveTrainConfig, type TrainConfig } from '../config'
import { Transformer } from '../transformer'
import { loadTokenizer } from '../tokenizer'
import { softmax } from '../neural-network/activation'
import { AdamOptimizer } from './optimizer'

export interface TrainerConfig {
  maxSteps: number
  batchSize: number
  sequenceLength: number
  learningRate: number
}

export class Trainer {
  dataLoader: DataLoader

  constructor(
    public model: Transformer,
    public config: TrainerConfig,
  ) {
    this.dataLoader = new DataLoader(
      path.resolve(process.cwd(), 'data/pretrain/train.bin'),
      config.batchSize,
      config.sequenceLength,
    )
  }
}

export class DataLoader {
  tokens: Uint16Array
  currentIndex: number

  constructor(
    public trainDataPath: string,
    public batchSize: number,
    public sequenceLength: number,
  ) {
    const data = fs.readFileSync(trainDataPath)
    this.tokens = new Uint16Array(data.buffer, data.byteOffset, data.byteLength / 2)
    this.currentIndex = 0
    console.log(`[train] Loaded ${this.tokens.length} tokens from ${trainDataPath}`)
  }

  nextBatch(): { inputIds: number[][], targetIds: number[][] } | null {
    if (this.currentIndex + this.batchSize * this.sequenceLength >= this.tokens.length) {
      return null // no more data
    }

    const inputIds: number[][] = []
    const targetIds: number[][] = []

    for (let i = 0; i < this.batchSize; i++) {
      const startIdx = this.currentIndex + i * this.sequenceLength
      const endIdx = startIdx + this.sequenceLength
      inputIds.push(Array.from(this.tokens.slice(startIdx, endIdx)))
      targetIds.push(Array.from(this.tokens.slice(startIdx + 1, endIdx + 1)))
    }

    this.currentIndex += this.batchSize * this.sequenceLength
    return { inputIds, targetIds }
  }

  reset(): void {
    this.currentIndex = 0
  }
}


const crossEntropyLoss = (logits: number[][], targetIds: number[]): number => {
  let loss = 0
  let loss_i = 0
  let probs: number[] = []
  let correctId: number = -1
  for (let i = 0; i < logits.length; i++) {
    probs = softmax(logits[i]!)
    correctId = targetIds[i]!
    loss_i = -Math.log(probs[correctId] ?? 1e-10) // add small value to prevent log(0)
    loss += loss_i
    // console.log(`[loss] Step ${i}: targetId=${correctId}, loss_i=${loss_i.toFixed(4)}`)
  }
  return loss / logits.length
}
const softmaxCrossEntropyGradient = (cfg: TrainConfig, logits: number[][], targetIds: number[]): number[][] => {
  // copy of zeros with same shape as logits
  const dLogits = logits.map(row => row.map(() => 0))

  let probs: number[] = []
  for (let i = 0; i < logits.length; i++) {
    probs = softmax(logits[i]!)
    for (let j = 0; j < cfg.vocabSize; j++) {
      dLogits[i]![j]! = probs[j]!
    }
    dLogits[i]![targetIds[i]!]! -= 1 // subtract 1 from the correct class
  }

  return dLogits.map(row => row.map(val => val / logits.length)) // average over batch
}

export async function runTrain() {
  const config = loadConfig()
  const trainCfg = resolveTrainConfig(config)
  const tokenizerCfg = resolveTokenizerConfig(config)
  const tokenizeCfg = resolvePretrainTokenizeConfig(config)

  const dataFile = resolve(process.cwd(), trainCfg.dataFile)
  const vocabDir = resolve(process.cwd(), tokenizeCfg.outputDir)

  if (!existsSync(dataFile)) {
    console.error(`[train] Data file not found: ${dataFile}`)
    console.error(`[train] Run "pretrain encode" first.`)
    process.exit(1)
  }

  manualTestTrain()

  // // manual test
  // const tokenizer = loadTokenizer(vocabDir, tokenizerCfg.charMarker)

  // const testText = 'jakarta adalah'
  // const testIds = tokenizer.encode(testText)
  // const testTokens = testIds.map(id => tokenizer.idToToken(id))

  // console.log(`\n[test] Input text   : "${testText}"`)
  // console.log(`[test] Token IDs    : [${testIds.join(', ')}]`)
  // console.log(`[test] Tokens       : [${testTokens.map(t => `"${t}"`).join(', ')}]`)
  // console.log(`[test] Sequence len : ${testIds.length}`)

  
  // const model = new Transformer({
  //   vocabSize: trainCfg.vocabSize,
  //   embedSize: trainCfg.embedSize,
  //   numHeads: trainCfg.numHeads,
  //   numLayers: trainCfg.numLayers,
  //   ffnDim: trainCfg.ffnDim,
  //   maxSeqLen: trainCfg.maxSeqLen,
  // })

  // const optimizer = new AdamOptimizer(model, trainCfg.learningRate)


  // // run backward pass with dLogits
  // model.backward(dLogits)

  // // update model parameters
  // optimizer.step()

  // // check
  // const resAfter = model.forward(testIds)
  // const lossAfter = crossEntropyLoss(resAfter, targetIds)
  // console.log(`[test] Loss after one update: ${lossAfter.toFixed(4)}`)
}



function manualTestTrain() {
  const config = loadConfig()
  const trainCfg = resolveTrainConfig(config)
  const tokenizerCfg = resolveTokenizerConfig(config)
  const tokenizeCfg = resolvePretrainTokenizeConfig(config)
  const dataFile = resolve(process.cwd(), trainCfg.dataFile)
  const vocabDir = resolve(process.cwd(), tokenizeCfg.outputDir)

  // model architecture
  const model = new Transformer({
    vocabSize: trainCfg.vocabSize,
    embedSize: trainCfg.embedSize,
    numHeads: trainCfg.numHeads,
    numLayers: trainCfg.numLayers,
    ffnDim: trainCfg.ffnDim,
    maxSeqLen: trainCfg.maxSeqLen,
  })
  const optimizer = new AdamOptimizer(model, trainCfg.learningRate)
  const tokenizer = loadTokenizer(vocabDir, tokenizerCfg.charMarker)
  
  // manual text
  const text = 'ibukota negara indonesia'
  const textIds = tokenizer.encode(text)
  const textTokens = textIds.map(id => tokenizer.idToToken(id))

  // batchs dataset examples
  const batchs = [
    'Biologi evolusioner adalah cabang dari ilmu biologi yang mempelajari proses-proses evolusioner seperti seleksi alam',
    'Gagasan-gagasan mengenai biologi evolusioner telah ada sejak masa Aristoteles',
    'Spesifisitas biologi adalah kecenderugan untuk suatu variasi perilaku atau biokimia untuk terjadi pada suatu spesies',
    'Geografi adalah ilmu yang mempelajari tentang hubungan, persamaan, dan perbedaan antar ruang di Bumi',
    'Indonesia, dengan nama resmi Republik Indonesia, adalah sebuah negara kepulauan di Asia Tenggara dan Oseania',
  ]


  // loop
  for (let step = 0; step < 5; step++) {
    // bacth data
    const batchText = batchs[step % batchs.length]!
    const batchIds = tokenizer.encode(batchText)

    // predict
    const res = model.forward(batchIds)
    const predictedIndex = res[res.length-1]
    const predictedTokenId = predictedIndex?.indexOf(Math.max(...predictedIndex)) ?? -1
    const predictedToken = tokenizer.idToToken(predictedTokenId)
    console.log(`[${step + 1}/${trainCfg.epochs}] [test] input : "${batchText}"`)

    // prepare machine learning
    const targetIds = batchIds.slice(1).concat([0])
    // calc loss
    const loss = crossEntropyLoss(res, targetIds)
    // calc dlogits (derivative of loss / gradient of logits)
    const dLogits = softmaxCrossEntropyGradient(trainCfg, res, targetIds)
    console.log(`[${step + 1}/${trainCfg.epochs}] [test] predicted token: "${predictedToken}" (id: ${predictedTokenId}, loss: ${loss.toFixed(4)})`)

    // backward pass
    model.backward(dLogits)

    // update parameters
    optimizer.step()
  }


  // test generate
  // while (true) {
  //   console.log(`${trainCfg.epochs}] [test] input : "${text}" (tokens: [${textIds.map(id => tokenizer.idToToken(id)).join(', ')}])`)
  //   const res = model.forward(textIds)
  //   const predictedIndex = res[res.length-1]
  //   const predictedTokenId = predictedIndex?.indexOf(Math.max(...predictedIndex)) ?? -1
  //   const predictedToken = tokenizer.idToToken(predictedTokenId)
  //   process.stdout.write(predictedToken)
  // }
}
