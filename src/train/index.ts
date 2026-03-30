import fs from 'fs'
import path from 'path'
import { resolve } from 'path'
import { existsSync } from 'fs'
import { loadConfig, resolveTokenizerConfig, resolvePretrainTokenizeConfig, resolveTrainConfig, type TrainConfig } from '../config'
import { Transformer } from '../transformer'
import { loadTokenizer } from '../tokenizer'
import { softmax } from '../neural-network/activation'
import { AdamOptimizer } from './optimizer'
import { saveCheckpoint, loadCheckpoint, latestCheckpoint } from './checkpoint'

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

function formatDuration(ms: number): string {
  const s = Math.floor(ms / 1000)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h > 0) return `${h}h ${m}m ${sec}s`
  if (m > 0) return `${m}m ${sec}s`
  return `${sec}s`
}

export async function runTrain(opts: { reset?: boolean } = {}) {
  const config = loadConfig()
  const trainCfg = resolveTrainConfig(config)
  const tokenizerCfg = resolveTokenizerConfig(config)
  const tokenizeCfg = resolvePretrainTokenizeConfig(config)

  const dataFile = resolve(process.cwd(), trainCfg.dataFile)
  const vocabDir = resolve(process.cwd(), tokenizeCfg.outputDir)
  const checkpointDir = resolve(process.cwd(), 'data/checkpoints')
  const saveEvery = 10000

  if (!existsSync(dataFile)) {
    console.error(`[train] Data file not found: ${dataFile}`)
    console.error(`[train] Run "pretrain encode" first.`)
    process.exit(1)
  }

  // model & komponen
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
  const dataLoader = new DataLoader(dataFile, 1, trainCfg.maxSeqLen)

  // reset: hapus semua checkpoint lama
  if (opts.reset && existsSync(checkpointDir)) {
    fs.rmSync(checkpointDir, { recursive: true, force: true })
    console.log(`[train] Checkpoint direset, mulai dari awal.`)
  }

  // resume dari checkpoint jika ada
  let startStep = 0
  const latest = latestCheckpoint(checkpointDir)
  if (latest) {
    const meta = loadCheckpoint(model, latest)
    startStep = meta.step + 1
  }

  const maxSteps = trainCfg.epochs * Math.floor(dataLoader.tokens.length / trainCfg.maxSeqLen)
  console.log(`[train] Steps: ${startStep} → ${maxSteps} | lr: ${trainCfg.learningRate} | seqLen: ${trainCfg.maxSeqLen}`)

  // training loop
  let lossAccum = 0
  let lossCount = 0
  const trainStart = Date.now()
  let cpuSnapshot = process.cpuUsage()
  let timeSnapshot = Date.now()

  for (let step = startStep; step < maxSteps; step++) {
    // ambil batch, reset jika data habis
    let batch = dataLoader.nextBatch()
    if (!batch) {
      dataLoader.reset()
      batch = dataLoader.nextBatch()!
    }

    const inputIds  = batch.inputIds[0]!
    const targetIds = batch.targetIds[0]!

    // forward
    const logits = model.forward(inputIds)
    const loss = crossEntropyLoss(logits, targetIds)
    lossAccum += loss
    lossCount++

    // backward + update
    const dLogits = softmaxCrossEntropyGradient(trainCfg, logits, targetIds)
    model.backward(dLogits)
    optimizer.step()

    // log setiap 10 step
    if (step % 10 === 0) {
      const avgLoss = lossAccum / lossCount

      // argmax tanpa spread agar aman untuk array besar
      const lastLogits = logits[logits.length - 1]!
      let maxVal = -Infinity, predictedId = 0
      for (let k = 0; k < lastLogits.length; k++) {
        if (lastLogits[k]! > maxVal) { maxVal = lastLogits[k]!; predictedId = k }
      }
      const predictedToken = tokenizer.idToToken(predictedId)

      // ETA
      const elapsed = Date.now() - trainStart
      const stepsCompleted = step - startStep + 1
      const msPerStep = elapsed / stepsCompleted
      const remaining = (maxSteps - step) * msPerStep
      const etaStr = formatDuration(remaining)

      // CPU usage sejak snapshot terakhir
      const cpuDelta = process.cpuUsage(cpuSnapshot)
      const timeDelta = (Date.now() - timeSnapshot) * 1000  // ke microseconds
      const cpuPct = timeDelta > 0
        ? ((cpuDelta.user + cpuDelta.system) / timeDelta * 100).toFixed(1)
        : '0.0'
      cpuSnapshot = process.cpuUsage()
      timeSnapshot = Date.now()

      const mem = process.memoryUsage()
      const heapMB = (mem.heapUsed / 1024 / 1024).toFixed(1)
      const rssMB  = (mem.rss      / 1024 / 1024).toFixed(1)

      console.log(
        `[train] step ${step}/${maxSteps}` +
        ` | loss: ${avgLoss.toFixed(4)}` +
        ` | predicted: "${predictedToken}"` +
        ` | eta: ${etaStr}` +
        ` | cpu: ${cpuPct}%` +
        ` | heap: ${heapMB}MB rss: ${rssMB}MB`
      )
      lossAccum = 0
      lossCount = 0
    }

    // simpan checkpoint setiap saveEvery step
    const saved = saveCheckpoint(model, { step, loss, savedAt: new Date().toISOString() }, checkpointDir, (step % saveEvery === 0 && step > 0))
    // if (step % saveEvery === 0 && step > 0) {
    //   const saved = saveCheckpoint(model, { step, loss, savedAt: new Date().toISOString() }, checkpointDir)
    //   console.log(`[checkpoint] Saved → ${saved}`)
    // }
  }

  // simpan checkpoint akhir
  saveCheckpoint(model, { step: maxSteps, loss: lossAccum / Math.max(lossCount, 1), savedAt: new Date().toISOString() }, checkpointDir)
  console.log(`[train] Done.`)
}

/**
 * Fungsi manual untuk testing dan pembelajaran step-by-step.
 * Tidak dipanggil di production — hanya untuk eksperimen.
 *
 * @example
 * // manualTestTrain()
 */



// dont delete, just for learning and testing :
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
