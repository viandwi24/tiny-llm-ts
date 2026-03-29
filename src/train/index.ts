import fs from 'fs'
import path from 'path'
import { readFileSync, existsSync } from 'fs'
import type { TrainConfig } from '../config'
import { Transformer } from '../transformer'

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
    console.log(`Loaded ${this.tokens.length} tokens from ${trainDataPath}`)
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
}

export async function runTrain(opts: TrainConfig) {
  if (!existsSync(opts.dataFile)) {
    console.error(`[train] Data file not found: ${opts.dataFile}`)
    console.error(`[train] Run "pretrain encode" first.`)
    process.exit(1)
  }

  console.log(`[train] Starting training with config:`, opts)

  const model = new Transformer({
    vocabSize: opts.vocabSize,
    embedSize: opts.embedSize,
    numHeads: opts.numHeads,
    numLayers: opts.numLayers,
    ffnDim: opts.ffnDim,
    maxSeqLen: opts.maxSeqLen,
  })

  const trainer = new Trainer(model, {
    batchSize: opts.batchSize,
    learningRate: opts.learningRate,
    maxSteps: opts.epochs * Math.ceil(opts.dataFile.length / (opts.batchSize * opts.maxSeqLen)),
    sequenceLength: opts.maxSeqLen,
  })

  const logits = model.forward([0, 1, 2])
  console.log(`[train] Sample logits shape`, logits)
}