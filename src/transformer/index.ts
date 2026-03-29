import { LayerNormalization, LinearLayer } from "../neural-network/layer"
import { MultiHeadAttention } from "./attention"
import  { Embedding } from "./embedding"
import { FeedForward } from "./feedforward"

export interface TransformerConfig {
  vocabSize: number
  embedSize: number
  numHeads: number
  numLayers: number
  ffnDim: number
  maxSeqLen: number
}

export class Transformer {
  embedding: Embedding
  blocks: TransformerBlock[]
  projection: LinearLayer

  constructor(public config: TransformerConfig) {
    this.embedding = new Embedding(config.vocabSize, config.embedSize, config.maxSeqLen)  
    this.blocks = Array.from({ length: config.numLayers }, () => new TransformerBlock(config.embedSize, config.numHeads, config.ffnDim))
    this.projection = new LinearLayer(config.embedSize, config.vocabSize)
  }

  forward(inputIds: number[]): number[][] {
    let x = this.embedding.forward(inputIds)
    for (const block of this.blocks) {
      x = block.forward(x)
    }
    return this.projection.forward(x) // output shape: [seqLen, vocabSize]  ← logits
  }
}

export class TransformerBlock {
  attention: MultiHeadAttention
  feedforward: FeedForward
  norm1: LayerNormalization
  norm2: LayerNormalization

  constructor(embedSize: number, numHeads: number, ffnDim: number) {
    this.attention = new MultiHeadAttention(embedSize, numHeads)
    this.feedforward = new FeedForward(embedSize, ffnDim)
    this.norm1 = new LayerNormalization(embedSize)
    this.norm2 = new LayerNormalization(embedSize)
  }
  
  forward(x: number[][]): number[][] {
    // x shape: [seqLen, embedDim]

    // attention + residual
    const attnOut = this.attention.forward(this.norm1.forward(x))
    x = x.map((row, i) => row.map((val, j) => val + attnOut[i]![j]!))

    // feedforward + residual
    const ffnOut = this.feedforward.forward(this.norm2.forward(x))
    x = x.map((row, i) => row.map((val, j) => val + ffnOut[i]![j]!))

    return x
  }
}