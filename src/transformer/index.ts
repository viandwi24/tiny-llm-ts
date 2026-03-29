import { LayerNormalization, LinearLayer } from "../neural-network/layer"
import { MultiHeadAttention } from "./attention"
import { Embedding } from "./embedding"
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

  // cache untuk backward
  cacheInputIds: number[] = []
  cachePreProjection: number[][] = []  // output block terakhir, input ke projection

  constructor(public config: TransformerConfig) {
    this.embedding = new Embedding(config.vocabSize, config.embedSize, config.maxSeqLen)
    this.blocks = Array.from({ length: config.numLayers }, () => new TransformerBlock(config.embedSize, config.numHeads, config.ffnDim))
    this.projection = new LinearLayer(config.embedSize, config.vocabSize)
  }

  forward(inputIds: number[]): number[][] {
    this.cacheInputIds = inputIds
    let x = this.embedding.forward(inputIds)
    for (const block of this.blocks) {
      x = block.forward(x)
    }
    this.cachePreProjection = x
    return this.projection.forward(x)  // output shape: [seqLen, vocabSize] ← logits
  }

  backward(dLogits: number[][]): void {
    // backward melalui projection
    const { dInput: dX } = this.projection.backward(this.cachePreProjection, dLogits)

    // backward melalui blocks (urutan terbalik)
    let dBlock = dX
    for (let i = this.blocks.length - 1; i >= 0; i--) {
      dBlock = this.blocks[i]!.backward(dBlock)
    }

    // backward melalui embedding
    this.embedding.backward(this.cacheInputIds, dBlock)
  }
}

export class TransformerBlock {
  attention: MultiHeadAttention
  feedforward: FeedForward
  norm1: LayerNormalization
  norm2: LayerNormalization

  // cache untuk backward
  private cacheXin: number[][] = []      // input ke block (sebelum norm1)
  private cacheNorm1Out: number[][] = [] // output norm1 (input ke attention)
  private cacheXmid: number[][] = []     // setelah residual attention (sebelum norm2)
  private cacheNorm2Out: number[][] = [] // output norm2 (input ke feedforward)

  constructor(embedSize: number, numHeads: number, ffnDim: number) {
    this.attention = new MultiHeadAttention(embedSize, numHeads)
    this.feedforward = new FeedForward(embedSize, ffnDim)
    this.norm1 = new LayerNormalization(embedSize)
    this.norm2 = new LayerNormalization(embedSize)
  }

  forward(x: number[][]): number[][] {
    // simpan input awal
    this.cacheXin = x

    // attention + residual
    const norm1Out = this.norm1.forward(x)
    this.cacheNorm1Out = norm1Out
    const attnOut = this.attention.forward(norm1Out)
    const xMid = x.map((row, i) => row.map((val, j) => val + attnOut[i]![j]!))
    this.cacheXmid = xMid

    // feedforward + residual
    const norm2Out = this.norm2.forward(xMid)
    this.cacheNorm2Out = norm2Out
    const ffnOut = this.feedforward.forward(norm2Out)
    const xOut = xMid.map((row, i) => row.map((val, j) => val + ffnOut[i]![j]!))

    return xOut
  }

  backward(dOutput: number[][]): number[][] {
    // ── bagian FFN ───────────────────────────────────────────────────
    // residual2: x_out = x_mid + ffn_out → kedua cabang dapat dOutput
    const dFfnOut = dOutput  // gradient ke cabang ffn
    const dXmidFromResidual2 = dOutput  // gradient ke cabang residual (langsung)

    // backward feedforward
    const { dInput: dNorm2Out } = this.feedforward.backward(dFfnOut)

    // backward norm2 — butuh cacheXmid (input asli ke norm2)
    const { dInput: dXmidFromNorm2 } = this.norm2.backward(this.cacheXmid, dNorm2Out)

    // jumlahkan dua cabang residual2
    const dXmid = dXmidFromResidual2.map((row, i) =>
      row.map((val, j) => val + dXmidFromNorm2[i]![j]!)
    )

    // ── bagian Attention ─────────────────────────────────────────────
    // residual1: x_mid = x_in + attn_out → kedua cabang dapat dXmid
    const dAttnOut = dXmid  // gradient ke cabang attention
    const dXinFromResidual1 = dXmid  // gradient ke cabang residual (langsung)

    // backward attention
    const { dInput: dNorm1Out } = this.attention.backward(dAttnOut)

    // backward norm1 — butuh cacheXin (input asli ke norm1)
    const { dInput: dXinFromNorm1 } = this.norm1.backward(this.cacheXin, dNorm1Out)

    // jumlahkan dua cabang residual1
    const dXin = dXinFromResidual1.map((row, i) =>
      row.map((val, j) => val + dXinFromNorm1[i]![j]!)
    )

    return dXin
  }
}
