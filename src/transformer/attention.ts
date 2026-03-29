import { Matrix } from "../matrix"
import { softmax } from "../neural-network/activation"
import { LinearLayer } from "../neural-network/layer"

export class MultiHeadAttention {
  numHeads: number
  headSize: number
  embedDim: number

  // 4 linear layer: Q, K, V, dan output projection
  wq: LinearLayer
  wk: LinearLayer
  wv: LinearLayer
  wo: LinearLayer

  constructor(embedDim: number, numHeads: number) {
    this.embedDim = embedDim
    this.numHeads = numHeads
    this.headSize = embedDim / numHeads
    this.wq = new LinearLayer(embedDim, embedDim)
    this.wk = new LinearLayer(embedDim, embedDim)
    this.wv = new LinearLayer(embedDim, embedDim)
    this.wo = new LinearLayer(embedDim, embedDim)
  }

  forward(x: number[][]): number[][] {
    const seqLen = x.length

    // buat Q, K, V dari input
    const Q = this.wq.forward(x)   // [seqLen, embedDim]
    const K = this.wk.forward(x)   // [seqLen, embedDim]
    const V = this.wv.forward(x)   // [seqLen, embedDim]

    // bagi jadi heads
    const scale = Math.sqrt(this.headSize)
    const headOutputs: number[][][] = []

    for (let h = 0; h < this.numHeads; h++) {
      const start = h * this.headSize
      const end = start + this.headSize

      // ambil slice tiap head
      const Qh = Q.map(row => row.slice(start, end))  // [seqLen, headSize]
      const Kh = K.map(row => row.slice(start, end))
      const Vh = V.map(row => row.slice(start, end))

      // hitung attention scores: Q·Kᵀ / √headSize
      const scores = Matrix.multiply(Qh, Matrix.transpose(Kh))
        .map(row => row.map(v => v / scale))          // [seqLen, seqLen]

      // causal mask: token tidak boleh lihat token berikutnya
      const masked = scores.map((row, i) =>
        row.map((v, j) => j > i ? -1e9 : v)
      )

      // softmax per baris
      const attnWeights = masked.map(row => softmax(row))  // [seqLen, seqLen]

      // weighted sum of V
      headOutputs.push(Matrix.multiply(attnWeights, Vh))            // [seqLen, headSize]
    }

    // gabung semua head
    const concat = x.map((_, i) =>
      headOutputs.flatMap(head => head[i]) as number[]
    )

    // output projection
    return this.wo.forward(concat)
  }
}