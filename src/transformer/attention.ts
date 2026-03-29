import { Matrix } from "../matrix"
import { softmax } from "../neural-network/activation"
import { LinearLayer } from "../neural-network/layer"

interface HeadCache {
  Qh: number[][]
  Kh: number[][]
  Vh: number[][]
  attnWeights: number[][]
}

export interface AttentionGradients {
  dInput: number[][]
  dWq: number[][]; dbq: number[]
  dWk: number[][]; dbk: number[]
  dWv: number[][]; dbv: number[]
  dWo: number[][]; dbo: number[]
}

export class MultiHeadAttention {
  numHeads: number
  headSize: number
  embedDim: number

  // 4 linear layer: Q, K, V, dan output projection
  wq: LinearLayer
  wk: LinearLayer
  wv: LinearLayer
  wo: LinearLayer

  // cache untuk backward
  private cacheX: number[][] = []
  private cacheQ: number[][] = []
  private cacheK: number[][] = []
  private cacheV: number[][] = []
  private cacheHeads: HeadCache[] = []
  private cacheConcat: number[][] = []

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
    this.cacheX = x
    const seqLen = x.length

    // buat Q, K, V dari input
    const Q = this.wq.forward(x)   // [seqLen, embedDim]
    const K = this.wk.forward(x)   // [seqLen, embedDim]
    const V = this.wv.forward(x)   // [seqLen, embedDim]
    this.cacheQ = Q
    this.cacheK = K
    this.cacheV = V

    // bagi jadi heads
    const scale = Math.sqrt(this.headSize)
    const headOutputs: number[][][] = []
    this.cacheHeads = []

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
      const headOut = Matrix.multiply(attnWeights, Vh)  // [seqLen, headSize]
      headOutputs.push(headOut)

      this.cacheHeads.push({ Qh, Kh, Vh, attnWeights })
    }

    // gabung semua head
    const concat = x.map((_, i) =>
      headOutputs.flatMap(head => head[i]) as number[]
    )
    this.cacheConcat = concat

    // output projection
    return this.wo.forward(concat)
  }

  backward(dOutput: number[][]): AttentionGradients {
    const seqLen = dOutput.length
    const scale = Math.sqrt(this.headSize)

    // Step 1: backward melalui wo
    const { dInput: dConcat, dWeights: dWo, dBias: dbo } = this.wo.backward(this.cacheConcat, dOutput)

    // Step 2: inisialisasi dQ, dK, dV (akan diakumulasi dari semua head)
    const dQ = Array.from({ length: seqLen }, () => Array(this.embedDim).fill(0) as number[])
    const dK = Array.from({ length: seqLen }, () => Array(this.embedDim).fill(0) as number[])
    const dV = Array.from({ length: seqLen }, () => Array(this.embedDim).fill(0) as number[])

    // Step 3: backward per head
    for (let h = 0; h < this.numHeads; h++) {
      const start = h * this.headSize
      const end = start + this.headSize
      const cache = this.cacheHeads[h]!

      // ambil slice dConcat untuk head ini
      const dHeadOut = dConcat.map(row => row.slice(start, end))  // [seqLen, headSize]

      // backward: headOut = attnWeights · Vh
      const dAttnWeights = Matrix.multiply(dHeadOut, Matrix.transpose(cache.Vh))       // [seqLen, seqLen]
      const dVh = Matrix.multiply(Matrix.transpose(cache.attnWeights), dHeadOut)        // [seqLen, headSize]

      // backward melalui softmax (per baris)
      // jika s = softmax(x), dan da = gradient masuk, maka dx[j] = s[j] * (da[j] - dot(da, s))
      const dMasked = cache.attnWeights.map((s, i) => {
        const da = dAttnWeights[i]!
        const dot = s.reduce((acc, sv, k) => acc + da[k]! * sv, 0)
        return s.map((sv, j) => sv * (da[j]! - dot))
      })

      // backward melalui causal mask: posisi yang di-mask (j > i) gradientnya = 0
      const dScores = dMasked.map((row, i) =>
        row.map((v, j) => j > i ? 0 : v)
      )

      // backward melalui pembagian scale: scores = (Qh·Khᵀ) / scale → d(Qh·Khᵀ) = dScores / scale
      const dPreScores = dScores.map(row => row.map(v => v / scale))

      // backward: preScores = Qh · Khᵀ
      // dQh = dPreScores · Kh        [seqLen,seqLen] · [seqLen,headSize] = [seqLen,headSize]
      // dKh = dPreScores^T · Qh      [seqLen,seqLen] · [seqLen,headSize] = [seqLen,headSize]
      const dQh = Matrix.multiply(dPreScores, cache.Kh)
      const dKh = Matrix.multiply(Matrix.transpose(dPreScores), cache.Qh)

      // akumulasi ke dQ, dK, dV di slice yang sesuai
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < this.headSize; j++) {
          dQ[i]![start + j]! += dQh[i]![j]!
          dK[i]![start + j]! += dKh[i]![j]!
          dV[i]![start + j]! += dVh[i]![j]!
        }
      }
    }

    // Step 4: backward melalui wq, wk, wv
    // x dipakai bersama oleh ketiga projection, jadi dInput dijumlahkan
    const { dInput: dXq, dWeights: dWq, dBias: dbq } = this.wq.backward(this.cacheX, dQ)
    const { dInput: dXk, dWeights: dWk, dBias: dbk } = this.wk.backward(this.cacheX, dK)
    const { dInput: dXv, dWeights: dWv, dBias: dbv } = this.wv.backward(this.cacheX, dV)

    // jumlahkan kontribusi gradient dari Q, K, V ke input x
    const dInput = dXq.map((row, i) =>
      row.map((v, j) => v + dXk[i]![j]! + dXv[i]![j]!)
    )

    return { dInput, dWq, dbq, dWk, dbk, dWv, dbv, dWo, dbo }
  }
}
