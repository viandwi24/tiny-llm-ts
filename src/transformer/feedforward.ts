import { LinearLayer } from '../neural-network/layer'
import { gelu } from '../neural-network/activation'

function geluGrad(x: number): number {
  const c = Math.sqrt(2 / Math.PI)
  const inner = c * (x + 0.044715 * x ** 3)
  const tanhInner = Math.tanh(inner)
  return 0.5 * (1 + tanhInner) + 0.5 * x * (1 - tanhInner ** 2) * c * (1 + 3 * 0.044715 * x ** 2)
}

export interface FeedForwardGradients {
  dInput: number[][]
  dW1: number[][]
  db1: number[]
  dW2: number[][]
  db2: number[]
}

export class FeedForward {
  linear1: LinearLayer
  linear2: LinearLayer

  // cache untuk backward
  private cacheX: number[][] = []
  private cacheHpre: number[][] = []  // output linear1 sebelum gelu
  private cacheH: number[][] = []     // output linear1 setelah gelu

  constructor(embedDim: number, ffnDim: number) {
    this.linear1 = new LinearLayer(embedDim, ffnDim)
    this.linear2 = new LinearLayer(ffnDim, embedDim)
  }

  forward(x: number[][]): number[][] {
    this.cacheX = x
    const hPre = this.linear1.forward(x)
    this.cacheHpre = hPre
    const h = hPre.map(row => row.map(v => gelu(v)))
    this.cacheH = h
    return this.linear2.forward(h)
  }

  backward(dOutput: number[][]): FeedForwardGradients {
    // backward linear2: butuh cacheH sebagai input-nya
    const { dInput: dH, dWeights: dW2, dBias: db2 } = this.linear2.backward(this.cacheH, dOutput)

    // backward gelu: kalikan dH dengan turunan gelu di setiap elemen h_pre
    const dHpre = dH.map((row, i) =>
      row.map((val, j) => val * geluGrad(this.cacheHpre[i]![j]!))
    )

    // backward linear1: butuh cacheX sebagai input-nya
    const { dInput, dWeights: dW1, dBias: db1 } = this.linear1.backward(this.cacheX, dHpre)

    return { dInput, dW1, db1, dW2, db2 }
  }
}
