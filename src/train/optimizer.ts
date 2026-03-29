import type { Transformer } from '../transformer'

interface Param2D {
  getW: () => number[][]
  getGrad: () => number[][]
  m: number[][]
  v: number[][]
}

interface Param1D {
  getW: () => number[]
  getGrad: () => number[]
  m: number[]
  v: number[]
}

export class AdamOptimizer {
  readonly beta1 = 0.9
  readonly beta2 = 0.999
  readonly eps = 1e-8
  t = 0

  private params2D: Param2D[] = []
  private params1D: Param1D[] = []

  constructor(
    private model: Transformer,
    readonly lr: number,
  ) {
    this.register()
  }

  private zeros2D(ref: number[][]): number[][] {
    return ref.map(row => row.map(() => 0))
  }

  private zeros1D(ref: number[]): number[] {
    return ref.map(() => 0)
  }

  private add2D(getW: () => number[][], getGrad: () => number[][]) {
    this.params2D.push({ getW, getGrad, m: this.zeros2D(getW()), v: this.zeros2D(getW()) })
  }

  private add1D(getW: () => number[], getGrad: () => number[]) {
    this.params1D.push({ getW, getGrad, m: this.zeros1D(getW()), v: this.zeros1D(getW()) })
  }

  private register() {
    const m = this.model

    // projection
    this.add2D(() => m.projection.weights, () => m.projection.gradWeights)
    this.add1D(() => m.projection.bias,    () => m.projection.gradBias)

    // blocks
    for (const block of m.blocks) {
      // attention — 4 linear layers
      for (const layer of [block.attention.wq, block.attention.wk, block.attention.wv, block.attention.wo]) {
        this.add2D(() => layer.weights, () => layer.gradWeights)
        this.add1D(() => layer.bias,    () => layer.gradBias)
      }

      // feedforward — 2 linear layers
      for (const layer of [block.feedforward.linear1, block.feedforward.linear2]) {
        this.add2D(() => layer.weights, () => layer.gradWeights)
        this.add1D(() => layer.bias,    () => layer.gradBias)
      }

      // layer norms — gamma dan beta (1D)
      for (const norm of [block.norm1, block.norm2]) {
        this.add1D(() => norm.gamma, () => norm.gradGamma)
        this.add1D(() => norm.beta,  () => norm.gradBeta)
      }
    }

    // embedding
    this.add2D(() => m.embedding.tokenWeights,    () => m.embedding.gradTokenWeights)
    this.add2D(() => m.embedding.positionWeights, () => m.embedding.gradPositionWeights)
  }

  step() {
    this.t++
    const { beta1, beta2, eps, lr, t } = this
    const b1t = 1 - Math.pow(beta1, t)
    const b2t = 1 - Math.pow(beta2, t)

    for (const p of this.params2D) {
      const W = p.getW()
      const grad = p.getGrad()
      if (grad.length === 0) continue  // backward belum pernah dipanggil
      for (let i = 0; i < W.length; i++) {
        for (let j = 0; j < W[i]!.length; j++) {
          const g = grad[i]?.[j] ?? 0
          p.m[i]![j]! = beta1 * p.m[i]![j]! + (1 - beta1) * g
          p.v[i]![j]! = beta2 * p.v[i]![j]! + (1 - beta2) * g * g
          W[i]![j]! -= lr * (p.m[i]![j]! / b1t) / (Math.sqrt(p.v[i]![j]! / b2t) + eps)
        }
      }
    }

    for (const p of this.params1D) {
      const W = p.getW()
      const grad = p.getGrad()
      if (grad.length === 0) continue
      for (let i = 0; i < W.length; i++) {
        const g = grad[i] ?? 0
        p.m[i]! = beta1 * p.m[i]! + (1 - beta1) * g
        p.v[i]! = beta2 * p.v[i]! + (1 - beta2) * g * g
        W[i]! -= lr * (p.m[i]! / b1t) / (Math.sqrt(p.v[i]! / b2t) + eps)
      }
    }
  }
}
