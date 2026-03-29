import { Matrix } from "../matrix"

export class LinearLayer {
  weights: number[][]
  bias: number[]
  inputSize: number
  outputSize: number

  // gradient — diisi saat backward(), dibaca oleh optimizer
  gradWeights: number[][] = []
  gradBias: number[] = []

  constructor(inputSize: number, outputSize: number) {
    this.inputSize = inputSize
    this.outputSize = outputSize

    // init use xavier initialization
    const scale = Math.sqrt(1 / inputSize)
    this.weights = Array.from({ length: inputSize }, () =>
      Array.from({ length: outputSize }, () =>
        (Math.random() * 2 - 1) * scale
      )
    )

    // init bias to zero
    this.bias = Array.from({ length: outputSize }, () => 0)
  }

  forward(input: number[][]): number[][] {
    // multiply input by weights
    const multiplied = Matrix.multiply(input, this.weights)

    // add bias
    const output = multiplied.map(row =>
      row.map((value, index) => value + this.bias[index]!)
    )

    // return
    return output
  }

  backward(input: number[][], dOutput: number[][]): { dInput: number[][]; dWeights: number[][]; dBias: number[] } {
    // dInput = dOutput * W^T
    const dInput = Matrix.multiply(dOutput, Matrix.transpose(this.weights))

    // dWeights = input^T * dOutput
    const dWeights = Matrix.multiply(Matrix.transpose(input), dOutput)

    // dBias = sum of dOutput across batch
    const dBias = Array(this.outputSize).fill(0)
    for (const row of dOutput) {
      for (let i = 0; i < this.outputSize; i++) {
        dBias[i]! += row[i]!
      }
    }

    this.gradWeights = dWeights
    this.gradBias = dBias
    return { dInput, dWeights, dBias }
  }
}

export class LayerNormalization {
  gamma: number[]   // scale, dipelajari saat training
  beta: number[]    // shift, dipelajari saat training
  eps: number       // nilai kecil biar tidak divide by zero

  // gradient — diisi saat backward(), dibaca oleh optimizer
  gradGamma: number[] = []
  gradBeta: number[] = []

  constructor(embedDim: number) {
    this.gamma = Array(embedDim).fill(1)   // init 1
    this.beta = Array(embedDim).fill(0)    // init 0
    this.eps = 1e-5
  }

  forward(x: number[][]): number[][] {
    // normalisasi tiap baris (tiap token) secara independen
    return x.map(row => {
      // hitung mean
      const mean = row.reduce((a, b) => a + b, 0) / row.length

      // hitung variance
      const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length

      // normalisasi
      const normalized = row.map(val => (val - mean) / Math.sqrt(variance + this.eps))

      // scale dan shift dengan gamma dan beta
      return normalized.map((val, i) => val * this.gamma[i]! + this.beta[i]!)
    })
  }

  backward(x: number[][], dOutput: number[][]): { dInput: number[][]; dGamma: number[]; dBeta: number[] } {
    const batchSize = x.length
    const embedDim = x[0]?.length ?? 0

    // hitung mean dan variance untuk backward pass
    const means = x.map(row => row.reduce((a, b) => a + b, 0) / embedDim)
    const variances = x.map((row, i) =>
      row.reduce((a, b) => a + (b - means[i]!) ** 2, 0) / embedDim
    )

    // hitung dGamma dan dBeta
    const dGamma = Array(embedDim).fill(0)
    const dBeta = Array(embedDim).fill(0)
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < embedDim; j++) {
        const normalized = (x[i]![j]! - means[i]!) / Math.sqrt(variances[i]! + this.eps)
        dGamma[j]! += dOutput[i]![j]! * normalized
        dBeta[j]! += dOutput[i]![j]!
      }
    }

    // hitung dInput — per baris i secara independen
    const dInput = Array.from({ length: batchSize }, () => Array(embedDim).fill(0))
    for (let i = 0; i < batchSize; i++) {
      const std = Math.sqrt(variances[i]! + this.eps)

      // dXNorm[j] = dOutput[i][j] * gamma[j]
      const dXNorm = Array.from({ length: embedDim }, (_, j) => dOutput[i]![j]! * this.gamma[j]!)

      // xNorm[j] = (x[i][j] - mean) / std
      const xNorm = Array.from({ length: embedDim }, (_, j) => (x[i]![j]! - means[i]!) / std)

      // sum1 = (1/N) * sum_j(dXNorm)
      const sum1 = dXNorm.reduce((a, b) => a + b, 0) / embedDim

      // sum2 = (1/N) * sum_j(dXNorm * xNorm)
      const sum2 = dXNorm.reduce((a, v, j) => a + v * xNorm[j]!, 0) / embedDim

      for (let j = 0; j < embedDim; j++) {
        dInput[i]![j]! = (1 / std) * (dXNorm[j]! - sum1 - xNorm[j]! * sum2)
      }
    }

    this.gradGamma = dGamma
    this.gradBeta = dBeta
    return { dInput, dGamma, dBeta }
  }
}