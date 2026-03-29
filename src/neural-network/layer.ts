import { Matrix } from "../matrix"

export class LinearLayer {
  weights: number[][]
  bias: number[]
  inputSize: number
  outputSize: number

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
}

export class LayerNormalization {
  gamma: number[]   // scale, dipelajari saat training
  beta: number[]    // shift, dipelajari saat training
  eps: number       // nilai kecil biar tidak divide by zero

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
}