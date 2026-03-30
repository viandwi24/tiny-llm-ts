import * as cpu from './backends/cpu'

// Set USE_TFJS=true to use TensorFlow.js (BLAS-backed, ~20-50x faster on CPU)
// Example: USE_TFJS=true bun run src/index.ts train
const USE_TFJS = process.env.USE_TFJS === 'true'

type MatrixBackend = {
  multiply(a: number[][], b: number[][]): number[][]
  transpose(matrix: number[][]): number[][]
}

function loadBackend(): MatrixBackend {
  if (USE_TFJS) {
    console.log('[matrix] Using TensorFlow.js backend (BLAS)')
    return require('./backends/tfjs')
  }
  return cpu
}

const backend: MatrixBackend = loadBackend()

export namespace Matrix {
  export function multiply(a: number[][], b: number[][]): number[][] {
    return backend.multiply(a, b)
  }

  export function transpose(matrix: number[][]): number[][]  {
    return backend.transpose(matrix)
  }
}
