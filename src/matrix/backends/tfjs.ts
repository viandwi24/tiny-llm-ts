import * as tf from '@tensorflow/tfjs-node'

export function multiply(a: number[][], b: number[][]): number[][] {
  const ta = tf.tensor2d(a)
  const tb = tf.tensor2d(b)
  const result = tf.matMul(ta, tb)
  const data = result.arraySync() as number[][]
  ta.dispose()
  tb.dispose()
  result.dispose()
  return data
}

export function transpose(matrix: number[][]): number[][] {
  const t = tf.tensor2d(matrix)
  const result = tf.transpose(t)
  const data = result.arraySync() as number[][]
  t.dispose()
  result.dispose()
  return data
}
