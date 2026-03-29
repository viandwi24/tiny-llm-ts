export function softmax(arr: number[]): number[] {
  const max = Math.max(...arr)
  const exps = arr.map(v => Math.exp(v - max)) // stabilisasi numerik
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map(v => v / sum)
}

export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))
}