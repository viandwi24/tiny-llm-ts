export function multiply(a: number[][], b: number[][]): number[][] {
  if (a.length === 0 || b.length === 0 || !a[0] || !b[0]) throw new Error('Input matrices cannot be empty')

  const aRows = a.length
  const aCols = a[0].length
  const bRows = b.length
  const bCols = b[0].length

  if (aCols !== bRows) {
    throw new Error('Incompatible matrix sizes for multiplication')
  }

  const result: number[][] = Array.from({ length: aRows }, () =>
    Array.from({ length: bCols }, () => 0)
  )

  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < bCols; j++) {
      for (let k = 0; k < aCols; k++) {
        result[i]![j]! += a[i]![k]! * b[k]![j]!
      }
    }
  }

  return result
}

export function transpose(matrix: number[][]): number[][] {
  if (matrix.length === 0 || !matrix[0]) throw new Error('Input matrix cannot be empty')

  const rows = matrix.length
  const cols = matrix[0].length

  const transposed: number[][] = Array.from({ length: cols }, () =>
    Array.from({ length: rows }, () => 0)
  )

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      transposed[j]![i]! = matrix[i]![j]!
    }
  }

  return transposed
}
