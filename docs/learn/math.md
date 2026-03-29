# Math Utils

## Overview

Semua operasi matematika yang dibutuhkan transformer dikumpulkan di satu file `src/transformer/math.ts`. Fungsi-fungsi ini muncul secara organik — tidak dirancang di awal, tapi ditambahkan satu per satu ketika komponen lain membutuhkannya.

---

## Daftar Fungsi

| Fungsi | Dibutuhkan Oleh | Kegunaan |
|---|---|---|
| `matMul` | LinearLayer | Perkalian dua matrix |
| `transpose` | MultiHeadAttention | Balik baris jadi kolom |
| `softmax` | MultiHeadAttention | Ubah skor jadi probabilitas |
| `gelu` | FeedForward | Fungsi aktivasi |

---

## matMul

Perkalian dua matrix. Operasi paling fundamental — dipakai di mana-mana.

```
A shape: [m, k]
B shape: [k, n]     ← kolom A harus sama dengan baris B
output shape: [m, n]

output[i][j] = sum(A[i][p] * B[p][j]) untuk semua p
```

Contoh konkret:
```
A = [[1, 2],    B = [[5, 6],
     [3, 4]]         [7, 8]]

output[0][0] = (1×5) + (2×7) = 19
output[0][1] = (1×6) + (2×8) = 22
output[1][0] = (3×5) + (4×7) = 43
output[1][1] = (3×6) + (4×8) = 50

output = [[19, 22],
           [43, 50]]
```

```ts
export function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length
  const k = A[0].length
  const n = B[0].length

  const output: number[][] = Array.from({ length: m }, () =>
    Array(n).fill(0)
  )

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      for (let p = 0; p < k; p++) {
        output[i][j] += A[i][p] * B[p][j]
      }
    }
  }

  return output
}
```

---

## transpose

Membalik baris jadi kolom. Dibutuhkan di attention untuk menghitung Q · Kᵀ.

```
A = [[1, 2, 3],    transpose(A) = [[1, 4],
     [4, 5, 6]]                    [2, 5],
                                   [3, 6]]

A shape: [m, n] → output shape: [n, m]
```

```ts
export function transpose(A: number[][]): number[][] {
  const rows = A.length
  const cols = A[0].length
  return Array.from({ length: cols }, (_, j) =>
    Array.from({ length: rows }, (_, i) => A[i][j])
  )
}
```

---

## softmax

Mengubah array angka jadi probabilitas yang jumlahnya = 1. Dibutuhkan di attention untuk mengubah skor relevansi jadi bobot.

```
input:  [2.0, 1.0, 0.5]
output: [0.59, 0.26, 0.15]   ← total = 1.0
```

Ada **stability trick** penting: kurangi nilai maksimum sebelum `exp()`. Ini mencegah nilai meledak jadi Infinity karena `exp()` dari angka besar sangat besar.

```
tanpa trick: exp(1000) = Infinity  ← meledak
dengan trick: exp(1000 - 1000) = exp(0) = 1  ← aman
```

```ts
export function softmax(vec: number[]): number[] {
  const max = Math.max(...vec)  // stability trick
  const exps = vec.map(v => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map(v => v / sum)
}
```

---

## gelu

Fungsi aktivasi yang dipakai di FeedForward. Menambah non-linearitas supaya model bisa belajar pola yang kompleks.

```
input negatif besar  → mendekati 0
input positif besar  → hampir sama dengan input
input sekitar 0      → diproses smooth
```

```ts
export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))
}
```

---

## Full File

```ts
// src/transformer/math.ts

export function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length
  const k = A[0].length
  const n = B[0].length
  const output: number[][] = Array.from({ length: m }, () => Array(n).fill(0))
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      for (let p = 0; p < k; p++)
        output[i][j] += A[i][p] * B[p][j]
  return output
}

export function transpose(A: number[][]): number[][] {
  const rows = A.length
  const cols = A[0].length
  return Array.from({ length: cols }, (_, j) =>
    Array.from({ length: rows }, (_, i) => A[i][j])
  )
}

export function softmax(vec: number[]): number[] {
  const max = Math.max(...vec)
  const exps = vec.map(v => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map(v => v / sum)
}

export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))
}
```