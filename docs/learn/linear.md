# Linear Layer

## Apa Itu?

Linear layer adalah operasi paling dasar di neural network. Rumusnya:

```
output = input · Weight + bias
```

Analoginya seperti fungsi linear di matematika `y = ax + b`, tapi dalam bentuk matrix. `Weight` dan `bias` inilah yang dipelajari saat training — nilainya terus diupdate supaya prediksi model makin akurat.

---

## Kenapa Perlu Weight dan Bias?

**Weight** menentukan seberapa penting tiap input dan mentransformasi dimensi:

```
input shape:  [3, 128]    (3 token, tiap token vector 128 dimensi)
weight shape: [128, 512]  (transformasi dari 128 → 512)
output shape: [3, 512]
```

**Bias** memberi model nilai dasar terlepas dari input. Tanpa bias, output selalu melewati titik nol dan model kurang fleksibel.

---

## Inisialisasi Weight

Weight tidak boleh diinisialisasi semua nol — kalau semua nol, semua neuron belajar hal yang sama dan model tidak berkembang.

Pakai **Xavier initialization** — nilai random kecil yang skalanya disesuaikan dengan ukuran layer:

```
nilai random antara -√(1/inputDim) sampai +√(1/inputDim)
```

Semakin besar inputDim, semakin kecil nilai random awalnya — supaya nilai tidak meledak saat dilewatkan banyak layer.

---

## Posisi di Transformer

LinearLayer adalah komponen paling banyak dipakai — total ada 9 instance dalam satu transformer dengan 4 block:

```
Transformer
├── Embedding
├── TransformerBlock × 4
│   ├── MultiHeadAttention
│   │   ├── LinearLayer  ← buat Query
│   │   ├── LinearLayer  ← buat Key
│   │   ├── LinearLayer  ← buat Value
│   │   └── LinearLayer  ← output projection
│   └── FeedForward
│       ├── LinearLayer  ← expand (embedDim → ffnDim)
│       └── LinearLayer  ← compress (ffnDim → embedDim)
└── LinearLayer          ← projection akhir (embedDim → vocabSize)
```

---

## Math Util yang Dibutuhkan

### matMul

Perkalian dua matrix — operasi paling sering dipakai di seluruh transformer:

```
A shape: [m, k]
B shape: [k, n]
output shape: [m, n]

output[i][j] = sum(A[i][p] * B[p][j]) untuk semua p
```

---

## Step by Step

**Step 1 — Inisialisasi**
```
Weight: matrix random shape [inputDim, outputDim] dengan Xavier init
Bias: vector nol shape [outputDim]
```

**Step 2 — Forward pass**
```
1. matMul(input, weight)   → [seqLen, outputDim]
2. tambah bias per kolom   → [seqLen, outputDim]
```

---

## Preview Code

```ts
// src/transformer/linear.ts

import { matMul } from './math'

export class LinearLayer {
  weights: number[][]
  bias: number[]
  inputDim: number
  outputDim: number

  constructor(inputDim: number, outputDim: number) {
    this.inputDim = inputDim
    this.outputDim = outputDim

    // xavier initialization
    const scale = Math.sqrt(1 / inputDim)
    this.weights = Array.from({ length: inputDim }, () =>
      Array.from({ length: outputDim }, () =>
        (Math.random() * 2 - 1) * scale
      )
    )

    // bias dimulai dari nol
    this.bias = Array(outputDim).fill(0)
  }

  forward(input: number[][]): number[][] {
    // input shape: [seqLen, inputDim]
    // output shape: [seqLen, outputDim]
    return matMul(input, this.weights).map(row =>
      row.map((val, j) => val + this.bias[j])
    )
  }
}
```

```ts
// src/transformer/math.ts

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

## Contoh Nyata

```ts
const linear = new LinearLayer(4, 2)

const input = [
  [0.2, -0.5, 0.8, 0.1],   // token 1
  [0.1,  0.3, -0.2, 0.4],  // token 2
  [-0.4, 0.7, 0.1, -0.3],  // token 3
]

const output = linear.forward(input)
// output shape: [3, 2]
// tiap token sekarang jadi vector dimensi 2
```