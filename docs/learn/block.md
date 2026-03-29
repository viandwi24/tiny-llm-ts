# Transformer Block

## Apa Itu?

Satu unit pemrosesan lengkap yang menggabungkan semua sub-komponen. Ini yang di-stack berkali-kali (misal 4x) untuk membentuk model yang lebih dalam dan powerful.

---

## Isi Satu Block

```
input x
  ↓
LayerNorm(x) → MultiHeadAttention → output_attn
  ↓
x = x + output_attn          ← Residual Connection 1
  ↓
LayerNorm(x) → FeedForward → output_ffn
  ↓
x = x + output_ffn           ← Residual Connection 2
  ↓
output x   (shape sama dengan input)
```

Ada 4 hal penting di sini: **LayerNorm**, **MultiHeadAttention**, **FeedForward**, dan **Residual Connection**.

---

## Residual Connection

Ini salah satu inovasi paling penting dalam deep learning. Idenya simpel — tambahkan input asli ke output setiap sub-layer:

```
x = x + sublayer(x)
```

Kenapa penting? Bayangkan kamu punya 12 layer. Tanpa residual connection, gradient harus mengalir lewat semua 12 layer saat backprop — lama-lama gradient mengecil dan menghilang (vanishing gradient), layer awal tidak belajar apa-apa.

Dengan residual connection, ada "jalan pintas" untuk gradient mengalir langsung tanpa harus lewat semua operasi — training jadi jauh lebih stabil.

```
tanpa residual:   input → L1 → L2 → L3 → ... → L12 → output
dengan residual:  input ──────────────────────────────→ +
                         ↓                              ↑
                         L1 → L2 → L3 → ... → L12 ────┘
```

---

## LayerNorm

**Layer Normalization** — normalisasi nilai dalam satu vector supaya mean = 0 dan variance = 1.

Kenapa perlu? Saat training, nilai-nilai dalam vector bisa jadi sangat besar atau sangat kecil tergantung weight. Kalau nilai meledak → gradient meledak → training tidak stabil. LayerNorm menjaga nilai tetap dalam range yang masuk akal.

Analoginya: bayangkan kamu punya tim dengan kemampuan sangat berbeda — satu orang bisa angkat 100kg, yang lain hanya 10kg. Kalau mereka harus kerja sama, perlu ada standarisasi supaya semua berkontribusi seimbang. LayerNorm melakukan itu untuk nilai-nilai dalam vector.

```
Rumus:
mean = rata-rata semua nilai dalam vector
var  = variance semua nilai

output[i] = (input[i] - mean) / √(var + ε)  ← normalisasi
output[i] = output[i] × γ + β               ← scale dan shift (dipelajari)

ε (epsilon) = angka sangat kecil (misal 1e-5) supaya tidak bagi nol
γ (gamma) dan β (beta) = parameter yang dipelajari saat training
```

**Catatan Pre-Norm:** Di implementasi modern (GPT-2+), LayerNorm diletakkan **sebelum** attention dan FFN (bukan sesudah). Ini disebut Pre-LN dan membuat training lebih stabil.

---

## Kenapa N Block Di-Stack?

Tiap block memperhalus representasi vector secara bertahap:

```
Block 1 → belajar pola permukaan (kata-kata yang sering berdampingan)
Block 2 → belajar pola yang lebih dalam (frasa dan struktur kalimat)
Block 3 → belajar pola semantik (makna dan konteks)
Block 4 → belajar pola abstrak (reasoning dan inferensi)
```

Semakin banyak block, semakin kompleks pola yang bisa dipelajari — tapi semakin berat komputasinya.

---

## Math Util yang Dibutuhkan

### mean dan variance

```ts
// mean dari satu vector
mean = sum(vec) / vec.length

// variance dari satu vector
variance = sum((x - mean)² untuk tiap x) / vec.length
```

---

## Step by Step

**Step 1 — Inisialisasi**
```
layerNorm1: LayerNorm(embedDim)
attention:  MultiHeadAttention(embedDim, numHeads)
layerNorm2: LayerNorm(embedDim)
ffn:        FeedForward(embedDim, ffnDim)
```

**Step 2 — Forward pass**
```
1. normed = layerNorm1(x)
2. attn_out = attention(normed)
3. x = x + attn_out              ← residual 1
4. normed = layerNorm2(x)
5. ffn_out = ffn(normed)
6. x = x + ffn_out               ← residual 2
7. return x
```

---

## Preview Code

```ts
// src/transformer/block.ts

import { MultiHeadAttention } from './attention'
import { FeedForward } from './feedforward'
import { LayerNorm } from './layernorm'

export class TransformerBlock {
  private layerNorm1: LayerNorm
  private attention: MultiHeadAttention
  private layerNorm2: LayerNorm
  private ffn: FeedForward

  constructor(embedDim: number, numHeads: number, ffnDim: number) {
    this.layerNorm1 = new LayerNorm(embedDim)
    this.attention = new MultiHeadAttention(embedDim, numHeads)
    this.layerNorm2 = new LayerNorm(embedDim)
    this.ffn = new FeedForward(embedDim, ffnDim)
  }

  forward(x: number[][]): number[][] {
    // x shape: [seqLen, embedDim]

    // Attention sub-layer (Pre-LN)
    const normed1 = this.layerNorm1.forward(x)
    const attnOut = this.attention.forward(normed1)
    x = x.map((row, i) => row.map((val, j) => val + attnOut[i][j]))  // residual 1

    // FFN sub-layer (Pre-LN)
    const normed2 = this.layerNorm2.forward(x)
    const ffnOut = this.ffn.forward(normed2)
    x = x.map((row, i) => row.map((val, j) => val + ffnOut[i][j]))   // residual 2

    return x
    // output shape: [seqLen, embedDim]  ← sama dengan input
  }
}
```

```ts
// src/transformer/layernorm.ts

export class LayerNorm {
  private gamma: number[]   // scale (dipelajari), init = 1
  private beta: number[]    // shift (dipelajari), init = 0
  private eps: number

  constructor(embedDim: number, eps = 1e-5) {
    this.gamma = Array(embedDim).fill(1)
    this.beta = Array(embedDim).fill(0)
    this.eps = eps
  }

  forward(input: number[][]): number[][] {
    // normalisasi tiap baris (tiap token) secara independen
    return input.map(row => {
      const mean = row.reduce((a, b) => a + b, 0) / row.length
      const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length

      return row.map((val, i) => {
        const normalized = (val - mean) / Math.sqrt(variance + this.eps)
        return normalized * this.gamma[i] + this.beta[i]
      })
    })
  }
}
```

---

## Dependency Graph Block

```
TransformerBlock
├── LayerNorm         ← butuh mean, variance (math)
├── MultiHeadAttention
│   ├── LinearLayer   ← butuh matMul
│   ├── softmax       ← math util
│   └── transpose     ← math util
└── FeedForward
    ├── LinearLayer   ← butuh matMul
    └── gelu          ← math util
```