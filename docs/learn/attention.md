# Multi-Head Attention

## Apa Itu?

Ini **jantung dari transformer** — mekanisme yang membuat setiap token bisa "melihat" dan "memperhatikan" token lain yang relevan di dalam sequence.

Contoh: waktu model memproses kata "dia" dalam kalimat:
```
"Soekarno lahir di Surabaya, dia adalah presiden pertama"
```
Attention membantu model tahu bahwa "dia" merujuk ke "Soekarno", bukan "Surabaya".

---

## Konsep Query, Key, Value

Cara kerjanya terinspirasi dari sistem pencarian (search engine). Tiap token membuat 3 vector dari dirinya sendiri:

**Query (Q)** — "saya sedang cari informasi tentang apa?"
**Key (K)** — "saya punya informasi tentang apa?"
**Value (V)** — "ini informasi saya yang sebenarnya"

Analoginya seperti perpustakaan:
- Query = pertanyaan yang kamu bawa ke perpustakaan
- Key = label di punggung tiap buku
- Value = isi buku yang sebenarnya

Kamu cocokkan Query dengan semua Key → buku yang labelnya paling cocok → ambil Value-nya (isi buku).

```
Token "dia" punya Query: "saya butuh referensi ke subjek manusia"
Token "Soekarno" punya Key: "saya adalah subjek manusia, tokoh sejarah"
Token "Surabaya" punya Key: "saya adalah nama tempat"

Skor cocok:
"dia" ↔ "Soekarno" = 0.92   ← tinggi, relevan
"dia" ↔ "Surabaya" = 0.11   ← rendah, tidak relevan
```

---

## Proses Attention Satu Head

```
1. Buat Q, K, V dari input lewat LinearLayer
   Q = input · Wq   shape: [seqLen, headDim]
   K = input · Wk   shape: [seqLen, headDim]
   V = input · Wv   shape: [seqLen, headDim]

2. Hitung skor relevansi (dot product Q dengan semua K)
   scores = Q · Kᵀ  shape: [seqLen, seqLen]
   setiap [i][j] = seberapa relevan token i terhadap token j

3. Scale supaya nilai tidak terlalu besar
   scores = scores / √headDim

4. Softmax → ubah skor jadi probabilitas (total = 1)
   weights = softmax(scores)  shape: [seqLen, seqLen]

5. Weighted sum of Values
   output = weights · V  shape: [seqLen, headDim]
   token yang lebih relevan dapat bobot lebih besar
```

---

## Kenapa "Multi-Head"?

Disebut multi-head karena proses attention dijalankan **H kali paralel** (misal 4 head). Tiap head belajar aspek relasi yang berbeda:

```
Head 1 → belajar relasi subjek-objek ("Soekarno" → "dia")
Head 2 → belajar relasi kata sifat-kata benda ("pertama" → "presiden")
Head 3 → belajar relasi temporal ("lahir" → "1901")
Head 4 → belajar pola lain yang ditemukan model sendiri
```

Model tidak diarahkan untuk belajar relasi tertentu di head tertentu — itu terbentuk sendiri lewat training.

**Tiap head punya dimensi lebih kecil:**
```
embedDim = 128
numHeads = 4
headDim  = 128 / 4 = 32   ← tiap head kerja di dimensi 32
```

Total komputasi sama, tapi dapat 4 perspektif berbeda.

---

## Causal Masking

Saat training, model harus prediksi token berikutnya **tanpa boleh lihat token yang belum ada** (token di masa depan). Ini namanya **causal masking** atau autoregressive masking.

Caranya: set skor attention ke `-infinity` untuk posisi yang tidak boleh dilihat, sehingga setelah softmax nilainya jadi 0.

```
Token ke-3 hanya boleh lihat token 1, 2, 3 — tidak boleh lihat 4, 5, dst

Mask (1 = boleh lihat, 0 = tidak boleh):
      1  2  3  4  5
  1 [ 1  0  0  0  0 ]
  2 [ 1  1  0  0  0 ]
  3 [ 1  1  1  0  0 ]
  4 [ 1  1  1  1  0 ]
  5 [ 1  1  1  1  1 ]
```

---

## Softmax — Math Util yang Dibutuhkan

Mengubah array angka jadi probabilitas yang jumlahnya = 1:

```
input:  [2.0, 1.0, 0.5]
output: [0.59, 0.26, 0.15]  ← total = 1.0

rumus: softmax(x_i) = e^x_i / sum(e^x_j)
```

Nilai yang lebih besar dapat probabilitas lebih tinggi — tapi semua tetap berkontribusi (tidak ada yang jadi nol persis).

---

## Step by Step

**Step 1 — Inisialisasi**
```
Untuk tiap head, buat 3 LinearLayer:
  Wq: Linear(embedDim, headDim)
  Wk: Linear(embedDim, headDim)
  Wv: Linear(embedDim, headDim)

Tambah 1 output projection:
  Wo: Linear(embedDim, embedDim)  ← gabung semua head
```

**Step 2 — Forward pass per head**
```
1. Q = input · Wq
2. K = input · Wk
3. V = input · Wv
4. scores = Q · Kᵀ / √headDim
5. Apply causal mask
6. weights = softmax(scores)
7. output = weights · V
```

**Step 3 — Gabung semua head**
```
concat semua output head → [seqLen, embedDim]
lewatkan ke Wo → [seqLen, embedDim]
```

---

## Preview Code

```ts
// src/transformer/attention.ts

import { LinearLayer } from './linear'
import { matMul, softmax, transpose } from './math'

export class MultiHeadAttention {
  private numHeads: number
  private headDim: number
  private embedDim: number

  // tiap head punya Wq, Wk, Wv sendiri
  private Wq: LinearLayer[]
  private Wk: LinearLayer[]
  private Wv: LinearLayer[]
  private Wo: LinearLayer

  constructor(embedDim: number, numHeads: number) {
    this.embedDim = embedDim
    this.numHeads = numHeads
    this.headDim = Math.floor(embedDim / numHeads)

    this.Wq = Array.from({ length: numHeads }, () => new LinearLayer(embedDim, this.headDim))
    this.Wk = Array.from({ length: numHeads }, () => new LinearLayer(embedDim, this.headDim))
    this.Wv = Array.from({ length: numHeads }, () => new LinearLayer(embedDim, this.headDim))
    this.Wo = new LinearLayer(embedDim, embedDim)
  }

  forward(input: number[][]): number[][] {
    // input shape: [seqLen, embedDim]
    const seqLen = input.length
    const headOutputs: number[][][] = []

    for (let h = 0; h < this.numHeads; h++) {
      // Step 1: buat Q, K, V
      const Q = this.Wq[h].forward(input)  // [seqLen, headDim]
      const K = this.Wk[h].forward(input)  // [seqLen, headDim]
      const V = this.Wv[h].forward(input)  // [seqLen, headDim]

      // Step 2: hitung scores = Q · Kᵀ / √headDim
      const Kt = transpose(K)              // [headDim, seqLen]
      const scale = Math.sqrt(this.headDim)
      const scores = matMul(Q, Kt).map(row =>
        row.map(val => val / scale)
      )                                    // [seqLen, seqLen]

      // Step 3: causal mask — token tidak boleh lihat masa depan
      const masked = scores.map((row, i) =>
        row.map((val, j) => j > i ? -Infinity : val)
      )

      // Step 4: softmax per baris
      const weights = masked.map(row => softmax(row))  // [seqLen, seqLen]

      // Step 5: weighted sum of V
      const headOut = matMul(weights, V)  // [seqLen, headDim]
      headOutputs.push(headOut)
    }

    // Step 6: concat semua head → [seqLen, embedDim]
    const concat = input.map((_, i) =>
      headOutputs.flatMap(head => head[i])
    )

    // Step 7: output projection
    return this.Wo.forward(concat)  // [seqLen, embedDim]
  }
}
```

```ts
// tambahkan ke src/transformer/math.ts

export function softmax(vec: number[]): number[] {
  const max = Math.max(...vec)  // stability trick: kurangi max sebelum exp
  const exps = vec.map(v => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map(v => v / sum)
}

export function transpose(A: number[][]): number[][] {
  const rows = A.length
  const cols = A[0].length
  return Array.from({ length: cols }, (_, j) =>
    Array.from({ length: rows }, (_, i) => A[i][j])
  )
}
```

---

## Math Utils Terkait

```
src/transformer/math.ts
├── matMul()     ← perkalian matrix Q·Kᵀ dan weights·V
├── transpose()  ← transpose K untuk dot product
├── softmax()    ← ubah scores jadi probabilitas
└── gelu()       ← (dari FFN)
```