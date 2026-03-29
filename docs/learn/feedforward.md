# FeedForward Network

## Apa Itu?

Setelah attention mengumpulkan informasi dari konteks, FeedForward bertugas **memproses dan mentransformasi informasi itu lebih dalam** — per token, secara independen.

Analoginya: attention itu seperti rapat tim — semua orang saling berbagi informasi. FeedForward itu seperti setelah rapat, tiap orang pulang ke mejanya masing-masing dan **berpikir sendiri** untuk mengolah info yang didapat dari rapat tadi.

---

## Struktur

```
input
  ↓
Linear(embedDim → ffnDim)   ← expand, "ruang berpikir" lebih luas
  ↓
GELU activation             ← tambah non-linearitas
  ↓
Linear(ffnDim → embedDim)   ← compress balik ke dimensi semula
  ↓
output
```

Kenapa perlu di-expand dulu baru dikompres? Karena dengan dimensi yang lebih besar, model punya "ruang" untuk merepresentasikan pola yang lebih kompleks sebelum akhirnya dikompres jadi representasi yang lebih padat dan bermakna.

`ffnDim` biasanya **4× embedDim** — jadi kalau embedDim 128, ffnDim 512.

---

## Apa Itu GELU?

GELU (Gaussian Error Linear Unit) adalah **fungsi aktivasi** — tugasnya menambah non-linearitas ke dalam model.

Kenapa perlu non-linearitas? Karena tanpa fungsi aktivasi, dua linear layer berturut-turut secara matematis sama saja dengan satu linear layer — tidak ada tambahan kemampuan belajar. Fungsi aktivasi "memecah" linearitas itu.

Analoginya: kalau kamu hanya bisa menggambar garis lurus, kamu tidak bisa gambar wajah orang. Fungsi aktivasi memberi model kemampuan untuk "menggambar kurva".

```
GELU:
- nilai positif besar → lewat hampir utuh
- nilai negatif besar → di-suppress mendekati nol
- nilai di sekitar nol → diproses secara smooth/halus

vs ReLU (lebih kasar):
- nilai positif → lewat utuh
- nilai negatif → langsung jadi 0
```

GPT-2 ke atas pakai GELU karena lebih smooth dibanding ReLU sehingga gradient mengalir lebih baik saat training.

---

## Math Util yang Dibutuhkan

### gelu

```
GELU(x) = x × Φ(x)
dimana Φ(x) adalah cumulative distribution function dari distribusi normal

Aproksimasi yang umum dipakai:
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

---

## Posisi di Transformer Block

```
input x
  ↓
LayerNorm → MultiHeadAttention → + x   ← residual 1
  ↓
LayerNorm → FeedForward                ← di sini
              ↓
           Linear(embedDim → ffnDim)
              ↓
           GELU
              ↓
           Linear(ffnDim → embedDim)
  ↓
+ x                                    ← residual 2
  ↓
output x
```

---

## Step by Step

**Step 1 — Inisialisasi dua LinearLayer**
```
layer1: Linear(embedDim → ffnDim)   expand
layer2: Linear(ffnDim → embedDim)   compress
```

**Step 2 — Forward pass**
```
1. input → layer1    → [seqLen, ffnDim]
2. → GELU per elemen → [seqLen, ffnDim]  ← shape sama, nilai berubah
3. → layer2          → [seqLen, embedDim]
```

---

## Preview Code

```ts
// src/transformer/feedforward.ts

import { LinearLayer } from './linear'
import { gelu } from './math'

export class FeedForward {
  private layer1: LinearLayer
  private layer2: LinearLayer

  constructor(embedDim: number, ffnDim: number) {
    this.layer1 = new LinearLayer(embedDim, ffnDim)
    this.layer2 = new LinearLayer(ffnDim, embedDim)
  }

  forward(input: number[][]): number[][] {
    // input shape: [seqLen, embedDim]

    // Step 1: expand
    const expanded = this.layer1.forward(input)
    // expanded shape: [seqLen, ffnDim]

    // Step 2: GELU per elemen
    const activated = expanded.map(row => row.map(gelu))
    // activated shape: [seqLen, ffnDim]

    // Step 3: compress balik
    const output = this.layer2.forward(activated)
    // output shape: [seqLen, embedDim]

    return output
  }
}
```

```ts
// tambahkan ke src/transformer/math.ts

export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))
}
```

---

## Contoh Nyata

```ts
const ffn = new FeedForward(4, 16)  // embedDim=4, ffnDim=16

const input = [
  [0.2, -0.5, 0.8, 0.1],   // token "biologi"
  [0.1,  0.3, -0.2, 0.4],  // token "adalah"
  [-0.4, 0.7, 0.1, -0.3],  // token "ilmu"
]

const output = ffn.forward(input)
// output shape: [3, 4]  ← shape sama dengan input
// tapi isinya sudah ditransformasi, mengandung informasi yang lebih kaya
```

---

## Math Utils Terkait

```
src/transformer/math.ts
├── matMul()   ← dipakai LinearLayer di dalam FFN
└── gelu()     ← dipakai langsung di FFN
```