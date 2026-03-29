# Transformer

## Definisi

Transformer adalah arsitektur model machine learning yang dirancang untuk memproses dan memahami urutan data — dalam konteks LLM, urutan kata/token.

Diciptakan oleh Google pada 2017 lewat paper **"Attention Is All You Need"**. Sebelum transformer, model bahasa pakai RNN yang memproses kata satu per satu secara berurutan — lambat dan susah ingat konteks panjang. Transformer memecahkan itu dengan memproses semua kata sekaligus paralel dan bisa "lihat" relasi antar kata di mana saja dalam kalimat.

GPT, BERT, LLaMA, Claude — semuanya berbasis transformer.

---

## Kenapa Pakai Neural Network?

Karena tujuannya adalah **belajar dari data** — bukan diprogram secara manual.

Kamu tidak bisa manually tulis aturan "kalau ada kata X diikuti Y maka Z" untuk semua kemungkinan bahasa — terlalu banyak dan terlalu kompleks. Neural network membiarkan data yang mengajarkan polanya sendiri lewat proses training.

Transformer secara spesifik dipilih karena:
- Bisa proses sequence panjang dengan efisien
- Bisa tangkap relasi antar kata yang jauh posisinya
- Mudah di-parallelize (training lebih cepat)
- Bisa di-scale — makin besar makin pintar

---

## Konsep Dasar Neural Network

Neural network terinspirasi dari cara kerja otak manusia. Otak punya miliaran neuron yang saling terhubung — sinyal listrik mengalir dari neuron ke neuron, dan dari sinyal-sinyal itu otak bisa "belajar" mengenali pola.

Neural network meniru konsep itu secara matematis:

```
Input → [neuron] → [neuron] → [neuron] → Output
```

Setiap "neuron" itu sebenernya cuma operasi matematika sederhana — kalikan input dengan angka, tambahkan angka lain, lewatkan fungsi. Yang bikin powerful adalah jutaan operasi kecil ini digabung dan diulang.

---

## Konsep Penting

### Weight (Bobot)

Bayangkan kamu seorang juri memasak yang harus menilai apakah masakan enak atau tidak. Kamu punya beberapa kriteria: rasa, tampilan, aroma, tekstur. Tapi tidak semua kriteria sama pentingnya — rasa bobotnya 60%, aroma 20%, tampilan 15%, tekstur 5%.

Angka-angka bobot itulah **weight** — mereka menentukan seberapa penting tiap input terhadap output.

```
nilai_akhir = (rasa × 0.6) + (aroma × 0.2) + (tampilan × 0.15) + (tekstur × 0.05)
```

Di neural network, weight ini awalnya random. Lewat training, weight diupdate terus sampai model tahu mana yang penting.

### Bias

Lanjut analogi juri — misalnya ada juri yang dasarnya orangnya picky, bahkan masakan sempurna pun dia beri nilai lebih rendah. Atau sebaliknya, ada juri yang murah hati. Kecenderungan dasar ini adalah **bias** — angka yang selalu ditambahkan ke output terlepas dari input apapun.

```
nilai_akhir = (rasa × 0.6) + (aroma × 0.2) + ... + bias
```

Bias memberi model fleksibilitas. Bias diinisialisasi nol di awal, lalu ikut diupdate lewat backpropagation sama seperti weight.

### Loss

Angka yang mengukur **seberapa salah** prediksi model. Kalau loss besar = model masih banyak salah. Kalau loss kecil = model sudah bagus. Tujuan training = minimasi loss.

### Gradient

Ketika bola meleset ke kiri, kamu instingtif tahu harus geser tangan ke kanan. Gradient itu seperti petunjuk arah itu — dia bilang ke model "weight ini perlu dinaikkan, weight itu perlu diturunkan, sebesar ini" supaya loss berkurang.

```
gradient positif → loss naik kalau weight naik → turunkan weight
gradient negatif → loss turun kalau weight naik → naikkan weight
```

### Backpropagation

Algoritma yang menghitung semua gradient sekaligus dengan cara menelusuri balik dari loss ke input:

```
Forward pass:
input → layer 1 → layer 2 → layer 3 → output → loss

Backward pass:
loss → layer 3 → layer 2 → layer 1
  ↓       ↓          ↓         ↓
grad3   grad2      grad1   grad_input
```

### Learning Rate

Seberapa besar langkah update weight setiap iterasi:

```
weight_baru = weight_lama - (learning_rate × gradient)
```

- Terlalu besar → melangkah terlalu jauh → melewati titik optimal
- Terlalu kecil → melangkah terlalu pelan → butuh waktu sangat lama
- Pas → pelan-pelan turun ke titik optimal dengan stabil

---

## Sub-Components

| Component | File | Kegunaan |
|---|---|---|
| Embedding | embedding.ts | Token ID → vector + positional info |
| MultiHeadAttention | attention.ts | Tiap token lihat konteks token lain |
| FeedForward | feedforward.ts | Proses info lebih dalam per token |
| LayerNorm | layernorm.ts | Normalisasi nilai supaya training stabil |
| LinearLayer | linear.ts | Transformasi matrix dasar (paling banyak dipakai) |

---

## Alur Forward Pass

```
[1205, 380, 445]              ← token IDs input
        ↓
   Embedding                  ← ID → vector + posisi
        ↓
TransformerBlock × N
  ├── LayerNorm
  ├── MultiHeadAttention      ← tiap token lihat konteks
  ├── + residual
  ├── LayerNorm
  ├── FeedForward             ← proses info lebih dalam
  └── + residual
        ↓
  LinearLayer                 ← project ke vocabSize
        ↓
[0.1, 0.3, 0.8, ...]         ← logits: skor tiap token sebagai prediksi berikutnya
```

---

## Config

```ts
interface TransformerConfig {
  vocabSize: number    // jumlah token unik (misal 8000)
  embedDim: number     // ukuran vector embedding (misal 128)
  numHeads: number     // jumlah attention head (misal 4)
  numLayers: number    // jumlah transformer block (misal 4)
  ffnDim: number       // ukuran hidden FFN (misal 512 = 4 × embedDim)
  maxSeqLen: number    // panjang konteks maksimal (misal 256)
}
```

---

## Preview Code

```ts
// src/transformer/index.ts

export class Transformer {
  private embedding: Embedding
  private blocks: TransformerBlock[]
  private projection: LinearLayer

  constructor(config: TransformerConfig) {
    this.embedding = new Embedding(config.vocabSize, config.embedDim, config.maxSeqLen)
    this.blocks = Array.from({ length: config.numLayers }, () =>
      new TransformerBlock(config.embedDim, config.numHeads, config.ffnDim)
    )
    this.projection = new LinearLayer(config.embedDim, config.vocabSize)
  }

  forward(tokenIds: number[]): number[][] {
    let x = this.embedding.forward(tokenIds)
    for (const block of this.blocks) {
      x = block.forward(x)
    }
    return this.projection.forward(x)
  }
}
```