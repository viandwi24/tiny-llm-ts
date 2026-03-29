# Embedding

## Apa Itu?

Pintu masuk pertama token ke model. Token ID (angka) tidak bisa langsung diproses transformer — perlu dikonversi ke vector berdimensi tinggi dulu yang mengandung makna semantik.

Ada dua bagian:
- **Token Embedding** — setiap token ID punya vector representasinya sendiri
- **Positional Encoding** — tambah info posisi token ke dalam vector

---

## Apa Itu Vector?

Vector adalah **representasi suatu hal dalam bentuk array angka berdimensi banyak**. Bayangkan merepresentasikan kepribadian seseorang dalam angka:

```
Soekarno = [0.9, 0.8, 0.2, 0.7]
             ↑     ↑     ↑    ↑
          berani pemimpin pendiam religius
```

Di LLM, tiap token punya vector dengan ratusan dimensi (misal 128). Dimensi-dimensi itu tidak punya label seperti "berani" — model sendiri yang menentukan maknanya lewat training.

---

## Bagaimana Vector Terbentuk?

### Awalnya: Random

Di awal training, vector untuk setiap token itu **pure random**. Tidak ada logika, tidak ada makna.

```
"biologi" → [0.23, -0.87, 0.41, 0.12, ...]  ← random
"kimia"   → [0.91, 0.34, -0.22, 0.67, ...]  ← random
"sepatu"  → [-0.44, 0.12, 0.88, -0.31, ...]  ← random
```

### Lalu Maknanya Terbentuk Lewat Training

Setiap kali model salah prediksi, weight diupdate — termasuk weight embedding. Lama-lama, vector tiap token bergeser ke arah yang membuat prediksi lebih akurat.

Contoh — di corpus ada kalimat-kalimat:
```
"biologi adalah ilmu yang mempelajari makhluk hidup"
"kimia adalah ilmu yang mempelajari zat dan reaksi"
"fisika adalah ilmu yang mempelajari energi dan materi"
```

Model sering melihat "biologi", "kimia", "fisika" muncul dalam konteks yang **sama**. Supaya prediksi akurat, model terpaksa menggeser vector ketiga kata itu ke arah yang mirip:

```
setelah training:
"biologi" → [0.82, 0.79, 0.11, ...]
"kimia"   → [0.80, 0.81, 0.09, ...]  ← mirip biologi!
"fisika"  → [0.78, 0.77, 0.13, ...]  ← mirip biologi!
"sepatu"  → [-0.44, 0.12, 0.88, ...]  ← tetap jauh
```

**Kemiripan konteks di data → kemiripan vector.** Bukan karena ada yang manually bilang "biologi dan kimia itu mirip" — model menyimpulkan sendiri dari pola kemunculan.

### Apakah Vector Diambil dari Token Penyusunnya?

Tidak langsung. Di BPE, "biologi" bisa jadi satu token utuh — vectornya berdiri sendiri dan dipelajari dari konteks kemunculannya, bukan dari karakter "b", "i", "o", dll.

Tapi kalau "biologi" tidak ada di vocab dan dipecah BPE jadi ["bio", "logi"], maka representasinya akan dipengaruhi vector "bio" dan "logi" secara tidak langsung.

---

## Token Embedding

Setiap token ID punya satu baris vector di **embedding table** — anggap seperti kamus raksasa:

```
embedding_table shape: [vocabSize, embedDim]
                        ↑           ↑
                       8000        128

token ID 1205 → ambil baris ke-1205 → vector [128 angka]
```

Cara ambilnya simpel — cukup indexing, tidak perlu perkalian matrix:

```ts
const vector = embeddingTable[tokenId]  // [embedDim]
```

---

## Positional Encoding

Transformer tidak tahu urutan token secara natural — mekanisme attention melihat semua token sekaligus tanpa peduli posisi. Tanpa positional encoding, kalimat:

```
"anjing menggigit orang"
"orang menggigit anjing"
```

Akan terlihat identik bagi model karena token-tokennya sama, hanya posisinya berbeda.

Positional encoding menambahkan **sinyal posisi** ke dalam vector token. Ada dua pendekatan:

**Sinusoidal (fixed)** — pakai fungsi sin/cos, tidak dipelajari:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/embedDim))
PE(pos, 2i+1) = cos(pos / 10000^(2i/embedDim))
```

**Learned** — posisi juga punya vector yang dipelajari saat training, mirip token embedding. Lebih fleksibel, dipakai GPT-2.

Untuk proyek ini kita pakai **learned positional encoding** karena lebih simpel diimplementasikan.

---

## Hasil Akhir Embedding

```
input: [1205, 380, 445]
         ↓ token embedding
[[0.2, -0.5, 0.8, ...],   ← vector token 1205
 [0.1,  0.3, -0.2, ...],  ← vector token 380
 [-0.4, 0.7, 0.1, ...]]   ← vector token 445
         ↓ + positional encoding
[[0.3, -0.4, 0.9, ...],   ← token 1205 + info posisi 0
 [0.2,  0.4, -0.1, ...],  ← token 380 + info posisi 1
 [-0.3, 0.8, 0.2, ...]]   ← token 445 + info posisi 2

output shape: [seqLen, embedDim]  →  [3, 128]
```

---

## Step by Step

**Step 1 — Inisialisasi token embedding table**
```
tokenEmbedding: matrix random shape [vocabSize, embedDim]
```

**Step 2 — Inisialisasi positional embedding table**
```
posEmbedding: matrix random shape [maxSeqLen, embedDim]
```

**Step 3 — Forward pass**
```
1. Untuk tiap tokenId → ambil baris dari tokenEmbedding
2. Untuk tiap posisi → ambil baris dari posEmbedding
3. Jumlahkan keduanya
4. Output: matrix [seqLen, embedDim]
```

---

## Preview Code

```ts
// src/transformer/embedding.ts

export class Embedding {
  tokenEmbedding: number[][]   // shape: [vocabSize, embedDim]
  posEmbedding: number[][]     // shape: [maxSeqLen, embedDim]
  embedDim: number

  constructor(vocabSize: number, embedDim: number, maxSeqLen: number) {
    this.embedDim = embedDim

    // inisialisasi random kecil
    const scale = Math.sqrt(1 / embedDim)

    this.tokenEmbedding = Array.from({ length: vocabSize }, () =>
      Array.from({ length: embedDim }, () =>
        (Math.random() * 2 - 1) * scale
      )
    )

    this.posEmbedding = Array.from({ length: maxSeqLen }, () =>
      Array.from({ length: embedDim }, () =>
        (Math.random() * 2 - 1) * scale
      )
    )
  }

  forward(tokenIds: number[]): number[][] {
    // tokenIds: [seqLen]
    // output: [seqLen, embedDim]

    return tokenIds.map((id, pos) => {
      const tokenVec = this.tokenEmbedding[id]
      const posVec = this.posEmbedding[pos]

      // jumlahkan token vector + positional vector
      return tokenVec.map((val, i) => val + posVec[i])
    })
  }
}
```

---

## Contoh Nyata

```ts
const embedding = new Embedding(8000, 128, 256)

const output = embedding.forward([1205, 380, 445])
// output shape: [3, 128]
// setiap token sekarang jadi vector 128 dimensi
// sudah mengandung info token + info posisi
```