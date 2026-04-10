# Bagian 3 — Embedding: Representasi Kata sebagai Angka

[← Sebelumnya](./02-tokenisasi.md) | [Selanjutnya →](./04-attention.md)

---

## Masalah: ID Tidak Mengandung Makna

Dari bagian sebelumnya, kita sudah bisa mengubah teks ke angka ID:

```
"kucing" → ID 500
"anjing" → ID 501
"mobil"  → ID 200
```

Tapi ada masalah besar: **angka-angka ini tidak menceritakan apapun tentang hubungan antar kata.**

Dari angka 500, 501, dan 200 — tidak ada informasi bahwa kucing dan anjing itu *mirip* (sama-sama hewan peliharaan), sedangkan mobil sangat berbeda dari keduanya.

Kalau model menerima angka-angka mentah ini, model akan kesulitan sekali belajar karena tidak ada struktur yang bisa dimanfaatkan.

---

## Analogi: Koordinat Peta vs Nomor Rumah

Bayangkan kamu punya daftar nama kota:
- Jakarta
- Bandung
- Surabaya

Kalau kita kasih nomor urut saja (1, 2, 3) — tidak ada info tentang seberapa dekat kota-kota itu.

Tapi kalau kita kasih **koordinat GPS**:
- Jakarta: (−6.2, 106.8)
- Bandung: (−6.9, 107.6)
- Surabaya: (−7.2, 112.7)

Sekarang dari angka-angkanya saja kita bisa tahu: Bandung lebih dekat ke Jakarta daripada ke Surabaya.

**Embedding bekerja dengan prinsip yang sama.** Setiap token direpresentasikan sebagai titik dalam ruang berdimensi banyak, di mana kata-kata yang maknanya mirip, posisinya juga berdekatan.

---

## Apa Itu Embedding?

**Embedding** adalah cara mengubah setiap token (ID angka) menjadi sebuah **vektor** — daftar angka floating point — yang merepresentasikan makna token tersebut.

Kalau kita pakai embedding 128 dimensi, setiap token jadi 128 angka:

```
Token "kucing" → [0.12, -0.34, 0.78, 0.02, ..., -0.55]  (128 angka)
Token "anjing" → [0.15, -0.31, 0.72, 0.04, ..., -0.50]  (128 angka)
Token "mobil"  → [-0.89, 0.45, -0.12, 0.67, ..., 0.23]  (128 angka)
```

"Kucing" dan "anjing" punya angka yang mirip. "Mobil" sangat berbeda. Model belajar ini otomatis dari data — kita tidak perlu menentukan nilainya secara manual.

---

## Embedding sebagai Tabel Lookup

Secara teknis, embedding adalah tabel besar:
- Jumlah baris = ukuran vocabulary (misal 8.000)
- Jumlah kolom = dimensi embedding (misal 128)

```
Tabel Embedding  (8000 baris × 128 kolom)

         dim_1   dim_2   dim_3  ...  dim_128
token_0: [0.1,  -0.2,   0.5,  ...,  0.3  ]
token_1: [0.7,   0.1,  -0.3,  ..., -0.1  ]
  ...
token_500: [0.12, -0.34,  0.78, ..., -0.55]  ← "kucing"
token_501: [0.15, -0.31,  0.72, ..., -0.50]  ← "anjing"
  ...
```

Saat kita kasih token ID 500, embedding cukup ambil baris ke-500.

---

## Embedding Dipelajari, Bukan Ditentukan

Nilai-nilai dalam tabel ini **tidak kita isi manual**. Di awal training, semua diisi acak. Seiring model belajar memprediksi token berikutnya dari data, nilai-nilai ini bergeser perlahan sampai kata-kata yang sering muncul di konteks yang sama, vektornya menjadi dekat.

Inilah yang membuat embedding menarik — model sendiri yang menemukan representasi paling berguna, tanpa kita perlu memberi tahu makna setiap kata.

---

## Sifat Unik: Aritmetika Kata

Embedding yang dilatih dengan baik punya sifat mengejutkan:

```
vektor("raja") − vektor("pria") + vektor("wanita") ≈ vektor("ratu")
```

Hubungan "raja adalah pria yang berkuasa" analog dengan "ratu adalah wanita yang berkuasa". Model menangkap analogi ini secara implisit hanya dari pola co-occurrence dalam teks.

Fenomena ini pertama kali ditemukan di word2vec (2013) dan sejak saat itu menjadi salah satu bukti paling kuat bahwa embedding bukan sekadar angka acak — ia mengandung struktur semantik yang nyata.

---

## Masalah Kedua: Urutan Kata

Bayangkan dua kalimat:
1. "Kucing mengejar anjing"
2. "Anjing mengejar kucing"

Kata-katanya sama. Kalau kita hanya pakai embedding per token, representasi kedua kalimat itu akan identik — model tidak tahu bedanya!

Padahal maknanya berlawanan total.

**Solusi: Positional Embedding**

Selain embedding untuk *apa* kata itu, kita tambahkan embedding untuk *di mana posisi* kata itu dalam kalimat.

```
Kalimat: "Anjing mengejar kucing"

Representasi akhir setiap posisi:
  Posisi 0 ("Anjing"):   embedding("anjing") + embedding_posisi(0)
  Posisi 1 ("mengejar"): embedding("mengejar") + embedding_posisi(1)
  Posisi 2 ("kucing"):   embedding("kucing") + embedding_posisi(2)
```

Sekarang "anjing" di posisi 0 punya representasi berbeda dari "anjing" di posisi 2.

---

## Dua Jenis Positional Embedding

### Learned Positional Embedding
Sama seperti token embedding — tabel lain yang dipelajari saat training. Satu baris per posisi.

```
Tabel Posisi (max_panjang_kalimat × dimensi_embedding)

         dim_1   dim_2  ...
posisi_0: [0.3,   0.1,  ...]
posisi_1: [0.7,  -0.2,  ...]
posisi_2: [0.1,   0.5,  ...]
```

Keunggulan: fleksibel, model bisa menyesuaikan.  
Kelemahan: tidak bisa generalize ke kalimat yang lebih panjang dari yang pernah dilihat saat training.

### Sinusoidal Positional Embedding
Menggunakan pola gelombang sinus dan kosinus:

```
PE(posisi, 2i)   = sin(posisi / 10000^(2i / d_model))
PE(posisi, 2i+1) = cos(posisi / 10000^(2i / d_model))
```

Keunggulan: bisa generalize ke kalimat lebih panjang, karena polanya matematis bukan tabel.  
Dipakai di: Transformer asli (paper "Attention is All You Need", 2017).

---

## Pseudocode Embedding Layer

```
KELAS EmbeddingLayer:

  tabel_token   = matriks acak [vocab_size, embed_dim]
  tabel_posisi  = matriks acak [max_seq_len, embed_dim]

  FUNGSI forward(token_ids):
    # token_ids: daftar ID, panjang = seq_len
    hasil = []

    UNTUK i, token_id DALAM enumerate(token_ids):
      v_token   = tabel_token[token_id]    # ambil baris token
      v_posisi  = tabel_posisi[i]          # ambil baris posisi
      hasil[i]  = v_token + v_posisi       # jumlahkan

    KEMBALIKAN hasil   # shape: [seq_len, embed_dim]
```

**Python Native** (murni Python, tanpa library):

```python
import random
import math

def buat_matriks_acak(baris, kolom, skala=0.01):
    """Buat matriks dengan nilai acak kecil."""
    return [[random.gauss(0, skala) for _ in range(kolom)]
            for _ in range(baris)]

def tambah_vektor(a, b):
    """Jumlahkan dua vektor elemen per elemen."""
    return [x + y for x, y in zip(a, b)]

class EmbeddingLayer:
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        self.embed_dim = embed_dim
        # Tabel embedding token: satu baris per token
        self.tabel_token  = buat_matriks_acak(vocab_size, embed_dim)
        # Tabel embedding posisi: satu baris per posisi
        self.tabel_posisi = buat_matriks_acak(max_seq_len, embed_dim)

    def forward(self, token_ids):
        """
        token_ids: list of int, panjang = seq_len
        return: list of list (shape [seq_len, embed_dim])
        """
        hasil = []
        for i, token_id in enumerate(token_ids):
            v_token  = self.tabel_token[token_id]   # ambil baris token
            v_posisi = self.tabel_posisi[i]          # ambil baris posisi
            hasil.append(tambah_vektor(v_token, v_posisi))
        return hasil  # shape: [seq_len, embed_dim]

# Contoh penggunaan
vocab_size  = 100
embed_dim   = 8
max_seq_len = 16

layer = EmbeddingLayer(vocab_size, embed_dim, max_seq_len)
token_ids = [5, 23, 47]   # tiga token
output = layer.forward(token_ids)

print(f"Input  shape: [{len(token_ids)}]")
print(f"Output shape: [{len(output)}, {len(output[0])}]")
print(f"Vektor token pertama: {[round(x, 3) for x in output[0]]}")
```

**Python + Library** (menggunakan PyTorch):

```python
import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        # PyTorch otomatis mengelola bobot dan gradient
        self.token_embedding    = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, token_ids):
        """
        token_ids: Tensor shape [seq_len] atau [batch, seq_len]
        """
        seq_len   = token_ids.shape[-1]
        posisi    = torch.arange(seq_len, device=token_ids.device)

        v_token  = self.token_embedding(token_ids)     # [..., seq_len, embed_dim]
        v_posisi = self.position_embedding(posisi)     # [seq_len, embed_dim]

        return v_token + v_posisi  # broadcast otomatis

# Contoh penggunaan
vocab_size  = 8000
embed_dim   = 128
max_seq_len = 512

layer = EmbeddingLayer(vocab_size, embed_dim, max_seq_len)

token_ids = torch.tensor([[5, 23, 47, 11]])  # batch=1, seq_len=4
output = layer(token_ids)

print(f"Input  shape: {token_ids.shape}")    # [1, 4]
print(f"Output shape: {output.shape}")       # [1, 4, 128]
print(f"Parameter count: {sum(p.numel() for p in layer.parameters()):,}")
```

---

## Visualisasi Alur

```
Token IDs: [500, 1205, 200]
            ↓
            ├── Lookup tabel_token
            │   [0.12, -0.34, ..., -0.55]   ← "kucing"
            │   [0.45,  0.23, ...,  0.67]   ← "makan"
            │   [-0.89, 0.45, ...,  0.23]   ← "mobil"
            │
            └── Lookup tabel_posisi
                [0.01,  0.02, ...,  0.03]   ← posisi 0
                [0.04, -0.01, ...,  0.02]   ← posisi 1
                [-0.02, 0.03, ...,  0.01]   ← posisi 2
                         ↓
                   Jumlahkan (token + posisi)
                         ↓
Output: matriks [3 × embed_dim]
(3 token, tiap token punya embed_dim angka)
```

---

## Shape yang Masuk ke Layer Berikutnya

Output dari embedding layer adalah matriks 2D:
```
shape: [seq_len, embed_dim]

seq_len  = panjang kalimat (jumlah token)
embed_dim = ukuran dimensi embedding (misal 128)
```

Matriks ini yang kemudian diproses oleh Transformer Blocks.

---

## Ringkasan

| Konsep | Penjelasan |
|--------|-----------|
| Token embedding | Setiap token ID → vektor bermakna |
| Positional embedding | Setiap posisi → vektor posisi |
| Embedding akhir | Token embedding + Positional embedding |
| Dipelajari | Nilai-nilai embedding otomatis terbentuk saat training |
| Shape output | [seq_len, embed_dim] |

---

> [Selanjutnya: Attention — Cara Model Memahami Konteks →](./04-attention.md)
