# Bagian 6 — Transformer Block: Menyatukan Semuanya

[← Sebelumnya](./05-feedforward.md) | [Selanjutnya →](./07-arsitektur-lengkap.md)

---

## Dari Komponen ke Blok

Kita sudah punya semua komponen:
- **Attention** — mengumpulkan informasi dari konteks
- **FFN** — memproses informasi yang dikumpulkan

Sekarang kita satukan keduanya dengan dua komponen tambahan yang krusial: **Residual Connection** dan **Layer Normalization**.

Gabungan keempatnya disebut satu **Transformer Block**.

---

## Komponen Tambahan 1: Residual Connection

### Masalahnya

Bayangkan kita menumpuk 12 atau 96 layer. Saat training, gradient harus mengalir mundur dari layer terakhir sampai ke layer pertama. Kalau gradient kecil di setiap layer, saat sampai di layer pertama ia sudah mengecil sampai hampir nol.

Akibatnya: **layer-layer awal hampir tidak belajar apapun**. Masalah ini disebut *vanishing gradient*.

### Solusinya: Jalur Pintas

Tambahkan "jalur langsung" yang menghubungkan input ke output setiap sublayer:

```
output = SubLayer(input) + input
                           ↑
                      tambahkan input asli!
```

### Analogi: Belajar Menambah, Bukan Belajar Mengganti

Bayangkan kamu sudah tahu cara memasak. Kamu belajar teknik baru. Alih-alih memulai dari nol, kamu *menambahkan* teknik baru ke kemampuan yang sudah ada.

Model tidak perlu belajar dari nol di setiap layer. Ia belajar "apa yang perlu ditambahkan" ke representasi yang sudah ada. Ini jauh lebih mudah.

Secara matematis, kalau sublayer tidak belajar apapun yang berguna, ia bisa menghasilkan nol — dan output tetap sama dengan input. Layer tidak menjadi *berbahaya*, hanya menjadi *netral*.

### Mengapa Ini Membantu Gradient?

Karena sekarang ada jalur langsung dari output ke input, gradient bisa mengalir melewati jalur itu tanpa melalui sublayer. Gradient tidak akan hilang sekalipun sublayernya kompleks.

---

## Komponen Tambahan 2: Layer Normalization

### Masalahnya

Selama training, nilai-nilai dalam model bisa bergerak tak terkendali. Layer tertentu mungkin menghasilkan angka sangat besar (ratusan, ribuan), sementara layer lain menghasilkan angka sangat kecil (mendekati nol). Ini membuat training tidak stabil — gradient bisa meledak atau menghilang.

### Solusinya: Normalisasi per Token

Untuk setiap token, normalisasi vektornya agar punya mean=0 dan std=1:

```
UNTUK setiap token x dalam batch:
  mean  = rata-rata(x)
  std   = standar_deviasi(x)
  x_norm = (x − mean) / (std + epsilon)
  output = gamma × x_norm + beta
```

`gamma` dan `beta` adalah parameter yang dipelajari. Mereka memberi model kebebasan untuk "membatalkan" normalisasi jika ternyata diperlukan.

`epsilon` (nilai kecil, misal 1e-5) mencegah pembagian dengan nol.

---

## Pre-Norm vs Post-Norm

Ada dua posisi untuk Layer Norm:

### Post-Norm (paper Transformer asli, 2017)
```
output = LayerNorm(input + SubLayer(input))
```

### Pre-Norm (GPT-2 dan seterusnya)
```
output = input + SubLayer(LayerNorm(input))
```

**Pre-Norm lebih stabil untuk model dalam** dan menjadi standar di hampir semua model modern. Alasannya: input ke sublayer selalu dalam skala yang terkontrol, sehingga gradient lebih mudah mengalir.

---

## Diagram Transformer Block Lengkap

```
Input X
  │
  ├──────────────────────────────────┐  residual
  │                                  │
  ↓                                  │
LayerNorm(X)                         │
  ↓                                  │
Multi-Head Self-Attention             │
  ↓                                  │
  ─────────────── + ─────────────────┘
  ↓
X' = X + Attention(LayerNorm(X))
  │
  ├──────────────────────────────────┐  residual
  │                                  │
  ↓                                  │
LayerNorm(X')                        │
  ↓                                  │
Feed Forward Network                 │
  ↓                                  │
  ─────────────── + ─────────────────┘
  ↓
Output = X' + FFN(LayerNorm(X'))
```

---

## Pseudocode Transformer Block

```
KELAS TransformerBlock:
  norm1     = LayerNorm(d_model)
  attention = MultiHeadAttention(d_model, num_heads)
  norm2     = LayerNorm(d_model)
  ffn       = FeedForward(d_model, d_ffn)

  FUNGSI forward(X):
    # Sub-layer 1: Attention dengan pre-norm + residual
    X_normed = norm1.forward(X)
    attn_out = attention.forward(X_normed)
    X_prime  = X + attn_out              ← residual!

    # Sub-layer 2: FFN dengan pre-norm + residual
    X_normed2 = norm2.forward(X_prime)
    ffn_out   = ffn.forward(X_normed2)
    output    = X_prime + ffn_out        ← residual!

    KEMBALIKAN output
```

**Python Native** (murni Python, tanpa library):

```python
import math

def tambah_matriks(A, B):
    """Penjumlahan dua matriks elemen per elemen."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.eps   = eps
        self.gamma = [1.0] * d_model   # skala (dipelajari)
        self.beta  = [0.0] * d_model   # geser (dipelajari)

    def forward(self, X):
        hasil = []
        for baris in X:
            mean = sum(baris) / len(baris)
            var  = sum((x - mean)**2 for x in baris) / len(baris)
            std  = math.sqrt(var + self.eps)
            norm = [(x - mean) / std for x in baris]
            scaled = [self.gamma[i] * norm[i] + self.beta[i]
                      for i in range(len(norm))]
            hasil.append(scaled)
        return hasil

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ffn):
        self.norm1     = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm2     = LayerNorm(d_model)
        self.ffn       = FeedForward(d_model, d_ffn)

    def forward(self, X):
        # Sub-layer 1: Attention + Pre-Norm + Residual
        X_normed = self.norm1.forward(X)
        attn_out = self.attention.forward(X_normed)
        X_prime  = tambah_matriks(X, attn_out)      # residual!

        # Sub-layer 2: FFN + Pre-Norm + Residual
        X_normed2 = self.norm2.forward(X_prime)
        ffn_out   = self.ffn.forward(X_normed2)
        output    = tambah_matriks(X_prime, ffn_out)  # residual!

        return output
```

**Python + Library** (menggunakan PyTorch):

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.1):
        super().__init__()
        self.norm1     = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)  # dari bagian sebelumnya
        self.norm2     = nn.LayerNorm(d_model)
        self.ffn       = FeedForward(d_model, d_ffn)             # dari bagian sebelumnya
        self.dropout   = nn.Dropout(dropout)

    def forward(self, X):
        # Sub-layer 1: Attention + Pre-Norm + Residual
        X_prime = X + self.dropout(self.attention(self.norm1(X)))

        # Sub-layer 2: FFN + Pre-Norm + Residual
        output  = X_prime + self.dropout(self.ffn(self.norm2(X_prime)))

        return output

# Contoh penggunaan
d_model   = 128
num_heads = 4
d_ffn     = 512
batch     = 2
seq_len   = 10

block = TransformerBlock(d_model, num_heads, d_ffn)
X     = torch.randn(batch, seq_len, d_model)
out   = block(X)
print(f"Input  shape: {X.shape}")    # [2, 10, 128]
print(f"Output shape: {out.shape}")  # [2, 10, 128]
# Shape tidak berubah — bisa ditumpuk sebanyak yang diinginkan!
```

---

## Apa yang Dipelajari Tiap Layer?

Penelitian interpretability (studi tentang apa yang dipelajari model) menemukan pola konsisten:

| Layer | Yang Dipelajari |
|-------|----------------|
| Layer awal | Pola lokal: n-gram, kata bersebelahan, frasa pendek |
| Layer tengah | Sintaksis: subjek-predikat, dependency, klausa |
| Layer akhir | Semantik: makna, coreference, konteks panjang, fakta |

Ini analog dengan hierarki pemrosesan di otak — dari persepsi kasar ke pemahaman abstrak.

---

## Penumpukan Block

Model terdiri dari N Transformer Block yang ditumpuk. Setiap block menerima output dari block sebelumnya:

```
Embedding Output [seq_len, d_model]
        ↓
  [Transformer Block 1]  [seq_len, d_model]
        ↓
  [Transformer Block 2]  [seq_len, d_model]
        ↓
  [Transformer Block 3]  [seq_len, d_model]
        ↓
  [Transformer Block N]  [seq_len, d_model]
        ↓
  (ke output projection)
```

Perhatikan bahwa **shape tidak berubah** di sepanjang proses. Ini yang memungkinkan kita menumpuk blok sebanyak yang kita mau.

---

## Berapa Banyak Block yang Dibutuhkan?

Untuk model kecil dengan tujuan Q&A sederhana:

| Jumlah Block | Karakter |
|-------------|---------|
| 2–4 | Bisa belajar pola dasar, kalimat pendek |
| 6–12 | Reasoning lebih dalam, koneksi jarak jauh |
| 24–96 | Model besar, pemahaman sangat dalam |

Untuk model kita, **4 block sudah cukup** sebagai titik awal yang bisa dilatih dalam waktu wajar.

---

## Ringkasan

| Komponen | Fungsi |
|----------|--------|
| Multi-Head Attention | Mengumpulkan informasi dari konteks |
| Feed Forward Network | Memproses dan mentransformasi informasi |
| Layer Normalization | Menstabilkan nilai agar training lancar |
| Residual Connection | Gradient bisa mengalir lancar ke layer awal |
| Pre-Norm | Norm sebelum sublayer (lebih stabil untuk model dalam) |

---

> [Selanjutnya: Arsitektur Lengkap →](./07-arsitektur-lengkap.md)
