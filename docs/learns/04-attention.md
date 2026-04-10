# Bagian 4 — Attention: Cara Model Memahami Konteks

[← Sebelumnya](./03-embedding.md) | [Selanjutnya →](./05-feedforward.md)

---

## Mengapa Konteks Itu Krusial

Baca dua kalimat ini:

- "Saya pergi ke **bank** untuk menabung"
- "Saya duduk di tepi **bank** sungai"

Kata "bank" sama persis, tapi maknanya berbeda total tergantung konteks di sekitarnya.

Embedding yang kita pelajari di bagian sebelumnya memberikan setiap token **satu vektor tetap** yang tidak bergantung konteks. "Bank" selalu jadi vektor yang sama, apapun kalimatnya.

Kita butuh mekanisme yang memungkinkan representasi setiap token *disesuaikan berdasarkan token-token di sekitarnya*. Inilah yang dilakukan **Attention**.

---

## Analogi: Rapat Tim

Kamu sedang di sebuah rapat. Ada 10 orang di ruangan. Ketika kamu mencari informasi untuk menjawab pertanyaan, kamu tidak memperhatikan semua orang dengan intensitas yang sama — kamu secara alami **memberi perhatian lebih pada orang yang paling relevan** dengan topik saat itu.

- Topik keuangan → fokus ke direktur keuangan
- Topik teknis → fokus ke tim engineer
- Topik desain → fokus ke desainer

Setiap token dalam kalimat bisa "memperhatikan" token lain, dengan bobot perhatian yang berbeda-beda. Inilah **Self-Attention** — setiap token memperhatikan semua token lain dalam kalimat yang sama.

---

## Tiga Komponen: Query, Key, Value

Ini adalah konsep paling penting dalam attention. Gunakan analogi **sistem pencarian di perpustakaan**:

### Query (Q) — Pertanyaan yang Kamu Bawa

Kamu datang ke perpustakaan dengan kebutuhan tertentu.
> "Saya mencari buku tentang sejarah Majapahit."

Query adalah representasi dari: *"Apa yang sedang saya cari?"*

### Key (K) — Label di Punggung Buku

Setiap buku punya label yang merangkum isinya.
> "Sejarah Majapahit", "Kerajaan Sriwijaya", "Resep Masakan", dll.

Key adalah representasi dari: *"Apa topik/identitas saya?"*

### Value (V) — Isi Buku yang Sesungguhnya

Ketika label buku cocok dengan pencarianmu, kamu membuka dan membaca isinya.

Value adalah representasi dari: *"Informasi apa yang sebenarnya saya miliki?"*

---

## Cara Kerja Self-Attention

Dalam "self" attention, setiap token membuat Q, K, dan V-nya *sendiri* dari embeddingnya.

### Langkah 1: Buat Q, K, V

Setiap token mengalikan vektornya dengan tiga matriks bobot berbeda:

```
Q = embedding × W_Q
K = embedding × W_K
V = embedding × W_V
```

`W_Q`, `W_K`, `W_V` adalah matriks yang dipelajari saat training.

### Langkah 2: Hitung Skor Kemiripan

Untuk setiap pasangan token (i, j), ukur seberapa relevan token j bagi token i:

```
skor(i, j) = Q_i · K_j    (dot product)
```

Hasilnya adalah matriks skor dengan shape `[seq_len, seq_len]`.

Contoh untuk kalimat "bank tempat menabung":
```
            bank  tempat  menabung
bank      [  8.2,   1.3,    6.5  ]
tempat    [  1.0,   9.4,    2.1  ]
menabung  [  7.8,   3.2,    8.9  ]
```

Token "menabung" memberi skor tinggi ke "bank" — ini berarti representasi "bank" akan banyak dipengaruhi oleh "menabung".

### Langkah 3: Skalakan dengan √d_k

```
skor_scaled = skor / √(dimensi_key)
```

Tanpa ini, dot product bisa menghasilkan angka sangat besar kalau dimensinya tinggi, membuat softmax berikutnya "terlalu yakin" dan menyulitkan training. Pembagian √d_k menstabilkan skalanya.

### Langkah 4: Softmax — Ubah ke Bobot Perhatian

```
bobot_perhatian = softmax(skor_scaled)
```

Setiap baris menjadi distribusi probabilitas yang jumlahnya 1.0:

```
Baris "bank":  [0.70, 0.04, 0.26]
  → "bank" memberi 70% perhatian ke dirinya sendiri
  → 4% ke "tempat"
  → 26% ke "menabung"
```

### Langkah 5: Ambil Informasi dari Value

```
output_i = Σ  bobot_perhatian[i][j]  ×  V_j
           j
```

Output setiap token adalah rata-rata tertimbang dari semua Value, dengan bobot sesuai perhatian.

Artinya: vektor "bank" di output sudah mengandung informasi dari "menabung" (dan token lain) sesuai relevansinya. Ini yang membuat representasi "bank" berbeda di setiap konteks!

---

## Rumus Lengkap

```
Attention(Q, K, V) = softmax( Q × K^T / √d_k ) × V
```

Inilah salah satu rumus paling berpengaruh dalam sejarah deep learning modern. Diperkenalkan di paper *"Attention is All You Need"* (Vaswani et al., 2017).

---

## Causal Masking: Tidak Boleh Intip Masa Depan

Untuk model yang tugasnya memprediksi token berikutnya, ada aturan penting: **token di posisi i tidak boleh melihat token di posisi i+1, i+2, dan seterusnya.**

Kalau tidak ada aturan ini, saat training model bisa langsung melihat jawabannya — seperti ujian dengan jawaban terbuka. Model tidak akan belajar apapun.

**Solusi: Causal Mask**

Sebelum softmax, kita set skor ke −∞ untuk semua posisi yang "di depan":

```
Kalimat 4 token — mask yang diaplikasikan sebelum softmax:

           tok_0   tok_1   tok_2   tok_3
tok_0  [   ada,    -∞,     -∞,     -∞   ]
tok_1  [   ada,    ada,    -∞,     -∞   ]
tok_2  [   ada,    ada,    ada,    -∞   ]
tok_3  [   ada,    ada,    ada,    ada  ]

-∞ setelah softmax → 0.0 (efektif diabaikan)
```

Token pertama hanya bisa melihat dirinya sendiri. Token terakhir bisa melihat semua yang sebelumnya. Ini disebut **triangular causal mask**.

---

## Multi-Head Attention

Satu mekanisme attention hanya bisa "fokus" pada satu jenis hubungan sekaligus. Tapi bahasa kaya dengan berbagai jenis relasi: subjek-objek, kata kerja-pelengkap, referensi ("dia" → siapa?), dan lainnya.

**Solusi: Jalankan beberapa attention secara paralel — masing-masing dengan parameter berbeda.**

Bayangkan beberapa analis membaca teks yang sama secara bersamaan, tapi tiap analis fokus pada aspek yang berbeda:
- Analis 1: hubungan gramatikal
- Analis 2: referensi dan kata ganti
- Analis 3: hubungan semantik
- Analis 4: pola jangka panjang

Hasilnya digabungkan untuk mendapat pemahaman yang lebih kaya.

### Cara Kerjanya

```
embed_dim = 128
num_heads = 4
head_dim  = embed_dim / num_heads = 32

Untuk setiap head h (h = 1 sampai 4):
  Q_h = X × W_Q_h     [seq_len, 32]
  K_h = X × W_K_h     [seq_len, 32]
  V_h = X × W_V_h     [seq_len, 32]
  
  head_output_h = Attention(Q_h, K_h, V_h)  [seq_len, 32]

Gabungkan semua head:
  concat([head_1, head_2, head_3, head_4])  [seq_len, 128]

Kalikan dengan W_O (output projection):
  output = concat × W_O                     [seq_len, 128]
```

---

## Pseudocode Multi-Head Attention Lengkap

```
KELAS MultiHeadAttention:

  W_Q, W_K, W_V  = matriks [embed_dim, embed_dim]
  W_O            = matriks [embed_dim, embed_dim]

  FUNGSI forward(X, num_heads):
    seq_len   = panjang X
    head_dim  = embed_dim / num_heads

    # Proyeksikan ke Q, K, V
    Q = X × W_Q        [seq_len, embed_dim]
    K = X × W_K        [seq_len, embed_dim]
    V = X × W_V        [seq_len, embed_dim]

    # Pecah ke beberapa head
    Q = reshape(Q, [seq_len, num_heads, head_dim])
    K = reshape(K, [seq_len, num_heads, head_dim])
    V = reshape(V, [seq_len, num_heads, head_dim])

    # Hitung attention per head
    skor = Q × K^T / sqrt(head_dim)  [num_heads, seq_len, seq_len]
    skor = terapkan_causal_mask(skor)
    bobot = softmax(skor)

    # Ambil informasi dari V
    head_outputs = bobot × V          [num_heads, seq_len, head_dim]

    # Gabungkan heads
    gabungan = reshape(head_outputs, [seq_len, embed_dim])

    # Output projection
    output = gabungan × W_O           [seq_len, embed_dim]

    KEMBALIKAN output
```

**Python Native** (murni Python, tanpa library):

```python
import math
import random

def matmul(A, B):
    """Perkalian matriks A [m,k] × B [k,n] → [m,n]."""
    m, k = len(A), len(A[0])
    n = len(B[0])
    return [[sum(A[i][p] * B[p][j] for p in range(k))
             for j in range(n)] for i in range(m)]

def transpose(M):
    """Transpose matriks."""
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

def softmax_baris(baris):
    """Softmax satu baris."""
    m = max(baris)  # stabilitasi numerik
    exp_vals = [math.exp(x - m) for x in baris]
    total = sum(exp_vals)
    return [v / total for v in exp_vals]

def buat_causal_mask(seq_len):
    """Mask segitiga bawah — True = boleh lihat."""
    return [[1 if j <= i else 0 for j in range(seq_len)]
            for i in range(seq_len)]

def satu_head_attention(Q, K, V, mask=None):
    """Hitung attention untuk satu head."""
    seq_len = len(Q)
    d_k = len(Q[0])
    skala = math.sqrt(d_k)

    # Skor = Q × K^T / sqrt(d_k)
    skor = matmul(Q, transpose(K))
    skor = [[s / skala for s in baris] for baris in skor]

    # Terapkan causal mask (set posisi masa depan ke -inf)
    if mask:
        for i in range(seq_len):
            for j in range(seq_len):
                if not mask[i][j]:
                    skor[i][j] = -1e9

    # Softmax per baris
    bobot = [softmax_baris(baris) for baris in skor]

    # Output = bobot × V
    return matmul(bobot, V)

def buat_matriks_acak(baris, kolom):
    return [[random.gauss(0, 0.02) for _ in range(kolom)]
            for _ in range(baris)]

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.W_Q = buat_matriks_acak(embed_dim, embed_dim)
        self.W_K = buat_matriks_acak(embed_dim, embed_dim)
        self.W_V = buat_matriks_acak(embed_dim, embed_dim)
        self.W_O = buat_matriks_acak(embed_dim, embed_dim)

    def forward(self, X):
        seq_len = len(X)
        Q = matmul(X, self.W_Q)  # [seq_len, embed_dim]
        K = matmul(X, self.W_K)
        V = matmul(X, self.W_V)
        mask = buat_causal_mask(seq_len)

        # Proses tiap head secara terpisah (versi sederhana)
        head_outputs = []
        for h in range(self.num_heads):
            start = h * self.head_dim
            end   = start + self.head_dim
            Q_h = [baris[start:end] for baris in Q]
            K_h = [baris[start:end] for baris in K]
            V_h = [baris[start:end] for baris in V]
            head_outputs.append(satu_head_attention(Q_h, K_h, V_h, mask))

        # Gabungkan semua head
        gabungan = []
        for i in range(seq_len):
            baris = []
            for h in range(self.num_heads):
                baris.extend(head_outputs[h][i])
            gabungan.append(baris)  # [embed_dim]

        return matmul(gabungan, self.W_O)  # [seq_len, embed_dim]

# Contoh penggunaan (dimensi kecil untuk demonstrasi)
embed_dim = 8
num_heads = 2
seq_len   = 3

mha = MultiHeadAttention(embed_dim, num_heads)
X   = [[random.gauss(0, 1) for _ in range(embed_dim)] for _ in range(seq_len)]
out = mha.forward(X)
print(f"Input  shape: [{seq_len}, {embed_dim}]")
print(f"Output shape: [{len(out)}, {len(out[0])}]")
```

**Python + Library** (menggunakan PyTorch):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, X):
        # X: [batch, seq_len, embed_dim]
        B, T, C = X.shape

        Q = self.W_Q(X).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(X).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(X).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Q,K,V: [B, num_heads, T, head_dim]

        # Hitung skor dan terapkan causal mask
        skor = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)  # [B, H, T, T]
        mask = torch.tril(torch.ones(T, T, device=X.device)).bool()
        skor = skor.masked_fill(~mask, float('-inf'))

        bobot = F.softmax(skor, dim=-1)

        out = bobot @ V  # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, embed_dim]
        return self.W_O(out)

# Contoh penggunaan
embed_dim = 128
num_heads = 4
batch     = 2
seq_len   = 16

mha = MultiHeadAttention(embed_dim, num_heads)
X   = torch.randn(batch, seq_len, embed_dim)
out = mha(X)
print(f"Input  shape: {X.shape}")    # [2, 16, 128]
print(f"Output shape: {out.shape}")  # [2, 16, 128]
```

---

## Kompleksitas dan Keterbatasan

Perhatikan bahwa dalam menghitung skor, kita melakukan `Q × K^T` yang menghasilkan matriks `[seq_len × seq_len]`. Setiap token dibandingkan dengan setiap token lain.

Ini berarti kompleksitas komputasi attention adalah **O(n²)** terhadap panjang sequence. Kalimat 2× lebih panjang = 4× lebih berat.

Inilah alasan mengapa semua LLM punya batasan **context window** — panjang maksimum teks yang bisa diproses sekaligus. Untuk model kecil kita, context window yang masuk akal adalah 128–512 token.

---

## Ringkasan

| Konsep | Penjelasan |
|--------|-----------|
| Self-Attention | Setiap token memperhatikan token lain dalam kalimat |
| Query (Q) | "Apa yang sedang saya cari?" |
| Key (K) | "Apa topik saya?" |
| Value (V) | "Informasi apa yang saya miliki?" |
| Skor | Dot product Q·K — ukuran relevansi |
| Causal Mask | Blokir token melihat masa depan saat training |
| Multi-Head | Beberapa attention paralel untuk berbagai jenis relasi |
| Rumus | `softmax(Q K^T / √d_k) × V` |

---

> [Selanjutnya: Feed Forward Network →](./05-feedforward.md)
