# Bagian 5 — Feed Forward Network

[← Sebelumnya](./04-attention.md) | [Selanjutnya →](./06-transformer-block.md)

---

## Peran FFN dalam Transformer

Attention mengumpulkan informasi dari konteks — setiap token melihat token-token lain dan memilih yang relevan. Tapi informasi yang terkumpul itu masih perlu *diproses* lebih lanjut.

**Feed Forward Network (FFN)** adalah komponen yang melakukan pemrosesan tersebut. Ia bekerja secara independen pada setiap token — tidak ada komunikasi antar token di sini.

---

## Analogi: Pencernaan vs Pengumpulan

Bayangkan kamu baru pulang belanja (attention = mengumpulkan bahan-bahan dari toko). Sekarang kamu perlu memasak — mengolah bahan-bahan itu menjadi sesuatu yang bisa disajikan.

FFN adalah proses memasak itu. Setiap "bahan" (token) diproses sendiri-sendiri menggunakan "resep" (bobot) yang dipelajari dari data.

---

## Struktur: Dua Linear Layer dengan Aktivasi di Tengah

```
Input  [seq_len, d_model]
    ↓
Linear1: d_model → d_ffn     ← ekspansi (perbesar dimensi)
    ↓
Aktivasi GELU
    ↓
Linear2: d_ffn → d_model     ← kompresi (kembali ke dimensi asal)
    ↓
Output [seq_len, d_model]
```

**Aturan umum:** `d_ffn = 4 × d_model`

Untuk model dengan `d_model = 128`: `d_ffn = 512`.

---

## Linear Layer: Operasi Dasar

Linear layer melakukan satu operasi sederhana:

```
output = input × W + b
```

- `input` = vektor masuk
- `W` = matriks bobot (dipelajari)
- `b` = vektor bias (dipelajari)
- `output` = vektor keluaran

Ini sama persis dengan persamaan garis `y = ax + b`, hanya saja dalam bentuk banyak dimensi.

### Pseudocode

```
KELAS LinearLayer:
  W = matriks acak [input_dim, output_dim]
  b = nol [output_dim]

  FUNGSI forward(x):
    KEMBALIKAN x × W + b

  FUNGSI backward(grad_output):
    grad_input = grad_output × W^T
    grad_W     = x^T × grad_output
    grad_b     = sum(grad_output, tiap baris)
    KEMBALIKAN grad_input
```

**Python Native** (murni Python, tanpa library):

```python
import random
import math

def matmul(A, B):
    m, k = len(A), len(A[0])
    n = len(B[0])
    return [[sum(A[i][p] * B[p][j] for p in range(k))
             for j in range(n)] for i in range(m)]

def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        skala = math.sqrt(2.0 / input_dim)  # inisialisasi He
        self.W = [[random.gauss(0, skala) for _ in range(output_dim)]
                  for _ in range(input_dim)]
        self.b = [0.0] * output_dim
        self.input_cache = None  # simpan untuk backward

    def forward(self, x):
        # x: [seq_len, input_dim]
        self.input_cache = x
        hasil = matmul(x, self.W)  # [seq_len, output_dim]
        # Tambahkan bias ke setiap baris
        return [[hasil[i][j] + self.b[j] for j in range(len(self.b))]
                for i in range(len(hasil))]

    def backward(self, grad_output):
        # grad_output: [seq_len, output_dim]
        x = self.input_cache
        grad_input = matmul(grad_output, transpose(self.W))     # [seq_len, input_dim]
        self.grad_W = matmul(transpose(x), grad_output)         # [input_dim, output_dim]
        self.grad_b = [sum(grad_output[i][j] for i in range(len(grad_output)))
                       for j in range(len(self.b))]
        return grad_input
```

**Python + Library** (menggunakan PyTorch):

```python
import torch
import torch.nn as nn

# nn.Linear sudah mengurus forward dan backward secara otomatis
layer = nn.Linear(in_features=128, out_features=512, bias=True)

x   = torch.randn(4, 128)   # [seq_len=4, input_dim=128]
out = layer(x)               # [4, 512]
print(f"Input  shape: {x.shape}")    # [4, 128]
print(f"Output shape: {out.shape}")  # [4, 512]

# Backward otomatis tersedia karena PyTorch autograd
loss = out.sum()
loss.backward()
print(f"Grad W shape: {layer.weight.grad.shape}")  # [512, 128]
```

---

## Mengapa Perlu Aktivasi?

Kalau kita hanya tumpuk linear layer tanpa aktivasi:

```
output = input × W1 × W2 × W3
       = input × (W1 × W2 × W3)
       = input × W_gabungan
```

Berapapun banyak lapisan, hasilnya ekuivalen dengan **satu** linear layer. Model tidak mendapat kapasitas tambahan.

**Fungsi aktivasi** memecahkan linearitas ini. Dengan aktivasi, model bisa mempelajari fungsi-fungsi yang lebih kompleks — kurva, ambang batas, hubungan non-linear.

---

## GELU: Aktivasi yang Dipakai

**GELU (Gaussian Error Linear Unit)** dipakai di BERT, GPT-2, GPT-3, dan keluarganya.

### Rumus

```
GELU(x) = 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

### Bentuk Grafik

```
Output
  │            /
  │           / ← nilai besar positif: dilewatkan hampir utuh
  │          /
  │   ______/ ← sekitar 0: transisi halus
──┼───/ ──────────────────  x
  │ /
  │/ ← nilai negatif besar: mendekati 0 (tapi tidak persis 0)
```

### Perbandingan dengan ReLU

**ReLU:** `max(0, x)` — sederhana, tapi ada "patahan" tajam di x=0 yang kadang menyulitkan training.

**GELU:** Lebih halus, tidak ada patahan. Nilai sedikit negatif masih bisa lewat sedikit (berbeda dengan ReLU yang langsung 0). Ini membantu gradient mengalir lebih baik saat backpropagation.

---

## Pseudocode FFN Lengkap

```
KELAS FeedForward:
  linear1 = LinearLayer(d_model, d_ffn)
  linear2 = LinearLayer(d_ffn, d_model)

  FUNGSI forward(X):
    # X shape: [seq_len, d_model]

    hidden     = linear1.forward(X)          # [seq_len, d_ffn]
    activated  = gelu(hidden)                # [seq_len, d_ffn]
    output     = linear2.forward(activated)  # [seq_len, d_model]

    KEMBALIKAN output

  FUNGSI backward(grad_output):
    grad_activated = linear2.backward(grad_output)
    grad_hidden    = grad_activated × gelu_turunan(hidden)
    grad_input     = linear1.backward(grad_hidden)
    KEMBALIKAN grad_input
```

**Python Native** (murni Python, tanpa library):

```python
import math

def gelu(x):
    """Gaussian Error Linear Unit — aktivasi yang dipakai GPT-2."""
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

def gelu_turunan(x):
    """Turunan GELU untuk backward pass."""
    tanh_val = math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))
    sech2    = 1 - tanh_val**2
    return (0.5 * (1 + tanh_val) +
            0.5 * x * sech2 * math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * x**2))

class FeedForward:
    def __init__(self, d_model, d_ffn):
        self.linear1 = LinearLayer(d_model, d_ffn)
        self.linear2 = LinearLayer(d_ffn, d_model)
        self.hidden_cache = None

    def forward(self, X):
        # X: [seq_len, d_model]
        hidden    = self.linear1.forward(X)    # [seq_len, d_ffn]
        activated = [[gelu(v) for v in baris] for baris in hidden]
        self.hidden_cache = hidden
        output    = self.linear2.forward(activated)  # [seq_len, d_model]
        return output

    def backward(self, grad_output):
        grad_activated = self.linear2.backward(grad_output)
        # Turunan GELU × gradient
        grad_hidden = [[grad_activated[i][j] * gelu_turunan(self.hidden_cache[i][j])
                        for j in range(len(grad_activated[0]))]
                       for i in range(len(grad_activated))]
        return self.linear1.backward(grad_hidden)

# Contoh penggunaan
d_model = 8
d_ffn   = 32   # 4× d_model
seq_len = 3

ffn = FeedForward(d_model, d_ffn)
X   = [[1.0] * d_model for _ in range(seq_len)]
out = ffn.forward(X)
print(f"Input  shape: [{seq_len}, {d_model}]")
print(f"Output shape: [{len(out)}, {len(out[0])}]")
```

**Python + Library** (menggunakan PyTorch):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)

    def forward(self, X):
        # X: [batch, seq_len, d_model]
        hidden    = self.linear1(X)           # [batch, seq_len, d_ffn]
        activated = F.gelu(hidden)            # GELU bawaan PyTorch
        output    = self.linear2(activated)   # [batch, seq_len, d_model]
        return output

# Contoh penggunaan
d_model = 128
d_ffn   = 512   # 4× d_model

ffn = FeedForward(d_model, d_ffn)
X   = torch.randn(2, 10, d_model)   # [batch=2, seq_len=10, d_model=128]
out = ffn(X)
print(f"Input  shape: {X.shape}")    # [2, 10, 128]
print(f"Output shape: {out.shape}")  # [2, 10, 128]
```

---

## Mengapa Ekspansi 4×?

Pertanyaan yang wajar: kenapa `d_ffn = 4 × d_model`?

Ini bukan aturan matematis yang bisa dibuktikan. Ini adalah hasil empiris dari eksperimen di paper Transformer asli (2017) yang terbukti bekerja dengan baik dan kemudian menjadi standar.

Intuisinya: ekspansi memberikan "ruang" yang lebih besar untuk model mengekspresikan fungsi yang kompleks sebelum dikembalikan ke dimensi asalnya. Terlalu kecil → kapasitas kurang. Terlalu besar → boros komputasi tanpa manfaat signifikan.

---

## FFN sebagai "Memori Faktual"

Penelitian terbaru (Geva et al., 2020) menunjukkan hipotesis menarik: **FFN dalam Transformer berperan sebagai penyimpan pengetahuan faktual.**

Ketika model melihat konteks "Paris adalah ibu kota dari ___", FFN-lah yang "mengingat" fakta bahwa Paris berasosiasi dengan Prancis — informasi ini tersimpan dalam bobot W1 dan W2.

Sementara Attention lebih berperan "mencari" informasi yang relevan dari konteks, FFN lebih berperan "mengingat" dan "mengambil" pengetahuan dari training data.

---

## Ringkasan

| Konsep | Penjelasan |
|--------|-----------|
| FFN | Dua linear layer dengan aktivasi di tengah |
| d_ffn | 4× d_model (ekspansi lalu kompresi) |
| GELU | Fungsi aktivasi — transisi halus, baik untuk training |
| Non-linearitas | Yang membuat model bisa belajar pola kompleks |
| Independen | FFN bekerja per-token, tidak ada komunikasi antar token |
| "Memori" | FFN menyimpan pengetahuan faktual dari data training |

---

> [Selanjutnya: Transformer Block — Menyatukan Semuanya →](./06-transformer-block.md)
