# Bagian 7 — Arsitektur Lengkap Model

[← Sebelumnya](./06-transformer-block.md) | [Selanjutnya →](./08-backpropagation.md)

---

## Saatnya Melihat Gambar Besar

Kita sudah pelajari semua potongannya. Sekarang kita susun menjadi model utuh.

Model yang kita bangun adalah **Decoder-Only Transformer** — arsitektur yang sama dengan GPT-2 dari OpenAI, hanya dalam skala jauh lebih kecil.

---

## Diagram Arsitektur Penuh

```
╔═══════════════════════════════════════════════════╗
║              DECODER-ONLY TRANSFORMER             ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  INPUT: Token IDs                                 ║
║  misal: [45, 1205, 342, 890]                      ║
║                                                   ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  EMBEDDING LAYER                                  ║
║  Token Embedding  [vocab_size × d_model]          ║
║         +                                         ║
║  Positional Embedding [max_len × d_model]         ║
║                                                   ║
║  Output shape: [seq_len, d_model]                 ║
║                                                   ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  TRANSFORMER BLOCK #1                             ║
║  ┌───────────────────────────────────────────┐    ║
║  │  LayerNorm → Multi-Head Attention         │    ║
║  │  + Residual                               │    ║
║  │  LayerNorm → Feed Forward Network         │    ║
║  │  + Residual                               │    ║
║  └───────────────────────────────────────────┘    ║
║  Output shape: [seq_len, d_model]                 ║
║                                                   ║
╠═══════════════════════════════════════════════════╣
║  TRANSFORMER BLOCK #2  ...                        ║
╠═══════════════════════════════════════════════════╣
║  TRANSFORMER BLOCK #N  ...                        ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  FINAL LAYER NORM                                 ║
║                                                   ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  OUTPUT PROJECTION (Linear)                       ║
║  d_model → vocab_size                             ║
║                                                   ║
║  Output shape: [seq_len, vocab_size]              ║
║  (disebut LOGITS)                                 ║
║                                                   ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  OUTPUT: Probabilitas token berikutnya            ║
║  softmax(logits[-1]) → pilih token                ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
```

---

## Konfigurasi dan Ekspektasi yang Jujur

> ⚠️ **Penting dibaca sebelum melanjutkan.**
>
> Ada kesalahpahaman umum yang perlu diluruskan: **ukuran model dan jumlah data latih sangat menentukan apa yang bisa dilakukan model — dan batasnya lebih ketat dari yang sering dikira.**
>
> Di bagian ini kita akan jujur tentang apa yang bisa dan tidak bisa dicapai di berbagai skala.

### Dua Tujuan yang Berbeda

**Tujuan A: Belajar cara kerja Transformer**
Kamu ingin memahami attention, backprop, dan training loop. Kamu tidak peduli apakah outputnya bermakna. Ini adalah tujuan yang **sangat valid** dan bisa dicapai dengan model kecil.

**Tujuan B: Membangun model yang benar-benar menjawab pertanyaan**
Kamu ingin model yang bisa menjawab *"Ibukota Indonesia?"* → *"Jakarta"* dengan benar dan konsisten. Ini butuh skala yang **jauh lebih besar**.

Kebanyakan tutorial pemula mencampurkan dua tujuan ini dan akhirnya menyesatkan.

---

### Kenyataan Skala Model

Ini adalah data nyata dari model-model yang sudah dipublikasikan:

| Model | Parameter | Data Latih | Bisa apa? |
|-------|-----------|------------|-----------|
| TinyStories-1M | 1 juta | ~2 miliar token (cerita anak) | Melanjutkan cerita anak saja |
| TinyStories-33M | 33 juta | ~2 miliar token (cerita anak) | Cerita anak yang lebih koheren |
| GPT-2 Small | 117 juta | ~8 miliar token | Teks koheren, *tidak* bisa QA andal |
| SmolLM2-135M | 135 juta | **2 triliun token** | MT-Bench 1.98/10 — masih sangat terbatas |
| Qwen2-0.5B | 500 juta | ~7 triliun token | MMLU 45% — mulai bisa QA tapi masih lemah |
| Phi-1.5 | 1,3 miliar | 30 miliar token (kualitas tinggi) | Mulai bisa reasoning |

**Kesimpulan kritis:** SmolLM2-135M dilatih dengan **2 triliun token** — setara dengan seluruh isi internet beberapa kali — dan hasilnya masih hanya 1.98 dari 10 di benchmark percakapan. Ini 45× lebih besar dari model kecil yang biasa dipakai untuk belajar.

---

### Konfigurasi untuk Belajar (Bukan untuk Produksi)

Konfigurasi ini bagus untuk **memahami cara kerja Transformer** dan **melihat loss turun**:

| Hyperparameter | Nilai | Penjelasan |
|----------------|-------|------------|
| `vocab_size` | 8.000 | Jumlah token yang dikenal |
| `d_model` | 128 | Dimensi vektor tiap token |
| `num_heads` | 4 | Jumlah attention head |
| `num_layers` | 4 | Jumlah Transformer Block |
| `d_ffn` | 512 | Dimensi hidden di FFN (4× d_model) |
| `max_seq_len` | 128 | Panjang konteks maksimum |
| `head_dim` | 32 | d_model / num_heads = 128/4 |

**Yang bisa diharapkan dari konfigurasi ini:**
- ✅ Loss turun dari ~8.9 ke ~3–4 dengan data yang cukup
- ✅ Model belajar pola frekuensi token dan sintaks dasar
- ✅ Kamu memahami seluruh mekanisme Transformer dari dalam
- ❌ Model tidak akan menjawab *"Ibukota Indonesia?"* → *"Jakarta"* secara andal
- ❌ Model tidak akan bercakap-cakap dengan koheren
- ❌ Model sering menghasilkan kalimat yang tidak masuk akal

---

## Menghitung Total Parameter

Mari kita hitung berapa angka yang harus dipelajari model ini:

### Embedding Layer
```
Token embedding:    8.000 × 128  =  1.024.000
Positional embed:     128 × 128  =     16.384
                                 ─────────────
Subtotal:                         1.040.384
```

### Per Transformer Block
```
LayerNorm 1 (gamma + beta):    128 + 128    =       256

Multi-Head Attention:
  W_Q:  128 × 128  =  16.384
  W_K:  128 × 128  =  16.384
  W_V:  128 × 128  =  16.384
  W_O:  128 × 128  =  16.384
  Biases (4 × 128) =     512
                    ─────────
  Subtotal MHA:         66.048

LayerNorm 2 (gamma + beta):    128 + 128    =       256

Feed Forward:
  Linear1: 128×512 + 512  =  66.048
  Linear2: 512×128 + 128  =  65.664
                            ────────
  Subtotal FFN:             131.712

Per block total: 256 + 66.048 + 256 + 131.712 = 198.272
```

### Output Projection
```
Linear 128 → 8.000:  128 × 8.000 + 8.000  =  1.032.000
```

### Total Parameter
```
Embedding:        1.040.384
4 Blocks × 4:       793.088
Output:           1.032.000
Final LayerNorm:        256
─────────────────────────────
TOTAL:           ~2.866.000 ≈ 2,9 juta parameter
```

**Perbandingan skala:**
- Model belajar kita (~3 juta): cukup untuk memahami mekanisme
- GPT-2 Small: 117 juta parameter
- SmolLM2 terkecil yang berguna: 135 juta parameter, dilatih 2 triliun token
- GPT-3: 175 miliar parameter
- Estimasi GPT-4: ~1 triliun parameter

Model kita sangat kecil, **arsitekturnya identik dengan GPT**, tapi kapasitasnya sangat jauh dari model produksi. Analoginya: kalkulator dan komputer modern sama-sama pakai transistor, tapi kapasitasnya berbeda jutaan kali lipat.

---

## Forward Pass: Langkah demi Langkah

Kita trace satu contoh nyata: input "Halo"

Anggap tokenizer mengubah "Halo" menjadi `[<|user|>, halo, <|end|>, <|assistant|>]` → ID = `[7995, 312, 7997, 7996]`

```
Step 1 — Embedding:
  Token 7995 → vektor [0.12, -0.34, ..., -0.55]   (128 angka)
  Token 312  → vektor [0.45,  0.23, ...,  0.67]
  Token 7997 → vektor [-0.1,  0.88, ...,  0.12]
  Token 7996 → vektor [0.73, -0.45, ...,  0.34]

  + positional embedding untuk posisi 0, 1, 2, 3

  Output X: matriks [4 × 128]

Step 2 — Transformer Block 1:
  - LayerNorm(X)
  - Multi-Head Attention (dengan causal mask):
      token 0 hanya lihat token 0
      token 1 lihat token 0 dan 1
      token 2 lihat token 0, 1, 2
      token 3 lihat semua
  - Residual: X += attention_output
  - LayerNorm
  - FFN: proses tiap token secara independen
  - Residual: X += ffn_output
  Output X: masih [4 × 128]

Step 3 — Block 2, 3, 4:
  Proses serupa, abstraksi semakin tinggi
  Output X: masih [4 × 128]

Step 4 — Final LayerNorm + Output Projection:
  logits = X × W_proj        shape: [4 × 8000]
  logits[3] = prediksi setelah token ke-3 (posisi terakhir)
           = [−1.2, 3.4, 0.7, ..., 8.1, ...]   (8000 angka)

Step 5 — Pilih token:
  probs = softmax(logits[3])
  token_berikutnya = argmax(probs)  atau  sample(probs)
```

---

## Output: Logits

Output model adalah matriks `[seq_len, vocab_size]` yang disebut **logits**.

Logits adalah "skor mentah" — bisa negatif, positif, besar, kecil. Belum dalam bentuk probabilitas.

```
logits[-1] = [−1.2, 3.4, 0.7, 8.1, 0.2, −3.5, ...]
                                  ↑
              skor token ID ke-3 sangat tinggi = token ini paling mungkin dipilih
```

Setelah softmax, logits berubah jadi distribusi probabilitas:
```
probs = softmax(logits[-1])
     → [0.001, 0.08, 0.03, 0.82, 0.01, 0.000, ...]
                                   ↑ 82% kemungkinan
```

---

## Pseudocode Model Lengkap

```
KELAS DecoderTransformer:
  embedding   = EmbeddingLayer(vocab_size, d_model, max_seq_len)
  blocks      = [TransformerBlock(d_model, num_heads, d_ffn)] × num_layers
  final_norm  = LayerNorm(d_model)
  projection  = LinearLayer(d_model, vocab_size)

  FUNGSI forward(token_ids):
    # Embedding
    X = embedding.forward(token_ids)        [seq_len, d_model]

    # Lewati semua blocks
    UNTUK setiap block DALAM blocks:
      X = block.forward(X)                  [seq_len, d_model]

    # Normalisasi akhir
    X = final_norm.forward(X)               [seq_len, d_model]

    # Proyeksikan ke vocab
    logits = projection.forward(X)          [seq_len, vocab_size]

    KEMBALIKAN logits

  FUNGSI backward(grad_logits):
    grad = projection.backward(grad_logits)
    grad = final_norm.backward(grad)

    UNTUK block DALAM blocks URUTAN TERBALIK:
      grad = block.backward(grad)

    embedding.backward(grad)
```

**Python Native** (murni Python, tanpa library):

```python
class DecoderTransformer:
    """Decoder-Only Transformer — arsitektur sama dengan GPT."""

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ffn, max_seq_len):
        self.embedding  = EmbeddingLayer(vocab_size, d_model, max_seq_len)
        self.blocks     = [TransformerBlock(d_model, num_heads, d_ffn)
                           for _ in range(num_layers)]
        self.final_norm = LayerNorm(d_model)
        self.projection = LinearLayer(d_model, vocab_size)

    def forward(self, token_ids):
        # token_ids: list of int [seq_len]
        X = self.embedding.forward(token_ids)      # [seq_len, d_model]

        for block in self.blocks:
            X = block.forward(X)                   # [seq_len, d_model]

        X = self.final_norm.forward(X)             # [seq_len, d_model]
        logits = self.projection.forward(X)        # [seq_len, vocab_size]
        return logits

    def hitung_parameter(self):
        """Hitung total parameter model."""
        total = 0
        # Estimasi sederhana
        total += len(self.embedding.tabel_token) * len(self.embedding.tabel_token[0])
        total += len(self.embedding.tabel_posisi) * len(self.embedding.tabel_posisi[0])
        for block in self.blocks:
            d = len(block.norm1.gamma)
            total += d * d * 4   # W_Q, W_K, W_V, W_O
            total += d * 4 * d + 4 * d * d  # FFN linear1 + linear2 (kira-kira)
        total += len(self.projection.W) * len(self.projection.W[0])
        return total

# Konfigurasi model kecil untuk belajar
config = {
    'vocab_size':  8000,
    'd_model':      128,
    'num_heads':      4,
    'num_layers':     4,
    'd_ffn':        512,
    'max_seq_len':  128,
}

model = DecoderTransformer(**config)
print(f"Model dibuat. Parameter perkiraan: ~{model.hitung_parameter():,}")

# Forward pass
token_ids = [7995, 312, 7997, 7996]   # contoh token IDs
logits = model.forward(token_ids)
print(f"Input  tokens: {token_ids}")
print(f"Output logits shape: [{len(logits)}, {len(logits[0])}]")
```

**Python + Library** (menggunakan PyTorch):

```python
import torch
import torch.nn as nn

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ffn, max_seq_len):
        super().__init__()
        self.embedding  = EmbeddingLayer(vocab_size, d_model, max_seq_len)
        self.blocks     = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ffn)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: tabel embedding token = transpos matriks proyeksi
        # Ini mengurangi parameter dan meningkatkan kualitas
        self.projection.weight = self.embedding.token_embedding.weight

    def forward(self, token_ids):
        # token_ids: [batch, seq_len]
        X = self.embedding(token_ids)         # [batch, seq_len, d_model]

        for block in self.blocks:
            X = block(X)                       # [batch, seq_len, d_model]

        X      = self.final_norm(X)            # [batch, seq_len, d_model]
        logits = self.projection(X)            # [batch, seq_len, vocab_size]
        return logits

# Konfigurasi dan test
config = dict(vocab_size=8000, d_model=128, num_heads=4,
              num_layers=4, d_ffn=512, max_seq_len=128)
model = DecoderTransformer(**config)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameter: {total_params:,}")   # ~2.9 juta

# Forward pass
token_ids = torch.tensor([[7995, 312, 7997, 7996]])   # [batch=1, seq_len=4]
logits    = model(token_ids)
print(f"Input  shape: {token_ids.shape}")    # [1, 4]
print(f"Output shape: {logits.shape}")       # [1, 4, 8000]
```

---

## Desain Keputusan Kunci

Beberapa keputusan arsitektur penting dan alasannya:

| Keputusan | Alasan |
|-----------|--------|
| Decoder-only (bukan encoder-decoder) | Lebih sederhana, cukup untuk Q&A dan chat |
| Pre-Norm | Lebih stabil untuk training model dalam |
| Learned positional embedding | Lebih fleksibel, cocok untuk training dari nol |
| Causal mask | Mencegah model "curang" melihat masa depan |
| 4 heads | Cukup banyak jenis relasi yang bisa ditangkap |

---

## Ringkasan

```
Token IDs
    ↓  Embedding (token + posisi)
[seq_len, d_model]
    ↓  × N Transformer Blocks
[seq_len, d_model]   ← shape tidak berubah!
    ↓  Final LayerNorm + Projection
[seq_len, vocab_size]
    ↓  Softmax pada posisi terakhir
Probabilitas token berikutnya
```

---

> [Selanjutnya: Backpropagation — Cara Model Belajar →](./08-backpropagation.md)
