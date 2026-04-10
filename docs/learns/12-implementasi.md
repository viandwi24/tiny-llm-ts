# Bagian 12 — Panduan Implementasi dari Nol

[← Sebelumnya](./11-inference.md) | [← Kembali ke Index](./index.md)

---

## Baca Ini Dulu: Ekspektasi yang Jujur

Ini adalah bagian yang paling sering diabaikan di tutorial LLM, tapi justru paling penting.

**Apakah model kecil yang kita bangun bisa menjawab "Ibukota Indonesia?" → "Jakarta"?**

Jawaban jujurnya: **tidak secara andal**, kalau kita membangun dari nol dengan skala kecil.

Ini bukan kegagalan desain. Ini adalah hukum fisika dari machine learning. Dan memahami *mengapa* adalah bagian penting dari belajar LLM yang sesungguhnya.

---

## Mengapa Model Kecil Tidak Bisa Menjawab Fakta?

### Masalah 1: Kapasitas Penyimpanan

Model dengan 3 juta parameter punya ~3 juta angka untuk "menyimpan semua yang dia tahu". Wikipedia bahasa Indonesia saja punya lebih dari 600 juta kata. Kamu tidak bisa memasukkan lautan ke dalam gelas.

Analoginya: bayangkan kamu harus menghafal seluruh ensiklopedia menggunakan hanya 3 juta neuron — sedangkan otak manusia punya 86 miliar neuron. Tidak cukup.

### Masalah 2: Data Latih Tidak Cukup

Berdasarkan **Chinchilla scaling laws** (riset dari DeepMind, 2022), jumlah data optimal untuk melatih model adalah sekitar **20 token per parameter**. Untuk model 3 juta parameter, itu berarti 60 juta token — sekitar 45 juta kata.

Tapi itu hanya untuk *mempelajari pola bahasa dasar*, bukan untuk *mengingat fakta secara andal*. Fakta seperti "ibukota Indonesia = Jakarta" harus dilihat berkali-kali dalam konteks yang beragam agar model bisa mengingat dan mengambilnya kembali dengan benar.

### Masalah 3: Text Completion ≠ Question Answering

Language model *dasar* adalah mesin pelengkap teks — ia memprediksi token berikutnya. Bukan mesin menjawab pertanyaan.

Ketika menerima input `"Ibukota Indonesia adalah"`, model akan melengkapi dengan apapun yang paling sering mengikuti pola itu di data latih. Kalau data latihnya bagus, bisa jadi "Jakarta". Tapi model juga bisa menjawab "kota terbesar di Asia Tenggara" atau kalimat yang sepenuhnya tidak relevan.

Kemampuan *mengikuti instruksi* ("jawab pertanyaan ini dengan format X") membutuhkan **fine-tuning tambahan** dengan data percakapan yang dikurasi, terlepas dari ukuran model.

---

## Data Nyata: Berapa Skala yang Dibutuhkan?

Berikut adalah perbandingan dari model-model yang sudah dipublikasikan dan dievaluasi secara terbuka:

| Model | Parameter | Data Latih | Skor MT-Bench* | Catatan |
|-------|-----------|------------|---------------|---------|
| Model belajar kita | ~3 juta | ~1–5 juta token | Tidak terukur | Untuk belajar arsitektur |
| TinyStories-33M | 33 juta | ~2 miliar token | Tidak berlaku | Hanya bisa cerita anak |
| GPT-2 Small | 117 juta | ~8 miliar token | ~1/10 | Teks koheren, QA tidak andal |
| SmolLM2-135M | 135 juta | **2 triliun token** | **1.98/10** | Masih sangat terbatas |
| Qwen2-0.5B | 500 juta | ~7 triliun token | ~3/10 | Mulai bisa menjawab |
| Phi-1.5 | 1,3 miliar | 30 miliar token (kualitas sangat tinggi) | ~4–5/10 | Data kualitas > kuantitas |

*MT-Bench: skala 1–10 untuk kualitas percakapan. GPT-4 ≈ 8.99/10.

**Takeaway:** SmolLM2-135M dilatih dengan data sebanyak **2 triliun token** — itu sekitar 400.000 kali lebih banyak dari skenario "5 juta token" — dan hasilnya masih 1.98 dari 10. Artinya bahkan dengan data sangat besar sekalipun, model di bawah ~500 juta parameter punya keterbatasan fundamental.

---

## Jadi Apa Gunanya Membangun Model Kecil?

Ini pertanyaan yang tepat. Jawabannya adalah: **nilai pembelajaran, bukan nilai produksi.**

### Yang Akan Kamu Pelajari

Dengan membangun model kecil dari nol, kamu akan benar-benar memahami:

- Bagaimana forward pass berjalan melalui setiap lapisan
- Bagaimana gradient mengalir mundur (backpropagation)
- Mengapa loss turun perlahan dan apa artinya
- Mengapa inisialisasi bobot penting
- Mengapa learning rate kritis
- Bagaimana tokenizer BPE bekerja
- Mengapa residual connection dan layer norm diperlukan

Semua ini adalah pengetahuan yang tidak bisa diperoleh hanya dengan *menggunakan* model besar.

### Apa yang Bisa Diharapkan dari Model Kecil

Dengan model ~3 juta parameter dilatih dengan data yang cukup (~10–50 juta token teks bersih):

- ✅ Loss turun secara konsisten selama training
- ✅ Model menghasilkan teks yang secara sintaksis masuk akal (dalam domain data latih)
- ✅ Kalau dilatih khusus pada satu topik sempit (misal: cerita pendek), model bisa menghasilkan teks yang mirip topik itu
- ❌ Model tidak akan menjawab pertanyaan faktual secara andal
- ❌ Model tidak akan mengikuti instruksi dengan benar
- ❌ Output di luar domain data latih akan kacau

Analoginya: model kecil seperti bayi yang baru belajar bicara. Ia meniru pola suara yang sering didengar, tapi belum bisa menjawab pertanyaan dengan pemahaman.

---

## Jalur Realistis Menuju Model yang Bisa Menjawab

Kalau tujuanmu adalah model yang *benar-benar bisa* menjawab pertanyaan sederhana, ada dua jalur yang realistis:

### Jalur A: Fine-tuning Model yang Sudah Ada (Direkomendasikan)

Ambil model pretrained yang sudah ada (misal GPT-2, atau model multilingual seperti mT5), lalu fine-tune dengan data QA Indonesia.

```
Model pretrained (sudah punya pengetahuan bahasa)
    ↓
Fine-tuning dengan ~10.000 pasang Q&A Indonesia
    ↓
Model yang bisa menjawab pertanyaan Indonesia
```

**Keunggulan:** Jauh lebih cepat, butuh jauh lebih sedikit data, hasilnya jauh lebih baik.

**Contoh nyata:** Banyak model Indonesia seperti IndoBERT dan IndoGPT dibangun dengan cara ini.

### Jalur B: Membangun dari Nol (Butuh Skala Besar)

Kalau ingin benar-benar dari nol dan hasilnya berguna:

```
Minimum yang realistis:
  Parameter: ~100–500 juta
  Data latih: ~1–10 miliar token teks Indonesia bersih
  Hardware: GPU, training bisa memakan waktu berminggu-minggu
```

Ini bukan tidak mungkin, tapi bukan proyek akhir pekan.

---

## Spesifikasi Proyek Belajar Kita

Dengan semua konteks di atas, berikut spesifikasi yang jujur untuk model belajar:

| Hyperparameter | Nilai |
|----------------|-------|
| `vocab_size` | 8.000 |
| `d_model` | 128 |
| `num_heads` | 4 |
| `num_layers` | 4 |
| `d_ffn` | 512 |
| `max_seq_len` | 128 |

**Total parameter:** ~2,9 juta  
**Tujuan:** Memahami cara kerja Transformer, bukan model produksi  
**Output yang diharapkan:** Teks yang secara sintaksis masuk akal dalam domain sempit  

---

## Struktur Komponen yang Perlu Diimplementasikan

```
model/
├── tokenizer
│   ├── bpe_trainer()       — latih tokenizer dari data
│   ├── encode()            — teks → token IDs
│   └── decode()            — token IDs → teks
│
├── layers
│   ├── LinearLayer         — output = input × W + b
│   ├── LayerNorm           — normalisasi per token
│   ├── EmbeddingLayer      — token + posisi → vektor
│   ├── MultiHeadAttention  — self-attention dengan causal mask
│   └── FeedForward         — 2-layer MLP dengan GELU
│
├── model
│   └── DecoderTransformer  — menyatukan semua komponen
│
├── training
│   ├── DataLoader          — baca data, buat batch
│   ├── cross_entropy()     — hitung loss
│   └── Adam                — update parameter
│
└── inference
    ├── generate()          — autoregressive generation
    └── chat_loop()         — antarmuka percakapan
```

---

## Implementasi Komponen: Pseudocode Lengkap

### 1. Linear Layer

```
KELAS LinearLayer(input_dim, output_dim):

  # Inisialisasi Xavier — mencegah nilai terlalu besar/kecil di awal
  skala = sqrt(1.0 / input_dim)
  W = random_normal(shape=[input_dim, output_dim]) × skala
  b = nol(shape=[output_dim])

  grad_W = nol_seperti(W)
  grad_b = nol_seperti(b)

  FUNGSI forward(x):
    simpan(x)           # untuk backward
    KEMBALIKAN x × W + b

  FUNGSI backward(grad_out):
    x = ambil_simpanan()
    grad_W = x^T × grad_out
    grad_b = sum(grad_out, axis=0)
    KEMBALIKAN grad_out × W^T
```

### 2. Layer Normalization

```
KELAS LayerNorm(dim):

  gamma = satu(shape=[dim])      # scale, dipelajari
  beta  = nol(shape=[dim])       # shift, dipelajari
  eps   = 1e-5

  FUNGSI forward(x):
    # x shape: [seq_len, dim] — normalisasi per token (per baris)
    mean   = rata_rata(x, per_baris)            [seq_len, 1]
    var    = varians(x, per_baris)              [seq_len, 1]
    x_norm = (x − mean) / sqrt(var + eps)       [seq_len, dim]

    simpan(x, x_norm, mean, var)
    KEMBALIKAN gamma × x_norm + beta
```

### 3. Embedding Layer

```
KELAS EmbeddingLayer(vocab_size, d_model, max_seq_len):

  tabel_token   = random_normal([vocab_size, d_model]) × 0.02
  tabel_posisi  = random_normal([max_seq_len, d_model]) × 0.01

  FUNGSI forward(token_ids):
    seq_len  = panjang(token_ids)
    v_token  = tabel_token[token_ids]       [seq_len, d_model]
    v_posisi = tabel_posisi[0:seq_len]      [seq_len, d_model]
    simpan(token_ids)
    KEMBALIKAN v_token + v_posisi
```

### 4. Multi-Head Attention (dengan Causal Mask)

```
KELAS MultiHeadAttention(d_model, num_heads):

  head_dim = d_model / num_heads
  W_Q = LinearLayer(d_model, d_model)
  W_K = LinearLayer(d_model, d_model)
  W_V = LinearLayer(d_model, d_model)
  W_O = LinearLayer(d_model, d_model)

  FUNGSI forward(X):
    seq_len = baris(X)

    Q = W_Q.forward(X)    # [seq_len, d_model]
    K = W_K.forward(X)
    V = W_V.forward(X)

    # Pecah ke heads → [num_heads, seq_len, head_dim]
    Q, K, V = reshape dan transpose ke (num_heads, seq_len, head_dim)

    skor = Q × K^T / sqrt(head_dim)   # [num_heads, seq_len, seq_len]

    # Causal mask: blokir posisi masa depan
    UNTUK setiap posisi (i, j) di mana j > i:
      skor[:, i, j] = −1e9

    bobot  = softmax(skor)
    output = bobot × V                # [num_heads, seq_len, head_dim]

    # Gabungkan heads → [seq_len, d_model]
    output = reshape(transpose(output), [seq_len, d_model])
    KEMBALIKAN W_O.forward(output)
```

### 5. Feed Forward Network

```
KELAS FeedForward(d_model, d_ffn):

  linear1 = LinearLayer(d_model, d_ffn)
  linear2 = LinearLayer(d_ffn, d_model)

  FUNGSI gelu(x):
    c = sqrt(2 / π)
    KEMBALIKAN 0.5 × x × (1 + tanh(c × (x + 0.044715 × x³)))

  FUNGSI forward(X):
    h     = linear1.forward(X)
    h_act = gelu(h)
    KEMBALIKAN linear2.forward(h_act)
```

### 6. Transformer Block

```
KELAS TransformerBlock(d_model, num_heads, d_ffn):

  norm1     = LayerNorm(d_model)
  attention = MultiHeadAttention(d_model, num_heads)
  norm2     = LayerNorm(d_model)
  ffn       = FeedForward(d_model, d_ffn)

  FUNGSI forward(X):
    X = X + attention.forward(norm1.forward(X))   # residual
    X = X + ffn.forward(norm2.forward(X))         # residual
    KEMBALIKAN X
```

### 7. Model Lengkap

```
KELAS DecoderTransformer(config):

  embedding  = EmbeddingLayer(vocab_size, d_model, max_seq_len)
  blocks     = [TransformerBlock(d_model, num_heads, d_ffn)] × num_layers
  final_norm = LayerNorm(d_model)
  projection = LinearLayer(d_model, vocab_size)

  FUNGSI forward(token_ids):
    X = embedding.forward(token_ids)
    UNTUK block DALAM blocks:
      X = block.forward(X)
    X = final_norm.forward(X)
    KEMBALIKAN projection.forward(X)   # [seq_len, vocab_size]
```

### 8. Training Loop

```
FUNGSI latih(config):

  model      = DecoderTransformer(config)
  optimizer  = Adam(lr=3e-4)
  dataloader = DataLoader(data_file, batch_size, seq_len)

  UNTUK step DALAM range(max_steps):

    input_ids, target_ids = dataloader.next_batch()
    logits = model.forward(input_ids)

    # Loss: cross-entropy
    probs  = softmax(logits)
    loss   = −rata_rata(log(probs diambil di posisi target_ids))

    # Backward
    grad_logits = probs
    grad_logits[target_ids] −= 1
    grad_logits /= seq_len
    model.backward(grad_logits)

    optimizer.langkah(model.semua_parameter())

    JIKA step % 100 == 0:
      tampilkan(f"Step {step} | Loss: {loss:.4f}")
    JIKA step % 5000 == 0:
      simpan_checkpoint(model, step, loss)
```

**Python Native** (murni Python, tanpa library ML):

```python
import json, math, os, random

def latih(config):
    model     = DecoderTransformer(config)
    optimizer = Adam(lr=3e-4)
    loader    = DataLoader("train.json", config['batch_size'], config['seq_len'])
    step = 0

    for step in range(config['max_steps']):
        input_ids, target_ids = loader.next_batch()
        logits = model.forward(input_ids)    # [batch, seq_len, vocab_size]

        # Cross-entropy loss (lihat Bagian 8)
        loss = cross_entropy(logits, target_ids)

        # Gradient logits (analitik)
        model.backward(logits, target_ids)

        optimizer.langkah(model.semua_parameter())

        if step % 100 == 0:
            print(f"Step {step:6d} | Loss: {loss:.4f}")

        if step % 5000 == 0 and step > 0:
            with open(f"ckpt-{step}.json", 'w') as f:
                json.dump({'step': step, 'loss': loss,
                           'bobot': model.ambil_bobot()}, f)

# Jalankan
config = {
    'vocab_size': 8000, 'd_model': 128, 'num_heads': 4,
    'num_layers': 4, 'd_ffn': 512, 'max_seq_len': 128,
    'batch_size': 8, 'max_steps': 50000
}
latih(config)
```

**Python + Library** (menggunakan PyTorch — cara yang umum dipakai):

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def latih_pytorch(config, device='cpu'):
    # Model (lihat kode PyTorch di Bagian 7)
    model = DecoderTransformer(**{k: config[k] for k in
                                   ['vocab_size','d_model','num_heads',
                                    'num_layers','d_ffn','max_seq_len']})
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Data — asumsikan sudah diload sebagai tensor
    data = torch.load("train.pt")   # [total_tokens]
    T    = config['seq_len']
    xs   = torch.stack([data[i:i+T]   for i in range(0, len(data)-T-1, T)])
    ys   = torch.stack([data[i+1:i+T+1] for i in range(0, len(data)-T-1, T)])
    loader = DataLoader(TensorDataset(xs, ys),
                        batch_size=config['batch_size'], shuffle=True)

    for step, (input_ids, target_ids) in enumerate(loader):
        if step >= config['max_steps']:
            break
        input_ids  = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)              # [B, T, vocab_size]
        B, T, V = logits.shape
        loss   = F.cross_entropy(logits.view(B*T, V), target_ids.view(B*T))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step:6d} | Loss: {loss.item():.4f}")

        if step % 5000 == 0 and step > 0:
            torch.save({'step': step, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()}, f"ckpt-{step}.pt")

config = {
    'vocab_size': 8000, 'd_model': 128, 'num_heads': 4,
    'num_layers': 4, 'd_ffn': 512, 'max_seq_len': 128,
    'batch_size': 16, 'max_steps': 50000
}
latih_pytorch(config, device='cuda' if torch.cuda.is_available() else 'cpu')
```

---

## Inisialisasi Bobot

Inisialisasi yang salah bisa membuat model tidak bisa belajar sama sekali:

```
Linear layer W  : normal(mean=0, std=sqrt(1/input_dim))   ← Xavier
Linear layer b  : nol
Embedding tabel : normal(mean=0, std=0.02)
LayerNorm gamma : satu
LayerNorm beta  : nol
```

---

## Membaca Loss saat Training

```
Step 0     | Loss ≈ 8.97   ← acak total (log(8000) ≈ 8.99)
Step 1000  | Loss ≈ 6.5    ← mulai belajar token umum
Step 10000 | Loss ≈ 4.5    ← pola sintaks dasar mulai terbentuk
Step 50000 | Loss ≈ 3.0    ← teks makin koheren di domain sempit
```

Loss yang bagus untuk model kecil di domain sempit: **sekitar 2.5–3.5**.  
Jangan harap loss bisa terus turun kalau model dan datanya tidak cukup besar.

---

## Tips Debugging

**Loss tidak turun dari langkah pertama:**
- Learning rate terlalu besar atau terlalu kecil — coba 3e-4 dulu
- Ada bug di backward pass — cek gradient dengan numerical gradient check
- Causal mask salah implementasi — kalau mask terbalik, loss akan sangat rendah palsu

**Loss turun lalu stuck di nilai tinggi:**
- Data terlalu sedikit atau tidak beragam
- Model terlalu kecil untuk menangkap pola data
- Coba naikkan `d_model` atau `num_layers`

**Model menghasilkan token berulang:**
- Naikkan `repetition_penalty` (coba 1.5 atau 2.0)
- Turunkan temperature

**Loss meledak menjadi NaN:**
- Learning rate terlalu besar — turunkan 10×
- Tambahkan gradient clipping: potong semua gradient yang `|nilai| > 1.0`

---

## Checklist Implementasi

```
Tokenizer:
  [ ] BPE training dari data
  [ ] Encode teks → token IDs
  [ ] Decode token IDs → teks
  [ ] Token spesial (<|user|>, <|assistant|>, <|end|>)

Layers:
  [ ] LinearLayer (forward + backward)
  [ ] LayerNorm (forward + backward)
  [ ] EmbeddingLayer (forward + backward)
  [ ] Causal mask
  [ ] MultiHeadAttention (forward + backward)
  [ ] FeedForward + GELU (forward + backward)

Model:
  [ ] TransformerBlock (forward + backward)
  [ ] DecoderTransformer (forward + backward)
  [ ] Kumpulkan semua parameter

Training:
  [ ] DataLoader (baca data, buat batch, geser satu posisi)
  [ ] Cross-entropy loss
  [ ] Adam optimizer
  [ ] Loop training + logging
  [ ] Checkpoint save + load

Inference:
  [ ] generate() dengan temperature sampling
  [ ] Repetition penalty
  [ ] Chat loop dengan format token spesial
```

---

## Langkah Berikutnya Setelah Memahami Ini

Setelah kamu berhasil membangun dan menjalankan model kecil ini dan memahami cara kerjanya, langkah logis berikutnya adalah:

### 1. Eksplorasi Model yang Sudah Jadi
Coba jalankan model kecil yang sudah pretrained:
- **GPT-2** via Hugging Face Transformers (Python)
- **SmolLM2** — model kecil tapi berguna dari HuggingFace
- **Ollama** — jalankan model seperti Llama 3.2 (1B/3B) di laptop

### 2. Fine-tuning
Ambil model pretrained, latih ulang hanya layer terakhir atau dengan teknik **LoRA** (Low-Rank Adaptation) menggunakan data QA Indonesia kecil (~10.000 pasang). Ini adalah jalur paling realistis untuk model QA Indonesia yang fungsional.

### 3. Baca Paper Kunci
- *TinyStories* (Eldan & Li, 2023) — batas bawah model yang bisa menghasilkan teks koheren
- *Chinchilla* (Hoffmann et al., 2022) — scaling laws: berapa data optimal per parameter
- *Phi-1* (Gunasekar et al., 2023) — data kualitas tinggi lebih efisien dari data banyak tapi rendah kualitas
- *Attention is All You Need* (Vaswani et al., 2017) — paper Transformer asli

---

> [← Kembali ke Index Materi](./index.md)
