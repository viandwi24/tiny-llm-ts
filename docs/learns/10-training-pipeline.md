# Bagian 10 — Pipeline Training Lengkap

[← Sebelumnya](./09-optimizer.md) | [Selanjutnya →](./11-inference.md)

---

## Gambaran Pipeline

Dari teks mentah hingga model yang bisa menjawab pertanyaan, ada empat fase:

```
FASE 1 — KUMPULKAN DATA
  Teks dari berbagai sumber

FASE 2 — SIAPKAN DATA
  Tokenisasi + Encoding

FASE 3 — TRAINING
  Loop belajar ribuan kali

FASE 4 — EVALUASI
  Uji model dengan percakapan nyata
```

---

## Fase 1: Mengumpulkan Data

### Jenis Data yang Diperlukan

Untuk model yang bisa menjawab pertanyaan pengetahuan dan percakapan sederhana, kita butuh dua jenis data:

**1. Data Pengetahuan (Pretraining Data)**
Teks deskriptif yang mengandung fakta — ensiklopedia, artikel, buku pelajaran.

Contoh:
```
Jakarta adalah ibu kota Republik Indonesia. Kota ini terletak di
bagian barat laut Pulau Jawa. Jakarta merupakan kota terbesar di
Indonesia dengan populasi sekitar 10 juta jiwa di area kota dan
30 juta di wilayah metropolitan...
```

**2. Data Percakapan (Instruction/Fine-tuning Data)**
Pasangan pertanyaan-jawaban dan percakapan yang terstruktur.

Contoh format:
```
<|user|> Apa ibukota Indonesia? <|end|>
<|assistant|> Ibukota Indonesia adalah Jakarta. <|end|>

<|user|> Halo <|end|>
<|assistant|> Halo! Ada yang bisa saya bantu? <|end|>
```

### Berapa Banyak Data yang Diperlukan?

Ini bergantung pada *apa yang ingin dicapai* — dan jawabannya sangat berbeda:

**Untuk belajar arsitektur (model ~3 juta parameter):**

| Tujuan | Data yang Dibutuhkan |
|--------|---------------------|
| Loss turun, model belajar pola token | 1–10 juta token |
| Teks koheren dalam domain sempit | 10–50 juta token |
| Menjawab fakta sederhana secara andal | ❌ Tidak bisa di skala ini |

**Untuk model yang benar-benar fungsional (fine-tuning dari model besar):**

| Tujuan | Data yang Dibutuhkan |
|--------|---------------------|
| Fine-tuning QA Indonesia | ~10.000–100.000 pasang Q&A |
| Model dasar yang perlu dipretrain | 1–10 miliar token teks bersih |

Sebagai referensi: GPT-2 Small (117 juta parameter) dilatih pada ~8 miliar token dan masih tidak bisa menjawab pertanyaan faktual secara andal — karena ia adalah *text completer*, bukan *question answerer*.

### Membersihkan Data

Sebelum digunakan, teks perlu dibersihkan:

```
FUNGSI bersihkan_teks(teks_mentah):
  hapus_HTML_dan_markup(teks)
  hapus_karakter_tidak_relevan(teks)
  perbaiki_encoding(teks)         # pastikan UTF-8
  hapus_baris_terlalu_pendek(teks) # kurang dari 10 karakter
  normalisasi_spasi_dan_baris_baru(teks)
  KEMBALIKAN teks_bersih
```

---

## Fase 2: Menyiapkan Data

### 2a. Latih Tokenizer

Dari seluruh data teks, jalankan BPE training (lihat Bagian 2). Hasilnya:
- `vocab.json` — kamus token → ID
- `merges.json` — aturan penggabungan

### 2b. Encode Semua Data ke Token IDs

```
FUNGSI encode_semua_data(semua_file_teks, vocab, merges):
  semua_ids = []

  UNTUK setiap file DALAM semua_file_teks:
    teks = baca(file)
    ids  = tokenize(teks, vocab, merges)
    semua_ids.extend(ids)

  # Simpan sebagai array binary yang efisien
  simpan_binary("train.bin", semua_ids)

  tampilkan(f"Total: {panjang(semua_ids)} token")
```

Mengapa binary? Lebih cepat dibaca dan jauh lebih hemat storage dibanding JSON atau CSV.

---

## Fase 3: Training Loop

### Konsep: Next-Token Prediction

Cara model belajar sangat elegan: dari satu data, hasilkan banyak pasang (input → target).

```
Data asli: [tok_A, tok_B, tok_C, tok_D, tok_E]

Pasang yang dihasilkan:
  Input: [tok_A]                Target: tok_B
  Input: [tok_A, tok_B]         Target: tok_C
  Input: [tok_A, tok_B, tok_C]  Target: tok_D
  ...dan seterusnya
```

Dalam praktiknya, ini dilakukan dengan "menggeser satu posisi":
```
Input:  [tok_A, tok_B, tok_C, tok_D]
Target: [tok_B, tok_C, tok_D, tok_E]
         ↑ digeser satu ke kanan
```

Satu sequence panjang N menghasilkan N pasang (input, target).

### DataLoader: Ambil Batch

```
KELAS DataLoader:
  data = baca_binary("train.bin")   # array panjang berisi semua token IDs
  posisi = 0

  FUNGSI next_batch(batch_size, seq_len):
    inputs  = []
    targets = []

    UNTUK i DARI 0 sampai batch_size:
      chunk  = data[posisi : posisi + seq_len + 1]   # ambil seq_len+1 token
      inputs.tambah(chunk[0 : seq_len])              # token 0..N-1
      targets.tambah(chunk[1 : seq_len + 1])         # token 1..N

      posisi += seq_len

      # Wrap jika sudah sampai akhir data
      JIKA posisi + seq_len + 1 >= panjang(data):
        posisi = 0

    KEMBALIKAN inputs, targets   # shape: [batch_size, seq_len]
```

### Loop Training Lengkap

```
FUNGSI training_loop(model, config):
  optimizer  = Adam(lr=3e-4)
  dataloader = DataLoader("train.bin")

  # Load checkpoint jika ada (untuk resume)
  step_awal = 0
  JIKA checkpoint_ada():
    muat_bobot(model, checkpoint)
    step_awal = checkpoint["step"] + 1
    tampilkan(f"Lanjut dari step {step_awal}")

  UNTUK step DARI step_awal sampai max_steps:

    # ── Ambil batch ──
    input_ids, target_ids = dataloader.next_batch(batch_size, seq_len)

    # ── Forward pass ──
    logits = model.forward(input_ids)   [seq_len, vocab_size]

    # ── Hitung loss ──
    loss = cross_entropy(logits, target_ids)

    # ── Backward pass ──
    grads = hitung_gradient(loss, logits, target_ids)
    model.backward(grads)

    # ── Update parameter ──
    optimizer.langkah(model.semua_parameter())

    # ── Logging ──
    JIKA step % 100 == 0:
      tampilkan(f"Step {step:6d} | Loss: {loss:.4f}")

    # ── Simpan checkpoint ──
    JIKA step % 5000 == 0:
      simpan_checkpoint(model, step, loss)
```

**Python Native** (murni Python, tanpa library):

```python
import json
import os

class DataLoader:
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)   # list of int (token IDs)
        self.posisi = 0

    def next_batch(self, batch_size, seq_len):
        inputs, targets = [], []
        for _ in range(batch_size):
            chunk = self.data[self.posisi : self.posisi + seq_len + 1]
            inputs.append(chunk[:seq_len])
            targets.append(chunk[1:seq_len + 1])
            self.posisi += seq_len
            if self.posisi + seq_len + 1 >= len(self.data):
                self.posisi = 0
        return inputs, targets

def training_loop(model, optimizer, config):
    loader   = DataLoader("train.json")
    step_awal = 0
    checkpoint_path = "checkpoint-latest.json"

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        model.muat_bobot(ckpt['bobot'])
        optimizer.t = ckpt['optimizer_step']
        step_awal   = ckpt['step'] + 1
        print(f"Lanjut dari step {step_awal}")

    for step in range(step_awal, config['max_steps']):
        input_ids, target_ids = loader.next_batch(config['batch_size'], config['seq_len'])
        logits = model.forward(input_ids)
        loss   = cross_entropy(logits, target_ids)
        model.backward(logits, target_ids)
        optimizer.langkah(model.semua_parameter())

        if step % 100 == 0:
            print(f"Step {step:6d} | Loss: {loss:.4f}")

        if step % 5000 == 0:
            ckpt = {'step': step, 'loss': loss,
                    'bobot': model.ambil_bobot(),
                    'optimizer_step': optimizer.t}
            with open(f"checkpoint-step{step}.json", 'w') as f:
                json.dump(ckpt, f)
            with open("checkpoint-latest.json", 'w') as f:
                json.dump(ckpt, f)
```

**Python + Library** (menggunakan PyTorch):

```python
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class TokenDataset(Dataset):
    def __init__(self, path, seq_len):
        data       = torch.load(path)   # tensor berisi semua token IDs
        self.data  = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk   = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]   # (input, target)

def training_loop(model, config, device='cpu'):
    dataset    = TokenDataset("train.pt", config['seq_len'])
    loader     = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    optimizer  = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.to(device)
    model.train()

    for step, (input_ids, target_ids) in enumerate(loader):
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

        if step % 5000 == 0:
            torch.save({'step': step, 'loss': loss.item(),
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       f"checkpoint-step{step}.pt")
```

---

## Checkpoint: Menyimpan Progress

Training bisa memakan waktu berjam-jam. Checkpoint memungkinkan kita melanjutkan jika terjadi gangguan.

```
FUNGSI simpan_checkpoint(model, step, loss):
  data = {
    "step": step,
    "loss": loss,
    "bobot_model": {
      "embedding_token": model.embedding.tabel_token,
      "embedding_posisi": model.embedding.tabel_posisi,
      "blocks": [
        {
          "attn_wq": ..., "attn_wk": ..., "attn_wv": ..., "attn_wo": ...,
          "ffn_w1": ..., "ffn_w2": ...,
          "norm1_gamma": ..., "norm1_beta": ...,
          "norm2_gamma": ..., "norm2_beta": ...
        }
        × num_layers
      ],
      "projection_w": ...,
      "final_norm_gamma": ..., "final_norm_beta": ...
    }
  }
  simpan_json(f"checkpoint-step{step}.json", data)
  simpan_json("checkpoint-latest.json", data)   # selalu update "latest"
```

---

## Memahami Loss saat Training

Loss adalah sinyal utama untuk menilai progress:

```
Step 0     | Loss: 8.97  ← awal, model menebak acak
                            log(8000) ≈ 8.99 (ekspektasi acak)
Step 500   | Loss: 6.20  ← mulai belajar token umum
Step 2000  | Loss: 4.80  ← pola frekuensi dasar terpelajari
Step 10000 | Loss: 3.50  ← mulai menangkap sintaks
Step 50000 | Loss: 2.80  ← pola semantik mulai muncul
```

**Perplexity** adalah cara lain mengekspresikan loss:
```
perplexity = exp(loss)

Loss 8.97 → perplexity ≈ 7900  (sama seperti menebak acak dari 7900 pilihan)
Loss 3.50 → perplexity ≈ 33    (rata-rata bingung antara 33 pilihan)
Loss 2.00 → perplexity ≈ 7     (rata-rata bingung antara 7 pilihan)
```

Makin kecil perplexity, makin baik model.

---

## Batch Size dan Pengaruhnya

**Batch size** = berapa banyak sequence yang diproses sebelum satu update parameter.

```
Batch kecil (4–16):
  + Hemat memori
  + Update lebih sering
  − Gradient "noisy" (estimasi dari sedikit contoh)
  − Training kurang stabil

Batch besar (128–512):
  + Gradient lebih akurat
  + Training lebih stabil
  − Butuh banyak memori
  − Perlu learning rate lebih besar untuk kompensasi
```

Untuk hardware terbatas (laptop/PC biasa), **batch size 8–16 sudah cukup**.

---

## Dua Fase Training: Pretraining + Fine-tuning

Model production dilatih dua tahap yang sangat berbeda:

### Tahap 1: Pretraining
- Data: teks umum dalam jumlah **sangat besar** — miliar hingga triliun token
- Tujuan: model belajar bahasa, tata bahasa, dan pengetahuan umum
- Skala: butuh GPU banyak, training berminggu-minggu hingga berbulan-bulan
- Contoh: GPT-2 Small dilatih pada ~8 miliar token teks web selama berminggu-minggu di GPU cluster

### Tahap 2: Fine-tuning / Instruction Tuning
- Data: puluhan ribu hingga jutaan pasang instruksi-respons yang dikurasi
- Tujuan: model belajar *mengikuti instruksi* — dari text completer menjadi asisten
- Skala: jauh lebih kecil dari pretraining, tapi hasilnya sangat berpengaruh
- Ini yang membuat perbedaan antara "GPT-2 mentah" dan "ChatGPT"

### Implikasi untuk Proyek Belajar

Untuk model kecil yang kita bangun, penting dipahami:

- Pretraining dari nol dengan data kecil (< 100 juta token) hanya cukup untuk **mempelajari pola bahasa dasar**
- Tanpa fine-tuning instruksi, model hanya bisa **melengkapi teks**, bukan menjawab pertanyaan
- Kalau ingin model yang benar-benar bisa bercakap-cakap: **fine-tune model yang sudah pretrained**, jangan mulai dari nol

Lihat Bagian 12 untuk penjelasan lengkap tentang jalur realistis.

---

## Ringkasan Pipeline

```
Teks mentah
    ↓  bersihkan
Teks bersih
    ↓  latih tokenizer (BPE)
vocab.json + merges.json
    ↓  encode semua teks
train.bin (array token IDs)
    ↓
DataLoader → batches [batch_size, seq_len]
    ↓
LOOP:
  forward → loss → backward → optimizer.step
    ↓  setiap N langkah
checkpoint tersimpan
    ↓  selesai
Model siap untuk inference!
```

---

> [Selanjutnya: Inference — Cara Model Menghasilkan Jawaban →](./11-inference.md)
