# Bagian 13 — Berapa Skala yang Dibutuhkan untuk Goals Faktual QA dan Chat?

[← Kembali ke Index](./index.md)

---

## Pertanyaan yang Tepat

Kita mau tahu: **dengan arsitektur Transformer Decoder-Only yang sudah kita pelajari, konfigurasi seperti apa yang diperlukan agar model benar-benar bisa:**

1. Menjawab pertanyaan faktual: *"Ibukota Indonesia?"* → *"Jakarta"*
2. Merespons percakapan: *"Halo"* → *"Halo juga!"*

Dan jawabannya adalah: **ya, bisa dicapai dengan arsitektur yang sama** — hanya skalanya berbeda dari model latihan kita.

---

## Bukti: Model Produksi Pakai Arsitektur yang Sama

Semua model sub-1B yang bisa melakukan kedua hal di atas menggunakan struktur yang **identik** dengan yang sudah kita pelajari:

```
Token Embedding
    ↓
N × Transformer Block:
    Pre-Norm → Attention → Residual
    Pre-Norm → Feed Forward → Residual
    ↓
Final Norm → Output Projection
```

Perbedaannya bukan di struktur — tapi di dua hal:

1. **Ukuran** (lebih dalam, lebih lebar)
2. **"Upgrade" teknis modern** yang meningkatkan efisiensi (dijelaskan di akhir bagian ini)

---

## Konfigurasi Nyata Model yang Sudah Terbukti

Berikut konfigurasi persis dari model-model produksi yang bisa faktual QA dan chat, diambil langsung dari `config.json` mereka di HuggingFace:

### SmolLM2-135M — bisa chat dasar, QA masih lemah

| Parameter | Nilai |
|-----------|-------|
| `num_layers` | 30 |
| `d_model` (hidden_size) | 576 |
| `num_heads` | 9 |
| `d_ffn` (intermediate_size) | 1.536 |
| `max_seq_len` | 8.192 |
| `vocab_size` | 49.152 |
| **Total parameter** | **~135 juta** |
| Data latih | 2 triliun token |
| Skor MMLU | 29.3% (hampir acak, baseline 25%) |
| Skor IFEval | 29.9% (instruction following) |

**Kesimpulan:** Cukup untuk percakapan sangat sederhana, tidak andal untuk faktual QA.

---

### SmolLM2-360M — bisa chat, QA mulai muncul tapi belum andal

| Parameter | Nilai |
|-----------|-------|
| `num_layers` | 32 |
| `d_model` | 960 |
| `num_heads` | 15 |
| `d_ffn` | 2.560 |
| `max_seq_len` | 8.192 |
| `vocab_size` | 49.152 |
| **Total parameter** | **~360 juta** |
| Data latih | 4 triliun token |
| Skor MMLU | 32.8% |
| Skor IFEval | 41.0% |

**Kesimpulan:** Chat mulai konsisten, faktual QA masih tidak andal.

---

### Qwen2.5-0.5B — bisa faktual QA dan chat dengan andal

| Parameter | Nilai |
|-----------|-------|
| `num_layers` | 24 |
| `d_model` | 896 |
| `num_heads` | 14 |
| `d_ffn` | 4.864 |
| `max_seq_len` | 32.768 |
| `vocab_size` | 151.936 |
| **Total parameter** | **~490 juta** |
| Data latih | 18 triliun token |
| Skor MMLU | 47.5% |
| Skor IFEval | — |

**Kesimpulan:** Ini titik minimum yang **terbukti bisa** untuk kedua goals. Mendukung 29+ bahasa termasuk Indonesia.

---

### Qwen2.5-1.5B — andal dan konsisten

| Parameter | Nilai |
|-----------|-------|
| `num_layers` | 28 |
| `d_model` | 1.536 |
| `num_heads` | 12 |
| `d_ffn` | 8.960 |
| `max_seq_len` | 131.072 |
| `vocab_size` | 151.936 |
| **Total parameter** | **~1,5 miliar** |
| Data latih | 18 triliun token |
| Skor MMLU | 60.9% |

**Kesimpulan:** Target ideal — andal untuk QA dan chat, masih bisa dijalankan di laptop.

---

## Pola yang Terlihat

Dari data di atas, ada dua faktor yang menentukan:

### Faktor 1: Jumlah Parameter

Lebih banyak parameter = lebih banyak "ruang" untuk menyimpan pengetahuan dan pola bahasa.

```
~135 juta → chat dasar saja
~360 juta → chat lumayan, QA mulai tapi tidak andal
~490 juta → chat + QA mulai andal (titik minimum yang berguna)
~1,5 miliar → chat + QA konsisten
```

### Faktor 2: Jumlah Data Latih (Ini yang Sering Diremehkan)

Perhatikan perbedaan mencolok ini:

```
SmolLM2-360M:  dilatih 4 triliun token  → MMLU 32.8%
Qwen2.5-0.5B:  dilatih 18 triliun token → MMLU 47.5%

Qwen2.5-0.5B lebih kecil dari SmolLM2-360M dalam jumlah parameter,
tapi jauh lebih baik karena dilatih dengan data 4.5× lebih banyak.
```

Data lebih banyak = model belajar lebih banyak fakta dan pola. **Data adalah bahan bakar, parameter adalah mesinnya.**

---

## Konfigurasi "Classical" untuk Arsitektur yang Kita Pelajari

Model produksi di atas menggunakan beberapa "upgrade teknis modern" (RoPE, GQA, RMSNorm, SwiGLU) yang sedikit berbeda dari versi klasik yang kita pelajari. Tapi arsitektur dasarnya **sama persis** — perbedaan itu hanya optimasi efisiensi, bukan perubahan fundamental.

Berikut adalah konfigurasi dalam arsitektur **klasik** (seperti yang kita pelajari: standard MHA, learned positional embedding, LayerNorm, GELU) yang menghasilkan jumlah parameter setara:

### Target ~360 juta parameter (setara SmolLM2-360M)

| Hyperparameter | Nilai |
|----------------|-------|
| `num_layers` | **24** |
| `d_model` | **1.024** |
| `num_heads` | **16** |
| `head_dim` | 64 (d_model / num_heads) |
| `d_ffn` | **4.096** (4× d_model) |
| `vocab_size` | ~50.000 |
| `max_seq_len` | 1.024 |
| **Total parameter** | **~355 juta** |

### Target ~490 juta parameter (setara Qwen2.5-0.5B)

| Hyperparameter | Nilai |
|----------------|-------|
| `num_layers` | **24** |
| `d_model` | **1.216** |
| `num_heads` | **16** |
| `head_dim` | 76 |
| `d_ffn` | **4.864** (4× d_model) |
| `vocab_size` | ~50.000 |
| `max_seq_len` | 1.024 |
| **Total parameter** | **~489 juta** |

---

## Perbandingan: Model Kecil (Latihan) vs Model yang Bisa Dipakai

| | Model Latihan | Target ~360M | Target ~490M |
|---|---|---|---|
| `num_layers` | 4 | 24 | 24 |
| `d_model` | 128 | 1.024 | 1.216 |
| `num_heads` | 4 | 16 | 16 |
| `d_ffn` | 512 | 4.096 | 4.864 |
| Parameter | ~3 juta | ~355 juta | ~489 juta |
| Bisa faktual QA? | ❌ | ⚠️ Mulai bisa | ✅ Ya |
| Bisa chat? | ❌ | ✅ Ya | ✅ Ya |

Perbedaan antara model latihan dan model yang bisa dipakai: **layer lebih banyak dan tiap layer lebih lebar**. Struktur blok-nya sama persis.

---

## Jadi Kenapa Model Latihan Kita Sangat Kecil?

Karena tujuannya berbeda: **memahami mekanisme, bukan menghasilkan produk.**

Model 3 juta parameter bisa di-training dalam beberapa jam di laptop biasa dan loss-nya sudah terlihat turun. Kalau kita langsung mulai dengan 360 juta parameter, training satu epoch saja butuh berminggu-minggu di CPU.

Analoginya: belajar cara kerja mesin mobil menggunakan model miniatur. Kamu tidak butuh mesin V8 sungguhan untuk memahami prinsip pembakaran. Tapi setelah kamu mengerti prinsipnya, kamu bisa naik ke mesin yang lebih besar.

---

## "Upgrade Teknis" yang Dipakai Model Modern

Semua model produksi di atas menggunakan empat upgrade dari versi klasik. Ini bukan mengubah arsitektur — hanya membuat yang sama jadi lebih efisien:

### 1. RoPE — Rotary Position Embedding

**Klasik:** Tabel posisi yang dipelajari, ditambahkan ke embedding.
**RoPE:** Informasi posisi dimasukkan langsung ke dalam matriks Q dan K saat menghitung attention, tanpa tabel terpisah.

Keunggulan: bisa generalize ke kalimat lebih panjang dari yang pernah dilihat saat training.

### 2. GQA — Grouped Query Attention

**Klasik:** Setiap head punya Q, K, V sendiri. Misal 16 head = 16 set K dan V.
**GQA:** Beberapa head berbagi K dan V yang sama. Misal 16 query heads tapi hanya 2 set KV.

Keunggulan: jauh lebih hemat memori saat inference, kecepatan tidak berubah banyak.

### 3. RMSNorm — Root Mean Square Normalization

**Klasik:** LayerNorm = normalisasi dengan mean dan std, punya parameter gamma dan beta.
**RMSNorm:** Normalisasi hanya dengan RMS (akar rata-rata kuadrat), hanya punya gamma. Tidak ada beta, tidak ada perhitungan mean.

Keunggulan: sedikit lebih cepat, hasil hampir sama.

### 4. SwiGLU — Swish Gated Linear Unit

**Klasik:** FFN = Linear → GELU → Linear (2 matriks).
**SwiGLU:** FFN = (Linear × swish(Linear)) → Linear (3 matriks, tapi yang tengah lebih kecil).

Keunggulan: kapasitas ekspresi lebih tinggi dengan jumlah parameter yang sama.

---

## Ringkasan Akhir

```
GOALS:
  Chat sederhana ("Halo" → "Halo juga!")  →  ~360M parameter minimum
  Faktual QA ("Ibukota Indonesia?")        →  ~490M parameter minimum
  Keduanya dengan andal                    →  ~490M–1.5B parameter

ARSITEKTUR:
  Sama persis dengan yang kita pelajari.
  Perbedaan hanya di ukuran dan 4 upgrade efisiensi (RoPE, GQA, RMSNorm, SwiGLU).

DATA LATIH:
  Ini yang paling sering diabaikan.
  360M parameter dengan 4 triliun token → MMLU 32.8% (belum andal)
  490M parameter dengan 18 triliun token → MMLU 47.5% (mulai andal)
  Data adalah faktor yang sama pentingnya dengan ukuran model.

JALUR PRAKTIS PALING REALISTIS:
  Gunakan model pretrained yang sudah ada (Qwen2.5-0.5B-Instruct)
  Fine-tune dengan data Indonesia ~10.000–50.000 pasang Q&A
  Hasilnya: model yang benar-benar menjawab dalam bahasa Indonesia
```

---

> [← Kembali ke Index Materi](./index.md)
