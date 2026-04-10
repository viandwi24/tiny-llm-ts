# Belajar Membuat LLM dari Nol — Panduan Lengkap

> **Untuk siapa panduan ini?**
> Untuk siapa saja yang ingin tahu bagaimana AI seperti ChatGPT bekerja *di dalam*. Tidak perlu latar belakang matematika tinggi. Yang dibutuhkan hanya rasa ingin tahu.

---

## Tujuan Akhir Materi Ini

Setelah membaca semua bagian, kamu akan **benar-benar memahami cara kerja Transformer dari dalam** — cukup untuk membangun sendiri setiap komponennya dari nol.

### Jujur tentang Ekspektasi

Materi ini mengajarkan cara membangun model kecil (~3 juta parameter) yang **identik secara arsitektur** dengan GPT. Ini bukan klaim kecil — arsitekturnya benar-benar sama.

Tapi ada satu hal yang perlu diluruskan sejak awal:

> **Model kecil yang kita bangun tidak akan bisa menjawab "Ibukota Indonesia?" secara andal.**

Menjawab pertanyaan faktual dengan andal butuh ratusan juta parameter dan data miliaran token. Model terkecil yang bisa melakukan ini (SmolLM2-135M dari HuggingFace) dilatih dengan **2 triliun token** — 400.000× lebih banyak dari yang realistis untuk proyek belajar.

**Yang akan kamu capai:**
- ✅ Memahami setiap komponen Transformer secara mendalam
- ✅ Melihat loss turun secara nyata saat training
- ✅ Model menghasilkan teks yang sintaksisnya masuk akal
- ✅ Fondasi untuk melangkah ke fine-tuning model yang sudah ada
- ❌ Model produksi yang bisa menjawab pertanyaan umum

Pemahaman tentang batas ini adalah bagian dari literasi LLM yang sesungguhnya.

---

## Daftar Materi

Ikuti urutan ini agar pemahamanmu runtut dari dasar:

### Fondasi
| # | Topik | Deskripsi |
|---|-------|-----------|
| 0 | [Matematika & Konsep Dasar](./00-matematika-dasar.md) | Vektor, matriks, softmax — konsep yang akan sering muncul |

### Bagian 1 — Memahami LLM
| # | Topik | Deskripsi |
|---|-------|-----------|
| 1 | [Apa Itu Language Model?](./01-apa-itu-llm.md) | Cara kerja LLM secara umum, analogi, dan tujuan |
| 2 | [Tokenisasi — Cara Komputer Membaca Teks](./02-tokenisasi.md) | Memotong teks jadi potongan yang bisa diproses |
| 3 | [Embedding — Representasi Kata sebagai Angka](./03-embedding.md) | Mengubah token jadi vektor yang mengandung makna |

### Bagian 2 — Arsitektur Model
| # | Topik | Deskripsi |
|---|-------|-----------|
| 4 | [Attention — Cara Model Memahami Konteks](./04-attention.md) | Mekanisme paling penting dalam Transformer |
| 5 | [Feed Forward Network](./05-feedforward.md) | Lapisan pemrosesan informasi |
| 6 | [Transformer Block](./06-transformer-block.md) | Menyatukan attention + FFN + normalisasi |
| 7 | [Arsitektur Lengkap Model](./07-arsitektur-lengkap.md) | Gambaran penuh dari input sampai output |

### Bagian 3 — Training
| # | Topik | Deskripsi |
|---|-------|-----------|
| 8 | [Backpropagation — Cara Model Belajar](./08-backpropagation.md) | Belajar dari kesalahan dengan matematika |
| 9 | [Optimizer — Adam](./09-optimizer.md) | Strategi cerdas memperbarui parameter |
| 10 | [Pipeline Training Lengkap](./10-training-pipeline.md) | Dari data mentah sampai model terlatih |

### Bagian 4 — Inference
| # | Topik | Deskripsi |
|---|-------|-----------|
| 11 | [Inference — Cara Model Menghasilkan Jawaban](./11-inference.md) | Temperature, sampling, autoregressive generation |

### Bagian 5 — Implementasi & Skala Nyata
| # | Topik | Deskripsi |
|---|-------|-----------|
| 12 | [Panduan Implementasi dari Nol](./12-implementasi.md) | Cetak biru implementasi + ekspektasi realistis |
| 13 | [Berapa Skala yang Dibutuhkan?](./13-berapa-skala-yang-dibutuhkan.md) | Parameter, data, dan konfigurasi nyata untuk goals faktual QA dan chat |

---

## Gambaran Besar

```
PHASE 1: PERSIAPAN DATA
  Teks (Wikipedia / buku / percakapan)
      ↓  [Bagian 2]
  Tokenisasi → potongan kata (token)
      ↓
  Encode → angka-angka
      ↓
  Disimpan sebagai data latih

PHASE 2: TRAINING
  Baca data → buat batch
      ↓  [Bagian 3–7]
  Forward pass melalui model
      ↓  [Bagian 8–9]
  Hitung kesalahan → perbaiki parameter
      ↓  [Bagian 10]
  Ulangi ratusan ribu kali

PHASE 3: INFERENCE
  Input teks dari user
      ↓  [Bagian 11]
  Model hasilkan token satu per satu
      ↓
  Token → teks → tampilkan ke user
```

---

> Mulai dari [Bagian 0 — Matematika Dasar](./00-matematika-dasar.md)
