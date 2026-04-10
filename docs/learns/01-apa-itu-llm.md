# Bagian 1 — Apa Itu Language Model?

[← Sebelumnya](./00-matematika-dasar.md) | [Selanjutnya →](./02-tokenisasi.md)

---

## Permainan Tebak Kata

Bayangkan kamu bermain permainan dengan teman. Aturannya: kamu menyebut kalimat yang belum selesai, temanmu melanjutkan.

> *"Saya mau pergi ke ___"*

Temanmu bisa menjawab "pasar", "sekolah", atau "rumah sakit" — karena dia sudah sering mendengar kalimat serupa. Dari ribuan percakapan yang pernah dia dengar, dia tahu kata apa yang biasanya mengikuti pola tersebut.

**Itulah inti dari Language Model** — sistem yang belajar dari banyak teks, lalu bisa memprediksi *kata apa yang paling mungkin muncul selanjutnya.*

---

## Definisi

**Language Model** adalah model matematika yang mempelajari pola dalam bahasa dan bisa memperkirakan probabilitas suatu token muncul setelah token-token sebelumnya.

Secara formal:

```
P(token berikutnya | token-token sebelumnya)
```

Dibaca: *"berapa peluang token X muncul, jika sebelumnya sudah ada kalimat Y?"*

**Large Language Model (LLM)** adalah Language Model yang sangat besar — dilatih dengan ratusan miliar kata dan memiliki miliaran parameter.

---

## Cara LLM Menghasilkan Teks

LLM tidak langsung "bicara". Mereka bekerja satu langkah kecil setiap kali:

```
1. Terima teks input
2. Ubah ke angka (tokenisasi)
3. Proses angka melalui banyak lapisan matematika
4. Hasilkan satu token berikutnya
5. Tambahkan token itu ke input
6. Ulangi dari langkah 3 — sampai kalimat selesai
```

Proses menghasilkan satu token per langkah ini disebut **autoregressive generation**.

---

## Analogi Autocomplete yang Sangat Canggih

Fitur autocomplete di keyboard HP menyarankan kata berikutnya. Ketik "Selamat" → HP menyarankan "pagi", "malam", "ulang tahun".

LLM adalah versi yang jauh lebih canggih:

| | Autocomplete HP | LLM |
|--|--|--|
| Konteks yang dilihat | 2-3 kata terakhir | Ribuan kata ke belakang |
| Pengetahuan | Kata-kata umum | Pola bahasa + pengetahuan faktual |
| Output | Satu kata | Paragraf panjang |

---

## Apa yang Dipelajari Model?

Ketika LLM dilatih, tidak ada yang memberitahu aturan bahasa secara eksplisit. Tidak ada yang bilang "kata benda biasanya diikuti kata kerja". Model harus *menemukan sendiri* semua pola itu dari data.

Yang dipelajari adalah jutaan **parameter** — angka-angka dalam matriks yang mewakili "pengetahuan" model. Sama seperti neuron di otak yang membentuk koneksi melalui pengalaman, parameter model terbentuk melalui proses training.

Dari data, model secara implisit belajar:
- Tata bahasa (tanpa pernah diajarkan)
- Fakta umum: ibukota negara, tokoh sejarah, definisi ilmiah
- Pola percakapan: cara menjawab pertanyaan, salam-balasan
- Logika sederhana: hubungan sebab-akibat

---

## Jenis-Jenis Language Model

Ada tiga arsitektur utama:

### Decoder-Only ← yang kita bangun
- Contoh populer: GPT-2, GPT-3, LLaMA, Mistral
- Cara kerja: membaca dari kiri ke kanan, prediksi token berikutnya
- Cocok untuk: chatbot, teks generation, tanya-jawab
- Analogi: penulis yang menulis kata demi kata dari kiri ke kanan

### Encoder-Only
- Contoh: BERT, RoBERTa
- Cara kerja: "membaca" seluruh kalimat sekaligus untuk memahami makna
- Cocok untuk: klasifikasi teks, sentiment analysis, pencarian
- Analogi: pembaca yang memahami makna keseluruhan

### Encoder-Decoder
- Contoh: T5, BART, model terjemahan
- Cara kerja: encode input → decode ke output baru
- Cocok untuk: terjemahan, ringkasan, parafrase
- Analogi: penerjemah — baca dulu, lalu tulis ulang dalam bahasa lain

---

## Model yang Akan Kita Bangun

Kita akan membangun model **Decoder-Only** kecil — arsitektur yang **persis sama** dengan GPT. Yang berbeda hanya skalanya.

### Jujur tentang Tujuan

Tujuan utama kita adalah **memahami cara kerja Transformer dari dalam**, bukan membangun produk. Ini penting untuk disampaikan dari awal.

Model kecil yang kita bangun (~3 juta parameter) akan:
- ✅ Menunjukkan cara kerja setiap komponen dengan benar
- ✅ Bisa dilatih dan loss-nya turun secara nyata
- ✅ Menghasilkan teks yang koheren dalam domain sempit
- ❌ **Tidak akan** menjawab *"Ibukota Indonesia?"* → *"Jakarta"* secara andal

Kenapa tidak bisa? Karena menjawab pertanyaan faktual secara andal butuh:
- Ratusan juta parameter minimum (model terkecil yang berguna saat ini: ~135 juta parameter dilatih 2 triliun token)
- Data latih miliaran token, bukan jutaan
- Fase fine-tuning khusus untuk mengikuti instruksi

Analoginya: kamu bisa memahami cara kerja mesin pesawat dengan mempelajari model miniaturnya. Tapi model miniatur itu tidak akan bisa terbang membawa penumpang. Nilai belajarnya tetap nyata, tapi fungsinya berbeda.

Pemahaman tentang batas ini justru adalah bagian dari *literasi LLM yang sesungguhnya* — banyak orang salah paham tentang ini.

---

## Komponen yang Akan Kita Pelajari

```
Input Teks
    ↓ [Bagian 2]
Tokenizer — memotong teks jadi token
    ↓ [Bagian 3]
Embedding — token jadi vektor angka
    ↓ [Bagian 4-6]
Transformer Blocks × N
    ├── Attention  — memahami konteks
    └── Feed Forward — memproses informasi
    ↓ [Bagian 7]
Output Projection — vektor jadi prediksi kata
    ↓ [Bagian 11]
Sampling — pilih token berikutnya
```

---

## Ringkasan

| Konsep | Penjelasan |
|--------|-----------|
| Language Model | Sistem yang memprediksi token berikutnya |
| Parameter | Angka-angka yang dipelajari saat training |
| Autoregressive | Menghasilkan output satu token per langkah |
| Decoder-Only | Arsitektur yang kita gunakan (seperti GPT) |

---

> [Selanjutnya: Tokenisasi — Cara Komputer Membaca Teks →](./02-tokenisasi.md)
