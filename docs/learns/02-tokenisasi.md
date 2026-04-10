# Bagian 2 — Tokenisasi: Cara Komputer Membaca Teks

[← Sebelumnya](./01-apa-itu-llm.md) | [Selanjutnya →](./03-embedding.md)

---

## Masalah Mendasar: Komputer Hanya Mengerti Angka

Komputer tidak bisa memproses huruf secara langsung. Semua yang ada di layar — teks, gambar, video — pada akhirnya adalah angka di balik layar.

Jadi sebelum teks bisa masuk ke model, teks harus diubah jadi angka dulu. Pertanyaannya: **cara apa yang terbaik?**

---

## Tiga Pendekatan, Tiga Masalah

### Pendekatan 1: Per Huruf

A=1, B=2, C=3, dst. "Halo" → [8, 1, 12, 15].

**Masalah:** Model harus belajar sendiri bahwa H-a-l-o adalah satu kesatuan bermakna. Kalimat pendek pun jadi sangat panjang dalam angka. Tidak efisien.

### Pendekatan 2: Per Kata

Setiap kata punya nomornya sendiri. "makan"=1, "tidur"=2, "berlari"=3, dst.

**Masalah:** Bahasa Indonesia punya ratusan ribu kata. Tambah nama orang, istilah teknis, slang baru — butuh kamus jutaan entri. Dan kata yang belum pernah dilihat saat training? Model tidak bisa menanganinya sama sekali.

### Pendekatan 3: Sub-kata (yang dipakai model modern)

Potong teks menjadi *potongan yang sering muncul bersama* — bisa satu huruf, satu suku kata, satu kata, atau lebih.

Hasilnya disebut **token**.

---

## Analogi: Balok Lego

Kata-kata adalah bangunan Lego. Kita bisa:
- Simpan setiap bangunan utuh → terlalu banyak varian
- Simpan satu bata satu-satu → terlalu kecil, susah membangun makna

**Atau:** Simpan *potongan yang sering muncul bersama* sebagai satu balok.

Di bahasa Indonesia, awalan "ber-", "me-", "ke-" sangat sering muncul. Akhiran "-an", "-kan", "-nya" juga. Kalau kita jadikan ini satu balok masing-masing, kita bisa membangun ribuan kata dengan sedikit balok.

Itulah token. **Token adalah potongan teks yang dipilih karena sering muncul.**

---

## BPE — Algoritma yang Dipakai

**Byte Pair Encoding (BPE)** adalah algoritma yang menemukan potongan-potongan terbaik secara otomatis dari data.

### Ide Dasarnya

Mulai dari huruf-huruf kecil. Lalu terus-menerus gabungkan dua simbol yang paling sering muncul *berdampingan* menjadi satu simbol baru. Ulangi sampai kita punya jumlah token yang kita mau.

### Contoh Langkah demi Langkah

Misalkan kita punya korpus kecil (teks latih). Setelah dihitung frekuensi kata:

```
"bermain"  muncul 200 kali
"berlari"  muncul 150 kali
"berjalan" muncul 120 kali
"berkata"  muncul 100 kali
```

**Iterasi 0: Mulai dari huruf**
```
b-e-r-m-a-i-n
b-e-r-l-a-r-i
b-e-r-j-a-l-a-n
b-e-r-k-a-t-a
```

**Iterasi 1: Cari pasangan paling sering berdampingan**

Pasangan `b` + `e` muncul 200+150+120+100 = 570 kali → gabungkan!

```
be-r-m-a-i-n
be-r-l-a-r-i
be-r-j-a-l-a-n
be-r-k-a-t-a
```

**Iterasi 2: Cari lagi**

Pasangan `be` + `r` muncul 570 kali → gabungkan!

```
ber-m-a-i-n
ber-l-a-r-i
ber-j-a-l-a-n
ber-k-a-t-a
```

Token `ber` lahir! Kini satu balok yang merepresentasikan awalan "ber-".

**Iterasi selanjutnya** terus berjalan sampai kita mencapai ukuran kosakata yang diinginkan (misalnya 8.000 atau 32.000 token).

---

## Pseudocode BPE Training

```
FUNGSI latih_bpe(semua_teks, target_vocab_size):

  # Langkah 1: Mulai dari huruf
  vocab = semua huruf unik dalam teks
  kata_tersegmen = pisahkan tiap kata jadi huruf-huruf

  # Langkah 2: Gabungkan berulang
  SELAMA ukuran(vocab) < target_vocab_size:

    # Hitung frekuensi semua pasangan yang berdampingan
    frekuensi = {}
    UNTUK setiap kata DALAM kata_tersegmen:
      UNTUK setiap pasangan bersebelahan DALAM kata:
        frekuensi[pasangan] += frekuensi_kata

    # Ambil pasangan paling sering
    pasangan_terbaik = pasangan dengan frekuensi tertinggi

    # Gabungkan jadi token baru
    token_baru = gabung(pasangan_terbaik)
    vocab.tambah(token_baru)
    merge_rules.tambah(pasangan_terbaik)

    # Terapkan ke semua kata
    ganti semua kemunculan pasangan_terbaik dengan token_baru

  KEMBALIKAN vocab, merge_rules
```

**Python Native** (murni Python, tanpa library):

```python
from collections import defaultdict

def hitung_frekuensi_pasangan(kata_tersegmen):
    """Hitung seberapa sering setiap pasangan token berdampingan."""
    frekuensi = defaultdict(int)
    for kata, freq in kata_tersegmen.items():
        simbol = kata.split()
        for i in range(len(simbol) - 1):
            frekuensi[(simbol[i], simbol[i+1])] += freq
    return frekuensi

def gabungkan_pasangan(kata_tersegmen, pasangan):
    """Ganti semua kemunculan pasangan dengan token baru."""
    a, b = pasangan
    token_baru = a + b
    kata_baru = {}
    for kata, freq in kata_tersegmen.items():
        simbol = kata.split()
        hasil = []
        i = 0
        while i < len(simbol):
            if i < len(simbol) - 1 and simbol[i] == a and simbol[i+1] == b:
                hasil.append(token_baru)
                i += 2
            else:
                hasil.append(simbol[i])
                i += 1
        kata_baru[' '.join(hasil)] = freq
    return kata_baru

def latih_bpe(teks, target_vocab_size):
    # Hitung frekuensi kata
    frekuensi_kata = defaultdict(int)
    for kata in teks.lower().split():
        frekuensi_kata[kata] += 1

    # Mulai dari huruf — pisahkan tiap kata jadi karakter
    kata_tersegmen = {' '.join(list(kata)): freq
                      for kata, freq in frekuensi_kata.items()}

    # Kumpulkan semua karakter unik sebagai vocab awal
    vocab = set()
    for kata in kata_tersegmen:
        for simbol in kata.split():
            vocab.add(simbol)

    merge_rules = []

    while len(vocab) < target_vocab_size:
        frekuensi = hitung_frekuensi_pasangan(kata_tersegmen)
        if not frekuensi:
            break
        pasangan_terbaik = max(frekuensi, key=frekuensi.get)
        kata_tersegmen = gabungkan_pasangan(kata_tersegmen, pasangan_terbaik)
        token_baru = ''.join(pasangan_terbaik)
        vocab.add(token_baru)
        merge_rules.append(pasangan_terbaik)

    return list(vocab), merge_rules

# Contoh penggunaan
teks = "bermain berlari berjalan berkata bermain bermain berlari"
vocab, rules = latih_bpe(teks, target_vocab_size=30)
print("Vocab:", sorted(vocab))
print("Merge rules pertama:", rules[:5])
```

**Python + Library** (menggunakan `tokenizers` dari HuggingFace):

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Buat tokenizer dengan model BPE
tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = Whitespace()

# Konfigurasi trainer
trainer = BpeTrainer(
    vocab_size=8000,
    special_tokens=["<|unk|>", "<|pad|>", "<|user|>", "<|assistant|>", "<|end|>"],
    show_progress=True
)

# Training dari file teks
# tokenizer.train(files=["data/teks_latih.txt"], trainer=trainer)

# Atau training dari list string
teks_latih = [
    "bermain berlari berjalan berkata",
    "saya makan nasi goreng di restoran",
    "kucing dan anjing adalah hewan peliharaan",
]
tokenizer.train_from_iterator(teks_latih, trainer=trainer)

# Simpan tokenizer
tokenizer.save("tokenizer.json")

# Encode teks
output = tokenizer.encode("bermain bola di lapangan")
print("Token IDs:", output.ids)
print("Tokens:", output.tokens)

# Decode kembali
teks_kembali = tokenizer.decode(output.ids)
print("Decoded:", teks_kembali)
```

---

## Hasil: Vocabulary dan Merge Rules

Setelah training tokenizer selesai, kita punya dua hal:

**1. Vocabulary** — kamus dari token ke ID angka:
```
{ "a": 0, "b": 1, ..., "ber": 342, "an": 45, "makan": 1205, ... }
```

**2. Merge Rules** — urutan aturan penggabungan:
```
[("b","e"), ("be","r"), ("a","n"), ("m","a"), ...]
```

Urutan sangat penting — aturan harus diterapkan dalam urutan yang sama persis saat encoding.

---

## Encoding: Teks → Token IDs

Saat kita ingin memberikan teks ke model:

```
Teks input: "saya makan"

Langkah 1: Pisah per kata
  → ["saya", "makan"]

Langkah 2: Tiap kata jadi huruf
  → [['s','a','y','a'], ['m','a','k','a','n']]

Langkah 3: Terapkan merge rules satu per satu
  Rule ("a","n") → "an"  :  ['m','a','k','an']
  Rule ("m","a") → "ma"  :  ['ma','k','an']
  ... dan seterusnya

Langkah 4: Ubah ke ID
  → [25, 0, 78, 0, 567, 89, 90]
```

---

## Decoding: Token IDs → Teks

Kebalikannya mudah — cukup lookup kamus:

```
IDs: [342, 45, 1205]
  342 → "ber"
   45 → "an"
 1205 → "makan"

Gabungkan → "bermakan"  (penggabungan teks mentah)
Bersihkan spasi → teks akhir
```

---

## Token Spesial

Selain kata-kata biasa, kita tambahkan token spesial untuk struktur percakapan:

| Token | Fungsi |
|-------|--------|
| `<\|user\|>` | Awal pesan dari pengguna |
| `<\|assistant\|>` | Awal respons model |
| `<\|end\|>` | Akhir sebuah giliran bicara |
| `<\|pad\|>` | Pengisi (padding) agar panjang batch seragam |
| `<\|unk\|>` | Token yang tidak dikenal |

Contoh percakapan yang sudah diformat:
```
<|user|> Ibukota Indonesia? <|end|>
<|assistant|> Jakarta. <|end|>
```

Ketika model dilatih dengan format ini, dia belajar bahwa setelah `<|assistant|>` harus ada jawaban.

---

## Perbandingan Metode

| Metode | Ukuran Vocab | Handle kata baru? | Panjang sequence |
|--------|-------------|-------------------|-----------------|
| Per huruf | ~100 | Ya | Sangat panjang |
| Per kata | ~500.000+ | Tidak | Pendek |
| BPE (sub-kata) | 8.000–50.000 | Ya | Sedang |

BPE memberikan keseimbangan terbaik. Kata umum jadi satu token ("makan"), kata langka dipecah ("antidisestablishmentarianism" → beberapa sub-kata).

---

## Berapa Ukuran Vocabulary yang Dibutuhkan?

Untuk model kecil yang menjawab pertanyaan sederhana dalam satu bahasa:
- **4.000–8.000 token**: Cukup, tapi kata-kata akan banyak terpecah
- **16.000–32.000 token**: Lebih baik, kata umum sudah punya token sendiri
- **50.000–100.000 token**: Dipakai model besar seperti GPT-3/4

Untuk tujuan kita (model kecil, satu bahasa), **8.000 token sudah memadai**.

---

## Ringkasan

- Teks tidak bisa langsung diproses komputer — harus diubah ke angka
- **Token** adalah potongan teks (bisa satu huruf, satu suku kata, satu kata)
- **BPE** menemukan potongan terbaik secara otomatis dari data
- Hasilnya: vocabulary (kamus) + merge rules (aturan)
- Token spesial digunakan untuk menandai struktur percakapan

---

> [Selanjutnya: Embedding — Representasi Kata sebagai Angka →](./03-embedding.md)
