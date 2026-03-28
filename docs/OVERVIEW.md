> Proyek ini murni untuk tujuan pembelajaran — memahami cara kerja LLM dari bawah ke atas.

# mini-llm — Gambaran Umum Proyek

## 1. Gambaran Proyek

Proyek **mini-llm** bertujuan membangun sebuah Large Language Model kecil dari nol, tanpa menggunakan library machine learning apapun — hanya pure Node.js. Tidak ada TensorFlow, tidak ada PyTorch, tidak ada bantuan framework apapun. Semua komponen dibangun dan dipahami satu per satu secara manual.

Tujuan utamanya bukan membuat model yang "canggih", melainkan **memahami cara kerja LLM dari dalam**. Dengan membangun sendiri setiap bagiannya, kita dipaksa untuk benar-benar mengerti apa yang terjadi di balik layar — bukan sekadar memanggil fungsi dan berharap hasilnya bagus.

Target kemampuan akhir model adalah sebuah **chatbot berbahasa Indonesia sederhana** yang bisa diajak ngobrol dan menjawab pertanyaan pengetahuan dasar. Bukan model GPT-4, tapi cukup untuk membuktikan bahwa arsitektur Transformer benar-benar bekerja bahkan ketika dibangun dari scratch.

Alur pengembangannya mengikuti cara LLM modern dibangun: pertama model dilatih dengan teks mentah dalam jumlah besar agar menyerap pengetahuan dan pola bahasa (pre-training), lalu dilanjutkan dengan pelatihan tambahan menggunakan data percakapan agar model bisa merespons seperti asisten (fine-tuning).

---

## 2. Apa yang Akan Dipelajari

Berikut adalah konsep-konsep inti yang akan dipahami secara mendalam melalui proyek ini:

### Byte Pair Encoding (BPE)
BPE adalah algoritma tokenisasi yang memecah teks menjadi potongan-potongan sub-kata. Daripada memotong per karakter atau per kata penuh, BPE mencari pasangan karakter yang paling sering muncul lalu menggabungkannya secara bertahap. Ini penting karena model tidak membaca teks mentah — ia butuh teks diubah ke urutan angka (token) terlebih dahulu, dan BPE menghasilkan representasi yang efisien untuk berbagai bahasa.

### Token Embedding dan Positional Encoding
Setelah teks dipecah jadi token, setiap token diubah menjadi vektor angka berdimensi tinggi — itulah embedding. Tapi model tidak tahu urutan token secara default, maka ditambahkan **positional encoding**: sinyal matematis yang memberi tahu model "token ini ada di posisi ke-3, yang itu di posisi ke-7". Keduanya digabung sebelum masuk ke lapisan Transformer.

### Multi-head Self-attention
Ini adalah jantung dari arsitektur Transformer. Self-attention memungkinkan setiap token "memperhatikan" token lain dalam kalimat yang sama — membangun konteks dan hubungan antar kata. "Multi-head" berarti proses ini dilakukan beberapa kali secara paralel dengan perspektif berbeda, lalu hasilnya digabung. Mekanisme ini yang membuat model bisa memahami makna dari konteks, bukan hanya urutan kata.

### Transformer Block (Attention + FFN + LayerNorm + Residual Connection)
Satu blok Transformer terdiri dari beberapa lapisan yang bekerja bersama: self-attention menangkap hubungan antar token, Feed-Forward Network (FFN) memproses setiap token secara independen untuk menambah kapasitas model, LayerNorm menstabilkan nilai selama pelatihan agar tidak meledak atau menghilang, dan residual connection menjaga agar gradien bisa mengalir lancar ke belakang. Gabungan keempat komponen ini membentuk satu "unit belajar" yang ditumpuk berkali-kali.

### Cross-entropy Loss
Loss adalah ukuran seberapa salah prediksi model saat ini. Cross-entropy mengukur perbedaan antara distribusi probabilitas yang diprediksi model dengan jawaban yang sebenarnya. Semakin kecil loss, semakin dekat prediksi model ke kenyataan. Ini adalah sinyal utama yang digunakan untuk melatih model.

### Backpropagation dan Gradient Descent
Setelah loss dihitung, backpropagation menelusuri balik seluruh jaringan untuk menghitung seberapa besar kontribusi setiap parameter terhadap kesalahan tersebut (gradien). Gradient descent kemudian menggunakan informasi itu untuk menggeser setiap parameter sedikit ke arah yang mengurangi loss. Proses ini diulang ribuan kali sampai model cukup pintar.

### Adam Optimizer
Adam adalah versi canggih dari gradient descent yang bekerja lebih cerdas. Ia menyimpan rata-rata bergerak dari gradien dan kuadratnya, sehingga bisa menyesuaikan laju belajar untuk setiap parameter secara adaptif. Dalam praktik, Adam hampir selalu lebih cepat dan lebih stabil daripada gradient descent biasa — maka hampir semua model modern menggunakannya.

### Special Tokens dan Instruction Format
Saat fine-tuning, model perlu diajarkan struktur percakapan — mana bagian "user berbicara" dan mana bagian "model menjawab". Ini dilakukan dengan menyisipkan special tokens seperti `<|user|>` dan `<|assistant|>` ke dalam data pelatihan. Model belajar bahwa setelah `<|user|>` ada pertanyaan, dan setelah `<|assistant|>` ia harus menjawab.

### Autoregressive Text Generation (Greedy, Top-k, Temperature Sampling)
Setelah model terlatih, cara ia menghasilkan teks adalah satu token per satu token: prediksi token berikutnya, tambahkan ke konteks, prediksi lagi, dan seterusnya. **Greedy** selalu memilih token dengan probabilitas tertinggi — deterministik tapi sering repetitif. **Top-k** hanya mempertimbangkan k kandidat teratas lalu memilih secara acak. **Temperature** mengontrol "kepercayaan diri" model: suhu tinggi menghasilkan teks yang lebih kreatif dan acak, suhu rendah menghasilkan teks yang lebih fokus dan konservatif.

---

## 3. Phase Pengerjaan

Proyek ini dibagi menjadi 7 phase yang dijalani secara berurutan. Setiap phase membangun fondasi untuk phase berikutnya. Alur besarnya mengikuti dua tahap utama: **pre-training** untuk menyerap pengetahuan dari teks mentah, dan **fine-tuning** untuk mengajarkan cara merespons sebagai asisten.

### Alur Besar

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRE-TRAINING PIPELINE                        │
│                                                                     │
│  ┌──────────────────────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │  data/raw/pretrain/      │  │ Cleaning │  │  train_pretrain.bin   │  │
│  │  ├── indonesia.txt       │─►│    &     │─►│   (teks mentah)       │  │
│  │  ├── fotosintesis.txt    │  │ Filtering│  │                       │  │
│  │  ├── soekarno.txt        │  └──────────┘  └──────────┬────────────┘  │
│  │  └── ... (ratusan file)  │                           │               │
│  └──────────────────────────┘                           │               │
│                                                     │               │
└─────────────────────────────────────────────────────│───────────────┘
                                                      │
                      ┌───────────────────────────────▼───────────────┐
                      │                   TOKENIZER                   │
                      │        BPE dari scratch → vocab.json          │
                      └───────────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              ▼                                                       ▼
┌─────────────────────────┐                           ┌──────────────────────────┐
│     ARSITEKTUR MODEL    │                           │   DATA FINE-TUNING       │
│                         │                           │                          │
│  Embedding              │                           │  Q&A + Percakapan JSONL  │
│    ↓                    │                           │    ↓                     │
│  Positional Encoding    │                           │  Inject Special Tokens   │
│    ↓                    │                           │    ↓                     │
│  [Transformer Block] x N│                           │  train_finetune.bin      │
│    ↓  Attention         │                           └──────────────┬───────────┘
│    ↓  FFN               │                                          │
│    ↓  LayerNorm         │                                          │
│    ↓  Residual          │                                          │
│  Output Logits          │                                          │
└──────────┬──────────────┘                                          │
           │                                                         │
           ▼                                                         │
┌─────────────────────────┐                                          │
│      PRE-TRAINING       │                                          │
│                         │                                          │
│  next-token prediction  │                                          │
│  loss → backprop        │                                          │
│  Adam optimizer         │                                          │
│                         │                                          │
│  → checkpoint_pretrain  │                                          │
└──────────┬──────────────┘                                          │
           │                                                         │
           └──────────────────────┬──────────────────────────────────┘
                                  ▼
                    ┌─────────────────────────┐
                    │      FINE-TUNING / SFT  │
                    │                         │
                    │  load checkpoint        │
                    │  training Q&A data      │
                    │  lr kecil               │
                    │                         │
                    │  → checkpoint_finetune  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    INFERENCE & CHAT     │
                    │                         │
                    │  load checkpoint        │
                    │  greedy / top-k / temp  │
                    │  terminal chat loop     │
                    │                         │
                    │   > tanya apa saja :)   │
                    └─────────────────────────┘
```

---

### Struktur Data

```
data/raw/
├── pretrain/
│   ├── indonesia.txt
│   ├── fotosintesis.txt
│   ├── soekarno.txt
│   ├── matematika.txt
│   └── ... (ratusan file)
├── conversations.jsonl
└── knowledge.jsonl
```

Dua jenis data digunakan di proyek ini, masing-masing untuk tahap yang berbeda:

**Pre-training** (`data/raw/pretrain/`) — teks mentah murni, satu topik per file, tanpa format apapun:

```
Indonesia adalah negara kepulauan di Asia Tenggara yang terdiri dari lebih dari 17.000 pulau.
Ibu kota Indonesia adalah Jakarta. Negara ini berbatasan dengan Malaysia, Papua Nugini, dan Timor Leste...
```

Tidak ada label, tidak ada struktur JSON — hanya teks bersih apa adanya. Ratusan file teks dikumpulkan dari berbagai topik pengetahuan umum, lalu digabung dan dibersihkan sebelum masuk ke training.

**Fine-tuning** (`data/raw/conversations.jsonl` dan `knowledge.jsonl`) — format percakapan Q&A:

```jsonl
{"user": "siapa soekarno?", "assistant": "Soekarno adalah presiden pertama Indonesia..."}
{"user": "halo", "assistant": "halo juga! ada yang bisa saya bantu?"}
```

---

### Phase 1 — Data Pipeline (Pre-training)

Titik awal dari seluruh proyek. Di sini kita mengumpulkan ratusan file teks mentah di `data/raw/pretrain/` — satu file per topik, isi murni teks tanpa format apapun. Semua file dibaca, dibersihkan dari noise (karakter aneh, duplikat, baris kosong berlebih), digabung, lalu disimpan sebagai dataset biner yang siap dikonsumsi tokenizer. Tidak ada label, tidak ada struktur JSON — hanya teks bersih.

**Input:** `data/raw/pretrain/*.txt` — ratusan file teks mentah
**Output:** `train_pretrain.bin` — dataset teks bersih dalam format biner
**Yang dipelajari:** Pipeline data mentah, pentingnya cleaning, dan betapa besarnya perbedaan antara "data ada" dengan "data siap pakai"

---

### Phase 2 — Tokenizer

Phase ini terdiri dari dua langkah yang dijalankan berurutan: **tokenize** lalu **encode**.

**Tokenize** (`pretrain tokenize`) membaca semua file `.txt` dari `data/pretrain/raw/`, menghitung frekuensi kata, lalu melatih algoritma BPE dari scratch. BPE secara iteratif menggabungkan pasangan karakter/sub-kata yang paling sering muncul hingga ukuran vocabulary yang diinginkan tercapai. Hasilnya disimpan sebagai `vocab.json` (mapping token → ID) dan `merges.json` (urutan merge rules).

**Encode** (`pretrain encode`) menerapkan tokenizer yang sudah dilatih ke seluruh dataset: setiap file teks dibaca, kata-katanya dipecah menggunakan merge rules yang sama persis seperti saat training, lalu diubah menjadi urutan angka. Semua token ID digabung dan disimpan sebagai file binary `train.bin` dalam format `Uint16Array` — format yang efisien dan langsung bisa dibaca saat training.

**Input:** `data/pretrain/raw/*.txt`
**Output:** `data/pretrain/tokenized/vocab.json`, `data/pretrain/tokenized/merges.json`, `data/pretrain/train.bin`
**Yang dipelajari:** Cara kerja BPE dari nol, kenapa word frequency jadi fondasi merge order, dan bagaimana teks mentah berubah menjadi urutan angka yang siap dikonsumsi model

---

### Phase 3 — Arsitektur Model

Phase paling teknis. Kita membangun seluruh arsitektur Transformer dari nol: embedding layer, positional encoding, multi-head self-attention, feed-forward network, dan menumpuk semuanya menjadi blok-blok Transformer yang membentuk model lengkap. Setiap operasi ditulis secara eksplisit — tidak ada yang tersembunyi di balik abstraksi library.

**Input:** Urutan token (angka)
**Output:** Distribusi probabilitas untuk token berikutnya
**Yang dipelajari:** Cara setiap komponen Transformer bekerja dan berinteraksi satu sama lain

---

### Phase 4 — Pre-training

Model dari Phase 3 masih kosong — parameternya acak, ia tidak tahu apa-apa. Di sini kita melatihnya pada teks mentah Wikipedia: model belajar memprediksi token berikutnya dari setiap urutan teks. Lewat proses inilah model menyerap pengetahuan: fakta tentang Indonesia, konsep sains, sejarah, dan pola bahasa Indonesia secara umum.

**Input:** `train_pretrain.bin` + model kosong
**Output:** Checkpoint model pre-trained
**Yang dipelajari:** Training loop, next-token prediction sebagai objective, dan bagaimana model "menyerap" pengetahuan dari teks mentah

---

### Phase 5 — Data Pipeline (Fine-tuning)

Sebelum fine-tuning bisa dimulai, data percakapan Q&A perlu disiapkan secara khusus. Di phase ini kita mengambil pasangan user-assistant, menyisipkan special tokens untuk menandai struktur percakapan, lalu mengubahnya ke format biner. Model perlu belajar bahwa ada "giliran user" dan "giliran asisten" — dan special tokens adalah cara mengajarkan itu.

**Input:** `data/raw/conversations.jsonl` + `data/raw/knowledge.jsonl`
**Output:** `train_finetune.bin` — dataset percakapan dengan special tokens
**Yang dipelajari:** Cara format instruksi bekerja dan mengapa special tokens penting untuk alignment

---

### Phase 6 — Fine-tuning / SFT

Dengan checkpoint dari pre-training sebagai titik awal, kita melanjutkan training menggunakan data percakapan. Model yang tadinya bisa "menulis teks" sekarang diajarkan cara "merespons pertanyaan". Proses ini disebut Supervised Fine-Tuning (SFT). Learning rate yang digunakan jauh lebih kecil dari pre-training — kita tidak ingin menghapus pengetahuan yang sudah diserap, hanya menambah kemampuan merespons.

**Input:** Checkpoint pre-training + `train_finetune.bin`
**Output:** Checkpoint model fine-tuned
**Yang dipelajari:** Perbedaan pre-training vs fine-tuning, transfer learning, dan mengapa urutan ini lebih efisien daripada melatih dari awal

---

### Phase 7 — Inference & Chat

Phase terakhir — saat model akhirnya "berbicara". Kita load checkpoint fine-tuning, implementasikan berbagai strategi sampling (greedy, top-k, temperature), lalu bungkus semuanya dalam antarmuka chat interaktif di terminal. Di sinilah semua kerja keras dari enam phase sebelumnya terbayar.

**Input:** Checkpoint fine-tuned + input pengguna dari terminal
**Output:** Antarmuka chat interaktif berbahasa Indonesia
**Yang dipelajari:** Inference autoregressive, efek parameter sampling terhadap output, dan cara menyusun prompt dengan special tokens

---

## 4. Urutan Belajar

Phase-phase ini harus dikerjakan secara berurutan karena setiap phase secara harfiah bergantung pada output dari phase sebelumnya.

Tanpa data bersih (Phase 1), tokenizer tidak bisa dilatih dengan baik. Tanpa tokenizer (Phase 2), tidak ada cara mengubah teks menjadi angka. Tanpa arsitektur (Phase 3), tidak ada yang bisa dilatih. Tanpa pre-training (Phase 4), model tidak punya pengetahuan dasar. Tanpa data fine-tuning yang disiapkan dengan benar (Phase 5), model tidak akan belajar pola percakapan. Tanpa fine-tuning (Phase 6), model hanya bisa "melanjutkan teks", bukan "menjawab pertanyaan". Dan tanpa Phase 7, model terlatih pun tidak bisa digunakan.

Ada juga logika belajar di balik urutan ini. Pre-training dulu sebelum fine-tuning mencerminkan cara LLM modern dibangun — dan memahaminya secara berurutan membuat konsep transfer learning terasa sangat intuitif. Kamu akan benar-benar merasakan perbedaan antara model sebelum dan sesudah fine-tuning, bukan hanya membaca teorinya.

Jangan tergoda untuk melompat ke phase yang lebih "menarik" — fondasi yang kuat dari awal akan membuat segalanya jauh lebih mudah dipahami di akhir.