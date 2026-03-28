# tiny-llm

LLM kecil dibangun dari nol menggunakan pure TypeScript — tanpa framework ML apapun. Proyek ini murni untuk tujuan pembelajaran: memahami cara kerja LLM dari bawah ke atas.

Baca [docs/OVERVIEW.md](docs/OVERVIEW.md) untuk gambaran lengkap arsitektur dan alur pengembangannya.

---

## Cara Menjalankan

### 1. Install dependencies

```bash
bun install
```

> `postinstall` otomatis menjalankan patch untuk library `wikipedia` agar header HTTP-nya benar.

### 2. Konfigurasi

Edit `data/config.json` untuk mengatur topics, bahasa, dan output:

```json
{
  "pretrain": {
    "providers": {
      "wikipedia": {
        "topics": ["Indonesia", "Fisika", "Biologi"],
        "language": "id",
        "outputDir": "data/pretrain/raw",
        "searchLimit": 100
      }
    }
  }
}
```

### 3. Fetch → Tokenize → Encode

Tiga langkah ini harus dijalankan berurutan. Output setiap langkah menjadi input langkah berikutnya.

---

#### Step 1 — Fetch data mentah dari Wikipedia

```bash
bun src/index.ts pretrain fetch wikipedia
```

Opsi:
```bash
bun src/index.ts pretrain fetch wikipedia --lang id --limit 50 --output data/pretrain/raw
```

Output: file `.txt` per artikel di `data/pretrain/raw/`, nama format `<topic>-<judul>.txt`.
Teks sudah di-clean otomatis — heading markup, referensi, karakter asing dihapus.

---

#### Step 2 — Tokenize (training BPE)

Membaca semua file `.txt` dari `data/pretrain/raw/`, melatih tokenizer BPE dari scratch, lalu menyimpan vocabulary dan merge rules.

```bash
bun src/index.ts pretrain tokenize
```

Opsi:
```bash
bun src/index.ts pretrain tokenize --input data/pretrain/raw --output data/pretrain/tokenized --vocab-size 8000
```

Output:
- `data/pretrain/tokenized/vocab.json` — mapping token → ID
- `data/pretrain/tokenized/merges.json` — merge rules BPE

---

#### Step 3 — Encode (teks → token ID binary)

Membaca semua file `.txt`, menerapkan tokenizer yang sudah dilatih, lalu menyimpan hasilnya sebagai file binary `uint16` siap pakai untuk training.

```bash
bun src/index.ts pretrain encode
```

Opsi:
```bash
bun src/index.ts pretrain encode --input data/pretrain/raw --vocab data/pretrain/tokenized --output data/pretrain/train.bin
```

Output: `data/pretrain/train.bin` — token ID dalam format binary `Uint16Array`.

---

#### Reset data

```bash
bun src/index.ts pretrain reset
```

---

## Struktur Workspace

```
data/
├── config.json               — konfigurasi project
├── pretrain/
│   ├── raw/                  — hasil fetch wikipedia (*.txt)
│   ├── tokenized/
│   │   ├── vocab.json        — token → ID mapping
│   │   └── merges.json       — BPE merge rules
│   └── train.bin             — token ID binary (Uint16Array)
├── finetune/                 — (mendatang) data Q&A fine-tuning
├── checkpoints/              — (mendatang) model checkpoint
└── output/                   — (mendatang) hasil inference

src/
├── cli/                      — CLI entry point (Commander)
├── pretrain/
│   ├── provider.ts           — interface PretrainProvider<TConfig>
│   ├── tokenize.ts           — BPE training
│   ├── encode.ts             — BPE encode teks → token ID binary
│   └── providers/
│       └── wikipedia.ts      — Wikipedia provider + cleanText
├── tui/
│   ├── index.tsx             — renderTUI() helper
│   └── components/
│       ├── ProgressBar.tsx       — progress bar reusable
│       ├── WikipediaScraper.tsx  — TUI fetch progress
│       ├── TokenizeProgress.tsx  — TUI tokenize progress
│       └── EncodeProgress.tsx    — TUI encode progress
├── config.ts                 — loadConfig() dari data/config.json
└── scripts/
    └── fix.ts                — patch wikipedia library
```

---

## Progress

### Phase 1 — Data Pipeline (Pre-training)

- [x] Provider system dengan interface `PretrainProvider<TConfig>`
- [x] Wikipedia provider — search → summary per topic
- [x] Config extensible via `data/config.json`
- [x] Dynamic CLI subcommand per provider
- [x] TUI interactive dengan progress per topic (ink)
- [x] Auto-cleaning teks: hapus heading markup, referensi, karakter asing
- [x] Output file dengan nama `<topic>-<judul>.txt`
- [x] Command `pretrain reset` untuk bersihkan data

### Phase 2 — Tokenizer

- [x] BPE (Byte Pair Encoding) dari scratch
- [x] Training tokenizer dari dataset pre-training (`pretrain tokenize`)
- [x] Encode teks ke token ID binary (`pretrain encode`)
- [x] Simpan vocabulary ke `data/pretrain/tokenized/vocab.json`
- [x] TUI progress untuk tokenize (phase label + progress bar BPE + eta)
- [x] TUI progress untuk encode (progress bar per file + format bytes)
- [ ] Decode token ID → teks (untuk inference)

### Phase 3 — Arsitektur Model

- [ ] Token embedding + positional encoding
- [ ] Multi-head self-attention
- [ ] Feed-forward network (FFN)
- [ ] Transformer block (attention + FFN + LayerNorm + residual)
- [ ] Stack N transformer blocks menjadi model lengkap

### Phase 4 — Pre-training

- [ ] DataLoader dari `data/pretrain/raw/`
- [ ] Next-token prediction loss (cross-entropy)
- [ ] Backpropagation manual
- [ ] Adam optimizer
- [ ] Training loop dengan logging loss
- [ ] Simpan checkpoint ke `data/checkpoints/pretrain.bin`

### Phase 5 — Data Pipeline (Fine-tuning)

- [ ] Format data Q&A (`data/finetune/conversations.jsonl`, `knowledge.jsonl`)
- [ ] Inject special tokens (`<|user|>`, `<|assistant|>`)
- [ ] Tokenisasi dan simpan ke `data/finetune/train.bin`

### Phase 6 — Fine-tuning / SFT

- [ ] Load checkpoint pre-training
- [ ] Training loop dengan data fine-tuning (lr lebih kecil)
- [ ] Simpan checkpoint ke `data/checkpoints/finetune.bin`

### Phase 7 — Inference & Chat

- [ ] Load checkpoint fine-tuning
- [ ] Greedy sampling
- [ ] Top-k sampling
- [ ] Temperature sampling
- [ ] Chat loop interaktif di terminal
