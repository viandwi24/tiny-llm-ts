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

Edit `data/config.json` untuk mengatur topics, bahasa, dan hyperparameter model. File ini dibuat otomatis dengan nilai default jika belum ada.

```json
{
  "tokenizer": {
    "charMarker": "Ġ",
    "vocabSize": 8000
  },
  "pretrain": {
    "providers": {
      "wikipedia": {
        "topics": ["Indonesia", "Fisika", "Biologi"],
        "language": "id",
        "outputDir": "data/pretrain/raw",
        "searchLimit": 100
      }
    }
  },
  "train": {
    "dataFile": "data/pretrain/train.bin",
    "vocabSize": 8000,
    "embedSize": 64,
    "numHeads": 2,
    "numLayers": 2,
    "ffnDim": 128,
    "maxSeqLen": 32,
    "epochs": 100,
    "learningRate": 0.0001,
    "batchSize": 1
  }
}
```

---

## Pipeline Lengkap

### Phase 1-2: Data & Tokenizer

Tiga langkah ini harus dijalankan berurutan. Output setiap langkah menjadi input langkah berikutnya.

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

#### Step 2 — Tokenize (training BPE)

Membaca semua file `.txt` dari `data/pretrain/raw/`, melatih tokenizer BPE dari scratch, lalu menyimpan vocabulary dan merge rules.

```bash
bun src/index.ts pretrain tokenize
```

Output:
- `data/pretrain/tokenized/vocab.json` — mapping token → ID
- `data/pretrain/tokenized/merges.json` — merge rules BPE

#### Step 3 — Encode (teks → token ID binary)

Membaca semua file `.txt`, menerapkan tokenizer yang sudah dilatih, lalu menyimpan hasilnya sebagai file binary `uint16` siap pakai untuk training.

```bash
bun src/index.ts pretrain encode
```

Output: `data/pretrain/train.bin` — token ID dalam format binary `Uint16Array`.

#### Reset data

```bash
bun src/index.ts pretrain reset
```

---

### Phase 4: Training

Melatih model Transformer pada data pre-training. Checkpoint disimpan otomatis.

```bash
bun src/index.ts train
```

Opsi:
```bash
bun src/index.ts train --reset    # hapus checkpoint lama, mulai dari awal
```

Checkpoint disimpan ke `data/checkpoints/latest.json` setiap N steps, dengan snapshot per-step di `data/checkpoints/checkpoint-step<N>.json`.

Akselerasi opsional via TensorFlow.js backend (BLAS):
```bash
USE_TFJS=true bun src/index.ts train
```

---

### Phase 7: Chat / Inference

Load checkpoint terbaru dan mulai sesi chat interaktif di terminal.

```bash
bun src/index.ts chat
```

Opsi:
```bash
bun src/index.ts chat --temperature 0.8 --max-tokens 100 --repetition-penalty 1.3
```

| Flag | Default | Keterangan |
|---|---|---|
| `--temperature` | `0` | 0 = greedy, >0 = sampling acak |
| `--max-tokens` | `50` | Jumlah token baru maksimal per reply |
| `--repetition-penalty` | `1.3` | Penalti token yang sudah muncul |

---

## Arsitektur Model

Transformer decoder-only, diimplementasi dari scratch tanpa library ML:

```
Input IDs
  → Token Embedding + Positional Embedding
  → [TransformerBlock × numLayers]
      ├── LayerNorm → MultiHeadAttention (causal mask) → Residual
      └── LayerNorm → FeedForward (GELU) → Residual
  → Linear Projection → Logits [vocabSize]
```

**Komponen utama:**

| File | Kelas/Fungsi | Keterangan |
|---|---|---|
| [src/transformer/index.ts](src/transformer/index.ts) | `Transformer`, `TransformerBlock` | Model utama |
| [src/transformer/attention.ts](src/transformer/attention.ts) | `MultiHeadAttention` | Self-attention dengan causal mask |
| [src/transformer/embedding.ts](src/transformer/embedding.ts) | `Embedding` | Token + positional embedding |
| [src/transformer/feedforward.ts](src/transformer/feedforward.ts) | `FeedForward` | MLP 2-layer dengan GELU |
| [src/neural-network/layer.ts](src/neural-network/layer.ts) | `LinearLayer`, `LayerNormalization` | Layer dasar |
| [src/neural-network/activation.ts](src/neural-network/activation.ts) | `softmax`, `gelu` | Fungsi aktivasi |
| [src/matrix/index.ts](src/matrix/index.ts) | `Matrix` | Backend matrix switchable (CPU / TF.js) |
| [src/train/optimizer.ts](src/train/optimizer.ts) | `AdamOptimizer` | Optimizer Adam |
| [src/train/checkpoint.ts](src/train/checkpoint.ts) | `saveCheckpoint`, `loadCheckpoint` | Simpan/load model |
| [src/tokenizer.ts](src/tokenizer.ts) | `loadTokenizer` | BPE tokenizer |
| [src/chat.ts](src/chat.ts) | `generate` | Autoregressive generation |

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
└── checkpoints/
    ├── latest.json           — checkpoint terbaru
    └── checkpoint-step<N>.json

src/
├── index.ts                  — entry point
├── cli/index.ts              — definisi CLI (Commander)
├── config.ts                 — loadConfig()
├── constants.ts              — special tokens
├── tokenizer.ts              — BPE tokenizer interface
├── chat.ts                   — chat loop & generate()
├── transformer/
│   ├── index.ts              — Transformer & TransformerBlock
│   ├── embedding.ts          — Token + positional embedding
│   ├── attention.ts          — MultiHeadAttention
│   └── feedforward.ts        — FeedForward MLP
├── neural-network/
│   ├── layer.ts              — LinearLayer, LayerNormalization
│   └── activation.ts         — softmax, gelu
├── matrix/
│   ├── index.ts              — switchable backend
│   └── backends/
│       ├── cpu.ts            — pure TS matrix ops
│       └── tfjs.ts           — TensorFlow.js backend
├── train/
│   ├── index.ts              — training loop, DataLoader, cross-entropy
│   ├── optimizer.ts          — Adam optimizer
│   └── checkpoint.ts         — save/load checkpoint
├── pretrain/
│   ├── provider.ts           — interface PretrainProvider<TConfig>
│   ├── tokenize.ts           — BPE training
│   ├── encode.ts             — BPE encode teks → token ID binary
│   └── providers/
│       └── wikipedia.ts      — Wikipedia provider + cleanText
├── tui/
│   ├── index.tsx             — renderTUI() helper
│   └── components/           — progress bar, scraper, tokenize, encode UI
└── scripts/fix.ts            — patch wikipedia library
```

---

## Testing

```bash
bun test
```

Test dasar mencakup forward pass Transformer: validasi shape output, deteksi NaN/Inf.

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
- [x] TUI progress untuk tokenize dan encode
- [x] Special tokens: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<|pad|>`, `<|unk|>`
- [x] Decode token ID → teks (untuk inference)

### Phase 3 — Arsitektur Model

- [x] Token embedding + positional encoding
- [x] Multi-head self-attention dengan causal mask
- [x] Feed-forward network (FFN) dengan GELU
- [x] Transformer block (attention + FFN + LayerNorm + residual)
- [x] Stack N transformer blocks menjadi model lengkap
- [x] Backpropagation manual untuk semua komponen

### Phase 4 — Pre-training

- [x] DataLoader dari `data/pretrain/train.bin`
- [x] Next-token prediction loss (cross-entropy)
- [x] Backpropagation manual
- [x] Adam optimizer
- [x] Training loop dengan logging loss
- [x] Save/load checkpoint (`data/checkpoints/`)
- [x] Matrix backend switchable (CPU / TF.js BLAS)

### Phase 5 — Data Pipeline (Fine-tuning)

- [ ] Format data Q&A (`data/finetune/conversations.jsonl`, `knowledge.jsonl`)
- [ ] Inject special tokens (`<|user|>`, `<|assistant|>`)
- [ ] Tokenisasi dan simpan ke `data/finetune/train.bin`

### Phase 6 — Fine-tuning / SFT

- [ ] Load checkpoint pre-training
- [ ] Training loop dengan data fine-tuning (lr lebih kecil)
- [ ] Simpan checkpoint fine-tuning

### Phase 7 — Inference & Chat

- [x] Load checkpoint terbaru
- [x] Greedy sampling
- [x] Temperature sampling
- [x] Repetition penalty
- [x] Chat loop interaktif di terminal
- [ ] Top-k / top-p sampling
