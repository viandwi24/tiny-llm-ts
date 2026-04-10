# Bagian 11 — Inference: Cara Model Menghasilkan Jawaban

[← Sebelumnya](./10-training-pipeline.md) | [Selanjutnya →](./12-implementasi.md)

---

## Training vs Inference

**Training** = model *belajar*. Parameter berubah setiap langkah.

**Inference** = model *digunakan*. Parameter tidak berubah — model sudah "matang".

Ketika kamu chatting dengan AI, kamu sedang menggunakan inference. Model yang kamu gunakan sudah selesai dilatih.

---

## Autoregressive Generation: Menulis Kata per Kata

LLM tidak menghasilkan seluruh jawaban sekaligus. Mereka menghasilkan **satu token per langkah**, lalu token itu dijadikan bagian dari input untuk langkah berikutnya.

### Analogi: Menulis Esai dengan Cara Khusus

Bayangkan menulis esai dengan aturan aneh: kamu hanya boleh menulis satu kata, lalu *baca ulang seluruh yang sudah ditulis*, baru tulis kata berikutnya.

1. [baca konteks] → tulis "Jakarta"
2. [baca "Jakarta"] → tulis "adalah"
3. [baca "Jakarta adalah"] → tulis "ibukota"
4. [baca "Jakarta adalah ibukota"] → tulis "Indonesia"
5. [baca seterusnya] → tulis "." → selesai

Itulah autoregressive generation. Setiap kata dihasilkan dengan mempertimbangkan **seluruh konteks sebelumnya**.

---

## Pseudocode Generate

```
FUNGSI generate(model, tokenizer, prompt, max_token_baru, temperature, rep_penalty):

  # Encode prompt ke token IDs
  context = tokenizer.encode(prompt)
  token_dihasilkan = []

  UNTUK _ DARI 0 sampai max_token_baru:

    # Trim jika konteks terlalu panjang
    input_ids = context[-max_seq_len:]

    # Forward pass — hanya butuh logits posisi terakhir
    logits = model.forward(input_ids)
    logits_terakhir = logits[-1]             [vocab_size]

    # Terapkan repetition penalty
    UNTUK token_id DALAM token_dihasilkan:
      JIKA logits_terakhir[token_id] > 0:
        logits_terakhir[token_id] /= rep_penalty
      TIDAK:
        logits_terakhir[token_id] *= rep_penalty

    # Pilih token berikutnya
    token_baru = pilih_token(logits_terakhir, temperature)

    # Cek kondisi berhenti
    JIKA token_baru == ID_END atau token_baru == ID_UNK:
      BERHENTI

    token_dihasilkan.tambah(token_baru)
    context.tambah(token_baru)

  KEMBALIKAN tokenizer.decode(token_dihasilkan)
```

**Python Native** (murni Python, tanpa library):

```python
import math
import random

def softmax(logits):
    m = max(logits)
    exp_vals = [math.exp(x - m) for x in logits]
    total = sum(exp_vals)
    return [v / total for v in exp_vals]

def pilih_token(logits, temperature=1.0):
    """Pilih token berdasarkan distribusi probabilitas."""
    if temperature == 0:
        return logits.index(max(logits))   # greedy
    logits_scaled = [l / temperature for l in logits]
    probs = softmax(logits_scaled)
    # Multinomial sampling
    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if r <= cumsum:
            return i
    return len(probs) - 1

def generate(model, tokenizer, prompt, max_token_baru=50,
             temperature=0.8, rep_penalty=1.3, max_seq_len=128,
             id_end=None):
    context = tokenizer.encode(prompt)
    token_dihasilkan = []

    for _ in range(max_token_baru):
        input_ids = context[-max_seq_len:]
        logits_semua = model.forward(input_ids)
        logits = list(logits_semua[-1])   # ambil posisi terakhir

        # Repetition penalty
        for token_id in set(token_dihasilkan):
            if logits[token_id] > 0:
                logits[token_id] /= rep_penalty
            else:
                logits[token_id] *= rep_penalty

        token_baru = pilih_token(logits, temperature)

        if token_baru == id_end:
            break

        token_dihasilkan.append(token_baru)
        context.append(token_baru)

    return tokenizer.decode(token_dihasilkan)
```

**Python + Library** (menggunakan PyTorch):

```python
import torch
import torch.nn.functional as F

@torch.no_grad()   # matikan gradient saat inference — hemat memori
def generate(model, tokenizer, prompt, max_token_baru=200,
             temperature=0.8, rep_penalty=1.3, max_seq_len=512,
             device='cpu'):
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    token_dihasilkan = []
    id_end = tokenizer.convert_tokens_to_ids('<|end|>')

    for _ in range(max_token_baru):
        # Trim ke max_seq_len
        ctx = input_ids[:, -max_seq_len:]

        logits = model(ctx)                      # [1, T, vocab_size]
        logits_terakhir = logits[0, -1, :].clone()   # [vocab_size]

        # Repetition penalty
        for token_id in set(token_dihasilkan):
            if logits_terakhir[token_id] > 0:
                logits_terakhir[token_id] /= rep_penalty
            else:
                logits_terakhir[token_id] *= rep_penalty

        # Sampling dengan temperature
        if temperature == 0:
            token_baru = logits_terakhir.argmax().item()
        else:
            probs = F.softmax(logits_terakhir / temperature, dim=-1)
            token_baru = torch.multinomial(probs, num_samples=1).item()

        if token_baru == id_end:
            break

        token_dihasilkan.append(token_baru)
        input_ids = torch.cat([input_ids,
                                torch.tensor([[token_baru]], device=device)], dim=1)

    return tokenizer.decode(token_dihasilkan, skip_special_tokens=True)

# Contoh penggunaan (dengan model HuggingFace)
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model     = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# print(generate(model, tokenizer, "<|user|> Ibukota Indonesia? <|end|>\n<|assistant|>"))
```

---

## Temperature: Kenali "Kepribadian" Model

**Temperature** mengontrol seberapa "berani" atau "konservatif" model dalam memilih token.

Cara kerjanya: logits dibagi dengan temperature sebelum softmax.

```
logits_scaled = logits / temperature
probs = softmax(logits_scaled)
```

### Efek Temperature pada Distribusi

```
Logits asli: [5.0, 3.0, 2.0, 1.0]
             "Jakarta" "Bandung" "Bogor" "Depok"

Temperature = 0.3 (rendah — percaya diri):
  Logits / 0.3 = [16.7, 10.0, 6.7, 3.3]
  Probs: [0.99, 0.006, 0.0003, 0.00003]
  → Model hampir selalu pilih "Jakarta"

Temperature = 1.0 (normal):
  Probs: [0.76, 0.14, 0.06, 0.03]
  → "Jakarta" masih paling mungkin, tapi variasi ada

Temperature = 2.0 (tinggi — kreatif):
  Logits / 2.0 = [2.5, 1.5, 1.0, 0.5]
  Probs: [0.47, 0.26, 0.17, 0.10]
  → Distribusi merata, banyak variasi
```

### Panduan Penggunaan

| Temperature | Perilaku | Cocok untuk |
|-------------|----------|-------------|
| 0 (greedy) | Deterministik, selalu pilih token terbaik | Fakta, kode, Q&A tepat |
| 0.3–0.7 | Fokus, sedikit variasi | Pertanyaan teknis, ringkasan |
| 0.8–1.0 | Seimbang | Percakapan umum |
| 1.2–2.0 | Kreatif, bervariasi | Cerita, puisi, brainstorming |

---

## Dua Cara Memilih Token

### Greedy Decoding (temperature = 0)
```
token = argmax(probs)
```
Selalu pilih token dengan probabilitas tertinggi. Cepat dan deterministik, tapi sering repetitif dan kurang natural.

### Multinomial Sampling (temperature > 0)
```
token = sample berdasarkan distribusi probs
```
Pilih secara probabilistik. Token dengan prob 70% punya peluang 70% terpilih. Menghasilkan teks yang lebih bervariasi dan natural.

**Analogi sampling:** Seperti melempar dadu yang sisi-sisinya tidak sama besar. Sisi "Jakarta" mungkin punya luas 70% dari total permukaan dadu — sering keluar tapi tidak selalu.

---

## Teknik Sampling Lanjutan (Referensi)

Meskipun tidak wajib diimplementasikan di model kecil, ini penting diketahui:

### Top-K Sampling
Hanya pertimbangkan K token teratas, buang sisanya, lalu sample:
```
Pilih 50 token dengan prob tertinggi
Normalisasi ulang probs-nya agar totalnya = 1
Sample dari 50 token itu
```
Memastikan model tidak pernah memilih token yang sangat tidak relevan.

### Top-P (Nucleus) Sampling
Pilih token terkecil yang total probnya mencapai P:
```
Urutkan token dari prob tertinggi ke terendah
Ambil token sampai total prob ≥ 0.9 (misalnya)
Sample dari token yang terpilih itu
```
Lebih dinamis dari Top-K karena jumlah token yang dipertimbangkan otomatis menyesuaikan.

---

## Repetition Penalty: Hindari Pengulangan

Tanpa pengendalian, model sering mengulang-ulang kata atau frasa yang sama:

```
"makan makan makan makan makan..."
"Indonesia Indonesia Indonesia..."
```

**Repetition Penalty** mengurangi probabilitas token yang sudah pernah muncul:

```
UNTUK setiap token_id yang sudah ada di output:
  JIKA logits[token_id] > 0:
    logits[token_id] = logits[token_id] / penalty
  TIDAK:
    logits[token_id] = logits[token_id] × penalty
```

`penalty = 1.3` berarti token yang sudah muncul probabilitasnya dikurangi sekitar 23%.

---

## Format Percakapan

Untuk model yang bisa bercakap-cakap, input harus diformat dengan token spesial:

```
Format single-turn:
  <|user|> Ibukota Indonesia? <|end|>
  <|assistant|>

Model generate sampai produksi <|end|>:
  → Jakarta. <|end|>
```

```
Format multi-turn (dengan sejarah percakapan):
  <|user|> Halo <|end|>
  <|assistant|> Halo! Ada yang bisa saya bantu? <|end|>
  <|user|> Apa ibukota Indonesia? <|end|>
  <|assistant|>
```

Model belajar dari data percakapan yang sudah diformat seperti ini. Ketika melihat `<|assistant|>` tanpa diikuti `<|end|>`, model tahu ia harus menghasilkan respons.

---

## Context Window: Batas Ingatan Model

LLM hanya bisa memproses sejumlah token sekaligus — disebut **context window**.

Mengapa terbatas? Karena attention punya kompleksitas O(n²). Dua kali lebih panjang = empat kali lebih berat.

```
Contoh context window 128 token:

Percakapan pendek (total 50 token): semua masuk ✓
  User: Halo
  AI: Halo!
  User: Apa itu fotosintesis?
  AI: Fotosintesis adalah proses...

Percakapan panjang (total 300 token):
  Token 1–172 terlupakan! ←───┐
  Hanya 128 token terakhir diproses
```

Untuk percakapan Q&A sederhana, **128–256 token sudah lebih dari cukup**.

---

## Pseudocode Chat Loop Lengkap

```
FUNGSI chat(model, tokenizer, config):

  # Muat model dari checkpoint
  bobot = baca_checkpoint("checkpoint-latest.json")
  muat_ke_model(model, bobot)
  tampilkan("Model siap! Loss: {bobot['loss']:.3f}")

  sejarah = []   # simpan riwayat percakapan

  SELAMA benar:
    input_user = minta_input("Kamu: ")

    JIKA input_user == "keluar":
      BERHENTI

    # Bangun prompt dengan sejarah
    prompt = ""
    UNTUK (user, ai) DALAM sejarah[-3:]:   # ambil 3 turn terakhir
      prompt += f"<|user|> {user} <|end|>\n"
      prompt += f"<|assistant|> {ai} <|end|>\n"
    prompt += f"<|user|> {input_user} <|end|>\n<|assistant|>"

    # Generate jawaban
    jawaban = generate(
      model, tokenizer, prompt,
      max_token_baru  = config.max_tokens,
      temperature     = config.temperature,
      rep_penalty     = config.repetition_penalty
    )

    tampilkan(f"AI: {jawaban}")
    sejarah.tambah((input_user, jawaban))
```

**Python Native** (murni Python, tanpa library):

```python
import json

def chat(model, tokenizer, config):
    # Muat checkpoint
    with open("checkpoint-latest.json") as f:
        ckpt = json.load(f)
    model.muat_bobot(ckpt['bobot'])
    print(f"Model siap! Loss terakhir: {ckpt['loss']:.3f}")

    sejarah = []

    while True:
        input_user = input("Kamu: ").strip()
        if input_user.lower() == "keluar":
            break

        # Bangun prompt dari sejarah (3 turn terakhir)
        prompt = ""
        for user, ai in sejarah[-3:]:
            prompt += f"<|user|> {user} <|end|>\n"
            prompt += f"<|assistant|> {ai} <|end|>\n"
        prompt += f"<|user|> {input_user} <|end|>\n<|assistant|>"

        jawaban = generate(
            model, tokenizer, prompt,
            max_token_baru = config['max_tokens'],
            temperature    = config['temperature'],
            rep_penalty    = config['rep_penalty']
        )

        print(f"AI: {jawaban}")
        sejarah.append((input_user, jawaban))
```

**Python + Library** (menggunakan PyTorch + HuggingFace Transformers):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def chat(model_path="Qwen/Qwen2.5-0.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("Model siap! Ketik 'keluar' untuk berhenti.")
    sejarah = []

    while True:
        input_user = input("Kamu: ").strip()
        if input_user.lower() == "keluar":
            break

        # Bangun prompt dari sejarah (format chat)
        messages = []
        for user, ai in sejarah[-3:]:
            messages.append({"role": "user",    "content": user})
            messages.append({"role": "assistant", "content": ai})
        messages.append({"role": "user", "content": input_user})

        # HuggingFace apply_chat_template otomatis format prompt
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **input_ids,
                max_new_tokens    = 200,
                temperature       = 0.8,
                repetition_penalty = 1.3,
                do_sample         = True
            )

        jawaban = tokenizer.decode(
            output_ids[0][input_ids['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        print(f"AI: {jawaban}")
        sejarah.append((input_user, jawaban))

# chat()  # uncomment untuk menjalankan
```

---

## Ringkasan

| Konsep | Penjelasan |
|--------|-----------|
| Inference | Menggunakan model yang sudah terlatih (tanpa update parameter) |
| Autoregressive | Hasilkan satu token, tambah ke konteks, ulangi |
| Temperature | Kontrol kreativitas — rendah = konservatif, tinggi = random |
| Greedy | Pilih token terbaik (deterministik) |
| Sampling | Pilih secara probabilistik (bervariasi) |
| Repetition penalty | Kurangi pengulangan token |
| Context window | Batas panjang teks yang bisa diproses sekaligus |

---

> [Selanjutnya: Panduan Implementasi dari Nol →](./12-implementasi.md)
