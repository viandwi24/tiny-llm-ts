# Bagian 8 — Backpropagation: Cara Model Belajar

[← Sebelumnya](./07-arsitektur-lengkap.md) | [Selanjutnya →](./09-optimizer.md)

---

## Inti dari Semua Machine Learning

Semua machine learning pada dasarnya satu siklus:

```
1. Coba    → buat prediksi
2. Ukur    → seberapa salah?
3. Perbaiki → sesuaikan parameter
4. Ulangi  → jutaan sampai miliaran kali
```

Tiga langkah ini disebut: **forward pass → loss → backward pass (backpropagation)**.

---

## Analogi: Belajar Melempar Bola Basket

Kamu baru belajar melempar bola ke ring:

1. **Lempar** — bola meluncur, meleset 50 cm ke kanan
2. **Ukur selisih** — "oke, terlalu ke kanan"
3. **Koreksi** — "lain kali arahkan lebih ke kiri"
4. **Lempar lagi** — lebih dekat! Ulangi terus.

Setiap kali melempar dan melihat hasilnya, otakmu membuat koreksi kecil. Setelah ribuan lemparan, akurasimu meningkat dramatis.

Model AI melakukan hal yang persis sama, tapi koreksinya bukan gerakan otot — melainkan angka-angka di dalam matriks (parameter).

---

## Step 1: Forward Pass

Beri model input, dapat prediksi. Sudah kita pelajari di bagian sebelumnya.

```
Input: "Ibukota Indonesia adalah ___"
Model prediksi: {"Jakarta": 30%, "Bandung": 5%, "Surabaya": 8%, ...}
```

---

## Step 2: Loss Function — Mengukur Kesalahan

**Loss** adalah satu angka yang meringkas seberapa buruk prediksi model. **Tujuan training: minimalkan loss.**

### Cross-Entropy Loss

Ini adalah fungsi loss standar untuk prediksi token.

Misalkan jawaban benar adalah "Jakarta" dan model memberikan probabilitas:

```
"Jakarta"  → 30%
"Bandung"  → 5%
"Surabaya" → 8%
...
```

```
Loss = −log(probabilitas jawaban benar)
     = −log(0.30)
     ≈ 1.20
```

### Mengapa Logaritma Negatif?

Grafik intuisinya:

```
Probabilitas jawaban benar → Loss

0.99  →  −log(0.99)  ≈ 0.01   (hampir benar → hukuman kecil)
0.50  →  −log(0.50)  ≈ 0.69   (ragu-ragu → hukuman sedang)
0.10  →  −log(0.10)  ≈ 2.30   (hampir salah → hukuman besar)
0.01  →  −log(0.01)  ≈ 4.61   (yakin tapi salah → hukuman sangat besar)
```

Loss yang baik mendorong model untuk tidak hanya *benar* tapi juga *yakin dengan jawabannya*.

### Loss untuk Seluruh Sequence

```
FUNGSI cross_entropy(logits, target_ids):
  total = 0
  UNTUK setiap posisi i:
    probs_i  = softmax(logits[i])
    loss_i   = −log(probs_i[target_ids[i]])
    total   += loss_i
  KEMBALIKAN total / panjang_sequence   (rata-rata)
```

**Python Native** (murni Python, tanpa library):

```python
import math

def softmax(logits):
    m = max(logits)  # stabilitasi numerik
    exp_vals = [math.exp(x - m) for x in logits]
    total = sum(exp_vals)
    return [v / total for v in exp_vals]

def cross_entropy(logits_per_posisi, target_ids):
    """
    logits_per_posisi: list of list — [seq_len, vocab_size]
    target_ids: list of int — [seq_len]
    """
    total = 0.0
    for i, logits in enumerate(logits_per_posisi):
        probs  = softmax(logits)
        target = target_ids[i]
        loss_i = -math.log(probs[target] + 1e-9)  # +eps cegah log(0)
        total += loss_i
    return total / len(target_ids)

# Contoh penggunaan
logits = [
    [2.0, 1.0, 5.0, 0.5],   # logits posisi 0
    [1.0, 4.0, 2.0, 0.3],   # logits posisi 1
]
targets = [2, 1]   # token yang benar untuk tiap posisi

loss = cross_entropy(logits, targets)
print(f"Loss: {loss:.4f}")  # ~0.08 — model sudah cukup yakin
```

**Python + Library** (menggunakan PyTorch):

```python
import torch
import torch.nn.functional as F

# PyTorch mengurus stabilitas numerik secara otomatis
logits  = torch.tensor([[2.0, 1.0, 5.0, 0.5],
                         [1.0, 4.0, 2.0, 0.3]])   # [seq_len=2, vocab_size=4]
targets = torch.tensor([2, 1])                      # token benar tiap posisi

loss = F.cross_entropy(logits, targets)
print(f"Loss: {loss.item():.4f}")

# Untuk training — backward otomatis tersedia
loss.backward()  # hitung gradient
```

---

## Step 3: Gradient — Petunjuk Perbaikan

Setelah tahu seberapa besar kesalahan (loss), kita perlu tahu: *parameter mana yang perlu diubah, dan ke arah mana?*

Jawabannya adalah **gradient** — turunan dari loss terhadap setiap parameter.

### Analogi: Turun Gunung dalam Kabut

Kamu berada di pegunungan berkabut tebal, ingin mencapai lembah (titik terendah). Kamu tidak bisa lihat keseluruhan peta, tapi bisa merasakan kemiringan tanah di kakimu.

**Strategi: selalu melangkah ke arah yang paling menurun.**

Gradient memberi tahu arah "naik". Kita melangkah ke arah *berlawanan* dengan gradient — itu arah "turun" menuju loss yang lebih kecil.

### Gradient Menunjukkan Dua Hal:
- **Tanda (+/−)**: Apakah menaikkan atau menurunkan parameter yang membuat loss mengecil
- **Besarnya**: Seberapa besar pengaruh parameter ini terhadap loss

---

## Backpropagation: Algoritma Menelusuri Gradient

**Backpropagation** adalah cara efisien menghitung gradient untuk semua parameter sekaligus.

Namanya "backward" karena gradient dihitung dari output ke input — kebalikan dari forward pass.

### Analogi: Detektif Menelusuri Jejak

Loss besar adalah "akibat". Backpropagation adalah detektif yang menelusuri ke belakang mencari "penyebab":

```
Loss besar
  ← disebabkan oleh output projection yang salah
    ← disebabkan oleh representasi di layer terakhir yang kurang tepat
      ← disebabkan oleh attention yang salah fokus
        ← disebabkan oleh embedding awal yang belum optimal
```

Setiap parameter mendapat "tagihan" — seberapa besar kontribusinya terhadap kesalahan.

---

## Chain Rule: Matematika di Balik Backprop

**Chain rule** (aturan rantai) dari kalkulus:

```
Jika y = f(g(x)), maka:
dy/dx = (dy/dg) × (dg/dx)
```

Dalam bahasa sederhana: gradient bisa diteruskan mundur melewati rantai fungsi dengan cara dikalikan di tiap tahap.

### Contoh dalam Model

```
Forward:
  x → embedding → block1 → block2 → projection → loss

Backward (gradient mengalir dari kanan ke kiri):
  ∂loss/∂projection  ← hitung dari loss
  ∂loss/∂block2      ← hitung menggunakan hasil sebelumnya × Jacobian block2
  ∂loss/∂block1      ← hitung menggunakan hasil sebelumnya × Jacobian block1
  ∂loss/∂embedding   ← hitung terakhir
```

Semua gradient akhirnya dihitung tanpa perlu melakukan forward pass berkali-kali.

---

## Gradient melalui Softmax + Cross-Entropy

Ada persamaan analitik yang elegan untuk gradient loss terhadap logits:

```
Jika target adalah token ke-t:
  grad_logit[i] = probs[i] − 1  (jika i == t)
  grad_logit[i] = probs[i]      (jika i ≠ t)
```

Atau lebih singkat:
```
grad_logit = probs − one_hot(target)
```

`one_hot(t)` adalah vektor nol kecuali di posisi t yang bernilai 1.

**Contoh:**
```
logits:  [2.0, 1.0, 5.0, 0.5]
target:  token ke-2

probs:   [0.03, 0.01, 0.93, 0.01]
one_hot: [   0,    0,    1,    0]

grad:    [0.03, 0.01, -0.07, 0.01]   (probs − one_hot)
                       ↑
               token yang benar dapat grad negatif
               → naikkan skor token ini
```

---

## Gradient melalui Linear Layer

```
Forward:  output = input × W + b

Diberikan grad_output, hitung:
  grad_input = grad_output × W^T          ← diteruskan ke layer sebelumnya
  grad_W     = input^T × grad_output      ← untuk update W
  grad_b     = sum(grad_output, per baris) ← untuk update b
```

---

## Pseudocode Training Step Lengkap

```
FUNGSI satu_langkah_training(model, input_ids, target_ids, optimizer):

  # ─── FORWARD PASS ───
  logits = model.forward(input_ids)         [seq_len, vocab_size]

  # ─── HITUNG LOSS ───
  loss = cross_entropy(logits, target_ids)

  # ─── BACKWARD PASS ───
  # Gradient dari loss ke logits (analitik)
  probs       = softmax(logits)
  grad_logits = probs − one_hot(target_ids)
  grad_logits /= seq_len                    (normalisasi)

  # Gradient mengalir mundur melalui model
  model.backward(grad_logits)
  # (ini secara otomatis mengisi .grad di setiap parameter)

  # ─── UPDATE PARAMETER ───
  optimizer.step(model.semua_parameter())

  KEMBALIKAN loss
```

**Python Native** (murni Python, tanpa library):

```python
import math

def satu_langkah_training(model, input_ids, target_ids, optimizer):
    # ─── FORWARD PASS ───
    logits = model.forward(input_ids)   # [seq_len, vocab_size]

    # ─── HITUNG LOSS ───
    loss = cross_entropy(logits, target_ids)

    # ─── HITUNG GRADIENT LOGITS (analitik) ───
    seq_len    = len(target_ids)
    vocab_size = len(logits[0])
    grad_logits = []
    for i in range(seq_len):
        probs = softmax(logits[i])
        grad  = [p for p in probs]   # copy
        grad[target_ids[i]] -= 1.0   # kurangi 1 pada posisi target
        grad = [g / seq_len for g in grad]   # normalisasi
        grad_logits.append(grad)

    # ─── BACKWARD PASS ───
    model.backward(grad_logits)

    # ─── UPDATE PARAMETER ───
    optimizer.langkah(model.semua_parameter())

    return loss
```

**Python + Library** (menggunakan PyTorch):

```python
import torch
import torch.nn.functional as F

def satu_langkah_training(model, input_ids, target_ids, optimizer):
    # ─── FORWARD PASS ───
    logits = model(input_ids)                  # [batch, seq_len, vocab_size]

    # ─── HITUNG LOSS ───
    # Reshape untuk F.cross_entropy: [batch*seq, vocab] dan [batch*seq]
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B*T, V), target_ids.view(B*T))

    # ─── BACKWARD PASS ───
    optimizer.zero_grad()  # reset gradient dari langkah sebelumnya
    loss.backward()         # PyTorch autograd hitung semua gradient

    # ─── GRADIENT CLIPPING (opsional tapi disarankan) ───
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # ─── UPDATE PARAMETER ───
    optimizer.step()

    return loss.item()
```

---

## Visualisasi Aliran Gradient

```
                        FORWARD (→)
Token IDs → Embed → Block1 → Block2 → Block3 → Block4 → Proj → Loss
                                                                  │
                        BACKWARD (←)                              │
Token IDs ← Embed ← Block1 ← Block2 ← Block3 ← Block4 ← Proj ←──┘
    │           │       │        │        │        │        │
  Update     Update  Update   Update   Update   Update   Update
 params     params  params   params   params   params   params
```

Setiap parameter mendapat gradient-nya sendiri, sesuai kontribusinya terhadap loss.

---

## Masalah Umum: Vanishing & Exploding Gradient

### Vanishing Gradient
Gradient mengecil saat mengalir mundur melewati banyak layer. Layer awal hampir tidak mendapat sinyal.

**Solusi:** Residual connection (memastikan ada jalur langsung), layer normalization, dan inisialisasi bobot yang tepat.

### Exploding Gradient
Gradient meledak menjadi sangat besar, membuat parameter berubah drastis dan training tidak stabil.

**Solusi:** Gradient clipping — potong gradient kalau melebihi threshold tertentu:
```
JIKA magnitude(gradient) > threshold:
  gradient = gradient × (threshold / magnitude(gradient))
```

---

## Ringkasan

| Konsep | Penjelasan |
|--------|-----------|
| Loss | Ukuran kesalahan model (ingin diminimalkan) |
| Cross-entropy | Fungsi loss standar untuk prediksi token |
| Gradient | Turunan — petunjuk arah dan besar perubahan |
| Backpropagation | Algoritma menghitung gradient semua parameter |
| Chain rule | Matematika yang memungkinkan gradient mengalir mundur |

---

> [Selanjutnya: Optimizer — Adam →](./09-optimizer.md)
