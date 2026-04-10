# Bagian 9 — Optimizer: Adam

[← Sebelumnya](./08-backpropagation.md) | [Selanjutnya →](./10-training-pipeline.md)

---

## Gradient Sudah Ada — Lalu Apa?

Backpropagation memberi kita gradient untuk setiap parameter. Sekarang kita perlu menggunakan gradient itu untuk *memperbarui* parameter.

Cara paling sederhana: langsung kurangi parameter dengan gradientnya.

```
parameter_baru = parameter_lama − learning_rate × gradient
```

Ini disebut **Stochastic Gradient Descent (SGD)**. Sederhana, tapi penuh masalah.

---

## Masalah SGD Sederhana

### 1. Learning Rate Sulit Dipilih

```
Terlalu besar:  zig-zag, tidak konvergen
  loss:  5.0 → 4.5 → 6.2 → 3.1 → 7.8 → ...  (kacau!)

Terlalu kecil:  konvergen tapi sangat lambat
  loss:  5.0 → 4.999 → 4.998 → 4.997 → ...

Goldilocks:    stabil dan cepat
  loss:  5.0 → 4.2 → 3.5 → 2.9 → 2.4 → ...
```

### 2. Semua Parameter Diperlakukan Sama

Ada parameter yang gradientnya konsisten, ada yang sering berubah arah. SGD tidak membedakannya — semua dapat update yang sama besar.

### 3. Mudah Terjebak di Local Minimum

Tidak punya "momentum" untuk melewati tonjolan kecil menuju minimum yang lebih dalam.

---

## Adam: Solusi Adaptif

**Adam (Adaptive Moment Estimation)** menggabungkan dua ide cerdas untuk mengatasi semua masalah di atas.

---

## Ide 1: Momentum

Bayangkan bola menggelinding di lereng. Bola tidak hanya terpengaruh kemiringan saat ini — ia juga membawa inersia dari gerakan sebelumnya. Kalau sudah menggelinding ke kanan, perlu gaya yang cukup besar untuk membeloknya.

Momentum dalam optimizer bekerja sama: kita simpan rata-rata bergerak dari gradient-gradient sebelumnya.

```
m_t = β₁ × m_{t-1}  +  (1 − β₁) × g_t

  m_t   = estimasi momentum saat ini
  g_t   = gradient saat ini
  β₁    = decay rate (biasanya 0.9)
           → "ingat 90% dari masa lalu, tambah 10% dari gradient baru"
```

Efeknya:
- Kalau gradient selalu ke satu arah → momentum membangun, langkah semakin besar
- Kalau gradient zig-zag → momentum meredam osilasi, langkah lebih stabil

---

## Ide 2: Adaptive Learning Rate

Berbeda dengan SGD yang memakai satu learning rate untuk semua parameter, Adam menyesuaikan learning rate **per parameter**.

Caranya: lacak seberapa besar gradient parameter ini selama ini.

```
v_t = β₂ × v_{t-1}  +  (1 − β₂) × g_t²

  v_t  = estimasi varians gradient
  β₂   = decay rate (biasanya 0.999)
```

Lalu gunakan ini untuk menyesuaikan learning rate:

```
learning_rate_efektif = lr / (√v_t + ε)
```

**Logikanya:**
- Parameter dengan gradient besar dan konsisten → `v` besar → learning rate efektif kecil → langkah hati-hati
- Parameter dengan gradient kecil dan jarang diupdate → `v` kecil → learning rate efektif besar → langkah lebih berani

---

## Bias Correction: Koreksi Awal Training

Di langkah pertama, `m` dan `v` diinisialisasi ke 0. Ini menyebabkan estimasi yang terlalu kecil di awal.

**Bias correction** memperbaiki ini:

```
m̂_t = m_t / (1 − β₁^t)
v̂_t = v_t / (1 − β₂^t)
```

Di langkah 1: `1 − 0.9^1 = 0.1`, jadi `m̂ = m/0.1 = 10×m` — ini mengimbangi nilai m yang masih kecil.

Setelah ratusan langkah: `1 − 0.9^100 ≈ 1`, koreksi hampir tidak berpengaruh.

---

## Formula Update Adam Lengkap

```
# Untuk setiap parameter θ, di setiap langkah t:

# 1. Hitung momen
m_t = β₁ × m_{t-1} + (1 − β₁) × g_t
v_t = β₂ × v_{t-1} + (1 − β₂) × g_t²

# 2. Bias correction
m̂_t = m_t / (1 − β₁^t)
v̂_t = v_t / (1 − β₂^t)

# 3. Update
θ_t = θ_{t-1} − lr × m̂_t / (√v̂_t + ε)
```

---

## Pseudocode Implementasi

```
KELAS Adam:
  lr   = 0.0003
  β₁   = 0.9
  β₂   = 0.999
  ε    = 1e-8
  t    = 0

  # Per parameter: simpan m dan v
  m = {}
  v = {}

  FUNGSI langkah(semua_parameter):
    t = t + 1

    UNTUK setiap parameter θ:
      g = θ.gradient

      # Inisialisasi jika belum ada
      JIKA θ tidak ada di m:
        m[θ] = nol (shape sama dengan θ)
        v[θ] = nol (shape sama dengan θ)

      # Update momen
      m[θ] = β₁ × m[θ] + (1 − β₁) × g
      v[θ] = β₂ × v[θ] + (1 − β₂) × g²

      # Bias correction
      m_hat = m[θ] / (1 − β₁^t)
      v_hat = v[θ] / (1 − β₂^t)

      # Update parameter
      θ -= lr × m_hat / (√v_hat + ε)

      # Reset gradient untuk langkah berikutnya
      θ.gradient = nol
```

**Python Native** (murni Python, tanpa library):

```python
import math

class Adam:
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m     = {}   # momen pertama per parameter
        self.v     = {}   # momen kedua per parameter

    def langkah(self, semua_parameter):
        """
        semua_parameter: list of dict, tiap dict punya 'nilai' dan 'grad'
        nilai dan grad: list of list (matriks) atau list (vektor)
        """
        self.t += 1
        b1t = self.beta1 ** self.t
        b2t = self.beta2 ** self.t

        for i, param in enumerate(semua_parameter):
            nilai = param['nilai']
            grad  = param['grad']

            # Inisialisasi state jika belum ada
            if i not in self.m:
                self.m[i] = [[0.0]*len(baris) for baris in nilai]
                self.v[i] = [[0.0]*len(baris) for baris in nilai]

            # Update per elemen
            for r in range(len(nilai)):
                for c in range(len(nilai[r])):
                    g = grad[r][c]

                    self.m[i][r][c] = self.beta1 * self.m[i][r][c] + (1 - self.beta1) * g
                    self.v[i][r][c] = self.beta2 * self.v[i][r][c] + (1 - self.beta2) * g**2

                    m_hat = self.m[i][r][c] / (1 - b1t)
                    v_hat = self.v[i][r][c] / (1 - b2t)

                    nilai[r][c] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

            # Reset gradient
            param['grad'] = [[0.0]*len(baris) for baris in grad]
```

**Python + Library** (menggunakan PyTorch):

```python
import torch
import torch.optim as optim

# Contoh: buat model sederhana dan optimizer Adam
model = torch.nn.Linear(128, 128)

optimizer = optim.Adam(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8
)

# Satu langkah training
x      = torch.randn(4, 128)
target = torch.randn(4, 128)
loss   = ((model(x) - target)**2).mean()   # MSE sederhana

optimizer.zero_grad()   # reset gradient
loss.backward()          # hitung gradient via autograd
optimizer.step()         # update semua parameter

print(f"Loss: {loss.item():.4f}")

# Learning rate scheduler (warmup + cosine decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)
scheduler.step()  # panggil setelah setiap optimizer.step()
```

---

## Hyperparameter Adam

| Parameter | Nilai Default | Keterangan |
|-----------|---------------|------------|
| `lr` | 3e-4 (0.0003) | Learning rate — **yang paling sering perlu di-tune** |
| `β₁` | 0.9 | Decay momen pertama |
| `β₂` | 0.999 | Decay momen kedua |
| `ε` | 1e-8 | Stabilitas numerik (jangan sampai bagi 0) |

Nilai β₁, β₂, dan ε hampir tidak pernah perlu diubah. Fokus tuning pada `lr`.

---

## Learning Rate: Hyperparameter Paling Penting

```
lr = 1e-2    terlalu besar → training tidak stabil
lr = 1e-3    besar → bisa berhasil untuk dataset kecil
lr = 3e-4    sweet spot yang umum dipakai (GPT-3, LLaMA menggunakan ini)
lr = 1e-4    kecil → stabil tapi lambat
lr = 1e-5    terlalu kecil → training sangat lambat
```

---

## Learning Rate Scheduling (Bonus)

Untuk hasil lebih baik, banyak model modern menggunakan jadwal learning rate:

### Warmup + Cosine Decay

```
Phase 1 — Warmup (500–2000 langkah pertama):
  lr naik perlahan dari 0 ke lr_max
  → Mencegah update besar di awal saat gradient masih berantakan

Phase 2 — Cosine Decay:
  lr turun mengikuti kurva cosine dari lr_max ke lr_min
  → Perlambat langkah saat mendekati minimum

  lr(t) = lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(π × t/T))
```

Untuk model kecil yang kita bangun, **learning rate konstan sudah cukup** sebagai titik mulai.

---

## Mengapa Bukan SGD?

| | SGD | Adam |
|--|-----|------|
| Learning rate | Satu untuk semua | Adaptif per parameter |
| Momentum | Tidak ada (vanilla SGD) | Ada |
| Konvergensi | Lambat, sensitif LR | Cepat, lebih robust |
| Memori ekstra | Tidak ada | 2× ukuran parameter (m dan v) |

Untuk jaringan dalam seperti Transformer, Adam hampir selalu lebih baik dari SGD sederhana.

---

## Ringkasan

| Konsep | Penjelasan |
|--------|-----------|
| SGD | Update langsung: θ = θ − lr × grad |
| Momentum (m) | Rata-rata bergerak gradient — "inersia" |
| Adaptive LR (v) | Rata-rata kuadrat gradient — sesuaikan per parameter |
| Bias correction | Koreksi estimasi di awal training |
| Adam | Gabungan momentum + adaptive LR |
| lr = 3e-4 | Nilai default yang umumnya baik |

---

> [Selanjutnya: Pipeline Training Lengkap →](./10-training-pipeline.md)
