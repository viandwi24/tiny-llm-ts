# Training Loop — Dari Logits ke Loss ke Predicted Token

> **Goal MVP:** Dari arsitektur Transformer yang sudah ada, kita ingin melihat dua hal:
> 1. **Predicted token** — token apa yang diprediksi model di setiap posisi
> 2. **Loss value** — seberapa salah prediksi model (semakin kecil = semakin baik)

---

## Gambaran Besar: Apa yang Sudah Ada vs Apa yang Perlu Dibuat

```
[DataLoader] → inputIds, targetIds
      ↓
[model.forward(inputIds)]          ← SUDAH ADA
      ↓
logits: number[][]                 ← shape [seqLen, vocabSize]
      ↓
┌─────────────────────────────────────────────────────────┐
│  INI YANG PERLU DIBUAT:                                  │
│                                                          │
│  logits  →  [Loss Function]  →  loss (angka tunggal)    │
│  logits  →  [argmax]         →  predicted token IDs     │
│  loss    →  [Backprop]       →  gradients               │
│  grads   →  [Optimizer]      →  update weights          │
└─────────────────────────────────────────────────────────┘
```

---

## Bagian 1: Memahami Output `model.forward()`

### Apa itu Logits?

`model.forward(inputIds)` mengembalikan array 2D dengan shape `[seqLen, vocabSize]`.

Setiap baris adalah **distribusi skor mentah** untuk posisi token tersebut. Skor ini belum dinormalisasi (bisa negatif, bisa lebih dari 1, tidak harus jumlahnya 1).

**Analogi:** Bayangkan ada 8000 kandidat jawaban (vocab size = 8000). Model memberikan "nilai ujian" ke tiap kandidat. Nilai ini belum berupa persentase — bisa nilai 150 atau nilai -30. Logits adalah nilai mentah tersebut.

```
inputIds = [42, 17, 305]   ← 3 token input, seqLen = 3

logits output:
  posisi 0: [0.1, -2.3, 0.8, ..., 1.2]   ← 8000 angka, prediksi token ke-1
  posisi 1: [-0.5, 0.3, 1.1, ..., -0.7]  ← prediksi token ke-2
  posisi 2: [2.1, 0.4, -1.2, ..., 0.9]   ← prediksi token ke-3
```

Hubungan dengan targetIds: logits di **posisi i** seharusnya memprediksi `targetIds[i]`, yaitu token yang muncul **setelah** inputIds[i] dalam teks asli.

---

## Bagian 2: Predicted Token — `argmax`

### Konsep

Dari logits per posisi, token yang "paling mungkin" menurut model adalah index dengan nilai tertinggi. Operasi ini disebut **argmax**.

**Analogi:** Dari 8000 kandidat yang sudah dinilai, kita pilih yang nilainya paling tinggi. Itu prediksi model.

### Fungsi yang Perlu Dibuat

**Nama:** `argmax(arr: number[]): number`

**Input:** array 1D berisi logits satu posisi, panjang = vocabSize

**Output:** index (token ID) dengan nilai tertinggi

**Pseudocode:**
```
function argmax(arr):
  maxVal = -Infinity
  maxIdx = 0
  for i from 0 to arr.length:
    if arr[i] > maxVal:
      maxVal = arr[i]
      maxIdx = i
  return maxIdx
```

**Untuk mendapat predicted token per posisi:**
```
for each position i in logits:
  predictedTokenId = argmax(logits[i])
  predictedToken = vocab[predictedTokenId]   ← lookup ke vocab.json
```

**Catatan penting:** argmax hanya untuk *melihat* prediksi. Untuk training, yang dipakai adalah distribusi probabilitas penuh (softmax), bukan argmax.

---

## Bagian 3: Cross-Entropy Loss

### Konsep

Loss mengukur "seberapa jauh" prediksi model dari jawaban yang benar. Untuk language model, kita pakai **cross-entropy loss**.

**Analogi:** Bayangkan lomba tebak kata. Jawaban benar adalah kata "kucing" (token ID = 305). Model memberikan skor ke semua 8000 kata. Cross-entropy menghukum model **sebanding dengan seberapa kecil skor yang diberikan ke kata "kucing"**. Kalau model yakin "kucing" tapi salah — hukuman ringan. Kalau model sama sekali tidak percaya "kucing" — hukuman berat.

### Langkah-langkah

**Step 1 — Softmax: ubah logits jadi probabilitas**

Softmax mengubah angka-angka logits menjadi distribusi probabilitas (0–1, jumlah = 1).

```
softmax(logits[i]) = exp(logits[i][j]) / sum(exp(logits[i]))  untuk setiap j
```

Fungsi `softmax` sudah ada di `src/neural-network/activation.ts`, bisa langsung dipakai.

**Step 2 — Negative Log Likelihood (NLL)**

Setelah dapat probabilitas, loss untuk satu token adalah:

```
loss_i = -log(probabilities[correctTokenId])
```

**Kenapa negatif log?** Probabilitas yang benar ada di rentang (0, 1]. Log dari angka kecil mendekati 0 akan sangat negatif (misal log(0.001) ≈ -6.9). Negatifnya dibalik sehingga loss jadi positif — semakin salah, loss semakin besar.

```
prob = 0.9  →  -log(0.9) ≈ 0.10  (hampir benar, loss kecil)
prob = 0.5  →  -log(0.5) ≈ 0.69  (ragu-ragu)
prob = 0.1  →  -log(0.1) ≈ 2.30  (sangat salah, loss besar)
prob = 0.01 →  -log(0.01) ≈ 4.60 (model tidak tahu sama sekali)
```

**Step 3 — Rata-rata atas semua posisi**

```
loss = mean(loss_0, loss_1, ..., loss_{seqLen-1})
```

### Fungsi yang Perlu Dibuat

**Nama:** `crossEntropyLoss(logits: number[][], targetIds: number[]): number`

**Input:**
- `logits` — output dari `model.forward()`, shape `[seqLen, vocabSize]`
- `targetIds` — token ID yang seharusnya diprediksi, length = `seqLen`

**Output:** satu angka (scalar) = rata-rata loss

**Pseudocode:**
```
function crossEntropyLoss(logits, targetIds):
  totalLoss = 0

  for i from 0 to seqLen:
    probs = softmax(logits[i])          ← ubah logits posisi i ke probabilitas
    correctId = targetIds[i]            ← index token yang benar
    correctProb = probs[correctId]      ← ambil prob token yang benar
    loss_i = -log(correctProb)          ← negative log likelihood
    totalLoss += loss_i

  return totalLoss / seqLen             ← rata-rata
```

**Catatan numerik:** Kalau `correctProb` sangat mendekati 0, `log(0)` akan jadi -Infinity. Untuk keamanan, clamp nilainya:
```
correctProb = max(correctProb, 1e-7)   ← hindari log(0)
```

### Apa Nilai Loss yang "Normal"?

Pada awal training, bobot model masih random. Distribusi probabilitas hampir seragam. Dengan vocabSize = 8000:

```
prob seragam = 1 / 8000 ≈ 0.000125
loss awal = -log(0.000125) ≈ 8.99
```

Jadi **loss sekitar 9 di awal training adalah normal**. Kalau loss turun ke < 5, model sudah mulai "belajar sesuatu".

---

## Bagian 4: Backpropagation — Dasar Konsep

> Bagian ini lebih konseptual. Untuk MVP "melihat loss dan predicted token", bagian ini **opsional** — bisa skip dulu, loss sudah bisa dihitung tanpa backprop. Backprop dibutuhkan agar loss **turun** dari waktu ke waktu.

### Konsep: Chain Rule

Backpropagation adalah cara menghitung "seberapa besar pengaruh setiap bobot terhadap loss". Ini menggunakan **chain rule** dari kalkulus:

```
∂Loss/∂W = ∂Loss/∂output × ∂output/∂W
```

**Analogi:** Bayangkan memasak nasi. Loss = "nasi terlalu asin". Kita ingin tahu: "siapa yang paling bertanggung jawab?" — apakah garam yang terlalu banyak? api yang terlalu besar? air yang kurang? Chain rule menelusuri kontribusi setiap bahan/tindakan ke hasil akhir.

### Aliran Gradient (Backward Pass)

Forward pass mengalir dari kiri ke kanan:
```
tokens → Embedding → Block(1..N) → Projection → Logits → Loss
```

Backward pass mengalir dari kanan ke kiri:
```
Loss → ∂Loss/∂Logits → ∂Loss/∂Block → ... → ∂Loss/∂Embedding
```

Di setiap langkah, kita menghitung dua hal:
1. **Gradient terhadap bobot layer ini** (untuk update bobot)
2. **Gradient terhadap input layer ini** (untuk diteruskan ke layer sebelumnya)

### Gradient Tiap Komponen

#### 4a. Gradient Cross-Entropy + Softmax (digabung)

Ini yang paling penting karena ini titik awal backward pass.

Rumus gabungan softmax + cross-entropy gradient sangat simpel:

```
∂Loss/∂logits[i][j] = probs[i][j] - 1  jika j == targetIds[i]
∂Loss/∂logits[i][j] = probs[i][j]      jika j != targetIds[i]
```

Dibagi seqLen karena loss adalah rata-rata:
```
dLogits[i][j] = (probs[i][j] - indicator(j == targetIds[i])) / seqLen
```

**Analogi:** Gradient ini bilang ke model: "kamu terlalu percaya diri ke token yang salah (kurangi probabilitasnya), dan terlalu tidak percaya ke token yang benar (tambah probabilitasnya)."

**Pseudocode:**
```
function softmaxCrossEntropyGrad(logits, targetIds):
  dLogits = copy of zeros with shape [seqLen, vocabSize]

  for i from 0 to seqLen:
    probs = softmax(logits[i])
    for j from 0 to vocabSize:
      dLogits[i][j] = probs[j]
    dLogits[i][targetIds[i]] -= 1   ← kurangi 1 di posisi yang benar

  divide all dLogits by seqLen
  return dLogits
```

#### 4b. Gradient Linear Layer

`LinearLayer.forward(x)` melakukan: `output = x · W + b`

Jika kita tahu `dOutput` (gradient dari layer setelahnya):

```
dW = xᵀ · dOutput         ← gradient terhadap weights
db = sum(dOutput, axis=0)  ← gradient terhadap bias
dX = dOutput · Wᵀ          ← gradient yang diteruskan ke layer sebelumnya
```

**Pseudocode:**
```
function linearBackward(dOutput, x, W):
  dW = transpose(x) · dOutput
  db = sum of each column in dOutput
  dX = dOutput · transpose(W)
  return { dW, db, dX }
```

#### 4c. Gradient LayerNorm

`LayerNorm` melakukan normalisasi + scale: `output = γ · x_norm + β`

```
dGamma = sum(dOutput * x_normalized, axis=0)
dBeta  = sum(dOutput, axis=0)
dX     = ... (lebih kompleks, melibatkan normalisasi ulang variance)
```

Rumus lengkap `dX` untuk LayerNorm:

```
dX_norm = dOutput * gamma
dVariance = sum(dX_norm * (x - mean) * -0.5 * (variance + eps)^(-1.5))
dMean = sum(dX_norm * -1/sqrt(variance + eps)) + dVariance * mean(-2*(x - mean))
dX = dX_norm / sqrt(variance + eps) + dVariance * 2*(x - mean)/N + dMean/N
```

**Pseudocode (ringkas):**
```
function layerNormBackward(dOutput, x, gamma, beta, mean, variance, eps):
  N = length of x per row
  x_norm = (x - mean) / sqrt(variance + eps)

  dGamma = sum over rows of (dOutput * x_norm)
  dBeta  = sum over rows of dOutput

  dX_norm = dOutput * gamma
  dX = ... (turunan dari normalisasi, 3 komponen)

  return { dGamma, dBeta, dX }
```

#### 4d. Gradient Attention (gambaran besar)

Ini yang paling kompleks. Alurnya:

```
dOutput
  → backward melalui wo (Linear)
  → dConcat
  → split per head
  → untuk tiap head:
      → backward melalui (attnWeights · Vh)
      → backward melalui softmax(masked scores)
      → backward melalui scores = Qh · Khᵀ / scale
      → dapat dQh, dKh, dVh
  → gabung dQ, dK, dV dari semua head
  → backward melalui wq, wk, wv (Linear)
  → dapat dX
```

#### 4e. Gradient Embedding

Embedding berfungsi sebagai "lookup table". Gradientnya sesederhana:

```
for i, tokenId in enumerate(inputIds):
  tokenWeights[tokenId] += dInput[i]
  positionWeights[i]    += dInput[i]
```

Karena embedding hanya *mengambil* baris tertentu, gradient-nya hanya *menambahkan* ke baris yang bersangkutan.

---

## Bagian 5: Optimizer — Adam

### Konsep

Setelah dapat gradient, kita perlu **update bobot** agar loss turun. Cara paling simpel adalah Stochastic Gradient Descent (SGD):

```
W = W - learningRate × gradient
```

Tapi ini tidak stabil. **Adam** (Adaptive Moment Estimation) lebih baik karena:
1. Menyimpan "momentum" (rata-rata bergerak dari gradient)
2. Menyimpan "kecepatan adaptif" (rata-rata bergerak dari kuadrat gradient)
3. Koreksi bias di awal training

**Analogi:** SGD itu seperti bola yang digelindingkan menuruni bukit langsung. Adam seperti bola dengan roda + suspensi — lebih mulus, tidak terpental saat menemui medan berbatu.

### State yang Dibutuhkan

Untuk setiap parameter `W`, Adam butuh menyimpan:
- `m` (moment pertama) — rata-rata gradient, init = 0
- `v` (moment kedua) — rata-rata kuadrat gradient, init = 0
- `t` (timestep) — counter berapa kali sudah update, init = 0

### Pseudocode Adam

```
hyperparameter: lr = 1e-4, beta1 = 0.9, beta2 = 0.999, eps = 1e-8

function adamUpdate(W, grad, m, v, t):
  t = t + 1
  m = beta1 * m + (1 - beta1) * grad          ← update momen pertama
  v = beta2 * v + (1 - beta2) * grad * grad    ← update momen kedua

  m_hat = m / (1 - beta1^t)                   ← koreksi bias
  v_hat = v / (1 - beta2^t)                   ← koreksi bias

  W = W - lr * m_hat / (sqrt(v_hat) + eps)    ← update bobot

  return W, m, v, t
```

### Cara Mengorganisir Parameter

Kumpulkan semua parameter model dalam satu struktur agar mudah di-update:

```
parameters = {
  "embedding.tokenWeights":     { W, m=0, v=0 },
  "embedding.positionWeights":  { W, m=0, v=0 },
  "block[0].wq.weights":        { W, m=0, v=0 },
  "block[0].wq.bias":           { W, m=0, v=0 },
  ...
}
```

Setelah backward pass, loop semua parameter dan panggil `adamUpdate`.

---

## Bagian 6: Training Loop Lengkap

### Alur Satu Step

```
step:
  1. ambil batch: { inputIds, targetIds } dari DataLoader
  2. forward:  logits = model.forward(inputIds[0])   ← untuk 1 sequence dulu
  3. loss:     loss = crossEntropyLoss(logits, targetIds[0])
  4. predict:  predictedIds = logits.map(row => argmax(row))
  5. backward: gradients = backward(logits, targetIds[0])
  6. update:   optimizer.step(gradients)
  7. log:      print step, loss, sample predicted tokens
```

### Pseudocode Training Loop

```
function trainLoop(model, dataLoader, optimizer, maxSteps):
  step = 0

  while step < maxSteps:
    batch = dataLoader.nextBatch()
    if batch is null:
      dataLoader.reset()         ← mulai dari awal data
      continue

    // untuk MVP: proses satu sequence dulu (batchSize = 1)
    inputIds  = batch.inputIds[0]    ← array of token IDs, length = seqLen
    targetIds = batch.targetIds[0]   ← array of token IDs, length = seqLen

    // forward pass
    logits = model.forward(inputIds)

    // hitung loss
    loss = crossEntropyLoss(logits, targetIds)

    // predicted token (argmax, tanpa perlu gradient)
    predictedTokenIds = map each row of logits to argmax
    predictedToken = vocab[predictedTokenIds[last]]    ← token terakhir paling relevan

    // backward pass
    dLogits = softmaxCrossEntropyGrad(logits, targetIds)
    model.backward(dLogits)    ← compute semua gradients

    // update weights
    optimizer.step()           ← Adam update semua parameter

    // zero gradients (penting! reset sebelum step berikutnya)
    model.zeroGrad()

    // logging setiap N step
    if step % 10 == 0:
      print `step ${step} | loss: ${loss.toFixed(4)} | predicted: "${predictedToken}"`

    step++
```

### Reset DataLoader

DataLoader perlu method `reset()` untuk kembali ke awal data saat sudah habis:

```
function reset():
  this.currentIndex = 0
```

---

## Bagian 7: Struktur File yang Disarankan

Berdasarkan kode yang sudah ada, ini struktur penambahan yang masuk akal:

```
src/train/
  index.ts          ← sudah ada, perlu diisi runTrain()
  loss.ts           ← crossEntropyLoss, softmaxCrossEntropyGrad, argmax
  optimizer.ts      ← Adam optimizer

src/transformer/
  index.ts          ← tambahkan method backward()
  attention.ts      ← tambahkan attentionBackward()
  feedforward.ts    ← tambahkan feedforwardBackward()
  embedding.ts      ← tambahkan embeddingBackward()

src/neural-network/
  layer.ts          ← tambahkan linearBackward(), layerNormBackward()
```

---

## Bagian 8: Urutan Implementasi yang Disarankan

Untuk mencapai goal MVP (melihat loss dan predicted token), urutannya:

### Fase 1 — Bisa Melihat Loss dan Predicted Token (tanpa belajar)
1. Buat `argmax(arr)` di `loss.ts`
2. Buat `crossEntropyLoss(logits, targetIds)` di `loss.ts`
3. Panggil keduanya di dalam `runTrain()` setelah `model.forward()`
4. Print loss dan predicted token

> **Checkpoint:** Seharusnya loss ≈ 9.0 dan predicted token random/tidak bermakna. Ini benar.

### Fase 2 — Model Bisa Belajar (loss turun)
5. Buat `softmaxCrossEntropyGrad(logits, targetIds)` di `loss.ts`
6. Tambahkan `linearBackward()` ke `LinearLayer` di `layer.ts`
7. Tambahkan `layerNormBackward()` ke `LayerNormalization` di `layer.ts`
8. Tambahkan backward ke `Embedding` di `embedding.ts`
9. Tambahkan backward ke `FeedForward` di `feedforward.ts`
10. Tambahkan backward ke `MultiHeadAttention` di `attention.ts` ← paling kompleks
11. Sambungkan semua di `Transformer.backward()` di `index.ts`
12. Buat `AdamOptimizer` di `optimizer.ts`
13. Integrasikan ke training loop di `runTrain()`

> **Checkpoint:** Loss seharusnya turun secara konsisten dari ~9 menuju < 6 dalam ratusan step.

---

## Ringkasan Rumus Penting

| Komponen | Rumus Forward | Rumus Backward |
|---|---|---|
| Softmax | `σ(x)_i = exp(x_i) / Σexp(x_j)` | `(σ - 1_target) / seqLen` |
| Cross-Entropy | `-log(p_correct)` | lihat baris atas (digabung) |
| Linear | `xW + b` | `dW=xᵀdO, db=Σ(dO), dX=dO·Wᵀ` |
| LayerNorm | `γ·(x-μ)/σ + β` | 3 komponen via chain rule |
| Embedding | `lookup(id)` | `accumulate at id` |
| Adam | — | `W -= lr * m̂ / (√v̂ + ε)` |