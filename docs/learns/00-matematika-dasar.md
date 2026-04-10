# Bagian 0 — Matematika & Konsep Dasar

[← Index](./index.md) | [Selanjutnya →](./01-apa-itu-llm.md)

---

> Baca sepintas dulu. Kamu tidak harus hapal semuanya — cukup tahu bahwa halaman ini ada, dan kembali ke sini kapanpun menemukan istilah yang asing.

---

## 1. Skalar, Vektor, Matriks

Tiga bentuk data yang akan terus muncul.

**Skalar** — satu angka saja.
```
x = 5
x = -3.14
```

**Vektor** — daftar angka berurutan.
```
v = [1.2, -0.5, 3.7, 0.9]
```
Bayangkan panah dalam ruang, atau daftar nilai. Panjangnya disebut **dimensi**. Contoh di atas: 4 dimensi.

**Matriks** — tabel angka, punya baris dan kolom.
```
M = [[1, 2, 3],
     [4, 5, 6]]
```
Shape-nya ditulis sebagai `baris × kolom`. Contoh di atas: shape **2×3**.

---

## 2. Dot Product

Kalikan tiap pasangan elemen, lalu jumlahkan hasilnya.

```
a = [1, 2, 3]
b = [4, 5, 6]

a · b = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

**Maknanya:** Dot product besar artinya dua vektor "searah" (mirip satu sama lain). Dot product kecil atau negatif artinya berbeda arah. Ini adalah cara model mengukur *kemiripan* — dan sangat penting di mekanisme attention.

---

## 3. Perkalian Matriks

Berbeda dari perkalian biasa. Setiap elemen hasil dihitung dari dot product satu baris dengan satu kolom.

```
A = [[1, 2],     B = [[5, 6],
     [3, 4]]          [7, 8]]

Elemen hasil (0,0) = baris-0 A · kolom-0 B = [1,2]·[5,7] = 1×5 + 2×7 = 19
Elemen hasil (0,1) = baris-0 A · kolom-1 B = [1,2]·[6,8] = 22
Elemen hasil (1,0) = baris-1 A · kolom-0 B = [3,4]·[5,7] = 43
Elemen hasil (1,1) = baris-1 A · kolom-1 B = [3,4]·[6,8] = 50

Hasil: [[19, 22],
        [43, 50]]
```

**Aturan shape yang wajib diingat:**
```
[m × k]  ×  [k × n]  =  [m × n]
   A             B         Hasil

Kolom A HARUS sama dengan baris B.
```

Hampir semua operasi dalam model (embedding lookup, attention, FFN) adalah perkalian matriks.

---

## 4. Transpose

Tukar baris dan kolom. Notasi: `A^T` atau `A.T`.

```
A = [[1, 2, 3],      A^T = [[1, 4],
     [4, 5, 6]]             [2, 5],
                             [3, 6]]
Shape 2×3 → Shape 3×2
```

---

## 5. Softmax — Ubah ke Probabilitas

Mengubah sekumpulan angka biasa (bisa negatif, bisa besar) menjadi probabilitas yang jumlahnya tepat 1.0.

```
Input: [2.0, 1.0, 0.1]

Langkah 1: hitung e^x untuk tiap elemen
  e^2.0 = 7.39
  e^1.0 = 2.72
  e^0.1 = 1.11
  Total  = 11.22

Langkah 2: bagi tiap elemen dengan total
  7.39 / 11.22 = 0.658
  2.72 / 11.22 = 0.242
  1.11 / 11.22 = 0.099

Output: [0.658, 0.242, 0.099]   jumlah = 1.0 ✓
```

Angka terbesar tetap punya probabilitas terbesar, tapi sekarang dalam bentuk persentase.

---

## 6. Log — Ukur Ketidakpastian

```
-log(0.99) ≈ 0.01   → model hampir pasti benar → hukuman kecil
-log(0.50) ≈ 0.69   → model ragu-ragu          → hukuman sedang
-log(0.10) ≈ 2.30   → model hampir pasti salah → hukuman besar
-log(0.01) ≈ 4.61   → model sangat yakin tapi salah → hukuman sangat besar
```

Dipakai di fungsi loss untuk menghukum prediksi yang salah dengan yakin.

---

## 7. Turunan (Gradient) — Petunjuk Arah

Turunan mengukur: "kalau aku ubah x sedikit, f(x) berubah seberapa dan ke arah mana?"

Analogi: kamu berdiri di lereng gunung. Turunan = kemiringan tanah di bawah kakimu.
- Kemiringan ke atas (+) → mundur untuk turun
- Kemiringan ke bawah (−) → maju untuk turun
- Rata (0) → kamu sudah di titik minimum (atau maximum)

Dalam training model, kita ingin *mengecilkan loss*. Gradient memberi tahu ke arah mana harus mengubah setiap parameter agar loss mengecil.

---

## 8. Normalisasi

Mengubah data agar rata-ratanya 0 dan sebarannya konsisten:

```
data = [2, 4, 6, 8]

mean = (2+4+6+8)/4 = 5
std  = √(((2−5)²+(4−5)²+(6−5)²+(8−5)²)/4) = √5 ≈ 2.24

normalized = (data − mean) / std
           = [(2−5)/2.24, (4−5)/2.24, (6−5)/2.24, (8−5)/2.24]
           = [−1.34, −0.45, 0.45, 1.34]
```

Dipakai di Layer Normalization agar nilai-nilai dalam model tidak meledak jadi terlalu besar atau mengecil jadi terlalu kecil.

---

## Ringkasan Notasi

| Simbol | Artinya |
|--------|---------|
| `[N, D]` | Matriks/tensor dengan N baris dan D kolom |
| `×` | Perkalian matriks |
| `·` | Dot product |
| `^T` | Transpose |
| `∑` | Penjumlahan |
| `∂L/∂w` | Gradient loss L terhadap parameter w |
| `argmax` | Posisi elemen terbesar |
| `exp(x)` / `e^x` | Eksponensial (e ≈ 2.718) |
| `log(x)` | Logaritma natural |
| `√` | Akar kuadrat |

---

> [Selanjutnya: Apa Itu Language Model? →](./01-apa-itu-llm.md)
