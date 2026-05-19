# Klasifikasi Rock Paper Scissors dengan CNN

Repositori ini merupakan implementasi **Convolutional Neural Network (CNN)** menggunakan TensorFlow/Keras untuk mengklasifikasikan gambar gestur tangan Rock-Paper-Scissors (Batu-Gunting-Kertas).

> **Praktikum Kecerdasan Buatan — Pertemuan 8**  
> Nama: Latifanika Nurafwi  
> NIM : H1D024099 
> Repositori: `https://github.com/LATIFANIKA-NURAFWI1/H1D024099-PraktikumKB-Pertemuan8.git` 
---

## Deskripsi

Pada praktikum ini, CNN digunakan untuk mengklasifikasikan gambar tangan ke dalam salah satu dari **tiga kelas**:
- **Rock** (Batu)
- **Paper** (Kertas)
- **Scissors** (Gunting)

---

## Persyaratan (Dependencies)

Instal semua dependensi dengan perintah:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

| Library | Versi Minimum | Kegunaan |
|---|---|---|
| TensorFlow | 2.x | Framework deep learning (CNN) |
| NumPy | 1.21 | Operasi array dan numerik |
| Pandas | 1.3 | Manipulasi data |
| Matplotlib | 3.4 | Visualisasi grafik dan gambar |
| Seaborn | 0.11 | Visualisasi confusion matrix |
| Scikit-learn | 0.24 | Metrik evaluasi klasifikasi |

---

## Dataset

Dataset yang digunakan adalah **Rock Paper Scissors Images** dari Kaggle.

**Unduh dataset:**
```
https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors
```

**Struktur folder setelah diekstrak:**
```
rockpaperscissors/
├── paper/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── rock/
│   ├── image1.png
│   └── ...
└── scissors/
    ├── image1.png
    └── ...
```

Total: **2.188 gambar** (726 Rock, 710 Paper, 752 Scissors), ukuran 300×200 piksel, latar belakang hijau.

---

## Cara Menjalankan Program

### Di Google Colab
```python
# 1. Upload dataset ke Colab atau mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Sesuaikan dataset_path di main.py:
dataset_path = "/content/drive/MyDrive/rockpaperscissors"

# 3. Jalankan semua cell
```

---

## Arsitektur CNN

```
Input (150 × 150 × 3)
        │
┌───────▼────────┐
│ Conv2D(32, 3×3)│  ← Mendeteksi fitur dasar: tepi, garis
│    ReLU        │
└───────┬────────┘
        │
┌───────▼────────┐
│ MaxPooling(2×2)│  ← Reduksi dimensi spasial
└───────┬────────┘
        │
┌───────▼────────┐
│ Conv2D(64, 3×3)│  ← Mendeteksi fitur menengah: tekstur
│    ReLU        │
└───────┬────────┘
        │
┌───────▼─────────┐
│ MaxPooling(2×2) │
└───────┬─────────┘
        │
┌────────▼─────────┐
│ Conv2D(128, 3×3) │  ← Mendeteksi fitur kompleks: bentuk jari
│    ReLU          │
└────────┬─────────┘
         │
┌────────▼─────────┐
│ MaxPooling(2×2)  │
└────────┬─────────┘
         │
┌────────▼─────────┐
│    Flatten       │  ← Ratakan ke vektor 1D
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Dense(512, ReLU)│  ← Fully Connected Layer
└────────┬─────────┘
         │
┌────────▼──────────┐
│ Dense(3, Softmax) │  ← Output: probabilitas 3 kelas
└───────────────────┘
```

**Kompilasi model:**
- **Loss:** `categorical_crossentropy`
- **Optimizer:** `Adam`
- **Metrik:** `accuracy`

---

## Contoh Output

### Akurasi & Loss Training
```
Epoch 1/10 — loss: 1.0821 — accuracy: 0.4123 — val_loss: 0.9134 — val_accuracy: 0.5412
Epoch 5/10 — loss: 0.3241 — accuracy: 0.8756 — val_loss: 0.2987 — val_accuracy: 0.8921
Epoch 10/10 — loss: 0.0834 — accuracy: 0.9712 — val_loss: 0.1523 — val_accuracy: 0.9534
```

### Evaluasi Akhir
```
Validation Loss    : 0.1523
Validation Accuracy: 0.9534 (95.34%)
```

### File yang dihasilkan
| File | Keterangan |
|---|---|
| `training_history.png` | Grafik akurasi dan loss per epoch |
| `confusion_matrix.png` | Confusion matrix hasil evaluasi |
| `sample_predictions.png` | Contoh prediksi 10 gambar validasi |

---

## Prediksi Gambar Tunggal

```python
# Panggil fungsi ini setelah model dilatih
predict_single_image("./rockpaperscissors/rock/image1.png")

# Output:
# Hasil Prediksi: rock (Kepercayaan: 98.72%)
```

---

## Struktur Repositori

```
H1D024099-PraktikumKB-Pertemuan8/
├── rps.ipynb                   # Script utama CNN
├── README.md                 # Dokumentasi proyek ini
├── training_history.png      # (dihasilkan setelah training)
├── confusion_matrix.png      # (dihasilkan setelah evaluasi)
└── sample_predictions.png    # (dihasilkan setelah prediksi)
```

---

