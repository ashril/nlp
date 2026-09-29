# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 1 — PERTEMUAN 1
## Eksplorasi Corpus

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 1 — Fundamental dan Text Processing |
| Pertemuan | 1 |
| Topik | Eksplorasi Corpus |
| Output | `eksplorasi_corpus.ipynb` |

---

## B. Tujuan

Mahasiswa mampu:

* membaca dataset CSV;
* memahami struktur corpus;
* mengetahui jumlah dokumen;
* mengetahui jumlah kolom;
* mengetahui jumlah data;
* mengetahui distribusi data;
* menghitung jumlah kalimat;
* menghitung jumlah kata;
* melakukan eksplorasi awal dataset.

---

## C. Materi Teori

### 1. Pengertian NLP

Natural Language Processing (NLP) adalah cabang kecerdasan buatan yang memungkinkan komputer untuk memahami, mengolah, dan menghasilkan bahasa alami (bahasa manusia).

Contoh aplikasi NLP:

* Mesin penerjemah (Google Translate)
* Asisten virtual (Siri, Google Assistant)
* Deteksi spam email
* Analisis sentimen ulasan produk
* Chatbot layanan pelanggan
* Ringkasan dokumen otomatis

### 2. NLP Klasik vs Modern

| Era | Pendekatan | Contoh |
|-----|-----------|--------|
| Klasik | Berbasis aturan (rule-based) | Regex, grammar rules |
| Statistik | Probabilitas & frekuensi | N-gram, Naive Bayes |
| Machine Learning | Representasi fitur manual | SVM, Decision Tree |
| Deep Learning | Representasi otomatis | RNN, LSTM |
| Modern | Pretrained Transformer | BERT, GPT, IndoBERT |

### 3. Level Pemrosesan Bahasa

NLP memproses bahasa pada berbagai level:

```text
Lexical    → Kata dan morfologi
Syntactic  → Struktur kalimat (grammar)
Semantic   → Makna kata dan kalimat
Discourse  → Hubungan antar kalimat
Pragmatic  → Konteks dan tujuan komunikasi
```

### 4. Apa Itu Corpus?

Dalam NLP, **corpus** adalah kumpulan data bahasa yang digunakan untuk analisis atau pengembangan sistem NLP.

```text
Dokumen 1
Dokumen 2
Dokumen 3
...
Dokumen N
```

Sebuah corpus dapat berupa:

* berita;
* review produk;
* komentar media sosial;
* tweet;
* dokumen akademik;
* percakapan;
* pengaduan masyarakat.

### 5. Dataset CSV

Dataset yang digunakan mahasiswa minimal memiliki satu kolom teks.

Contoh:

| id | text | label |
| -: | ---- | ----- |
| 1 | Pelayanan hotel sangat bagus | positif |
| 2 | Kamar hotel sangat kotor | negatif |
| 3 | Lokasi hotel cukup strategis | positif |

Nama kolom setiap dataset dapat berbeda. Mahasiswa harus mengidentifikasi sendiri kolom yang berisi teks.

---

## D. Praktikum
### 0. Persiapan Notebook di VSCode
#### Persiapan Environment

Sebelum memulai praktikum Natural Language Processing (NLP), mahasiswa perlu menyiapkan environment Python yang akan digunakan untuk menjalankan seluruh notebook praktikum.

Praktikum menggunakan:

* Python
* Visual Studio Code
* Jupyter Notebook
* Virtual Environment (`.venv`)

---

#### 1. Membuat Folder Project

Buka **Terminal** pada Visual Studio Code.

Buat folder utama untuk project NLP:

```bash
mkdir NLP
cd NLP
```

Kemudian buat folder untuk Modul 1:

```bash
mkdir modul-1
cd modul-1
```

Struktur folder sementara:

```text
NLP/
└── modul-1/
```

> **Catatan:** Pastikan terminal berada di dalam folder `modul-1` sebelum membuat virtual environment.

---

#### 2. Membuat Virtual Environment

Virtual environment digunakan agar library yang digunakan dalam praktikum NLP terisolasi dari instalasi Python lainnya pada komputer.

##### Windows

Jalankan:

```bash
python -m venv .venv
```

Aktifkan virtual environment:

```bash
.venv\Scripts\activate
```

Jika berhasil, terminal akan menampilkan:

```text
(.venv)
```

Contoh:

```text
(.venv) C:\Users\Nama\NLP\modul-1>
```

##### Linux / macOS

Jalankan:

```bash
python3 -m venv .venv
```

Kemudian aktifkan:

```bash
source .venv/bin/activate
```

Jika berhasil:

```text
(.venv)
```

Contoh:

```text
(.venv) user@computer:~/NLP/modul-1$
```

##### Memastikan Virtual Environment Aktif

Untuk memastikan Python yang digunakan berasal dari virtual environment, jalankan:

```bash
python --version
```

Kemudian:

```bash
python -c "import sys; print(sys.executable)"
```

Path yang ditampilkan seharusnya mengarah ke folder `.venv`.

Contoh Windows:

```text
C:\Users\Nama\NLP\modul-1\.venv\Scripts\python.exe
```

Contoh Linux/macOS:

```text
/home/user/NLP/modul-1/.venv/bin/python
```

---

#### 3. Install Library

Setelah virtual environment aktif, install library yang diperlukan untuk praktikum Modul 1.

Jalankan:

```bash
pip install pandas numpy matplotlib seaborn nltk spacy Sastrawi scikit-learn jupyter
```

Library yang digunakan:

| Library        | Kegunaan                         |
| -------------- | -------------------------------- |
| `pandas`       | Membaca dan mengolah dataset     |
| `numpy`        | Komputasi numerik                |
| `matplotlib`   | Visualisasi data                 |
| `seaborn`      | Visualisasi statistik            |
| `nltk`         | Pemrosesan bahasa dan tokenisasi |
| `spacy`        | Linguistic processing            |
| `Sastrawi`     | Stemming Bahasa Indonesia        |
| `scikit-learn` | Preprocessing dan evaluasi dasar |
| `jupyter`      | Menjalankan Jupyter Notebook     |

Untuk memastikan seluruh library terinstall, dapat digunakan:

```bash
pip list
```

Cari library yang diperlukan pada daftar tersebut.

---

#### 4. Memeriksa Instalasi

Buat sebuah notebook baru dengan nama:

```text
test_environment.ipynb
```

Pada cell pertama, masukkan kode berikut:

```python
import pandas
import numpy
import matplotlib
import nltk
import spacy
import Sastrawi
import sklearn

print("Environment NLP siap digunakan.")
```

Jalankan cell tersebut.

Jika tidak muncul error dan menghasilkan:

```text
Environment NLP siap digunakan.
```

maka environment NLP telah berhasil disiapkan.

##### Jika Muncul Error

Jika salah satu library menghasilkan error seperti:

```text
ModuleNotFoundError
```

periksa kembali apakah virtual environment sudah aktif.

Contoh:

```text
(.venv)
```

Jika sudah aktif, install kembali library yang bermasalah.

Misalnya:

```bash
pip install pandas
```

atau:

```bash
pip install nltk
```

Setelah proses instalasi selesai, jalankan kembali notebook.

---

#### 5. Memilih Kernel Jupyter

Jupyter Notebook pada VS Code harus menggunakan Python environment yang sama dengan tempat library di-install.

Buka file:

```text
test_environment.ipynb
```

Pada bagian kanan atas notebook, pilih:

**Kernel / Select Kernel**

Kemudian pilih:

```text
Python (.venv)
```

atau interpreter Python yang berada di dalam folder:

```text
modul-1/.venv/
```

Contoh pada Windows:

```text
.venv\Scripts\python.exe
```

Contoh pada Linux/macOS:

```text
.venv/bin/python
```

##### Memeriksa Kernel yang Digunakan

Untuk memastikan notebook menggunakan Python dari `.venv`, jalankan:

```python
import sys

print(sys.executable)
```

Hasilnya harus mengarah ke folder `.venv`.

Contoh:

```text
C:\Users\Nama\NLP\modul-1\.venv\Scripts\python.exe
```

atau:

```text
/home/user/NLP/modul-1/.venv/bin/python
```

Jika path tersebut muncul, maka notebook sudah menggunakan environment yang benar.

---

#### 6. Struktur Folder Setelah Persiapan

Setelah seluruh tahap persiapan selesai, struktur folder Modul 1 akan menjadi:

```text
NLP/
└── modul-1/
    ├── .venv/
    └── test_environment.ipynb
```

Folder `.venv` merupakan virtual environment dan **tidak perlu di-upload ke GitHub**.

> **Penting:** Jangan mengunggah folder `.venv` ke GitHub karena ukurannya besar dan berisi file khusus untuk komputer masing-masing.

---

#### 7. Checklist Persiapan

Sebelum melanjutkan ke praktikum Pertemuan 1, pastikan:

* [ ] VS Code sudah terinstall
* [ ] Python sudah terinstall
* [ ] Folder `NLP` sudah dibuat
* [ ] Folder `modul-1` sudah dibuat
* [ ] Virtual environment `.venv` sudah dibuat
* [ ] Virtual environment sudah aktif
* [ ] `pandas` sudah terinstall
* [ ] `numpy` sudah terinstall
* [ ] `matplotlib` sudah terinstall
* [ ] `seaborn` sudah terinstall
* [ ] `nltk` sudah terinstall
* [ ] `spacy` sudah terinstall
* [ ] `Sastrawi` sudah terinstall
* [ ] `scikit-learn` sudah terinstall
* [ ] `jupyter` sudah terinstall
* [ ] `test_environment.ipynb` berhasil dijalankan
* [ ] Kernel notebook menggunakan `.venv`
* [ ] `sys.executable` menunjukkan Python dari `.venv`

Jika seluruh checklist sudah terpenuhi, environment praktikum NLP **siap digunakan**.

---

#### 8. Langkah Berikutnya

Setelah environment berhasil disiapkan, mahasiswa siap melanjutkan praktikum. Pada praktikum ini, dataset CSV dari Kaggle yang telah dipersiapkan akan digunakan untuk:

1. Membaca dataset dengan `pandas`.
2. Memahami struktur dataset.
3. Mengidentifikasi kolom teks.
4. Memeriksa data kosong dan data duplikat.
5. Menghitung jumlah dokumen.
6. Menghitung jumlah karakter dan kata.
7. Melakukan eksplorasi awal terhadap corpus.
8. Membuat visualisasi sederhana.
9. Menyusun kesimpulan awal mengenai karakteristik dataset.

### 1. Membuat Notebook

Buat file:

```text
pertemuan-01/eksplorasi_corpus.ipynb
```

### 2. Import Library

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
```

### 3. Membaca Dataset

```python
df = pd.read_csv("../dataset/dataset.csv")
```

Jika delimiter menggunakan titik koma:

```python
df = pd.read_csv("../dataset/dataset.csv", sep=";")
```

### 4. Melihat Data Awal

```python
# 10 data pertama
df.head(10)
```

```python
# 10 data terakhir
df.tail(10)
```

Pertanyaan:

1. Apa nama kolom dataset?
2. Kolom mana yang berisi teks?
3. Apakah terdapat label?
4. Apakah terdapat nilai kosong?

### 5. Informasi Dataset

```python
df.info()
```

Perhatikan:
* jumlah baris;
* jumlah kolom;
* tipe data;
* jumlah nilai non-null.

```python
df.shape
```

Contoh output:

```text
(10000, 3)
```

Artinya: 10.000 baris, 3 kolom.

### 6. Melihat Nama Kolom

```python
df.columns
```

### 7. Jumlah Dokumen

```python
jumlah_dokumen = len(df)
print("Jumlah dokumen:", jumlah_dokumen)
```

### 8. Memilih Kolom Teks

Sesuaikan dengan nama kolom pada dataset masing-masing:

```python
text_column = "text"  # sesuaikan dengan nama kolom
texts = df[text_column]
```

Melihat contoh teks:

```python
for text in texts.head(10):
    print(text)
    print("-" * 80)
```

### 9. Memeriksa Missing Value

```python
df.isnull().sum()
```

Memeriksa khusus pada kolom teks:

```python
df[text_column].isnull().sum()
```

Menghapus data kosong:

```python
df = df.dropna(subset=[text_column])
df[text_column].isnull().sum()
```

### 10. Memeriksa Duplikasi

```python
df.duplicated().sum()
```

Memeriksa duplikasi pada kolom teks:

```python
df[text_column].duplicated().sum()
```

Menghapus duplikasi:

```python
df = df.drop_duplicates(subset=[text_column])
```

### 11. Menghitung Jumlah Karakter

```python
df["jumlah_karakter"] = df[text_column].astype(str).apply(len)
df[[text_column, "jumlah_karakter"]].head()
```

### 12. Menghitung Jumlah Kata

```python
df["jumlah_kata"] = df[text_column].astype(str).apply(
    lambda x: len(x.split())
)
df[[text_column, "jumlah_kata"]].head()
```

Statistik jumlah kata:

```python
df["jumlah_kata"].describe()
```

Perhatikan: mean, minimum, maksimum, median, quartile.

### 13. Menghitung Jumlah Kalimat

```python
from nltk.tokenize import sent_tokenize

df["jumlah_kalimat"] = df[text_column].astype(str).apply(
    lambda x: len(sent_tokenize(x))
)
```

### 14. Statistik Corpus

```python
print("Jumlah dokumen :", len(df))
print("Jumlah kata    :", df["jumlah_kata"].sum())
print("Jumlah kalimat :", df["jumlah_kalimat"].sum())
print("Jumlah karakter:", df["jumlah_karakter"].sum())
```

### 15. Visualisasi Distribusi Panjang Dokumen

```python
plt.figure(figsize=(10, 5))
plt.hist(df["jumlah_kata"], bins=30, color="steelblue", edgecolor="white")
plt.xlabel("Jumlah Kata")
plt.ylabel("Jumlah Dokumen")
plt.title("Distribusi Panjang Dokumen")
plt.tight_layout()
plt.show()
```

### 16. Distribusi Label (Jika Ada)

```python
# Ganti "label" dengan nama kolom label di dataset Anda
df["label"].value_counts()
```

```python
df["label"].value_counts().plot(kind="bar", color="steelblue")
plt.xlabel("Label")
plt.ylabel("Jumlah Data")
plt.title("Distribusi Label")
plt.tight_layout()
plt.show()
```

---

## E. Tugas Analisis

Mahasiswa harus menjawab:

1. Apa nama dataset?
2. Dataset berasal dari mana?
3. Berapa jumlah dokumen?
4. Berapa jumlah kolom?
5. Apa nama kolom teks?
6. Apakah terdapat missing value?
7. Apakah terdapat data duplikat?
8. Berapa rata-rata jumlah kata per dokumen?
9. Berapa dokumen terpendek?
10. Berapa dokumen terpanjang?
11. Apakah dataset memiliki label?
12. Bagaimana distribusi labelnya?
13. Apa karakteristik bahasa yang ditemukan?

---

## F. Output Pertemuan 1

File yang dikumpulkan:

```text
eksplorasi_corpus.ipynb
```

Notebook minimal berisi:

```text
1. Import Library
2. Load Dataset
3. Dataset Information
4. Missing Value
5. Duplicate Data
6. Corpus Statistics
7. Word Statistics
8. Sentence Statistics
9. Visualization
10. Analysis
```

---

## G. Pertanyaan Diskusi

1. Mengapa eksplorasi corpus penting sebelum memulai proses NLP?
2. Apa perbedaan dokumen, kalimat, kata, dan karakter dalam konteks NLP?
3. Apakah dataset yang seimbang (balanced) selalu lebih baik?
4. Mengapa perlu memeriksa missing value dan duplikasi?
5. Apa yang dapat disimpulkan dari distribusi panjang dokumen?

---

> **Pertemuan berikutnya:** [Pertemuan 2 — Tokenization](praktikum2.md)
