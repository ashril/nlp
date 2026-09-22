# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 1 — FUNDAMENTAL DAN TEXT PROCESSING

---

## A. Identitas Modul

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Program Studi | S1 Teknologi Informasi |
| Modul | Modul 1 |
| Topik | Fundamental dan Text Processing |
| Platform | VS Code |
| Environment | Jupyter Notebook |
| Bahasa | Python |
| Dataset | Dataset teks CSV dari Kaggle |
| Pertemuan | 1–5 |

---

## B. Deskripsi Modul

Modul ini merupakan tahap pertama dalam pembelajaran Natural Language Processing (NLP). Sebelum sebuah model NLP dapat digunakan, data teks harus dipahami dan dipersiapkan terlebih dahulu. Secara umum, proses NLP dimulai dari:

```text
Raw Text
    ↓
Corpus
    ↓
Exploration
    ↓
Tokenization
    ↓
Normalization
    ↓
Stopword Removal
    ↓
Stemming / Lemmatization
    ↓
Linguistic Processing
    ↓
N-Gram
    ↓
Language Model
```

Pada modul ini mahasiswa belum diarahkan untuk membangun model Machine Learning secara mendalam.

Fokus utama adalah:

> **memahami data bahasa dan melakukan text processing menggunakan Python.**

---

## C. Tujuan Modul

Setelah menyelesaikan Modul 1, mahasiswa diharapkan mampu:

1. Membuka dan menjalankan Jupyter Notebook di VS Code.
2. Membaca dataset CSV menggunakan Python.
3. Mengeksplorasi struktur dataset teks.
4. Mengidentifikasi kolom teks dan label.
5. Menghitung jumlah dokumen, kalimat, dan kata.
6. Melakukan sentence segmentation dan word tokenization.
7. Melakukan case folding dan membersihkan teks.
8. Melakukan normalisasi, stopword removal, dan stemming Bahasa Indonesia.
9. Memahami perbedaan stemming dan lemmatization.
10. Melakukan POS Tagging dan dependency parsing.
11. Membuat unigram, bigram, dan trigram.
12. Menghitung probabilitas sederhana pada n-gram.
13. Membangun mini language model sederhana.
14. Mendokumentasikan hasil eksperimen.

---

## D. Pembagian Praktikum

Modul 1 terdiri dari lima praktikum.

| Pertemuan | File | Materi | Output |
| --------- | ---- | ------ | ------ |
| 1 | [praktikum1.md](praktikum1.md) | Eksplorasi Corpus | Notebook eksplorasi corpus |
| 2 | [praktikum2.md](praktikum2.md) | Tokenization | Program text tokenizer |
| 3 | [praktikum3.md](praktikum3.md) | Text Normalization | Dataset hasil preprocessing |
| 4 | [praktikum4.md](praktikum4.md) | Linguistic Processing | Analisis struktur linguistik |
| 5 | [praktikum5.md](praktikum5.md) | N-Gram & Language Model | Mini language model |

---

## E. Software dan Library

Praktikum menggunakan:

* Python
* VS Code
* Jupyter Notebook
* pandas
* numpy
* matplotlib
* seaborn
* NLTK
* spaCy
* Sastrawi
* scikit-learn

---

## F. Struktur Folder Praktikum

Mahasiswa disarankan membuat struktur:

```text
NLP/
│
├── modul-1/
│   │
│   ├── dataset/
│   │   └── dataset.csv
│   │
│   ├── pertemuan-01/
│   │   └── eksplorasi_corpus.ipynb
│   │
│   ├── pertemuan-02/
│   │   └── tokenization.ipynb
│   │
│   ├── pertemuan-03/
│   │   └── text_normalization.ipynb
│   │
│   ├── pertemuan-04/
│   │   └── linguistic_processing.ipynb
│   │
│   └── pertemuan-05/
│       └── ngram_language_model.ipynb
│
└── README.md
```

---

## G. Persiapan Environment

### 1. Membuat Folder Project

Buka terminal pada VS Code.

```bash
mkdir NLP
cd NLP
```

Kemudian:

```bash
mkdir modul-1
cd modul-1
```

### 2. Membuat Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Jika berhasil, terminal akan menunjukkan:

```text
(.venv)
```

### 3. Install Library

```bash
pip install pandas numpy matplotlib seaborn nltk spacy Sastrawi scikit-learn jupyter
```

### 4. Memeriksa Instalasi

Buat notebook `test_environment.ipynb` dan jalankan:

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

Jika output menampilkan:

```text
Environment NLP siap digunakan.
```

maka environment telah siap.

### 5. Memilih Kernel Jupyter

Pada VS Code:

1. Buka file `.ipynb`.
2. Klik pilihan **Kernel** di bagian kanan atas.
3. Pilih environment: `Python (.venv)`

Pastikan notebook menggunakan environment yang sama dengan tempat library di-install.

---

## H. Alur Kompetensi Modul 1

```text
LANGUAGE
    ↓
TEXT
    ↓
CORPUS
    ↓
TOKENIZATION
    ↓
NORMALIZATION
    ↓
LINGUISTIC PROCESSING
    ↓
REPRESENTATION
    ↓
LANGUAGE MODEL
```

Modul 1 menjadi fondasi untuk Modul 2 (Representasi Teks & Neural NLP) dan Modul 3 (Transformer & Pretrained Model).
