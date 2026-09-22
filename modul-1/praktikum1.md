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
