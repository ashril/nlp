# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 1 — FUNDAMENTAL DAN TEXT PROCESSING
## Pengolahan Awal Dataset Teks Menggunakan Python

---

## A. Identitas Praktikum

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

# B. Deskripsi Modul

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
````

Pada modul ini mahasiswa belum diarahkan untuk membangun model Machine Learning secara mendalam.

Fokus utama adalah:

> **memahami data bahasa dan melakukan text processing menggunakan Python.**

---

# C. Tujuan Praktikum

Setelah menyelesaikan Modul 1, mahasiswa diharapkan mampu:

1. Membuka dan menjalankan Jupyter Notebook di VS Code.
2. Membaca dataset CSV menggunakan Python.
3. Mengeksplorasi struktur dataset teks.
4. Mengidentifikasi kolom teks dan label.
5. Menghitung jumlah dokumen.
6. Menghitung jumlah kalimat.
7. Menghitung jumlah kata.
8. Melakukan sentence segmentation.
9. Melakukan word tokenization.
10. Melakukan character tokenization.
11. Melakukan case folding.
12. Membersihkan tanda baca.
13. Membersihkan URL dan karakter yang tidak diperlukan.
14. Melakukan normalisasi teks.
15. Menghilangkan stopword.
16. Melakukan stemming Bahasa Indonesia.
17. Memahami perbedaan stemming dan lemmatization.
18. Melakukan POS Tagging.
19. Memahami konsep parsing.
20. Membuat unigram, bigram, dan trigram.
21. Menghitung probabilitas sederhana pada n-gram.
22. Membuat mini language model sederhana.
23. Mendokumentasikan hasil eksperimen.

---

# D. Pembagian Praktikum

Modul 1 terdiri dari lima praktikum.

| Pertemuan | Materi                  | Output                       |
| --------- | ----------------------- | ---------------------------- |
| 1         | Eksplorasi Corpus       | Notebook eksplorasi corpus   |
| 2         | Tokenization            | Program text tokenizer       |
| 3         | Text Normalization      | Dataset hasil preprocessing  |
| 4         | Linguistic Processing   | Analisis struktur linguistik |
| 5         | N-Gram & Language Model | Mini language model          |

---

# E. Software dan Library

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

Beberapa library digunakan hanya untuk mendukung praktikum.

---

# F. Struktur Folder Praktikum

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

# G. Persiapan Environment

## 1. Membuat Folder Project

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

---

# 2. Membuat Virtual Environment

Windows:

```bash
python -m venv .venv
```

Aktivasi:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
```

Aktivasi:

```bash
source .venv/bin/activate
```

Jika berhasil, terminal akan menunjukkan:

```text
(.venv)
```

---

# 3. Install Library

Jalankan:

```bash
pip install pandas numpy matplotlib seaborn nltk spacy Sastrawi scikit-learn jupyter
```

---

# 4. Memeriksa Instalasi

Buat notebook:

```text
test_environment.ipynb
```

Kemudian jalankan:

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

Jika tidak muncul error:

```text
Environment NLP siap digunakan.
```

maka environment telah siap.

---

# 5. Memilih Kernel Jupyter

Pada VS Code:

1. Buka file `.ipynb`.
2. Klik pilihan **Kernel** di bagian kanan atas.
3. Pilih environment:

```text
Python (.venv)
```

Pastikan notebook menggunakan environment yang sama dengan tempat library di-install.

---

# PERTEMUAN 1

# EKSPLORASI CORPUS

---

# A. Tujuan

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

# B. Konsep Corpus

Dalam NLP, **corpus** adalah kumpulan data bahasa yang digunakan untuk analisis atau pengembangan sistem NLP.

Contoh corpus:

```text
Dokumen 1
Dokumen 2
Dokumen 3
Dokumen 4
...
Dokumen N
```

Sebuah corpus dapat berupa:

* berita;
* review;
* komentar;
* tweet;
* dokumen akademik;
* percakapan;
* pengaduan;
* deskripsi produk.

---

# C. Dataset CSV

Dataset Kaggle yang digunakan mahasiswa minimal memiliki satu kolom teks.

Contoh:

| id | text                         | label   |
| -: | ---------------------------- | ------- |
|  1 | Pelayanan hotel sangat bagus | positif |
|  2 | Kamar hotel sangat kotor     | negatif |
|  3 | Lokasi hotel cukup strategis | positif |

Nama kolom setiap dataset dapat berbeda.

Misalnya:

```text
text
review
comment
content
sentence
tweet
description
```

Mahasiswa harus mengidentifikasi sendiri kolom yang berisi teks.

---

# D. Membaca Dataset

Import pandas:

```python
import pandas as pd
```

Baca CSV:

```python
df = pd.read_csv("../dataset/dataset.csv")
```

Jika delimiter menggunakan titik koma:

```python
df = pd.read_csv("../dataset/dataset.csv", sep=";")
```

Tampilkan data:

```python
df.head()
```

---

# E. Melihat 10 Data Pertama

```python
df.head(10)
```

Pertanyaan:

1. Apa nama kolom dataset?
2. Kolom mana yang berisi teks?
3. Apakah terdapat label?
4. Apakah terdapat nilai kosong?
5. Apakah terdapat data duplikat?

---

# F. Melihat 10 Data Terakhir

```python
df.tail(10)
```

---

# G. Melihat Nama Kolom

```python
df.columns
```

Contoh output:

```text
Index(['id', 'text', 'label'], dtype='object')
```

---

# H. Informasi Dataset

```python
df.info()
```

Perhatikan:

* jumlah baris;
* jumlah kolom;
* tipe data;
* jumlah nilai non-null.

---

# I. Ukuran Dataset

```python
df.shape
```

Contoh:

```text
(10000, 3)
```

Artinya:

```text
10.000 baris
3 kolom
```

---

# J. Jumlah Dokumen

Jika satu baris dianggap sebagai satu dokumen:

```python
jumlah_dokumen = len(df)

print("Jumlah dokumen:", jumlah_dokumen)
```

---

# K. Memilih Kolom Teks

Misalnya kolom teks bernama:

```text
text
```

Buat variabel:

```python
text_column = "text"
```

Kemudian:

```python
texts = df[text_column]
```

---

# L. Melihat Contoh Teks

```python
for text in texts.head(10):
    print(text)
    print("-" * 80)
```

---

# M. Memeriksa Missing Value

```python
df.isnull().sum()
```

Contoh:

```text
id        0
text     15
label     0
```

Artinya terdapat 15 data teks kosong.

---

# N. Menghitung Missing Value pada Kolom Teks

```python
df[text_column].isnull().sum()
```

---

# O. Menghapus Data Kosong

Sebelum processing, data teks kosong dapat dihapus:

```python
df = df.dropna(subset=[text_column])
```

Periksa kembali:

```python
df[text_column].isnull().sum()
```

Hasil:

```text
0
```

---

# P. Memeriksa Duplikasi

```python
df.duplicated().sum()
```

Jika ingin memeriksa duplikasi khusus teks:

```python
df[text_column].duplicated().sum()
```

---

# Q. Menghapus Duplikasi

```python
df = df.drop_duplicates(subset=[text_column])
```

---

# R. Menghitung Jumlah Karakter

```python
df["jumlah_karakter"] = df[text_column].astype(str).apply(len)
```

Tampilkan:

```python
df[[text_column, "jumlah_karakter"]].head()
```

---

# S. Menghitung Jumlah Kata

Versi sederhana:

```python
df["jumlah_kata"] = df[text_column].astype(str).apply(
    lambda x: len(x.split())
)
```

Tampilkan:

```python
df[[text_column, "jumlah_kata"]].head()
```

---

# T. Statistik Jumlah Kata

```python
df["jumlah_kata"].describe()
```

Perhatikan:

* mean;
* minimum;
* maksimum;
* median;
* quartile.

---

# U. Menghitung Jumlah Kalimat

Untuk tahap awal dapat menggunakan NLTK.

```python
import nltk
```

Download tokenizer:

```python
nltk.download("punkt")
nltk.download("punkt_tab")
```

Kemudian:

```python
from nltk.tokenize import sent_tokenize
```

Contoh:

```python
text = "Saya belajar NLP. NLP sangat menarik."

sentences = sent_tokenize(text)

print(sentences)
```

Output:

```text
['Saya belajar NLP.', 'NLP sangat menarik.']
```

---

# V. Menghitung Jumlah Kalimat dalam Dataset

```python
df["jumlah_kalimat"] = df[text_column].astype(str).apply(
    lambda x: len(sent_tokenize(x))
)
```

---

# W. Melihat Statistik Corpus

```python
print("Jumlah dokumen :", len(df))
print("Jumlah kata    :", df["jumlah_kata"].sum())
print("Jumlah kalimat :", df["jumlah_kalimat"].sum())
print("Jumlah karakter:", df["jumlah_karakter"].sum())
```

---

# X. Visualisasi Distribusi Panjang Dokumen

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.hist(df["jumlah_kata"], bins=30)

plt.xlabel("Jumlah Kata")
plt.ylabel("Jumlah Dokumen")
plt.title("Distribusi Panjang Dokumen")

plt.show()
```

---

# Y. Jika Dataset Memiliki Label

Misalnya:

```text
label
```

Periksa:

```python
df["label"].value_counts()
```

Visualisasi:

```python
df["label"].value_counts().plot(kind="bar")

plt.xlabel("Label")
plt.ylabel("Jumlah Data")
plt.title("Distribusi Label")

plt.show()
```

---

# Z. Tugas Analisis Pertemuan 1

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

# Output Pertemuan 1

File:

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

# PERTEMUAN 2

# TOKENIZATION

---

# A. Tujuan

Mahasiswa mampu:

* memahami token;
* melakukan sentence segmentation;
* melakukan word tokenization;
* melakukan character tokenization;
* memahami subword tokenization;
* membandingkan tokenizer.

---

# B. Apa Itu Token?

Token adalah unit teks yang diproses oleh sistem NLP.

Contoh:

```text
Saya belajar NLP
```

dapat menjadi:

```text
["Saya", "belajar", "NLP"]
```

Token tidak selalu sama dengan kata.

---

# C. Level Token

Token dapat berupa:

### Character

```text
S
a
y
a
```

### Word

```text
Saya
belajar
NLP
```

### Subword

Contohnya:

```text
bermainnya
```

dapat dipecah menjadi beberapa subword tergantung tokenizer.

---

# D. Sentence Segmentation

Kalimat:

```text
Saya belajar NLP. NLP menarik.
```

menjadi:

```text
[
    "Saya belajar NLP.",
    "NLP menarik."
]
```

---

# E. Sentence Tokenization dengan NLTK

```python
from nltk.tokenize import sent_tokenize
```

Contoh:

```python
text = "Saya belajar NLP. NLP sangat menarik."

sentences = sent_tokenize(text)

for sentence in sentences:
    print(sentence)
```

---

# F. Word Tokenization

```python
from nltk.tokenize import word_tokenize
```

Contoh:

```python
text = "Saya belajar Natural Language Processing."

tokens = word_tokenize(text)

print(tokens)
```

---

# G. Mengamati Hasil Tokenisasi

Perhatikan:

```text
[
'Saya',
'belajar',
'Natural',
'Language',
'Processing',
'.'
]
```

Tanda baca dapat dianggap sebagai token tersendiri.

---

# H. Character Tokenization

```python
text = "NLP"

characters = list(text)

print(characters)
```

Output:

```text
['N', 'L', 'P']
```

---

# I. Tokenisasi Dataset

```python
df["tokens"] = df[text_column].astype(str).apply(word_tokenize)
```

Lihat:

```python
df[[text_column, "tokens"]].head()
```

---

# J. Menghitung Jumlah Token

```python
df["jumlah_token"] = df["tokens"].apply(len)
```

---

# K. Membandingkan Kata dan Token

Bandingkan:

```python
df["jumlah_kata"]
```

dengan:

```python
df["jumlah_token"]
```

Mengapa hasilnya berbeda?

Karena tokenizer dapat memisahkan:

* tanda baca;
* simbol;
* angka;
* karakter tertentu.

---

# L. Tokenisasi Bahasa Indonesia

Bahasa Indonesia memiliki tantangan seperti:

```text
tidak
nggak
gak
ga
tdk
```

Secara makna dapat berkaitan, tetapi secara bentuk berbeda.

Contoh:

```text
"pelayanannya bagus banget!!!"
```

Tokenizer dapat menghasilkan:

```text
pelayanannya
bagus
banget
!
!
!
```

---

# M. Eksperimen Tokenisasi

Gunakan beberapa contoh:

```python
texts = [
    "Saya suka produk ini.",
    "Bagus banget!!!",
    "Tidak bagus...",
    "Harga Rp50.000.",
    "email@example.com",
    "https://example.com"
]

for text in texts:
    print("TEXT :", text)
    print("TOKEN:", word_tokenize(text))
    print()
```

Amati hasilnya.

---

# N. spaCy

Install model Bahasa Indonesia jika tersedia pada environment yang digunakan.

Mahasiswa dapat menggunakan pipeline spaCy untuk membandingkan hasil tokenisasi.

Contoh konsep:

```python
import spacy
```

Kemudian menggunakan model yang tersedia pada environment.

---

# O. Perbandingan Tokenizer

Mahasiswa membuat tabel:

| Teks              | NLTK | spaCy |
| ----------------- | ---- | ----- |
| Saya belajar NLP. | ...  | ...   |
| Bagus banget!!!   | ...  | ...   |
| Harga Rp50.000    | ...  | ...   |

---

# P. Tugas Analisis

Jawab:

1. Apakah setiap kata selalu menghasilkan satu token?
2. Bagaimana tokenizer menangani tanda baca?
3. Bagaimana tokenizer menangani angka?
4. Bagaimana tokenizer menangani URL?
5. Bagaimana tokenizer menangani emoji?
6. Apa masalah tokenisasi Bahasa Indonesia?
7. Mengapa subword tokenizer diperlukan?

---

# Output Pertemuan 2

File:

```text
tokenization.ipynb
```

Output:

> Program text tokenizer yang mampu melakukan sentence, word, dan character tokenization serta analisis hasilnya.

---

# PERTEMUAN 3

# TEXT NORMALIZATION

---

# A. Tujuan

Mahasiswa mampu:

* melakukan case folding;
* membersihkan teks;
* menghapus punctuation;
* menangani angka;
* menangani URL;
* menangani slang;
* melakukan stopword removal;
* melakukan stemming;
* memahami lemmatization;
* membangun preprocessing pipeline.

---

# B. Mengapa Normalisasi?

Data teks dunia nyata biasanya:

```text
Tidak bersih
Tidak konsisten
Banyak typo
Banyak slang
Banyak simbol
```

Contoh:

```text
"PELAYANANNYA BAGUUUS BANGET!!! 😍😍"
```

Sistem NLP perlu menentukan:

> informasi mana yang penting dan mana yang dapat dihilangkan.

---

# C. Case Folding

Mengubah huruf menjadi lowercase.

```python
text = "Saya Belajar NLP"

text_lower = text.lower()

print(text_lower)
```

Output:

```text
saya belajar nlp
```

---

# D. Cleaning

Contoh:

```text
"Harga murah!!! https://example.com"
```

dapat dibersihkan menjadi:

```text
"Harga murah"
```

---

# E. Menghapus URL

Gunakan regular expression:

```python
import re
```

Function:

```python
def remove_url(text):
    return re.sub(r"http\S+|www\S+", "", text)
```

Contoh:

```python
text = "Kunjungi https://example.com sekarang"

print(remove_url(text))
```

---

# F. Menghapus Mention

Untuk data media sosial:

```python
def remove_mention(text):
    return re.sub(r"@\w+", "", text)
```

---

# G. Menghapus Hashtag

```python
def remove_hashtag(text):
    return re.sub(r"#\w+", "", text)
```

Catatan:

> Jangan selalu menghapus hashtag jika hashtag mengandung informasi penting.

Contoh:

```text
#Lombok
#Pariwisata
```

dapat menjadi informasi yang berguna.

---

# H. Menghapus Angka

```python
def remove_number(text):
    return re.sub(r"\d+", "", text)
```

Namun hati-hati.

Angka dapat memiliki makna:

```text
2026
Rp100.000
5 bintang
```

Karena itu keputusan menghapus angka harus disesuaikan dengan task.

---

# I. Menghapus Punctuation

```python
import string

def remove_punctuation(text):
    return text.translate(
        str.maketrans("", "", string.punctuation)
    )
```

---

# J. Membersihkan Whitespace

```python
def clean_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()
```

---

# K. Membuat Fungsi Cleaning

```python
def clean_text(text):
    text = str(text)

    text = text.lower()
    text = remove_url(text)
    text = remove_mention(text)
    text = remove_hashtag(text)
    text = remove_number(text)
    text = remove_punctuation(text)
    text = clean_whitespace(text)

    return text
```

---

# L. Menerapkan Cleaning

```python
df["clean_text"] = df[text_column].apply(clean_text)
```

Lihat:

```python
df[[text_column, "clean_text"]].head(10)
```

---

# M. Stopword

Stopword adalah kata yang sering muncul dan dalam task tertentu dianggap kurang informatif.

Contoh:

```text
yang
dan
di
ke
dari
untuk
dengan
```

Namun:

> stopword tidak selalu harus dihapus.

Contoh sentiment:

```text
tidak bagus
```

Jika kata:

```text
tidak
```

dihapus:

```text
bagus
```

maka makna berubah.

---

# N. Stopword Bahasa Indonesia

NLTK memiliki daftar stopword Bahasa Indonesia.

```python
nltk.download("stopwords")
```

Kemudian:

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words("indonesian"))
```

---

# O. Stopword Filtering

```python
def remove_stopwords(tokens):
    return [
        token for token in tokens
        if token not in stop_words
    ]
```

---

# P. Stemming

Stemming bertujuan mengubah kata menjadi bentuk dasar/stem.

Contoh:

```text
berlari
berlari-lari
lari
```

dapat diarahkan menuju:

```text
lari
```

---

# Q. Stemming dengan Sastrawi

```python
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()

stemmer = factory.create_stemmer()
```

Contoh:

```python
text = "Mahasiswa sedang mempelajari pemrograman."

hasil = stemmer.stem(text)

print(hasil)
```

---

# R. Stemming Token

```python
def stem_tokens(tokens):
    return [
        stemmer.stem(token)
        for token in tokens
    ]
```

---

# S. Lemmatization

Lemmatization mengubah kata ke bentuk lemma berdasarkan informasi linguistik.

Contoh konsep:

```text
better → good
running → run
```

Lemmatization berbeda dari stemming.

---

# T. Stemming vs Lemmatization

| Stemming                   | Lemmatization                |
| -------------------------- | ---------------------------- |
| Berbasis aturan pemotongan | Berbasis analisis linguistik |
| Lebih sederhana            | Lebih kompleks               |
| Cepat                      | Relatif lebih mahal          |
| Hasil dapat berupa stem    | Hasil berupa lemma           |

Untuk Bahasa Indonesia:

> Sastrawi sering digunakan sebagai stemmer dalam praktikum.

---

# U. Pipeline Preprocessing

Pipeline yang digunakan:

```text
Raw Text
   ↓
Cleaning
   ↓
Normalization
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Stemming
   ↓
Processed Text
```

---

# V. Implementasi Pipeline

```python
def preprocessing_pipeline(text):

    # Cleaning
    text = clean_text(text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal
    tokens = remove_stopwords(tokens)

    # Stemming
    tokens = stem_tokens(tokens)

    return tokens
```

---

# W. Menerapkan Pipeline

```python
df["processed_tokens"] = df[text_column].apply(
    preprocessing_pipeline
)
```

Lihat:

```python
df[[text_column, "processed_tokens"]].head()
```

---

# X. Menggabungkan Token Kembali

```python
df["processed_text"] = df["processed_tokens"].apply(
    lambda tokens: " ".join(tokens)
)
```

---

# Y. Perbandingan Raw vs Processed

```python
for i in range(5):

    print("RAW:")
    print(df.iloc[i][text_column])

    print("\nPROCESSED:")
    print(df.iloc[i]["processed_text"])

    print("=" * 80)
```

---

# Z. Tugas Eksperimen

Mahasiswa mengambil minimal 20 data.

Bandingkan:

1. Raw text.
2. Setelah lowercase.
3. Setelah cleaning.
4. Setelah tokenization.
5. Setelah stopword removal.
6. Setelah stemming.

Buat tabel:

| Raw | Clean | Token | Stopword | Stem |
| --- | ----- | ----- | -------- | ---- |

---

# AA. Analisis Penting

Mahasiswa harus menjawab:

1. Apakah semua stopword harus dihapus?
2. Apakah stemming selalu meningkatkan kualitas data?
3. Apakah angka harus selalu dihapus?
4. Apakah URL selalu tidak berguna?
5. Apakah hashtag harus selalu dihapus?
6. Apakah preprocessing yang sama cocok untuk semua task NLP?

Kesimpulan yang diharapkan:

> **Preprocessing bergantung pada karakteristik data dan tujuan NLP task.**

---

# AB. Menyimpan Dataset Hasil Preprocessing

```python
output_columns = [
    text_column,
    "clean_text",
    "processed_text"
]

df[output_columns].to_csv(
    "../dataset/dataset_preprocessed.csv",
    index=False
)
```

---

# Output Pertemuan 3

File:

```text
text_normalization.ipynb
```

dan:

```text
dataset_preprocessed.csv
```

---

# PERTEMUAN 4

# LINGUISTIC PROCESSING

---

# A. Tujuan

Mahasiswa mampu:

* memahami lexical analysis;
* memahami morphology;
* memahami Part-of-Speech;
* melakukan POS tagging;
* memahami syntax;
* memahami dependency parsing;
* membaca struktur kalimat;
* menganalisis kesalahan linguistic processing.

---

# B. Lexical Analysis

Lexical analysis berkaitan dengan:

> unit-unit kata dalam bahasa.

Contoh:

```text
Mahasiswa belajar NLP.
```

Token:

```text
Mahasiswa
belajar
NLP
```

---

# C. Morphology

Morphology mempelajari:

> struktur pembentukan kata.

Bahasa Indonesia banyak menggunakan imbuhan.

Contoh:

```text
ajar
 ↓
belajar
 ↓
mempelajari
 ↓
pembelajaran
```

---

# D. Root Word

Contoh:

```text
bermain
permainan
memainkan
dimainkan
```

memiliki hubungan dengan:

```text
main
```

Stemming mencoba menemukan:

> bentuk dasar/stem.

---

# E. Part-of-Speech

POS menunjukkan kelas kata.

Contoh:

```text
Saya membaca buku.
```

Secara konseptual:

```text
Saya      → PRON
membaca   → VERB
buku      → NOUN
```

---

# F. POS Tagging

POS Tagging adalah:

> proses memberikan label kelas kata kepada token.

Contoh:

```text
Saya/PRON
membaca/VERB
buku/NOUN
```

---

# G. POS Tagging Bahasa Indonesia

Tag dapat mencakup:

```text
NOUN
VERB
ADJ
ADV
PRON
DET
ADP
NUM
CONJ
```

Label aktual bergantung pada:

> model/tagset yang digunakan.

---

# H. spaCy

Mahasiswa dapat menggunakan spaCy dengan model Bahasa Indonesia yang tersedia.

Konsep penggunaan:

```python
import spacy
```

Kemudian load model Bahasa Indonesia yang telah di-install.

---

# I. Memproses Kalimat

Secara umum:

```python
doc = nlp("Mahasiswa belajar Natural Language Processing.")
```

Kemudian:

```python
for token in doc:
    print(token.text, token.pos_)
```

---

# J. Analisis POS

Buat tabel:

| Token      | POS  |
| ---------- | ---- |
| Mahasiswa  | NOUN |
| belajar    | VERB |
| Natural    | ...  |
| Language   | ...  |
| Processing | ...  |

---

# K. Dependency Parsing

Dependency parsing mencoba menemukan:

> hubungan gramatikal antar token.

Contoh sederhana:

```text
Mahasiswa → membaca → buku
```

Hubungan:

```text
subjek
objek
```

---

# L. Dependency Tree

Konsep:

```text
             membaca
             /     \
            /       \
      Mahasiswa     buku
```

---

# M. Melihat Dependency

```python
for token in doc:
    print(
        token.text,
        token.dep_,
        token.head.text
    )
```

---

# N. Visualisasi Dependency

spaCy menyediakan visualisasi dependency.

```python
from spacy import displacy
```

Kemudian:

```python
displacy.render(
    doc,
    style="dep",
    jupyter=True
)
```

---

# O. Contoh Kalimat

Gunakan beberapa kalimat:

```python
sentences = [
    "Mahasiswa belajar NLP.",
    "Dosen mengajar Natural Language Processing.",
    "Mahasiswa membaca buku.",
    "Sistem memproses data teks."
]
```

Analisis setiap kalimat.

---

# P. Analisis Kesalahan

Linguistic processing tidak selalu sempurna.

Kesalahan dapat disebabkan oleh:

* typo;
* slang;
* bahasa informal;
* campuran bahasa;
* nama orang;
* nama organisasi;
* singkatan;
* domain khusus.

Contoh:

```text
"aku lg belajar NLP nih"
```

berbeda dengan:

```text
"Saya sedang mempelajari NLP."
```

---

# Q. Tugas Analisis Linguistik

Pilih minimal 10 kalimat dari dataset.

Untuk setiap kalimat:

1. lakukan tokenization;
2. lakukan POS tagging;
3. lakukan dependency parsing;
4. visualisasikan dependency;
5. identifikasi kesalahan.

Buat tabel:

| Kalimat | POS | Dependency | Error |
| ------- | --- | ---------- | ----- |

---

# R. Pertanyaan Diskusi

1. Apakah POS Tagging selalu benar?
2. Apa penyebab kesalahan POS Tagging?
3. Bagaimana slang mempengaruhi POS Tagging?
4. Bagaimana nama organisasi mempengaruhi POS Tagging?
5. Mengapa parsing penting dalam NLP?
6. Apa hubungan morphology dengan stemming?
7. Apakah stemming dan POS Tagging menyelesaikan masalah yang sama?

---

# Output Pertemuan 4

File:

```text
linguistic_processing.ipynb
```

Output:

> Analisis struktur linguistik corpus yang mencakup token, POS, dependency, visualisasi, dan analisis kesalahan.

---

# PERTEMUAN 5

# N-GRAM DAN LANGUAGE MODEL

---

# A. Tujuan

Mahasiswa mampu:

1. memahami konsep language model;
2. memahami probabilitas kata;
3. membuat unigram;
4. membuat bigram;
5. membuat trigram;
6. menghitung conditional probability;
7. menghitung probabilitas kalimat;
8. memahami context;
9. memahami perplexity;
10. memahami keterbatasan n-gram.

---

# B. Apa Itu Language Model?

Language model mencoba menjawab:

> Seberapa mungkin sebuah sequence kata muncul?

Contoh:

```text
Saya belajar NLP
```

Model ingin mengetahui:

> P("Saya belajar NLP")

---

# C. Probabilitas Sequence

Secara sederhana:

$$
P(w_1,w_2,...,w_n)
$$

dapat diuraikan menjadi:

$$
P(w_1)
P(w_2|w_1)
P(w_3|w_1,w_2)
...
$$

---

# D. Mengapa Membutuhkan Context?

Bandingkan:

```text
Saya makan ...
```

Kata berikutnya mungkin:

```text
nasi
roti
bakso
```

Context membantu model memperkirakan:

> kata berikutnya.

---

# E. Unigram

Unigram hanya memperhatikan satu kata.

Contoh corpus:

```text
saya belajar nlp
saya belajar python
```

Frekuensi:

```text
saya      = 2
belajar   = 2
nlp       = 1
python    = 1
```

---

# F. Membuat Unigram

Gabungkan seluruh teks:

```python
all_text = " ".join(
    df["processed_text"].astype(str)
)
```

Token:

```python
tokens = all_text.split()
```

Hitung frekuensi:

```python
from collections import Counter

unigram = Counter(tokens)

print(unigram.most_common(20))
```

---

# G. Probabilitas Unigram

$$
P(w)=\frac{count(w)}{N}
$$

Implementasi:

```python
total_tokens = sum(unigram.values())

unigram_probability = {
    word: count / total_tokens
    for word, count in unigram.items()
}
```

---

# H. Melihat Probabilitas

```python
for word, probability in list(
    unigram_probability.items()
)[:20]:

    print(word, probability)
```

---

# I. Bigram

Bigram menggunakan:

> dua token berturut-turut.

Contoh:

```text
saya belajar
belajar nlp
```

---

# J. Membuat Bigram

```python
from nltk.util import bigrams

bigram_tokens = list(bigrams(tokens))

bigram_frequency = Counter(bigram_tokens)

print(
    bigram_frequency.most_common(20)
)
```

---

# K. Trigram

Trigram:

> tiga token berturut-turut.

Contoh:

```text
saya belajar nlp
belajar nlp menggunakan
```

---

# L. Membuat Trigram

```python
from nltk.util import trigrams

trigram_tokens = list(trigrams(tokens))

trigram_frequency = Counter(trigram_tokens)

print(
    trigram_frequency.most_common(20)
)
```

---

# M. Conditional Probability

Bigram:

$$
P(w_n|w_{n-1})
$$

Contoh:

$$
P(bermain|saya)
$$

dihitung:

$$
\frac{Count(saya, bermain)}
{Count(saya)}
$$

---

# N. Implementasi Probabilitas Bigram

```python
def bigram_probability(word1, word2):

    numerator = bigram_frequency[(word1, word2)]
    denominator = unigram[word1]

    if denominator == 0:
        return 0

    return numerator / denominator
```

---

# O. Menguji Probabilitas

```python
probability = bigram_probability(
    "saya",
    "belajar"
)

print(probability)
```

---

# P. Sentence Probability

Misalkan:

```text
saya belajar nlp
```

Secara sederhana:

$$
P(saya)
\times
P(belajar|saya)
\times
P(nlp|belajar)
$$

---

# Q. Fungsi Sentence Probability

```python
def sentence_probability(sentence):

    words = sentence.lower().split()

    probability = 1.0

    for i in range(len(words) - 1):

        p = bigram_probability(
            words[i],
            words[i + 1]
        )

        probability *= p

    return probability
```

---

# R. Pengujian

```python
sentence = "saya belajar nlp"

p = sentence_probability(sentence)

print("Probability:", p)
```

---

# S. Masalah Zero Probability

Misalkan:

```text
saya belajar deep learning
```

tetapi:

```text
deep learning
```

tidak pernah muncul dalam corpus.

Maka:

```text
P(learning | deep) = 0
```

Akibatnya:

```text
P(sentence) = 0
```

Walaupun kalimat tersebut sebenarnya masuk akal.

---

# T. Smoothing

Smoothing digunakan untuk:

> menghindari probabilitas nol.

Salah satu konsep sederhana:

> Laplace smoothing.

---

# U. Laplace Smoothing

Formula:

$$
P(w_i|w_{i-1})
=
\frac{C(w_{i-1},w_i)+1}
{C(w_{i-1})+V}
$$

dengan:

* C = count;
* V = ukuran vocabulary.

---

# V. Implementasi

```python
vocab_size = len(unigram)

def smoothed_bigram_probability(
    word1,
    word2
):

    numerator = (
        bigram_frequency[(word1, word2)] + 1
    )

    denominator = (
        unigram[word1] + vocab_size
    )

    return numerator / denominator
```

---

# W. Prediksi Kata Sederhana

Misalnya context:

```text
saya
```

Cari kata yang memiliki probabilitas tertinggi:

```python
candidates = []

for (word1, word2), count in bigram_frequency.items():

    if word1 == "saya":

        p = smoothed_bigram_probability(
            word1,
            word2
        )

        candidates.append(
            (word2, p)
        )
```

Urutkan:

```python
candidates.sort(
    key=lambda x: x[1],
    reverse=True
)

print(candidates[:10])
```

---

# X. Perplexity

Perplexity digunakan untuk mengukur:

> seberapa baik language model memprediksi sequence.

Secara sederhana:

$$
PP(W)
=
P(W)^{-1/N}
$$

Interpretasi umum:

> perplexity lebih rendah menunjukkan model lebih baik dalam memprediksi data evaluasi, dengan catatan model dan dataset pembanding sama.

---

# Y. Menghitung Perplexity Sederhana

```python
import math

def perplexity(probability, N):

    if probability == 0:
        return float("inf")

    return probability ** (-1 / N)
```

Contoh:

```python
p = 0.0001
N = 4

print(perplexity(p, N))
```

---

# Z. Eksperimen Language Model

Mahasiswa membuat:

### Model 1

Unigram

### Model 2

Bigram

### Model 3

Trigram

Kemudian membandingkan:

* jumlah vocabulary;
* jumlah kombinasi;
* context;
* kemampuan prediksi;
* sparsity.

---

# AA. Keterbatasan N-Gram

N-gram memiliki beberapa masalah.

### 1. Sparsity

Banyak kombinasi kata tidak pernah muncul.

### 2. Context terbatas

Bigram hanya melihat satu kata sebelumnya.

Trigram hanya melihat dua kata sebelumnya.

### 3. Vocabulary

Vocabulary besar menghasilkan kombinasi sangat banyak.

### 4. Tidak memahami makna secara mendalam

Contoh:

```text
mobil
```

dan:

```text
kendaraan
```

dianggap token berbeda.

---

# AB. N-Gram vs Neural Language Model

```text
N-Gram
   ↓
Frekuensi
   ↓
Probabilitas
```

Sedangkan neural language model:

```text
Text
 ↓
Embedding
 ↓
Neural Network
 ↓
Contextual Representation
 ↓
Prediction
```

Pembahasan neural language model akan dilanjutkan pada Modul 2 dan Modul 3.

---

# AC. Tugas Pertemuan 5

Mahasiswa harus:

1. Membuat unigram.
2. Membuat bigram.
3. Membuat trigram.
4. Menghitung probabilitas.
5. Menguji minimal 5 kalimat.
6. Menghitung perplexity.
7. Membuat prediksi kata sederhana.
8. Menganalisis zero probability.
9. Menerapkan smoothing.
10. Membandingkan unigram, bigram, dan trigram.

---

# AD. Tabel Analisis

Buat tabel:

| Model   | Context | Kelebihan             | Kekurangan             |
| ------- | ------- | --------------------- | ---------------------- |
| Unigram | 1 kata  | sederhana             | tidak memahami konteks |
| Bigram  | 2 kata  | lebih kontekstual     | context pendek         |
| Trigram | 3 kata  | context lebih panjang | sparsity meningkat     |

---

# AE. Pertanyaan Diskusi

1. Mengapa unigram tidak cukup untuk memahami bahasa?
2. Mengapa bigram lebih baik daripada unigram?
3. Mengapa trigram dapat mengalami sparsity?
4. Apa yang terjadi jika sebuah bigram belum pernah muncul?
5. Mengapa smoothing diperlukan?
6. Apa hubungan n-gram dengan language model?
7. Apa keterbatasan n-gram dibandingkan neural language model?
8. Mengapa embedding dibutuhkan pada neural NLP?

---

# Output Pertemuan 5

File:

```text
ngram_language_model.ipynb
```

Output:

> Mini language model berbasis unigram, bigram, dan trigram.

---

# TUGAS AKHIR MODUL 1

# TEXT PROCESSING PIPELINE

Pada akhir Modul 1 mahasiswa harus menggabungkan seluruh konsep.

Pipeline:

```text
                    DATASET CSV
                         ↓
                    LOAD DATA
                         ↓
                  CORPUS EXPLORATION
                         ↓
                    RAW TEXT
                         ↓
                  SENTENCE SEGMENT
                         ↓
                    TOKENIZATION
                         ↓
                    NORMALIZATION
                         ↓
                   STOPWORD REMOVAL
                         ↓
                      STEMMING
                         ↓
                  LINGUISTIC ANALYSIS
                         ↓
                     N-GRAM
                         ↓
                 LANGUAGE MODEL
```

---

# A. Struktur Notebook Final

Mahasiswa membuat:

```text
modul1_final.ipynb
```

Isi notebook:

```text
1. Dataset Description
2. Corpus Exploration
3. Data Cleaning
4. Sentence Segmentation
5. Tokenization
6. Normalization
7. Stopword Removal
8. Stemming
9. POS Tagging
10. Dependency Parsing
11. N-Gram
12. Language Model
13. Visualization
14. Analysis
15. Conclusion
```

---

# B. Dataset Description

Mahasiswa wajib menjelaskan:

```text
Nama Dataset:
Sumber:
URL Kaggle:
Jumlah Data:
Jumlah Kolom:
Kolom Teks:
Kolom Label:
Bahasa:
Jenis Data:
```

---

# C. Contoh Format

```markdown
## Dataset Description

Nama Dataset:
Indonesian Hotel Reviews

Sumber:
Kaggle

Bahasa:
Bahasa Indonesia

Jumlah Data:
10.000

Kolom Teks:
review

Kolom Label:
sentiment
```

Mahasiswa harus menggunakan informasi sesuai dataset masing-masing.

---

# D. Statistik Corpus

Minimal tampilkan:

```text
Jumlah dokumen
Jumlah kalimat
Jumlah token
Jumlah karakter
Rata-rata kata/dokumen
Dokumen terpendek
Dokumen terpanjang
```

---

# E. Analisis Preprocessing

Mahasiswa harus menunjukkan contoh:

```text
RAW TEXT

↓

CLEAN TEXT

↓

TOKENIZED TEXT

↓

STOPWORD REMOVAL

↓

STEMMED TEXT
```

Minimal 10 contoh.

---

# F. Analisis Linguistik

Minimal 10 kalimat.

Tampilkan:

```text
Token
POS
Dependency
Head
```

Kemudian berikan analisis terhadap minimal:

> 3 kesalahan linguistic processing.

---

# G. Analisis N-Gram

Tampilkan:

### Top 20 Unigram

```text
kata → frekuensi
```

### Top 20 Bigram

```text
kata1 kata2 → frekuensi
```

### Top 20 Trigram

```text
kata1 kata2 kata3 → frekuensi
```

---

# H. Visualisasi

Minimal tiga visualisasi:

1. Distribusi panjang dokumen.
2. Distribusi label jika tersedia.
3. Frekuensi unigram.

Contoh:

```python
top_words = unigram.most_common(20)

words = [x[0] for x in top_words]
counts = [x[1] for x in top_words]

plt.figure(figsize=(12, 6))

plt.bar(words, counts)

plt.xticks(rotation=45)

plt.xlabel("Kata")
plt.ylabel("Frekuensi")
plt.title("20 Kata Paling Sering Muncul")

plt.show()
```

---

# I. Kesimpulan Modul 1

Mahasiswa harus menulis kesimpulan sendiri.

Kesimpulan minimal menjawab:

1. Apa karakteristik dataset?
2. Apa masalah terbesar pada data?
3. Bagaimana preprocessing memengaruhi teks?
4. Apa masalah tokenisasi?
5. Apa masalah stemming?
6. Apa hasil linguistic processing?
7. Apa pola yang ditemukan melalui n-gram?
8. Apa keterbatasan n-gram?

---

# J. Ketentuan Pengumpulan

Struktur:

```text
nama_nim_modul1/
│
├── modul1_final.ipynb
│
├── dataset/
│   └── dataset.csv
│
├── output/
│   └── dataset_preprocessed.csv
│
└── README.md
```

Jika ukuran dataset terlalu besar:

> dataset tidak perlu dimasukkan ke repository GitHub.

Cukup cantumkan:

* nama dataset;
* sumber Kaggle;
* link dataset;
* cara mahasiswa lain memperoleh dataset.

---

# K. Ketentuan Notebook

Notebook harus dapat dijalankan dari:

> **Cell pertama sampai cell terakhir tanpa error.**

Mahasiswa tidak diperbolehkan hanya mengumpulkan:

```text
screenshot hasil
```

Notebook harus berisi:

* kode;
* output;
* penjelasan;
* analisis.

---

# L. Prinsip Penting Praktikum

## 1. Jangan hanya menjalankan kode

Mahasiswa harus memahami:

> mengapa kode tersebut digunakan.

---

## 2. Jangan menghapus informasi tanpa alasan

Contoh:

```text
tidak
```

bisa sangat penting untuk sentiment analysis.

---

## 3. Jangan menganggap preprocessing selalu sama

Pipeline:

```text
Cleaning
→ Stopword
→ Stemming
```

bukan aturan mutlak.

Pipeline harus disesuaikan dengan:

> dataset + task NLP.

---

## 4. Jangan menganggap hasil tokenizer selalu benar

Tokenizer adalah alat.

Mahasiswa harus:

> memeriksa hasilnya.

---

# M. Checklist Mahasiswa

Sebelum mengumpulkan Modul 1:

```text
[ ] Dataset berhasil dibaca
[ ] Corpus berhasil dieksplorasi
[ ] Missing value diperiksa
[ ] Duplicate diperiksa
[ ] Jumlah dokumen dihitung
[ ] Jumlah kata dihitung
[ ] Jumlah kalimat dihitung
[ ] Sentence segmentation dilakukan
[ ] Word tokenization dilakukan
[ ] Character tokenization dilakukan
[ ] Case folding dilakukan
[ ] Cleaning dilakukan
[ ] Stopword dianalisis
[ ] Stemming dilakukan
[ ] POS tagging dilakukan
[ ] Dependency parsing dilakukan
[ ] Unigram dibuat
[ ] Bigram dibuat
[ ] Trigram dibuat
[ ] Probabilitas dihitung
[ ] Smoothing dipahami
[ ] Perplexity dihitung
[ ] Visualisasi dibuat
[ ] Error analysis dilakukan
[ ] Kesimpulan ditulis
```

---

# N. Pertanyaan Refleksi Modul 1

Jawab dengan bahasa sendiri.

### 1.

Mengapa komputer tidak dapat langsung memahami teks mentah?

### 2.

Apa perbedaan antara:

```text
document
sentence
word
token
character
```

### 3.

Mengapa tokenization merupakan tahap penting dalam NLP?

### 4.

Apakah semua punctuation harus dihapus?

Jelaskan berdasarkan dataset yang digunakan.

### 5.

Apakah semua stopword harus dihapus?

Berikan contoh ketika penghapusan stopword dapat mengubah makna.

### 6.

Apa perbedaan stemming dan lemmatization?

### 7.

Apa hubungan morphology dengan stemming?

### 8.

Apa manfaat POS Tagging?

### 9.

Apa manfaat dependency parsing?

### 10.

Mengapa n-gram mengalami masalah sparsity?

### 11.

Apa perbedaan unigram, bigram, dan trigram?

### 12.

Mengapa neural language model diperlukan setelah n-gram?

---

# O. KOMPETENSI AKHIR MODUL 1

Setelah menyelesaikan modul ini, mahasiswa harus memahami bahwa NLP bukan langsung:

```text
Text → AI
```

tetapi melalui proses:

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

Modul 1 menjadi fondasi untuk Modul 2.

Pada Modul 2 mahasiswa akan mempelajari bagaimana teks yang sudah diproses dapat direpresentasikan menjadi bentuk numerik dan kemudian digunakan dalam:

```text
TF-IDF
   ↓
Word Embedding
   ↓
Word2Vec
   ↓
Contextual Representation
   ↓
RNN
   ↓
LSTM
   ↓
GRU
   ↓
Attention
```

Kemudian pada Modul 3:

```text
Attention
   ↓
Transformer
   ↓
BERT / IndoBERT
   ↓
Pretrained Language Model
   ↓
Fine-Tuning
   ↓
NLP Application
```

---

# END OF MODUL 1

**Output akhir:**

```text
Dataset
    ↓
Corpus Exploration
    ↓
Tokenization
    ↓
Normalization
    ↓
Linguistic Processing
    ↓
N-Gram
    ↓
Language Model
```

**File utama:**

```text
modul1_final.ipynb
```

**Target kompetensi:**

> Mahasiswa mampu mengambil dataset teks nyata, memahami karakteristiknya, melakukan text preprocessing, melakukan analisis linguistik dasar, dan membangun language model sederhana berbasis n-gram.

```
```
