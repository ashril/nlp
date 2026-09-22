# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 1 — PERTEMUAN 2
## Tokenization

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 1 — Fundamental dan Text Processing |
| Pertemuan | 2 |
| Topik | Tokenization |
| Output | `tokenization.ipynb` |

---

## B. Tujuan

Mahasiswa mampu:

* memahami konsep token;
* melakukan sentence segmentation;
* melakukan word tokenization;
* melakukan character tokenization;
* memahami subword tokenization;
* membandingkan tokenizer yang berbeda.

---

## C. Materi Teori

### 1. Apa Itu Token?

Token adalah unit teks yang diproses oleh sistem NLP.

Contoh kalimat:

```text
Saya belajar NLP
```

dapat menjadi token:

```text
["Saya", "belajar", "NLP"]
```

Token tidak selalu sama dengan kata.

### 2. Level Token

Token dapat berada pada berbagai level:

**Character:**
```text
S, a, y, a
```

**Word:**
```text
Saya, belajar, NLP
```

**Subword:**

Contoh kata `bermainnya` dapat dipecah menjadi beberapa subword tergantung tokenizer yang digunakan. Subword tokenization digunakan oleh model Transformer modern (BPE, WordPiece, SentencePiece).

### 3. Jenis Tokenization

| Jenis | Deskripsi | Contoh Input | Contoh Output |
|-------|-----------|-------------|--------------|
| Word tokenization | Memecah teks menjadi kata | `"Saya belajar NLP."` | `["Saya", "belajar", "NLP", "."]` |
| Sentence tokenization | Memecah teks menjadi kalimat | `"Saya belajar. NLP menarik."` | `["Saya belajar.", "NLP menarik."]` |
| Character tokenization | Memecah teks menjadi karakter | `"NLP"` | `["N", "L", "P"]` |
| Subword tokenization | Memecah kata menjadi subword | `"bermainnya"` | bergantung tokenizer |

### 4. Permasalahan Tokenisasi Bahasa Indonesia

Bahasa Indonesia memiliki tantangan khusus dalam tokenisasi:

* **Variasi ejaan informal:** `tidak`, `nggak`, `gak`, `ga`, `tdk` memiliki makna yang sama
* **Imbuhan kompleks:** `mempelajarinya`, `ketidakhadiran`
* **Campuran bahasa:** campur kode Indonesia-Inggris, Indonesia-daerah
* **Singkatan:** `yg`, `dgn`, `utk`, `krn`
* **Tanda baca ganda:** `!!!`, `...`, `???`

Contoh:

```text
Input : "pelayanannya bagus banget!!!"
Output: ["pelayanannya", "bagus", "banget", "!", "!", "!"]
```

---

## D. Praktikum

### 1. Membuat Notebook

Buat file:

```text
pertemuan-02/tokenization.ipynb
```

### 2. Import Library

```python
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
```

### 3. Load Dataset

```python
df = pd.read_csv("../dataset/dataset.csv")
text_column = "text"  # sesuaikan dengan nama kolom
```

### 4. Sentence Segmentation

Memecah teks menjadi kalimat:

```python
text = "Saya belajar NLP. NLP sangat menarik. Saya ingin menjadi NLP Engineer."

sentences = sent_tokenize(text)

for sentence in sentences:
    print(sentence)
```

Output:

```text
Saya belajar NLP.
NLP sangat menarik.
Saya ingin menjadi NLP Engineer.
```

### 5. Word Tokenization

```python
from nltk.tokenize import word_tokenize

text = "Saya belajar Natural Language Processing."

tokens = word_tokenize(text)

print(tokens)
```

Output:

```text
['Saya', 'belajar', 'Natural', 'Language', 'Processing', '.']
```

Perhatikan bahwa tanda baca (`.`) dianggap sebagai token tersendiri.

### 6. Character Tokenization

```python
text = "NLP"

characters = list(text)

print(characters)
```

Output:

```text
['N', 'L', 'P']
```

### 7. Tokenisasi Dataset

Menerapkan word tokenization ke seluruh dataset:

```python
df["tokens"] = df[text_column].astype(str).apply(word_tokenize)

df[[text_column, "tokens"]].head()
```

Menghitung jumlah token per dokumen:

```python
df["jumlah_token"] = df["tokens"].apply(len)
df["jumlah_token"].describe()
```

### 8. Membandingkan Kata dan Token

Hitung jumlah kata dengan split sederhana:

```python
df["jumlah_kata"] = df[text_column].astype(str).apply(
    lambda x: len(x.split())
)
```

Bandingkan:

```python
df[["jumlah_kata", "jumlah_token"]].describe()
```

**Mengapa hasilnya bisa berbeda?**

Karena tokenizer memisahkan tanda baca, simbol, dan karakter tertentu sebagai token tersendiri.

### 9. Eksperimen Tokenisasi

Amati bagaimana tokenizer menangani berbagai jenis teks:

```python
texts = [
    "Saya suka produk ini.",
    "Bagus banget!!!",
    "Tidak bagus...",
    "Harga Rp50.000.",
    "email@example.com",
    "https://example.com",
    "yg bagus bgt sih",
    "aku udah beli nih kak"
]

for text in texts:
    print("TEXT :", text)
    print("TOKEN:", word_tokenize(text))
    print()
```

### 10. Perbandingan Tokenizer

Jika menggunakan spaCy, bandingkan hasilnya:

```python
import spacy

# Load model (sesuaikan dengan model yang tersedia)
# nlp = spacy.load("xx_ent_wiki_sm")  # multilingual

# Tokenisasi dengan spaCy
# doc = nlp("Saya belajar NLP.")
# spacy_tokens = [token.text for token in doc]
```

Buat tabel perbandingan:

| Teks | NLTK word_tokenize | split() |
|------|--------------------|---------|
| Saya belajar NLP. | ... | ... |
| Bagus banget!!! | ... | ... |
| Harga Rp50.000 | ... | ... |

---

## E. Tugas Analisis

Jawab pertanyaan berikut berdasarkan eksperimen:

1. Apakah setiap kata selalu menghasilkan satu token?
2. Bagaimana tokenizer menangani tanda baca?
3. Bagaimana tokenizer menangani angka (contoh: `Rp50.000`)?
4. Bagaimana tokenizer menangani URL?
5. Bagaimana tokenizer menangani singkatan informal (`yg`, `dgn`, `kak`)?
6. Apa masalah tokenisasi yang spesifik untuk Bahasa Indonesia?
7. Mengapa subword tokenizer diperlukan pada model Transformer?

---

## F. Output Pertemuan 2

File yang dikumpulkan:

```text
tokenization.ipynb
```

Notebook berisi:

```text
1. Import Library
2. Load Dataset
3. Sentence Segmentation
4. Word Tokenization
5. Character Tokenization
6. Dataset Tokenization
7. Tokenizer Comparison
8. Analysis
```

> Output: Program text tokenizer yang mampu melakukan sentence, word, dan character tokenization serta analisis hasilnya.

---

## G. Pertanyaan Diskusi

1. Apakah tokenizer yang baik untuk Bahasa Inggris otomatis baik untuk Bahasa Indonesia?
2. Apa yang terjadi jika teks tidak di-tokenize terlebih dahulu?
3. Mengapa tanda baca bisa menjadi token tersendiri?
4. Apa hubungan tokenization dengan vocabulary dalam model NLP?
5. Mengapa model Transformer menggunakan subword tokenization?

---

> **Pertemuan sebelumnya:** [Pertemuan 1 — Eksplorasi Corpus](praktikum1.md)
>
> **Pertemuan berikutnya:** [Pertemuan 3 — Text Normalization](praktikum3.md)
