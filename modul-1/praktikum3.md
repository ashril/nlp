# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 1 — PERTEMUAN 3
## Text Normalization

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 1 — Fundamental dan Text Processing |
| Pertemuan | 3 |
| Topik | Text Normalization |
| Output | `text_normalization.ipynb`, `dataset_preprocessed.csv` |

---

## B. Tujuan

Mahasiswa mampu:

* melakukan case folding;
* membersihkan teks (URL, mention, hashtag, angka, punctuation);
* menangani slang words;
* melakukan stopword removal Bahasa Indonesia;
* melakukan stemming Bahasa Indonesia dengan Sastrawi;
* memahami perbedaan stemming dan lemmatization;
* membangun preprocessing pipeline lengkap.

---

## C. Materi Teori

### 1. Mengapa Normalisasi Diperlukan?

Data teks dunia nyata biasanya tidak bersih:

```text
Input: "PELAYANANNYA BAGUUUS BANGET!!! 😍 https://toko.com"
```

Sistem NLP perlu menentukan informasi mana yang penting dan mana yang dapat dihilangkan sebelum diproses lebih lanjut.

Masalah umum pada data teks:

* Penggunaan huruf kapital yang tidak konsisten
* Tanda baca berlebihan
* URL, mention, hashtag yang tidak relevan
* Angka dalam berbagai format
* Slang dan singkatan informal
* Typographical error

### 2. Tahapan Normalisasi

```text
Raw Text
   ↓
Case Folding (lowercase)
   ↓
Cleaning (URL, mention, hashtag, angka)
   ↓
Punctuation Removal
   ↓
Whitespace Cleaning
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Stemming
   ↓
Processed Text
```

### 3. Stopword

Stopword adalah kata yang sering muncul dan dalam konteks tertentu dianggap kurang informatif.

Contoh stopword Bahasa Indonesia:

```text
yang, dan, di, ke, dari, untuk, dengan, adalah, ini, itu
```

**Perhatian penting:**

> Stopword **tidak selalu** harus dihapus.

Contoh pada sentiment analysis:

```text
Input : "tidak bagus"
Setelah hapus "tidak" → "bagus"
```

Makna berubah secara drastis. Keputusan menghapus stopword harus disesuaikan dengan task NLP.

### 4. Stemming vs Lemmatization

| Aspek | Stemming | Lemmatization |
|-------|----------|---------------|
| Pendekatan | Berbasis aturan pemotongan | Berbasis analisis linguistik |
| Kompleksitas | Lebih sederhana | Lebih kompleks |
| Kecepatan | Cepat | Relatif lebih lambat |
| Hasil | Stem (mungkin bukan kata valid) | Lemma (kata dasar yang valid) |
| Contoh (EN) | `running` → `run` | `better` → `good` |
| Untuk Bahasa Indonesia | Sastrawi (stemmer) | Terbatas, umumnya menggunakan stemmer |

---

## D. Praktikum

### 1. Membuat Notebook

Buat file:

```text
pertemuan-03/text_normalization.ipynb
```

### 2. Import Library

```python
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

### 3. Load Dataset

```python
df = pd.read_csv("../dataset/dataset.csv")
text_column = "text"  # sesuaikan
```

### 4. Case Folding

Mengubah seluruh teks menjadi huruf kecil (lowercase):

```python
def case_folding(text):
    return str(text).lower()

# Contoh
text = "Saya Belajar NLP"
print(case_folding(text))
# Output: saya belajar nlp
```

### 5. Menghapus URL

```python
def remove_url(text):
    return re.sub(r"http\S+|www\S+", "", text)

# Contoh
text = "Kunjungi https://example.com sekarang"
print(remove_url(text))
# Output: Kunjungi  sekarang
```

### 6. Menghapus Mention

```python
def remove_mention(text):
    return re.sub(r"@\w+", "", text)

# Contoh
text = "@user1 produk ini bagus"
print(remove_mention(text))
```

### 7. Menghapus Hashtag

```python
def remove_hashtag(text):
    return re.sub(r"#\w+", "", text)
```

> **Catatan:** Jangan selalu menghapus hashtag. Hashtag seperti `#Lombok` atau `#Pariwisata` dapat mengandung informasi penting tentang topik atau lokasi.

### 8. Menghapus Angka

```python
def remove_number(text):
    return re.sub(r"\d+", "", text)
```

> **Catatan:** Angka dapat memiliki makna penting: `2026`, `Rp100.000`, `5 bintang`. Keputusan menghapus angka harus disesuaikan dengan task.

### 9. Menghapus Punctuation

```python
def remove_punctuation(text):
    return text.translate(
        str.maketrans("", "", string.punctuation)
    )
```

### 10. Membersihkan Whitespace

```python
def clean_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()
```

### 11. Fungsi Cleaning Lengkap

Gabungkan semua fungsi menjadi satu pipeline:

```python
def clean_text(text):
    text = str(text)
    text = text.lower()           # case folding
    text = remove_url(text)       # hapus URL
    text = remove_mention(text)   # hapus mention
    text = remove_hashtag(text)   # hapus hashtag
    text = remove_number(text)    # hapus angka
    text = remove_punctuation(text)  # hapus punctuation
    text = clean_whitespace(text) # bersihkan whitespace
    return text
```

Menerapkan ke dataset:

```python
df["clean_text"] = df[text_column].apply(clean_text)

df[[text_column, "clean_text"]].head(10)
```

### 12. Stopword Removal

```python
stop_words = set(stopwords.words("indonesian"))

def remove_stopwords(tokens):
    return [
        token for token in tokens
        if token not in stop_words
    ]

# Contoh
tokens = ["saya", "sedang", "belajar", "nlp", "yang", "menarik"]
print(remove_stopwords(tokens))
```

### 13. Stemming dengan Sastrawi

```python
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

# Contoh
text = "Mahasiswa sedang mempelajari pemrograman."
print(stemmer.stem(text))
```

Contoh hasil stemming:

```text
Input : mempelajari
Output: ajar

Input : berlari-lari
Output: lari

Input : ketidakhadiran
Output: hadir
```

### 14. Pipeline Preprocessing Lengkap

```python
def preprocessing_pipeline(text):
    # 1. Cleaning
    text = clean_text(text)

    # 2. Tokenization
    tokens = word_tokenize(text)

    # 3. Stopword removal
    tokens = remove_stopwords(tokens)

    # 4. Stemming
    tokens = stem_tokens(tokens)

    return tokens

# Menerapkan ke dataset
df["processed_tokens"] = df[text_column].apply(preprocessing_pipeline)
```

### 15. Menggabungkan Token Kembali

```python
df["processed_text"] = df["processed_tokens"].apply(
    lambda tokens: " ".join(tokens)
)
```

### 16. Perbandingan Raw vs Processed

```python
for i in range(5):
    print("RAW:")
    print(df.iloc[i][text_column])

    print("\nPROCESSED:")
    print(df.iloc[i]["processed_text"])

    print("=" * 80)
```

### 17. Menyimpan Dataset Hasil Preprocessing

```python
output_columns = [text_column, "clean_text", "processed_text"]

df[output_columns].to_csv(
    "../dataset/dataset_preprocessed.csv",
    index=False
)

print("Dataset berhasil disimpan.")
```

---

## E. Tugas Eksperimen

Ambil minimal 20 data dan buat tabel perbandingan:

| Raw Text | Setelah Lowercase | Setelah Cleaning | Setelah Tokenisasi | Setelah Stopword | Setelah Stemming |
|----------|------------------|-----------------|--------------------|-----------------|-----------------|
| ... | ... | ... | ... | ... | ... |

---

## F. Analisis Penting

Jawab pertanyaan berikut:

1. Apakah semua stopword harus dihapus? Berikan contoh ketika stopword penting untuk dipertahankan.
2. Apakah stemming selalu meningkatkan kualitas data? Kapan stemming dapat merugikan?
3. Apakah angka harus selalu dihapus? Berikan contoh kasus di mana angka penting.
4. Apakah URL selalu tidak berguna? Kapan URL dapat menjadi fitur yang informatif?
5. Apakah hashtag harus selalu dihapus?
6. Apakah preprocessing yang sama cocok untuk semua task NLP?

**Kesimpulan yang diharapkan:**

> **Preprocessing bergantung pada karakteristik data dan tujuan NLP task.**

---

## G. Output Pertemuan 3

File yang dikumpulkan:

```text
text_normalization.ipynb
dataset_preprocessed.csv
```

Notebook berisi:

```text
1. Import Library
2. Load Dataset
3. Case Folding
4. URL Removal
5. Mention & Hashtag Removal
6. Punctuation Removal
7. Stopword Removal
8. Stemming
9. Preprocessing Pipeline
10. Before vs After Comparison
11. Save Preprocessed Dataset
12. Analysis
```

---

## H. Pertanyaan Diskusi

1. Mengapa preprocessing berbeda untuk task sentiment analysis vs text classification?
2. Apa dampak menghapus kata `"tidak"` dalam kalimat `"tidak bagus"`?
3. Bagaimana cara menangani slang Bahasa Indonesia (`gue`, `lu`, `bgt`)?
4. Apakah Sastrawi cocok digunakan untuk semua domain teks?
5. Apa perbedaan pendekatan normalisasi untuk teks Twitter vs teks berita formal?

---

> **Pertemuan sebelumnya:** [Pertemuan 2 — Tokenization](praktikum2.md)
>
> **Pertemuan berikutnya:** [Pertemuan 4 — Linguistic Processing](praktikum4.md)
