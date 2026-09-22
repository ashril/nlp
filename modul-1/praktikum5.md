# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 1 — PERTEMUAN 5
## N-Gram dan Language Model

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 1 — Fundamental dan Text Processing |
| Pertemuan | 5 |
| Topik | N-Gram dan Language Model |
| Output | `ngram_language_model.ipynb` |

---

## B. Tujuan

Mahasiswa mampu:

1. memahami konsep language model;
2. memahami probabilitas kata dalam sequence;
3. membuat unigram, bigram, dan trigram;
4. menghitung conditional probability;
5. menghitung probabilitas kalimat;
6. memahami masalah zero probability dan smoothing;
7. membuat prediksi kata sederhana;
8. menghitung perplexity;
9. memahami keterbatasan n-gram.

---

## C. Materi Teori

### 1. Apa Itu Language Model?

Language model (model bahasa) mencoba menjawab pertanyaan:

> *Seberapa mungkin sebuah sequence kata muncul?*

Contoh:

```text
P("Saya belajar NLP")   → probabilitas tinggi
P("belajar Saya NLP")   → probabilitas rendah
P("xkjz qqrt mnop")     → probabilitas mendekati nol
```

Aplikasi language model:
* Autocomplete (prediksi kata berikutnya)
* Spell checker
* Machine translation
* Text generation

### 2. Probabilitas Sequence

Secara matematis, probabilitas sebuah sequence kata:

$$P(w_1, w_2, ..., w_n)$$

dapat diuraikan menggunakan chain rule:

$$P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1,w_2) \cdot ... \cdot P(w_n|w_1,...,w_{n-1})$$

### 3. N-Gram

N-gram adalah sequence berurutan dari N token.

| N | Nama | Contoh |
|---|------|--------|
| 1 | Unigram | `saya`, `belajar`, `nlp` |
| 2 | Bigram | `saya belajar`, `belajar nlp` |
| 3 | Trigram | `saya belajar nlp` |
| N | N-gram | sequence N token berturut-turut |

### 4. Asumsi Markov

N-gram menggunakan asumsi Markov: probabilitas kata berikutnya hanya bergantung pada N-1 kata sebelumnya.

**Bigram (Markov order 1):**

$$P(w_n | w_1,...,w_{n-1}) \approx P(w_n | w_{n-1})$$

**Trigram (Markov order 2):**

$$P(w_n | w_1,...,w_{n-1}) \approx P(w_n | w_{n-2}, w_{n-1})$$

### 5. Probabilitas Unigram

$$P(w) = \frac{Count(w)}{N}$$

dengan N = total token dalam corpus.

### 6. Probabilitas Bigram (Conditional Probability)

$$P(w_n | w_{n-1}) = \frac{Count(w_{n-1}, w_n)}{Count(w_{n-1})}$$

### 7. Masalah Zero Probability

Jika bigram `(deep, learning)` tidak pernah muncul dalam corpus:

```text
P(learning | deep) = 0
```

Akibatnya probabilitas seluruh kalimat menjadi nol, meskipun kalimatnya masuk akal secara linguistik.

### 8. Laplace Smoothing

Smoothing digunakan untuk menghindari probabilitas nol dengan menambahkan nilai kecil pada semua frekuensi:

$$P(w_i | w_{i-1}) = \frac{Count(w_{i-1}, w_i) + 1}{Count(w_{i-1}) + V}$$

dengan V = ukuran vocabulary (jumlah token unik).

### 9. Perplexity

Perplexity mengukur seberapa baik language model memprediksi data:

$$PP(W) = P(W)^{-1/N}$$

> Perplexity lebih rendah menunjukkan model lebih baik dalam memprediksi data evaluasi (dengan dataset dan model pembanding yang sama).

---

## D. Praktikum

### 1. Membuat Notebook

Buat file:

```text
pertemuan-05/ngram_language_model.ipynb
```

### 2. Import Library

```python
import pandas as pd
import math
from collections import Counter
from nltk.util import bigrams, trigrams, ngrams
```

### 3. Load Dataset

```python
df = pd.read_csv("../dataset/dataset_preprocessed.csv")
text_column = "processed_text"  # gunakan hasil preprocessing

# Hapus nilai kosong
df = df.dropna(subset=[text_column])

print(f"Jumlah dokumen: {len(df)}")
```

### 4. Membuat Corpus Token

Gabungkan seluruh teks menjadi satu corpus:

```python
all_text = " ".join(df[text_column].astype(str))
tokens = all_text.split()

print(f"Total token: {len(tokens)}")
print(f"Contoh token: {tokens[:20]}")
```

### 5. Membuat Unigram

```python
unigram = Counter(tokens)

print("20 Kata Paling Sering Muncul:")
for word, count in unigram.most_common(20):
    print(f"  {word:<20} : {count}")
```

### 6. Probabilitas Unigram

```python
total_tokens = sum(unigram.values())

unigram_probability = {
    word: count / total_tokens
    for word, count in unigram.items()
}

print("\n10 Kata dengan Probabilitas Tertinggi:")
sorted_prob = sorted(
    unigram_probability.items(),
    key=lambda x: x[1],
    reverse=True
)
for word, prob in sorted_prob[:10]:
    print(f"  {word:<20} : {prob:.6f}")
```

### 7. Membuat Bigram

```python
bigram_tokens = list(bigrams(tokens))
bigram_frequency = Counter(bigram_tokens)

print("20 Bigram Paling Sering Muncul:")
for bigram, count in bigram_frequency.most_common(20):
    print(f"  {' '.join(bigram):<30} : {count}")
```

### 8. Membuat Trigram

```python
trigram_tokens = list(trigrams(tokens))
trigram_frequency = Counter(trigram_tokens)

print("20 Trigram Paling Sering Muncul:")
for trigram, count in trigram_frequency.most_common(20):
    print(f"  {' '.join(trigram):<40} : {count}")
```

### 9. Conditional Probability Bigram

$$P(w_n | w_{n-1}) = \frac{Count(w_{n-1}, w_n)}{Count(w_{n-1})}$$

```python
def bigram_probability(word1, word2):
    numerator = bigram_frequency[(word1, word2)]
    denominator = unigram[word1]

    if denominator == 0:
        return 0.0

    return numerator / denominator

# Contoh
word1, word2 = "belajar", "nlp"
prob = bigram_probability(word1, word2)
print(f"P({word2} | {word1}) = {prob:.6f}")
```

### 10. Probabilitas Kalimat

Menggunakan bigram untuk menghitung probabilitas kalimat:

```python
def sentence_probability(sentence):
    words = sentence.lower().split()
    probability = 1.0

    for i in range(len(words) - 1):
        p = bigram_probability(words[i], words[i + 1])
        probability *= p

    return probability

# Uji
test_sentences = [
    "saya belajar nlp",
    "nlp menarik sekali",
    "model bahasa sederhana",
]

for sentence in test_sentences:
    p = sentence_probability(sentence)
    print(f"P('{sentence}') = {p:.8f}")
```

### 11. Laplace Smoothing

```python
vocab_size = len(unigram)

def smoothed_bigram_probability(word1, word2):
    numerator = bigram_frequency[(word1, word2)] + 1
    denominator = unigram[word1] + vocab_size
    return numerator / denominator

# Uji dengan kata yang mungkin tidak ada dalam corpus
word1, word2 = "deep", "learning"
prob = smoothed_bigram_probability(word1, word2)
print(f"Smoothed P({word2} | {word1}) = {prob:.8f}")
```

### 12. Prediksi Kata Sederhana

Diberikan kata terakhir, prediksi kata berikutnya:

```python
def predict_next_word(context_word, top_n=5):
    candidates = []

    for (word1, word2), count in bigram_frequency.items():
        if word1 == context_word:
            p = smoothed_bigram_probability(word1, word2)
            candidates.append((word2, p))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_n]

# Contoh
context = "belajar"
predictions = predict_next_word(context)
print(f"\nPrediksi kata setelah '{context}':")
for word, prob in predictions:
    print(f"  {word:<20} : {prob:.6f}")
```

### 13. Menghitung Perplexity

```python
def perplexity(probability, N):
    if probability == 0 or probability is None:
        return float("inf")
    return probability ** (-1 / N)

# Contoh
sentence = "saya belajar nlp"
words = sentence.split()
prob = sentence_probability(sentence)
N = len(words)

pp = perplexity(prob, N)
print(f"Kalimat : '{sentence}'")
print(f"P(W)    : {prob:.8f}")
print(f"N       : {N}")
print(f"PP(W)   : {pp:.4f}")
```

### 14. Perbandingan Unigram, Bigram, Trigram

```python
print("=" * 60)
print("PERBANDINGAN MODEL N-GRAM")
print("=" * 60)
print(f"Ukuran Vocabulary     : {len(unigram):,}")
print(f"Jumlah Token          : {len(tokens):,}")
print(f"Jumlah Unigram unik   : {len(unigram):,}")
print(f"Jumlah Bigram unik    : {len(bigram_frequency):,}")
print(f"Jumlah Trigram unik   : {len(trigram_frequency):,}")
print()

# Cek sparsity
max_bigram = len(unigram) ** 2
max_trigram = len(unigram) ** 3
print(f"Kemungkinan Bigram maks  : {max_bigram:,}")
print(f"Bigram yang ada          : {len(bigram_frequency):,}")
print(f"Bigram coverage          : {len(bigram_frequency)/max_bigram*100:.4f}%")
```

### 15. Visualisasi

```python
import matplotlib.pyplot as plt

# Top 20 unigram
top_words = unigram.most_common(20)
words_list = [w[0] for w in top_words]
counts_list = [w[1] for w in top_words]

plt.figure(figsize=(12, 5))
plt.bar(words_list, counts_list, color="steelblue")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Kata")
plt.ylabel("Frekuensi")
plt.title("20 Kata Paling Sering Muncul (Unigram)")
plt.tight_layout()
plt.show()
```

---

## E. Analisis dan Perbandingan

Buat tabel perbandingan:

| Model | Context | Kelebihan | Kekurangan |
|-------|---------|-----------|------------|
| Unigram | 1 kata | Sederhana, cepat | Tidak memahami konteks |
| Bigram | 2 kata | Lebih kontekstual | Context sangat pendek |
| Trigram | 3 kata | Context lebih panjang | Sparsity meningkat drastis |

### Keterbatasan N-Gram

1. **Sparsity** — Banyak kombinasi kata tidak pernah muncul dalam corpus
2. **Context terbatas** — Bigram hanya melihat 1 kata sebelumnya, trigram 2 kata
3. **Tidak memahami makna** — `mobil` dan `kendaraan` dianggap token yang sama sekali berbeda
4. **Vocabulary besar** — Jumlah kombinasi meledak seiring bertambahnya N

---

## F. Tugas Pertemuan 5

Mahasiswa harus:

1. Membuat unigram dari dataset preprocessed.
2. Membuat bigram dari dataset preprocessed.
3. Membuat trigram dari dataset preprocessed.
4. Menghitung probabilitas minimal 5 kata.
5. Menguji sentence probability pada minimal 5 kalimat.
6. Menghitung perplexity untuk setiap kalimat uji.
7. Membuat prediksi kata untuk minimal 5 context word.
8. Menerapkan Laplace smoothing.
9. Menganalisis zero probability problem.
10. Membandingkan unigram, bigram, dan trigram.

---

## G. Pertanyaan Diskusi

1. Mengapa unigram tidak cukup untuk memahami bahasa?
2. Mengapa bigram lebih baik daripada unigram dalam konteks prediksi?
3. Mengapa trigram dapat mengalami masalah sparsity yang lebih parah dari bigram?
4. Apa yang terjadi jika sebuah bigram belum pernah muncul dalam corpus?
5. Mengapa Laplace smoothing diperlukan?
6. Apa hubungan n-gram dengan konsep language model?
7. Apa keterbatasan mendasar n-gram dibandingkan neural language model?
8. Mengapa embedding diperlukan pada neural NLP yang tidak ada dalam n-gram?

---

## H. Output Pertemuan 5

File yang dikumpulkan:

```text
ngram_language_model.ipynb
```

> Output: Mini language model berbasis unigram, bigram, dan trigram yang mampu menghitung probabilitas, memprediksi kata, dan mengevaluasi dengan perplexity.

---

## I. Tugas Akhir Modul 1 — TEXT PROCESSING PIPELINE

Pada akhir Modul 1, mahasiswa harus menggabungkan **seluruh konsep** dari pertemuan 1–5 menjadi satu pipeline lengkap.

### Pipeline End-to-End

```text
DATASET CSV
     ↓
LOAD DATA
     ↓
CORPUS EXPLORATION
     ↓
RAW TEXT
     ↓
SENTENCE SEGMENTATION
     ↓
TOKENIZATION
     ↓
NORMALIZATION
     ↓
STOPWORD REMOVAL
     ↓
STEMMING
     ↓
LINGUISTIC ANALYSIS (POS, Parsing)
     ↓
N-GRAM
     ↓
LANGUAGE MODEL
```

### Struktur Notebook Final

Buat file:

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

### Format Dataset Description

Mahasiswa wajib menjelaskan:

```text
Nama Dataset   :
Sumber         :
URL Kaggle     :
Jumlah Data    :
Jumlah Kolom   :
Kolom Teks     :
Kolom Label    :
Bahasa         :
Jenis Data     :
```

### Statistik Corpus yang Harus Ditampilkan

```text
Jumlah dokumen
Jumlah kalimat
Jumlah token
Jumlah karakter
Rata-rata kata/dokumen
Dokumen terpendek
Dokumen terpanjang
```

### Visualisasi yang Harus Dibuat (Minimal 3)

1. Distribusi panjang dokumen
2. Distribusi label (jika tersedia)
3. Frekuensi unigram (top 20 kata)

### Analisis N-Gram yang Harus Ditampilkan

* Top 20 Unigram (kata → frekuensi)
* Top 20 Bigram (pasangan kata → frekuensi)
* Top 20 Trigram (tiga kata → frekuensi)

### Pertanyaan Refleksi Modul 1

Mahasiswa harus menjawab dengan kalimat sendiri:

1. Mengapa komputer tidak dapat langsung memahami teks mentah?
2. Apa perbedaan antara document, sentence, word, token, dan character?
3. Mengapa tokenization merupakan tahap penting dalam NLP?
4. Apakah semua punctuation harus dihapus? Jelaskan berdasarkan dataset yang digunakan.
5. Apakah semua stopword harus dihapus? Berikan contoh ketika penghapusan stopword mengubah makna.
6. Apa perbedaan stemming dan lemmatization?
7. Apa hubungan morphology dengan stemming?
8. Apa manfaat POS Tagging dalam NLP?
9. Apa manfaat dependency parsing?
10. Mengapa n-gram mengalami masalah sparsity?
11. Apa perbedaan unigram, bigram, dan trigram?
12. Mengapa neural language model diperlukan setelah n-gram?

---

## J. Ketentuan Pengumpulan

### Struktur Folder

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

> Jika ukuran dataset terlalu besar, tidak perlu dimasukkan ke GitHub. Cukup cantumkan nama dataset, sumber Kaggle, link, dan cara memperolehnya.

### Ketentuan Notebook

Notebook harus dapat dijalankan dari **cell pertama sampai cell terakhir tanpa error**.

Notebook harus berisi:
* kode yang berfungsi;
* output yang terlihat;
* penjelasan (markdown cell);
* analisis dan kesimpulan.

---

## K. Prinsip Penting Praktikum

**1. Jangan hanya menjalankan kode**
> Mahasiswa harus memahami *mengapa* kode tersebut digunakan, bukan sekadar menyalin dan menjalankan.

**2. Jangan menghapus informasi tanpa alasan**
> Contoh: kata `"tidak"` bisa sangat penting untuk sentiment analysis.

**3. Jangan menganggap preprocessing selalu sama**
> Pipeline `Cleaning → Stopword → Stemming` bukan aturan mutlak. Pipeline harus disesuaikan dengan dataset dan task NLP.

**4. Jangan menganggap hasil tokenizer selalu benar**
> Tokenizer adalah alat. Mahasiswa harus memeriksa dan menganalisis hasilnya.

---

## L. Checklist Mahasiswa

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

## M. Kompetensi Akhir Modul 1

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

Modul 1 menjadi fondasi untuk:

* **Modul 2** — Representasi Teks (TF-IDF, Word Embedding, RNN, LSTM, Attention)
* **Modul 3** — Transformer, Pretrained Model, dan Aplikasi NLP

---

> **Pertemuan sebelumnya:** [Pertemuan 4 — Linguistic Processing](praktikum4.md)
>
> **Modul berikutnya:** [Modul 2 — Representasi Teks dan Neural NLP](../modul-2/README.md)
