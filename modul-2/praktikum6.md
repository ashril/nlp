# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 2 — PERTEMUAN 6
## Representasi Teks: Bag of Words dan TF-IDF

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 2 — Representasi Teks dan Neural NLP |
| Pertemuan | 6 |
| Topik | Representasi Teks: BoW dan TF-IDF |
| Output | `representasi_teks.ipynb` |

---

## B. Tujuan

Mahasiswa mampu:

* memahami mengapa teks perlu diubah ke bentuk numerik;
* membuat representasi Bag of Words (BoW);
* menghitung Term Frequency (TF) dan Inverse Document Frequency (IDF);
* membuat matriks TF-IDF menggunakan scikit-learn;
* menghitung cosine similarity antar dokumen;
* memahami keterbatasan representasi berbasis frekuensi.

---

## C. Materi Teori

### 1. Mengapa Teks Harus Diubah ke Numerik?

Model Machine Learning dan Neural Network tidak dapat memproses teks mentah secara langsung. Teks harus diubah menjadi representasi numerik.

```text
"Saya belajar NLP"  →  [0, 1, 0, 1, 0, 1]  (BoW)
                    →  [0.0, 0.45, 0.0, 0.63, 0.0, 0.45]  (TF-IDF)
```

### 2. Vocabulary

Vocabulary adalah kumpulan semua kata unik dalam corpus.

Contoh corpus:

```text
Dokumen 1: "Saya belajar NLP"
Dokumen 2: "Saya belajar Python"
Dokumen 3: "NLP menggunakan Python"
```

Vocabulary:

```text
["Saya", "belajar", "NLP", "Python", "menggunakan"]
```

### 3. Bag of Words (BoW)

BoW merepresentasikan setiap dokumen sebagai vektor frekuensi kata dari vocabulary.

Contoh:

| Dokumen | Saya | belajar | NLP | Python | menggunakan |
|---------|------|---------|-----|--------|------------|
| D1 | 1 | 1 | 1 | 0 | 0 |
| D2 | 1 | 1 | 0 | 1 | 0 |
| D3 | 0 | 0 | 1 | 1 | 1 |

**Keterbatasan BoW:**
* Tidak mempertimbangkan urutan kata (order)
* Tidak membedakan kata penting dan kata umum
* Dimensi sangat besar (seukuran vocabulary)
* Matriks sangat sparse (banyak nilai nol)

### 4. Term Frequency (TF)

TF mengukur seberapa sering sebuah kata muncul dalam satu dokumen:

$$TF(t, d) = \frac{\text{Jumlah kemunculan } t \text{ dalam } d}{\text{Total kata dalam } d}$$

### 5. Inverse Document Frequency (IDF)

IDF mengukur seberapa langka sebuah kata di seluruh corpus:

$$IDF(t) = \log\left(\frac{N}{df(t)}\right)$$

dengan:
* N = jumlah total dokumen
* df(t) = jumlah dokumen yang mengandung kata t

Kata yang muncul di banyak dokumen (seperti stopword) akan mendapat IDF rendah. Kata yang langka mendapat IDF tinggi.

### 6. TF-IDF

TF-IDF adalah perkalian TF dan IDF:

$$TFIDF(t, d) = TF(t, d) \times IDF(t)$$

TF-IDF memberikan bobot tinggi pada kata yang sering muncul dalam dokumen tertentu, tetapi jarang di seluruh corpus — kata tersebut dianggap kata kunci dokumen.

### 7. Cosine Similarity

Cosine similarity mengukur kemiripan antara dua vektor berdasarkan sudut di antara keduanya:

$$\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{A}| \cdot |\mathbf{B}|}$$

Nilai:
* 1.0 = dokumen identik
* 0.0 = dokumen sama sekali berbeda
* -1.0 = dokumen berlawanan arah (jarang terjadi pada representasi positif)

---

## D. Praktikum

### 1. Membuat Notebook

Buat file:

```text
pertemuan-06/representasi_teks.ipynb
```

### 2. Import Library

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

### 3. Load Dataset

```python
df = pd.read_csv("../dataset/dataset_preprocessed.csv")
text_column = "processed_text"

df = df.dropna(subset=[text_column])

# Gunakan sample untuk eksperimen awal
sample = df[text_column].head(1000).astype(str).tolist()
print(f"Jumlah dokumen sample: {len(sample)}")
```

### 4. Bag of Words dengan CountVectorizer

```python
# Inisialisasi CountVectorizer
vectorizer_bow = CountVectorizer(max_features=5000)

# Fit dan transform
bow_matrix = vectorizer_bow.fit_transform(sample)

print("Shape matriks BoW:", bow_matrix.shape)
print("Jumlah vocabulary :", len(vectorizer_bow.vocabulary_))
```

Melihat vocabulary:

```python
vocab = vectorizer_bow.get_feature_names_out()
print("Contoh vocabulary:", vocab[:20])
```

Melihat representasi satu dokumen:

```python
# Dokumen pertama dalam bentuk dense array
doc_vector = bow_matrix[0].toarray()[0]

# Tampilkan kata yang memiliki nilai > 0
nonzero_indices = np.where(doc_vector > 0)[0]
print("\nDokumen 1:")
print(sample[0])
print("\nRepresentasi BoW (kata dengan frekuensi > 0):")
for idx in nonzero_indices:
    print(f"  {vocab[idx]:<20} : {int(doc_vector[idx])}")
```

### 5. Statistik Matriks BoW

```python
# Sparsity
total_elements = bow_matrix.shape[0] * bow_matrix.shape[1]
nonzero_elements = bow_matrix.nnz
sparsity = 1 - (nonzero_elements / total_elements)

print(f"Total elemen       : {total_elements:,}")
print(f"Elemen non-zero    : {nonzero_elements:,}")
print(f"Sparsity           : {sparsity:.4%}")
```

### 6. TF-IDF dengan TfidfVectorizer

```python
# Inisialisasi TfidfVectorizer
vectorizer_tfidf = TfidfVectorizer(max_features=5000)

# Fit dan transform
tfidf_matrix = vectorizer_tfidf.fit_transform(sample)

print("Shape matriks TF-IDF:", tfidf_matrix.shape)
```

Melihat kata dengan skor TF-IDF tertinggi pada dokumen tertentu:

```python
def top_tfidf_words(doc_index, tfidf_mat, feature_names, top_n=10):
    doc_vector = tfidf_mat[doc_index].toarray()[0]
    top_indices = doc_vector.argsort()[::-1][:top_n]
    return [(feature_names[i], doc_vector[i]) for i in top_indices if doc_vector[i] > 0]

feature_names = vectorizer_tfidf.get_feature_names_out()

print("Top 10 kata TF-IDF Dokumen 0:")
print(sample[0][:150])
print()
for word, score in top_tfidf_words(0, tfidf_matrix, feature_names):
    print(f"  {word:<20} : {score:.4f}")
```

### 7. Cosine Similarity

Hitung similarity antar dokumen:

```python
# Ambil 5 dokumen pertama
subset = tfidf_matrix[:5]

# Hitung cosine similarity
sim_matrix = cosine_similarity(subset)

print("Cosine Similarity Matrix (5 dokumen):")
print(np.round(sim_matrix, 3))
```

### 8. Visualisasi Similarity Matrix

```python
plt.figure(figsize=(8, 6))
sns.heatmap(
    sim_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=[f"D{i}" for i in range(5)],
    yticklabels=[f"D{i}" for i in range(5)]
)
plt.title("Cosine Similarity antar Dokumen (TF-IDF)")
plt.tight_layout()
plt.show()
```

### 9. Document Retrieval Sederhana

Diberikan query, cari dokumen paling mirip:

```python
def search_documents(query, vectorizer, tfidf_mat, docs, top_n=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_mat)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    results = []
    for idx in top_indices:
        results.append({
            "index": idx,
            "similarity": similarities[idx],
            "text": docs[idx][:100]
        })
    return results

# Contoh
query = "produk bagus kualitas"
results = search_documents(query, vectorizer_tfidf, tfidf_matrix, sample)

print(f"Query: '{query}'")
print("=" * 60)
for i, res in enumerate(results, 1):
    print(f"{i}. [Score: {res['similarity']:.4f}]")
    print(f"   {res['text']}...")
    print()
```

### 10. Perbandingan BoW vs TF-IDF

```python
print("=" * 50)
print("PERBANDINGAN BOW vs TF-IDF")
print("=" * 50)

# Ambil 1 dokumen
doc_idx = 0
bow_vec = vectorizer_bow.transform([sample[doc_idx]]).toarray()[0]
tfidf_vec = vectorizer_tfidf.transform([sample[doc_idx]]).toarray()[0]

bow_vocab = vectorizer_bow.get_feature_names_out()
tfidf_vocab = vectorizer_tfidf.get_feature_names_out()

print(f"\nDokumen: {sample[doc_idx][:100]}...")
print("\nTop 10 Kata — BoW (frekuensi):")
top_bow = sorted(
    [(bow_vocab[i], bow_vec[i]) for i in range(len(bow_vec)) if bow_vec[i] > 0],
    key=lambda x: x[1], reverse=True
)[:10]
for word, val in top_bow:
    print(f"  {word:<20} : {int(val)}")

print("\nTop 10 Kata — TF-IDF (bobot):")
for word, score in top_tfidf_words(doc_idx, tfidf_matrix, tfidf_vocab):
    print(f"  {word:<20} : {score:.4f}")
```

---

## E. Tugas Analisis

1. Pilih 3 dokumen dari dataset. Hitung cosine similarity di antara keduanya menggunakan TF-IDF. Mana yang paling mirip?
2. Buat query baru dan jalankan document retrieval. Apakah hasilnya relevan?
3. Bandingkan BoW dan TF-IDF pada dokumen yang sama. Kata apa yang skornya berubah paling signifikan?
4. Apa dampak sparsity pada performa komputasi?

---

## F. Pertanyaan Diskusi

1. Mengapa BoW tidak mempertimbangkan urutan kata, dan apakah hal tersebut menjadi masalah?
2. Kata apa yang mendapat skor TF-IDF rendah meskipun sering muncul? Mengapa?
3. Apakah cosine similarity cocok untuk semua jenis teks?
4. Mengapa matriks BoW/TF-IDF sangat sparse? Apa dampaknya?
5. Apa keterbatasan TF-IDF yang tidak dapat diselesaikan tanpa word embedding?

---

## G. Output Pertemuan 6

File yang dikumpulkan:

```text
representasi_teks.ipynb
```

> Output: Matriks TF-IDF, document similarity matrix, dan sistem retrieval dokumen sederhana.

---

> **Pertemuan sebelumnya:** [Modul 1 — Pertemuan 5](../modul-1/praktikum5.md)
>
> **Pertemuan berikutnya:** [Pertemuan 7 — Word Embedding](praktikum7.md)
