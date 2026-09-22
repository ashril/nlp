# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 2 — PERTEMUAN 7
## Word Embedding: Word2Vec dan GloVe

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 2 — Representasi Teks dan Neural NLP |
| Pertemuan | 7 |
| Topik | Word Embedding: Word2Vec dan GloVe |
| Output | `word_embedding.ipynb` |

---

## B. Tujuan

Mahasiswa mampu:

* memahami keterbatasan representasi sparse (BoW/TF-IDF);
* memahami konsep word embedding dan dense representation;
* melatih model Word2Vec dari corpus sendiri;
* menggunakan pretrained GloVe atau FastText;
* menemukan kata-kata yang semantically similar;
* melakukan operasi vektor kata (king - man + woman);
* memvisualisasikan embedding dengan PCA dan t-SNE.

---

## C. Materi Teori

### 1. Keterbatasan BoW dan TF-IDF

| Masalah | Penjelasan |
|---------|-----------|
| Dimensi tinggi | Vektor sebesar ukuran vocabulary (bisa ribuan) |
| Sparse | Sebagian besar nilai adalah nol |
| Tidak memahami makna | "bagus" dan "baik" dianggap tidak berhubungan |
| Tidak memahami konteks | Urutan kata diabaikan |
| Tidak ada relasi kata | Tidak ada informasi bahwa "kucing" dekat dengan "anjing" |

### 2. Word Embedding

Word embedding adalah representasi kata dalam bentuk **dense vector** berdimensi rendah (misalnya 100–300 dimensi), di mana kata-kata yang bermakna mirip memiliki vektor yang berdekatan.

Contoh representasi:

```text
"raja"   → [0.12, -0.45, 0.78, 0.33, ...]  (300 dimensi)
"ratu"   → [0.11, -0.43, 0.75, 0.41, ...]
"pria"   → [0.09, -0.40, 0.22, 0.10, ...]
"wanita" → [0.08, -0.38, 0.21, 0.18, ...]
```

Kata "raja" dan "ratu" memiliki vektor yang lebih dekat satu sama lain dibandingkan dengan "meja".

### 3. Distributional Hypothesis

Word embedding berdasarkan pada prinsip:

> *"You shall know a word by the company it keeps."* — J.R. Firth

Kata-kata yang muncul dalam konteks yang sama cenderung memiliki makna yang serupa.

### 4. Word2Vec

Word2Vec (Mikolov et al., 2013) adalah metode untuk melatih word embedding menggunakan neural network sederhana.

Dua arsitektur Word2Vec:

| Arsitektur | Cara Kerja | Cocok Untuk |
|-----------|-----------|------------|
| CBOW (Continuous Bag of Words) | Prediksi kata target dari konteks | Corpus besar |
| Skip-gram | Prediksi konteks dari kata target | Kata langka, corpus kecil |

### 5. Operasi Vektor Kata

Salah satu properti menarik word embedding:

$$\vec{raja} - \vec{pria} + \vec{wanita} \approx \vec{ratu}$$

Ini menunjukkan bahwa relasi semantik antar kata dapat direpresentasikan secara aljabar dalam ruang vektor.

### 6. GloVe (Global Vectors)

GloVe (Pennington et al., 2014) adalah metode embedding berbasis matriks co-occurrence global dari corpus. GloVe tersedia sebagai pretrained model untuk berbagai bahasa.

### 7. FastText

FastText (Facebook AI) adalah ekstensi Word2Vec yang mempertimbangkan subword. Cocok untuk Bahasa Indonesia yang kaya morfologi karena dapat menangani kata-kata baru (OOV — Out of Vocabulary).

---

## D. Praktikum

### 1. Membuat Notebook

Buat file:

```text
pertemuan-07/word_embedding.ipynb
```

### 2. Import Library

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.fasttext import FastText
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
```

### 3. Load Dataset

```python
df = pd.read_csv("../dataset/dataset_preprocessed.csv")
text_column = "processed_text"
df = df.dropna(subset=[text_column])

# Buat list of list untuk Word2Vec
sentences = [
    str(text).split()
    for text in df[text_column]
]

print(f"Jumlah kalimat: {len(sentences)}")
print(f"Contoh kalimat: {sentences[0][:10]}")
```

### 4. Melatih Model Word2Vec

```python
# Latih model Word2Vec
model_w2v = Word2Vec(
    sentences=sentences,
    vector_size=100,    # dimensi embedding
    window=5,           # ukuran konteks kiri-kanan
    min_count=2,        # abaikan kata yang muncul < 2 kali
    workers=4,          # jumlah thread
    epochs=10,          # iterasi training
    sg=1                # 1 = Skip-gram, 0 = CBOW
)

print("Model Word2Vec berhasil dilatih.")
print(f"Ukuran vocabulary: {len(model_w2v.wv)}")
```

### 5. Menyimpan dan Load Model

```python
# Simpan model
model_w2v.save("word2vec_model.bin")

# Load model
# model_w2v = Word2Vec.load("word2vec_model.bin")
```

### 6. Melihat Vektor Kata

```python
# Contoh kata dari dataset
sample_word = "bagus"

if sample_word in model_w2v.wv:
    vector = model_w2v.wv[sample_word]
    print(f"Vektor '{sample_word}':")
    print(f"  Dimensi : {vector.shape}")
    print(f"  Nilai   : {vector[:10]}...")  # tampilkan 10 dimensi pertama
else:
    print(f"Kata '{sample_word}' tidak ada dalam vocabulary.")
```

### 7. Mencari Kata Paling Mirip

```python
def find_similar_words(word, model, top_n=10):
    if word not in model.wv:
        print(f"Kata '{word}' tidak ada dalam vocabulary.")
        return []
    similar = model.wv.most_similar(word, topn=top_n)
    return similar

# Contoh
test_words = ["bagus", "buruk", "hotel", "pelayanan"]

for word in test_words:
    print(f"\nKata yang mirip dengan '{word}':")
    similar = find_similar_words(word, model_w2v)
    for similar_word, score in similar:
        print(f"  {similar_word:<20} : {score:.4f}")
```

### 8. Similarity antar Kata

```python
def word_similarity(word1, word2, model):
    if word1 not in model.wv or word2 not in model.wv:
        return None
    return model.wv.similarity(word1, word2)

# Uji beberapa pasangan kata
pairs = [
    ("bagus", "baik"),
    ("bagus", "buruk"),
    ("hotel", "penginapan"),
    ("hotel", "mobil"),
]

print("Similarity antar kata:")
print("-" * 40)
for w1, w2 in pairs:
    sim = word_similarity(w1, w2, model_w2v)
    if sim is not None:
        print(f"  {w1} ↔ {w2:<20}: {sim:.4f}")
```

### 9. Operasi Vektor Kata

```python
# Cari kata yang paling mendekati: positif - negatif + ...
try:
    result = model_w2v.wv.most_similar(
        positive=["bagus", "pelayanan"],
        negative=["buruk"],
        topn=5
    )
    print("Hasil operasi vektor:")
    for word, score in result:
        print(f"  {word:<20} : {score:.4f}")
except Exception as e:
    print(f"Error: {e}")
    print("Pastikan kata-kata tersebut ada dalam vocabulary.")
```

### 10. Visualisasi dengan PCA

```python
def visualize_embeddings_pca(words, model, title="Word Embedding PCA"):
    # Filter kata yang ada dalam vocabulary
    valid_words = [w for w in words if w in model.wv]
    if not valid_words:
        print("Tidak ada kata yang tersedia dalam vocabulary.")
        return

    vectors = np.array([model.wv[w] for w in valid_words])

    # Reduksi ke 2D dengan PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)
    for i, word in enumerate(valid_words):
        plt.annotate(
            word,
            xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
            fontsize=9
        )
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.show()

# Ambil top 50 kata paling sering
top_words = [word for word, _ in model_w2v.wv.key_to_index.items()][:50]
visualize_embeddings_pca(top_words, model_w2v, "Word2Vec — Visualisasi PCA")
```

### 11. Visualisasi dengan t-SNE

```python
def visualize_embeddings_tsne(words, model, title="Word Embedding t-SNE"):
    valid_words = [w for w in words if w in model.wv]
    if len(valid_words) < 10:
        print("Terlalu sedikit kata untuk t-SNE.")
        return

    vectors = np.array([model.wv[w] for w in valid_words])

    tsne = TSNE(n_components=2, perplexity=15, random_state=42, n_iter=1000)
    vectors_2d = tsne.fit_transform(vectors)

    plt.figure(figsize=(14, 10))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6)
    for i, word in enumerate(valid_words):
        plt.annotate(
            word,
            xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
            fontsize=8
        )
    plt.title(title)
    plt.tight_layout()
    plt.show()

top_words = [word for word, _ in model_w2v.wv.key_to_index.items()][:100]
visualize_embeddings_tsne(top_words, model_w2v, "Word2Vec — Visualisasi t-SNE")
```

### 12. FastText (Menangani OOV)

```python
# Latih model FastText
model_ft = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10
)

# FastText dapat menangani kata baru (OOV)
oov_word = "ketidakberhasilannya"  # kata yang mungkin tidak ada di corpus
vector_oov = model_ft.wv[oov_word]
print(f"FastText berhasil menghasilkan vektor untuk '{oov_word}'")
print(f"Dimensi: {vector_oov.shape}")
```

### 13. Perbandingan Word2Vec vs FastText

```python
test_oov = "pemrogramannya"  # kata dengan imbuhan kompleks

print("Perbandingan Word2Vec vs FastText:")
print("-" * 50)

if test_oov in model_w2v.wv:
    w2v_similar = model_w2v.wv.most_similar(test_oov, topn=5)
    print(f"\nWord2Vec — '{test_oov}':")
    for w, s in w2v_similar:
        print(f"  {w:<20}: {s:.4f}")
else:
    print(f"\nWord2Vec: '{test_oov}' tidak ada dalam vocabulary (OOV)")

ft_similar = model_ft.wv.most_similar(test_oov, topn=5)
print(f"\nFastText — '{test_oov}':")
for w, s in ft_similar:
    print(f"  {w:<20}: {s:.4f}")
```

---

## E. Tugas Analisis

1. Latih model Word2Vec dengan parameter `sg=0` (CBOW). Bandingkan kata-kata yang paling mirip dengan `sg=1` (Skip-gram). Apa perbedaannya?
2. Cari 5 pasang kata yang secara intuisi mirip. Hitung similarity-nya. Apakah model menangkap relasi tersebut?
3. Temukan minimal 3 contoh operasi vektor yang menghasilkan kata bermakna.
4. Visualisasikan embedding dan identifikasi cluster kata yang bermakna.
5. Bandingkan kemampuan Word2Vec dan FastText dalam menangani kata berimbuhan Bahasa Indonesia.

---

## F. Pertanyaan Diskusi

1. Mengapa representasi dense (embedding) lebih baik dari representasi sparse (BoW) untuk memahami makna kata?
2. Apa yang dimaksud dengan "konteks" dalam Word2Vec? Bagaimana konteks menentukan makna kata?
3. Mengapa operasi `raja - pria + wanita ≈ ratu` bisa terjadi?
4. Apa keterbatasan Word2Vec statis? (Petunjuk: pikirkan kata-kata yang memiliki makna berbeda dalam konteks berbeda)
5. Mengapa FastText lebih cocok untuk Bahasa Indonesia dibandingkan Word2Vec standar?
6. Apa yang akan diperbaiki oleh contextual embedding pada pertemuan berikutnya?

---

## G. Output Pertemuan 7

File yang dikumpulkan:

```text
word_embedding.ipynb
word2vec_model.bin
```

> Output: Model Word2Vec yang dilatih dari corpus, visualisasi embedding, dan analisis relasi semantik kata.

---

> **Pertemuan sebelumnya:** [Pertemuan 6 — Representasi Teks](praktikum6.md)
>
> **Pertemuan berikutnya:** [Pertemuan 8 — Contextual Representation](praktikum8.md)
