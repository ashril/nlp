# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 2 — PERTEMUAN 8
## Contextual Representation

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 2 — Representasi Teks dan Neural NLP |
| Pertemuan | 8 |
| Topik | Contextual Representation |
| Output | `contextual_representation.ipynb` |

---

## B. Tujuan

Mahasiswa mampu:

* memahami keterbatasan word embedding statis;
* memahami konsep polisemi dan ambiguitas konteks;
* memahami mengapa representasi kontekstual diperlukan;
* menggunakan model pretrained sederhana untuk mendapatkan contextual embedding;
* membandingkan embedding statis dan embedding kontekstual;
* memvisualisasikan perbedaan representasi kata yang sama dalam konteks berbeda.

---

## C. Materi Teori

### 1. Keterbatasan Word Embedding Statis

Word2Vec dan GloVe adalah **embedding statis** — setiap kata selalu memiliki vektor yang sama, terlepas dari konteksnya.

**Masalah polisemi:**

```text
"Bank" dalam "Saya pergi ke bank untuk menabung."
   ↓
vektor["bank"] = [0.12, 0.45, ...]

"Bank" dalam "Kami duduk di bank sungai."
   ↓
vektor["bank"] = [0.12, 0.45, ...]  ← SAMA!
```

Padahal makna kedua kalimat tersebut berbeda.

**Contoh lain dalam Bahasa Indonesia:**

| Kata | Kalimat 1 | Kalimat 2 |
|------|-----------|-----------|
| bisa | "Ular itu berbisa." (racun) | "Saya bisa berenang." (mampu) |
| bunga | "Bunga mawar merah." (tanaman) | "Bunga tabungan 5%." (interest) |
| kali | "Tiga kali lima." (perkalian) | "Di tepi kali itu." (sungai kecil) |

### 2. Apa Itu Contextual Embedding?

Contextual embedding menghasilkan representasi berbeda untuk kata yang sama berdasarkan konteks kalimatnya.

```text
Kalimat: "Ular itu berbisa."
  "bisa" → vektor_bisa_1 = [0.23, -0.41, ...]  (makna: racun)

Kalimat: "Saya bisa berenang."
  "bisa" → vektor_bisa_2 = [0.71,  0.12, ...]  (makna: mampu)
```

Vektor yang berbeda untuk kata yang sama!

### 3. Evolusi Representasi Bahasa

```text
One-Hot Encoding
      ↓ (masalah: tidak ada makna, dimensi sangat besar)
Bag of Words / TF-IDF
      ↓ (masalah: tidak ada konteks urutan kata)
Word2Vec / GloVe (Static Embedding)
      ↓ (masalah: satu vektor per kata, tidak ada konteks kalimat)
ELMo (2018) — Contextual Embedding pertama
      ↓
BERT / GPT (2018–2019) — Transformer-based Contextual Embedding
      ↓
GPT-3, GPT-4, LLaMA, IndoBERT (2020–sekarang)
```

### 4. ELMo (Embeddings from Language Models)

ELMo adalah salah satu model pertama yang menghasilkan contextual embedding. ELMo menggunakan bidirectional LSTM yang dilatih pada corpus besar.

### 5. Sentence Embedding

Selain word-level embedding, kita juga dapat menghasilkan representasi pada level kalimat:

```text
"Produk ini bagus sekali."
        ↓
[0.12, 0.45, -0.33, ...]  (vektor 768 dimensi, misalnya dari BERT)
```

Sentence embedding berguna untuk:
* Document classification
* Semantic similarity
* Clustering dokumen
* Information retrieval

### 6. Sentence Transformers

Sentence Transformers adalah library yang menyediakan model pretrained untuk menghasilkan sentence embedding berkualitas tinggi.

---

## D. Praktikum

### 1. Membuat Notebook

Buat file:

```text
pertemuan-08/contextual_representation.ipynb
```

### 2. Install Library Tambahan

```bash
pip install sentence-transformers
```

### 3. Import Library

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
```

### 4. Load Dataset

```python
df = pd.read_csv("../dataset/dataset_preprocessed.csv")
text_column = "text"  # gunakan teks asli (sebelum preprocessing)
df = df.dropna(subset=[text_column])
sample_texts = df[text_column].head(100).astype(str).tolist()
print(f"Jumlah sample: {len(sample_texts)}")
```

### 5. Demonstrasi Masalah Static Embedding

```python
from gensim.models import Word2Vec

# Load model Word2Vec yang sudah dilatih sebelumnya
# (atau latih ulang jika diperlukan)
model_w2v = Word2Vec.load("../pertemuan-07/word2vec_model.bin")

# Kata yang sama dalam konteks berbeda
ambiguous_sentences = [
    "ular itu berbisa mematikan",
    "saya bisa berenang jauh",
    "obat itu manjur bisa menyembuhkan"
]

print("Demonstrasi Keterbatasan Static Embedding:")
print("=" * 60)
print("Kata 'bisa' selalu memiliki vektor yang SAMA:")

if "bisa" in model_w2v.wv:
    vec = model_w2v.wv["bisa"]
    print(f"  Vektor 'bisa' (10 dim pertama): {vec[:10].round(3)}")
else:
    print("  Kata 'bisa' tidak ada dalam vocabulary model ini.")
    print("  Gunakan kata lain yang tersedia sebagai contoh.")
```

### 6. Load Sentence Transformer Model

```python
# Model multilingual yang mendukung Bahasa Indonesia
model_st = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Model Sentence Transformer berhasil dimuat.")
```

### 7. Menghasilkan Sentence Embedding

```python
# Encode kalimat menjadi vektor
embeddings = model_st.encode(sample_texts, show_progress_bar=True)

print(f"Shape embeddings: {embeddings.shape}")
print(f"Dimensi per kalimat: {embeddings.shape[1]}")
```

### 8. Semantic Similarity antar Kalimat

```python
# Ambil 5 kalimat pertama
test_sentences = sample_texts[:5]
test_embeddings = model_st.encode(test_sentences)

# Hitung cosine similarity
sim_matrix = cosine_similarity(test_embeddings)

# Visualisasi
plt.figure(figsize=(8, 6))
sns.heatmap(
    sim_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=0, vmax=1,
    xticklabels=[f"K{i+1}" for i in range(5)],
    yticklabels=[f"K{i+1}" for i in range(5)]
)
plt.title("Cosine Similarity — Sentence Embedding")
plt.tight_layout()
plt.show()

# Tampilkan kalimat
for i, text in enumerate(test_sentences):
    print(f"K{i+1}: {text[:80]}...")
```

### 9. Demonstrasi Contextual Embedding

```python
# Kata ambigu dalam konteks berbeda
context_pairs = [
    ("Bunga mawar merah itu sangat indah.", "Bunga tabungan bank naik bulan ini."),
    ("Kita harus bisa menyelesaikan masalah ini.", "Ular kobra itu sangat berbisa."),
    ("Dia pergi ke bank untuk menabung.", "Mereka duduk di tepi kali yang jernih."),
]

print("Perbandingan Semantic Similarity Konteks Berbeda:")
print("=" * 60)
for s1, s2 in context_pairs:
    e1 = model_st.encode([s1])
    e2 = model_st.encode([s2])
    sim = cosine_similarity(e1, e2)[0][0]
    print(f"\nK1: {s1}")
    print(f"K2: {s2}")
    print(f"Similarity: {sim:.4f} ({'mirip' if sim > 0.7 else 'berbeda'})")
```

### 10. Document Retrieval dengan Sentence Embedding

```python
def semantic_search(query, corpus_embeddings, corpus_texts, top_n=5):
    query_embedding = model_st.encode([query])
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    results = []
    for idx in top_indices:
        results.append({
            "score": similarities[idx],
            "text": corpus_texts[idx]
        })
    return results

# Contoh pencarian semantik
query = "pelayanan hotel yang memuaskan"
results = semantic_search(query, embeddings, sample_texts)

print(f"Query: '{query}'")
print("=" * 60)
for i, res in enumerate(results, 1):
    print(f"\n{i}. [Score: {res['score']:.4f}]")
    print(f"   {res['text'][:120]}...")
```

### 11. Visualisasi Sentence Embedding dengan PCA

```python
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20)

# Highlight beberapa dokumen
for i in range(min(10, len(sample_texts))):
    plt.annotate(
        f"D{i}",
        xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
        fontsize=8
    )

plt.title("Sentence Embedding — PCA 2D")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.tight_layout()
plt.show()

print(f"Variansi yang dijelaskan: {pca.explained_variance_ratio_.sum():.2%}")
```

### 12. Visualisasi dengan t-SNE

```python
tsne = TSNE(n_components=2, perplexity=20, random_state=42, n_iter=1000)
embeddings_tsne = tsne.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], alpha=0.6, s=20)
plt.title("Sentence Embedding — t-SNE 2D")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.show()
```

### 13. Clustering Dokumen Berdasarkan Embedding

```python
from sklearn.cluster import KMeans

# Cluster dokumen menjadi 5 kelompok
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)

# Visualisasi cluster
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    embeddings_tsne[:, 0],
    embeddings_tsne[:, 1],
    c=cluster_labels,
    cmap="tab10",
    alpha=0.7,
    s=20
)
plt.colorbar(scatter, label="Cluster")
plt.title(f"Clustering Dokumen ({n_clusters} cluster) — t-SNE")
plt.tight_layout()
plt.show()

# Contoh dokumen per cluster
for cluster_id in range(n_clusters):
    indices = np.where(cluster_labels == cluster_id)[0][:3]
    print(f"\nCluster {cluster_id} (contoh dokumen):")
    for idx in indices:
        print(f"  - {sample_texts[idx][:80]}...")
```

---

## E. Perbandingan Representasi

```text
Representasi         | Dimensi  | Memahami makna? | Memahami konteks?
---------------------|----------|-----------------|-----------------
One-Hot              | |V|       | Tidak           | Tidak
BoW / TF-IDF         | |V|       | Tidak           | Tidak
Word2Vec / GloVe     | 100-300   | Sebagian        | Tidak
Contextual Embedding | 768-1024  | Ya              | Ya
```

---

## F. Tugas Analisis

1. Cari 3 pasang kalimat yang memiliki makna mirip meskipun menggunakan kata-kata berbeda. Hitung similarity-nya dengan Sentence Transformer.
2. Cari 3 pasang kalimat yang berisi kata yang sama tetapi bermakna berbeda. Apakah similarity-nya rendah?
3. Bandingkan hasil document retrieval antara TF-IDF (pertemuan 6) dan Sentence Transformer untuk query yang sama. Mana yang lebih relevan?
4. Analisis cluster yang terbentuk. Apakah dokumen dalam satu cluster memiliki tema yang sama?

---

## G. Pertanyaan Diskusi

1. Apa perbedaan mendasar antara static embedding dan contextual embedding?
2. Bagaimana model Sentence Transformer menghasilkan representasi kalimat yang berbeda untuk kata yang sama?
3. Mengapa contextual embedding sangat berguna untuk task NLP modern?
4. Apa trade-off antara efisiensi komputasi dan kualitas embedding?
5. Bagaimana BERT (yang akan dipelajari di Modul 3) menghasilkan contextual embedding?

---

## H. Output Pertemuan 8

File yang dikumpulkan:

```text
contextual_representation.ipynb
```

> Output: Analisis perbandingan static vs contextual embedding, semantic search, dan visualisasi clustering dokumen.

---

> **Pertemuan sebelumnya:** [Pertemuan 7 — Word Embedding](praktikum7.md)
>
> **Pertemuan berikutnya:** [Pertemuan 9 — RNN, LSTM, GRU](praktikum9.md)
