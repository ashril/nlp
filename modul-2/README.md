# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 2 — REPRESENTASI TEKS DAN NEURAL NLP

---

## A. Identitas Modul

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Program Studi | S1 Teknologi Informasi |
| Modul | Modul 2 |
| Topik | Representasi Teks dan Neural NLP |
| Platform | VS Code |
| Environment | Jupyter Notebook |
| Bahasa | Python |
| Pertemuan | 6–10 |

---

## B. Deskripsi Modul

Modul 2 melanjutkan fondasi dari Modul 1. Teks yang sudah diproses kini perlu direpresentasikan dalam bentuk numerik agar dapat diproses oleh model Machine Learning dan Neural Network.

Alur Modul 2:

```text
Processed Text (dari Modul 1)
        ↓
Bag of Words / TF-IDF
        ↓
Word Embedding (Word2Vec / GloVe)
        ↓
Contextual Representation
        ↓
Sequence Modeling (RNN / LSTM / GRU)
        ↓
Attention Mechanism
```

---

## C. Tujuan Modul

Setelah menyelesaikan Modul 2, mahasiswa diharapkan mampu:

1. Memahami konsep Bag of Words dan keterbatasannya.
2. Menerapkan TF-IDF untuk representasi teks.
3. Menghitung document similarity menggunakan cosine similarity.
4. Memahami konsep word embedding dan perbedaannya dengan representasi sparse.
5. Menggunakan Gensim untuk Word2Vec dan GloVe.
6. Memvisualisasikan embedding dengan PCA dan t-SNE.
7. Memahami keterbatasan word embedding statis.
8. Memahami dan mengimplementasikan RNN, LSTM, dan GRU.
9. Memahami encoder-decoder dan bottleneck problem.
10. Mengimplementasikan dan memvisualisasikan attention mechanism.

---

## D. Pembagian Praktikum

| Pertemuan | File | Materi | Output |
|-----------|------|--------|--------|
| 6 | [praktikum6.md](praktikum6.md) | Representasi Teks (BoW & TF-IDF) | Dokumen similarity matrix |
| 7 | [praktikum7.md](praktikum7.md) | Word Embedding | Model Word2Vec & visualisasi |
| 8 | [praktikum8.md](praktikum8.md) | Contextual Representation | Analisis embedding kontekstual |
| 9 | [praktikum9.md](praktikum9.md) | RNN, LSTM, GRU | Model sequence processing |
| 10 | [praktikum10.md](praktikum10.md) | Seq2Seq & Attention | Visualisasi attention |

---

## E. Library Tambahan Modul 2

Install library berikut (tambahan dari Modul 1):

```bash
pip install gensim torch torchvision scikit-learn
```

Verifikasi:

```python
import gensim
import torch
import sklearn

print("Gensim version :", gensim.__version__)
print("PyTorch version:", torch.__version__)
print("sklearn version:", sklearn.__version__)
```

---

## F. Struktur Folder

```text
NLP/
│
├── modul-2/
│   │
│   ├── pertemuan-06/
│   │   └── representasi_teks.ipynb
│   │
│   ├── pertemuan-07/
│   │   └── word_embedding.ipynb
│   │
│   ├── pertemuan-08/
│   │   └── contextual_representation.ipynb
│   │
│   ├── pertemuan-09/
│   │   └── rnn_lstm_gru.ipynb
│   │
│   └── pertemuan-10/
│       └── seq2seq_attention.ipynb
```

---

## G. Posisi Modul 2 dalam Roadmap NLP

```text
[Modul 1] Text Processing
    ↓
[Modul 2] Representasi & Neural NLP  ← Anda di sini
    ↓
[Modul 3] Transformer & Pretrained Model
```

Modul 2 menjawab pertanyaan:

> *Bagaimana teks yang sudah diproses dapat diubah menjadi bentuk numerik yang bermakna bagi komputer?*
