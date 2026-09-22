# RENCANA PEMBELAJARAN SEMESTER (RPS)
## NATURAL LANGUAGE PROCESSING
### Program Studi S1 Teknologi Informasi

---

## Informasi Mata Kuliah

| Komponen | Keterangan |
|---|---|
| **Mata Kuliah** | Natural Language Processing |
| **Program Studi** | S1 Teknologi Informasi |
| **Bobot** | 3 SKS |
| **Komposisi** | 2 SKS Teori + 1 SKS Praktikum |
| **Semester** | Disesuaikan kurikulum |
| **Prasyarat** | Pemrograman, Struktur Data, Basis Data, Kecerdasan Buatan dan/atau Machine Learning |
| **Bentuk Pembelajaran** | Kuliah, diskusi, praktikum, studi kasus, *project-based learning* |
| **Tools** | Python, Jupyter/Google Colab, NLTK, spaCy, Scikit-learn, PyTorch, Hugging Face |

---

## A. DESKRIPSI MATA KULIAH

Mata kuliah **Natural Language Processing (NLP)** membahas konsep, teknik, dan penerapan komputasional untuk memahami, mengolah, dan menghasilkan bahasa alami.

Pembelajaran dimulai dari karakteristik bahasa alami, *text processing*, tokenisasi, normalisasi, *stemming*, *lemmatization*, *n-gram*, representasi teks, *word embedding*, serta pemodelan bahasa. Selanjutnya mahasiswa mempelajari pendekatan *neural network* untuk bahasa, termasuk *sequence modeling*, RNN, LSTM, GRU, *attention mechanism*, dan Transformer.

Pada bagian akhir mahasiswa mempelajari *pretrained language model*, BERT dan turunannya, *fine-tuning*, *Named Entity Recognition*, *text classification*, *sentiment analysis*, *text summarization*, *question answering*, serta pengembangan aplikasi NLP.

Mata kuliah menekankan pemahaman konsep NLP dan implementasi, bukan pembahasan mendalam algoritma Machine Learning yang telah menjadi materi pada mata kuliah tersendiri.

---

## B. CAPAIAN PEMBELAJARAN MATA KULIAH (CPMK)

Setelah menyelesaikan mata kuliah ini, mahasiswa mampu:

* **CPMK 1**: Menjelaskan konsep, ruang lingkup, sejarah, dan permasalahan utama Natural Language Processing.
* **CPMK 2**: Menerapkan teknik *preprocessing* dan *linguistic processing* terhadap data bahasa alami.
* **CPMK 3**: Menerapkan berbagai metode representasi bahasa untuk mengubah teks menjadi representasi yang dapat diproses komputer.
* **CPMK 4**: Menjelaskan dan mengimplementasikan konsep *language modeling* dan *neural network* untuk pemrosesan bahasa.
* **CPMK 5**: Menjelaskan prinsip *attention* dan Transformer serta perkembangannya menjadi *pretrained language model*.
* **CPMK 6**: Menggunakan *pretrained language model* untuk menyelesaikan berbagai permasalahan NLP melalui *prompting*, *feature extraction*, atau *fine-tuning*.
* **CPMK 7**: Mengembangkan dan mengevaluasi prototipe aplikasi NLP untuk menyelesaikan permasalahan nyata.

---

## C. STRUKTUR MODUL

### MODUL 1: FUNDAMENTAL DAN TEXT PROCESSING

#### Tujuan
Mahasiswa mampu memahami karakteristik bahasa alami dan melakukan pengolahan awal terhadap data teks.

#### Daftar Materi
* Konsep Natural Language Processing
* Bahasa alami dan bahasa formal
* Level analisis bahasa
* Corpus dan dataset
* Text preprocessing
* Tokenization
* Sentence segmentation
* Normalisasi
* Stopword
* Stemming
* Lemmatization
* Part-of-Speech Tagging
* N-gram
* Linguistic ambiguity

---

#### Rincian Pertemuan Modul 1

##### Pertemuan 1 — Pengantar Natural Language Processing
* **Materi Teori:**
  * Pengertian NLP
  * Sejarah perkembangan NLP
  * NLP sebagai bagian dari AI
  * Contoh aplikasi NLP
  * NLP klasik sampai modern
  * Permasalahan NLP
  * Bahasa alami vs bahasa formal
  * Level pemrosesan bahasa:
    * *lexical*
    * *syntactic*
    * *semantic*
    * *discourse*
    * *pragmatic*
* **Praktikum:**
  * Setup environment NLP
  * Membaca dataset teks
  * Eksplorasi corpus
  * Menghitung jumlah dokumen, kata, dan kalimat
* **Output:**
  * Notebook eksplorasi corpus.

##### Pertemuan 2 — Text Processing dan Tokenization
* **Materi Teori:**
  * Document, Sentence, Word, Token, Character
  * Word tokenization
  * Sentence tokenization
  * Character tokenization
  * Subword tokenization
  * Permasalahan tokenisasi Bahasa Indonesia
* **Praktikum:**
  * Tokenisasi menggunakan NLTK/spaCy
  * Sentence segmentation
  * Word tokenization
  * Character tokenization
  * Eksperimen tokenizer Bahasa Indonesia
* **Output:**
  * Program text tokenizer.

##### Pertemuan 3 — Text Normalization
* **Materi Teori:**
  * Case folding
  * Cleaning
  * Punctuation & number normalization
  * Slang words & typographical errors
  * Stopword
  * Stemming & Lemmatization
  * Normalisasi Bahasa Indonesia
* **Praktikum:**
  * Membangun pipeline:  
    `Raw Text → Cleaning → Normalization → Tokenization → Stopword → Stemming`
* **Output:**
  * Dataset teks hasil preprocessing.

##### Pertemuan 4 — Linguistic Processing
* **Materi Teori:**
  * Lexical analysis
  * Part-of-Speech & POS Tagging
  * Morphology (Root word, Affix)
  * Syntax & Parsing (Dependency parsing, Constituency parsing)
* **Praktikum:**
  * POS tagging
  * Dependency parsing
  * Visualisasi struktur kalimat
  * Analisis kesalahan linguistic processing
* **Output:**
  * Analisis struktur linguistik sebuah corpus.

##### Pertemuan 5 — N-Gram dan Language Modeling Dasar
* **Materi Teori:**
  * Konsep language model
  * Probability of words (Unigram, Bigram, Trigram)
  * Conditional probability & Context
  * Sentence probability
  * Perplexity
  * Keterbatasan n-gram language model
* **Praktikum:**
  * Membuat unigram/bigram/trigram
  * Menghitung probabilitas
  * Membuat prediksi kata sederhana
  * Menghitung perplexity
* **Output:**
  * Mini language model berbasis n-gram.

---

### MODUL 2: REPRESENTASI BAHASA DAN NEURAL NLP

#### Tujuan
Mahasiswa mampu memahami bagaimana komputer merepresentasikan makna dan konteks bahasa serta menerapkan *neural network* untuk pemrosesan bahasa.

#### Daftar Materi
* Bag-of-Words sebagai representasi
* TF-IDF
* Word representation & Word embedding
* Word2Vec (CBOW, Skip-Gram)
* GloVe
* Contextual representation
* Sequence modeling
* RNN, LSTM, GRU
* Sequence-to-sequence
* Attention mechanism

---

#### Rincian Pertemuan Modul 2

##### Pertemuan 6 — Representasi Teks
* **Materi Teori:**
  * Representasi teks & sparse representation
  * Bag-of-Words
  * TF-IDF
  * Keterbatasan representasi berbasis frekuensi
  * Semantic similarity
* **Praktikum:**
  * Bag-of-Words
  * TF-IDF
  * Document similarity
  * Cosine similarity
* **Output:**
  * Program representasi dan similarity dokumen.

##### Pertemuan 7 — Word Embedding
* **Materi Teori:**
  * Distributed representation
  * Word embedding & semantic relationship
  * Word2Vec (CBOW, Skip-Gram)
  * GloVe
  * Static embedding & OOV problem
* **Praktikum:**
  * Menggunakan pretrained Word2Vec
  * Word similarity
  * Analisis semantic relationship
  * Visualisasi embedding (PCA/t-SNE)
* **Output:**
  * Visualisasi semantic word space.

##### Pertemuan 8 — Contextual Representation dan Neural NLP
* **Materi Teori:**
  * Keterbatasan static embedding
  * Contextual representation
  * Word representation berdasarkan konteks
  * Neural language representation
  * Embedding layer & sequence representation
* **Praktikum:**
  * Menggunakan pretrained embedding
  * Membandingkan static dan contextual representation
  * Eksperimen representasi kata berdasarkan konteks
* **Output:**
  * Analisis perbedaan static dan contextual embedding.

##### Pertemuan 9 — RNN, LSTM dan GRU
* **Materi Teori:**
  * Sequential data
  * Recurrent Neural Network & Hidden state
  * Vanishing gradient problem
  * Long Short-Term Memory & LSTM gates
  * GRU (Gated Recurrent Unit)
  * Bidirectional RNN
  * Sequence classification
* **Praktikum:**
  * Implementasi RNN
  * Implementasi LSTM
  * Implementasi GRU
  * Membandingkan kemampuan menangkap konteks
* **Output:**
  * Model sequence processing sederhana.

##### Pertemuan 10 — Sequence-to-Sequence dan Attention
* **Materi Teori:**
  * Encoder-decoder architecture
  * Sequence-to-sequence
  * Context vector & bottleneck problem
  * Attention mechanism (Query, Key, Value)
  * Self-attention
  * Bahaya kehilangan konteks pada sequence panjang
* **Praktikum:**
  * Implementasi sederhana attention
  * Eksperimen sequence-to-sequence
  * Visualisasi attention
* **Output:**
  * Implementasi dan visualisasi attention.

---

### MODUL 3: TRANSFORMER, PRETRAINED MODEL DAN APLIKASI NLP

#### Tujuan
Mahasiswa mampu memahami perkembangan NLP modern berbasis Transformer dan menggunakan *pretrained language model* untuk membangun solusi NLP.

#### Daftar Materi
* Transformer architecture (Self-attention, Encoder, Decoder, Positional encoding)
* BERT, RoBERTa, IndoBERT, GPT
* Pretrained language model & Transfer learning
* Fine-tuning vs Prompting
* Named Entity Recognition
* Text classification & Sentiment analysis
* Text summarization
* Question answering
* NLP application

---

#### Rincian Pertemuan Modul 3

##### Pertemuan 11 — Transformer
* **Materi Teori:**
  * Keterbatasan RNN
  * Attention sebagai solusi
  * Transformer architecture
  * Self-attention & Multi-head attention
  * Positional encoding
  * Encoder & Decoder
  * Transformer untuk NLP
* **Praktikum:**
  * Menggunakan Transformer
  * Eksplorasi tokenizer
  * Eksplorasi attention
  * Analisis output Transformer
* **Output:**
  * Eksperimen Transformer.

##### Pertemuan 12 — Pretrained Language Model
* **Materi Teori:**
  * Pretraining & Transfer learning
  * Language representation
  * BERT, RoBERTa, ALBERT, GPT
  * Model Bahasa Indonesia (IndoBERT)
  * Tokenizer subword
  * Masked Language Modeling (MLM) vs Causal Language Modeling (CLM)
* **Praktikum:**
  * Menggunakan Hugging Face Transformers
  * Load pretrained model
  * Tokenisasi
  * Inference
  * Analisis embedding
* **Output:**
  * Notebook eksplorasi pretrained language model.

##### Pertemuan 13 — Fine-Tuning dan NLP Tasks
* **Materi Teori:**
  * Fine-tuning
  * Classification head
  * Sequence classification vs Token classification
  * Fine-tuning vs feature extraction
  * Training data & Validation data
  * Overfitting & Model evaluation
* **Praktikum:**
  * Fine-tuning pretrained model untuk salah satu task:
    * Sentiment analysis
    * Text classification
    * Topic classification
* **Output:**
  * Model NLP hasil fine-tuning.

##### Pertemuan 14 — Named Entity Recognition dan Information Extraction
* **Materi Teori:**
  * Information extraction
  * Named Entity Recognition (NER) & Entity
  * BIO tagging
  * Token classification
  * Entity linking & Relation extraction
  * Information extraction pipeline
* **Praktikum:**
  * NER Bahasa Indonesia
  * Menggunakan pretrained model & Fine-tuning NER
  * Ekstraksi entitas:
    * PERSON
    * ORGANIZATION
    * LOCATION
    * DATE dan entitas lainnya
* **Output:**
  * Prototype information extraction.

##### Pertemuan 15 — Advanced NLP Tasks
* **Materi Teori:**
  * Sentiment analysis & Text classification
  * Text summarization
  * Machine translation
  * Question answering & Text generation
  * Semantic similarity & Text embedding
  * Retrieval & RAG sebagai perkembangan NLP modern
* **Praktikum:**
  * Mahasiswa memilih salah satu task:
    * Sentiment analysis
    * Text classification
    * Text summarization
    * Question answering
    * Semantic similarity
    * Information extraction
* **Output:**
  * Prototype aplikasi NLP tahap pertama.

##### Pertemuan 16 — NLP Project: Implementation and Presentation
* **Materi & Aktivitas:**
  * Pertemuan terakhir difokuskan pada penyelesaian dan demonstrasi project.
  * Finalisasi dataset & preprocessing
  * Finalisasi model & evaluasi (Error analysis)
  * Implementasi inference & integrasi model ke aplikasi sederhana
  * Presentasi & demonstrasi aplikasi
* **Output Akhir:**
  * `Dataset → NLP Pipeline → Model → Evaluation → Application`

---

## D. RINGKASAN 16 PERTEMUAN

| Pertemuan | Modul | Materi Utama | Praktikum |
|:---:|:---:|---|---|
| **1** | Modul 1 | Pengantar NLP | Eksplorasi corpus |
| **2** | Modul 1 | Tokenization | Tokenizer |
| **3** | Modul 1 | Text Normalization | Preprocessing pipeline |
| **4** | Modul 1 | Linguistic Processing | POS & Parsing |
| **5** | Modul 1 | N-Gram & Language Model | N-Gram model |
| **6** | Modul 2 | Representasi Teks | BoW, TF-IDF, similarity |
| **7** | Modul 2 | Word Embedding | Word2Vec/GloVe |
| **8** | Modul 2 | Contextual Representation | Contextual embedding |
| **9** | Modul 2 | RNN, LSTM, GRU | Sequence model |
| **10** | Modul 2 | Seq2Seq & Attention | Attention |
| **11** | Modul 3 | Transformer | Transformer |
| **12** | Modul 3 | Pretrained Language Model | BERT/IndoBERT/GPT |
| **13** | Modul 3 | Fine-Tuning | Fine-tuning NLP |
| **14** | Modul 3 | NER & Information Extraction | NER |
| **15** | Modul 3 | Advanced NLP Tasks | NLP application |
| **16** | Modul 3 | Final NLP Project | Demo & presentasi |

---

## E. PENDEKATAN PEMBELAJARAN

Mata kuliah menggunakan tiga pilar pendekatan:

1. **Theory-to-Practice**  
   Setiap konsep teori langsung diikuti implementasi.  
   *Contoh:* Tokenization → teori → implementasi Python → analisis hasil.

2. **Case-Based Learning**  
   Kasus nyata yang digunakan dapat berupa:
   * Berita Bahasa Indonesia
   * Review produk / e-commerce
   * Komentar media sosial
   * Ulasan destinasi wisata
   * Dokumen akademik
   * Pengaduan masyarakat
   * Data layanan universitas

3. **Project-Based Learning**  
   Project dibangun secara bertahap sepanjang semester:
   ```text
   Modul 1: Dataset → Preprocessing
              ↓
   Modul 2: Representation → Neural NLP
              ↓
   Modul 3: Transformer → Pretrained Model → Application
   ```

---

## F. KOMPONEN PENILAIAN

Mata kuliah ini menerapkan penilaian berbasis proses dan proyek (*continuous & project-based assessment*), tidak menggunakan UTS/UAS sebagai ujian terpisah:

| Komponen | Bobot |
|---|:---:|
| Partisipasi & Diskusi | 5% |
| Praktikum Modul 1 | 15% |
| Praktikum Modul 2 | 20% |
| Praktikum Modul 3 | 20% |
| Tugas / Case Study | 10% |
| Final NLP Project | 25% |
| Presentasi & Demonstrasi | 5% |
| **Total** | **100%** |

---

## G. FINAL PROJECT

Project akhir harus menunjukkan penerapan konsep NLP secara *end-to-end*. Contoh topik project yang dapat dipilih:

1. **Sentiment Analysis**: Review → preprocessing → Transformer → sentiment
2. **News Classification**: Berita → tokenizer → pretrained model → kategori berita
3. **Named Entity Recognition**: Berita → Transformer → NER → entity extraction
4. **Academic Document NLP**: Dokumen → preprocessing → embedding → semantic similarity
5. **Tourism NLP**: Ulasan wisatawan → NLP → sentiment → insight
6. **Question Answering**: Dokumen → embedding → retrieval → question answering
7. **Text Summarization**: Dokumen panjang → Transformer → ringkasan

---

## H. BATASAN MATERI MACHINE LEARNING

Karena Machine Learning merupakan mata kuliah tersendiri, materi berikut **tidak menjadi fokus**:
* Teori Naive Bayes secara mendalam
* Teori Support Vector Machine (SVM) secara mendalam
* Decision Tree
* Random Forest
* K-Means
* Gradient descent secara matematis mendalam
* Hyperparameter optimization secara mendalam
* Ensemble learning
* SMOTE dan teknik *imbalance learning*

> **Catatan:** Jika diperlukan dalam NLP, algoritma di atas hanya diperkenalkan sebagai *baseline* atau alat pembanding.

**Fokus utama mata kuliah ini adalah alur linguistik komputasional modern:**  
`Language → Text → Representation → Context → Sequence → Attention → Transformer → Pretrained Language Model → NLP Application`

---

## I. ROADMAP KOMPETENSI

Mahasiswa diharapkan berkembang melalui empat tingkat kompetensi:

```text
Level 1 — Memproses Bahasa
Raw Text → Tokenization → Normalization → Linguistic Processing

Level 2 — Merepresentasikan Bahasa
Text → N-Gram → TF-IDF → Word Embedding → Contextual Representation

Level 3 — Memahami Bahasa dengan Neural NLP
Sequence → RNN → LSTM → Attention → Transformer

Level 4 — Membangun NLP Modern
Pretrained Model → Fine-Tuning → NLP Task → Evaluation → Application
```

Dengan struktur tersebut, mata kuliah NLP menjadi benar-benar berdiri sendiri sebagai mata kuliah NLP, sementara Machine Learning hanya digunakan sebagai fondasi yang sudah dipelajari mahasiswa pada mata kuliah lain.
