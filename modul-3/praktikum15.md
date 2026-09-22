# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 3 — PERTEMUAN 15
## Advanced NLP Tasks: Summarization, Question Answering, dan Semantic Retrieval

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 3 — Transformer, Pretrained Model, dan Aplikasi NLP |
| Pertemuan | 15 |
| Topik | Advanced NLP Tasks (Text Summarization, QA, dan Dense Retrieval) |
| Output | `advanced_nlp_tasks.ipynb` |

---

## B. Tujuan

Mahasiswa mampu:

* memahami perbedaan *extractive* dan *abstractive* text summarization;
* mengevaluasi ringkasan teks menggunakan metrik ROUGE (ROUGE-1, ROUGE-2, ROUGE-L);
* mengimplementasikan model *Question Answering* berbasis *extractive span prediction*;
* memahami konsep *Dense Retrieval* dan fondasi *Retrieval-Augmented Generation (RAG)* menggunakan Vector Search;
* membangun pipeline pencarian semantik menggunakan `sentence-transformers` dan pencarian vektor;
* membuat antarmuka (*User Interface*) interaktif sederhana menggunakan `Gradio` untuk mendemonstrasikan aplikasi NLP.

---

## C. Materi Teori

### 1. Text Summarization: Extractive vs Abstractive

```text
Pendekatan 1: Extractive Summarization
[Dokumen Panjang] → Pilih kalimat-kalimat paling representatif → Gabungkan
- Kelebihan: Faktual terjamin, kalimat gramatikal.
- Kekurangan: Kurang kohesif, tidak dapat parafrase.

Pendekatan 2: Abstractive Summarization (Sequence-to-Sequence / Encoder-Decoder)
[Dokumen Panjang] → Encoder memahami makna → Decoder menulis ringkasan baru
- Kelebihan: Dapat memadatkan informasi, membuat parafrase alami.
- Kekurangan: Rentan terhadap halusinasi (*factual inaccuracy*).
```

Metrik evaluasi: **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**:
* **ROUGE-1**: Overlap unigram antara ringkasan kandidat dan ringkasan referensi.
* **ROUGE-2**: Overlap bigram.
* **ROUGE-L**: Panjang *Longest Common Subsequence* (LCS).

### 2. Extractive Question Answering (QA)

Model menerima sepasang input: **Pertanyaan ($Q$)** dan **Konteks Dokumen ($C$)**:

$$\text{Input} = \text{[CLS]} \circ Q \circ \text{[SEP]} \circ C \circ \text{[SEP]}$$

Model memprediksi probabilitas posisi token awal (*start token index*) dan token akhir (*end token index*) dari jawaban di dalam konteks:

$$P_{\text{start}}(i) = \frac{\exp(\mathbf{s} \cdot \mathbf{h}_i)}{\sum_j \exp(\mathbf{s} \cdot \mathbf{h}_j)}, \quad P_{\text{end}}(k) = \frac{\exp(\mathbf{e} \cdot \mathbf{h}_k)}{\sum_j \exp(\mathbf{e} \cdot \mathbf{h}_j)}$$

### 3. Dense Retrieval & Dasar RAG (Retrieval-Augmented Generation)

Pencarian kata kunci tradisional (BM25 / TF-IDF) rentan terhadap *vocabulary mismatch* (misal: mencari *"kendaraan"* tidak menemukan dokumen yang hanya menulis *"mobil"*).

**Dense Retrieval** mengatasi hal ini dengan merepresentasikan query dan dokumen ke dalam ruang vektor kontinu berdimensi padat (*dense embedding*):

```text
Query: "Bagaimana cara refund tiket pesawat?"
   ↓ Bi-Encoder (Embedding)
q ∈ R^768
   ↓ Cosine Similarity / Dot Product terhadap Korpus Vektor Dokumen
[Dokumen 1, Dokumen 2, ...] → Ambil Top-K Dokumen Relevan
```

---

## D. Praktikum

### 1. Persiapan Environment

```python
!pip install -q transformers datasets evaluate rouge-score sentence-transformers gradio
```

Verifikasi GPU:

```python
import torch

device = 0 if torch.cuda.is_available() else -1
print(f"Device index: {device}")
```

### 2. Praktikum 1: Abstractive Text Summarization

Kita menggunakan model Seq2Seq pretrained multibahasa/Bahasa Indonesia (misal IndoBART atau mT5):

```python
from transformers import pipeline

# Inisialisasi pipeline summarization
summarizer = pipeline(
    "summarization",
    model="cahya/t5-base-indonesian-summarization",
    device=device
)

artikel_panjang = """
Kementerian Komunikasi dan Digital terus mempercepat pemerataan akses internet broadband di wilayah 
tertinggal, terdepan, dan terluar (3T). Program ini bertujuan untuk mengurangi kesenjangan digital 
antara kawasan perkotaan dan perdesaan. Hingga kuartal ketiga tahun ini, ribuan menara Base Transceiver 
Station (BTS) 4G telah berhasil dibangun dan dioperasikan. Kehadiran infrastruktur digital ini diharapkan 
mampu menggerakkan roda perekonomian lokal, memfasilitasi digitalisasi UMKM, serta mendukung pembelajaran 
jarak jauh bagi para pelajar di pelosok nusantara. Pemerintah juga menggandeng penyedia layanan satelit 
orbit rendah guna menyediakan koneksi darurat bagi fasilitas kesehatan dan kantor pemerintahan desa.
"""

summary = summarizer(
    artikel_panjang,
    max_length=60,
    min_length=20,
    do_sample=False
)

print("=== TEKS ASLI ===")
print(artikel_panjang.strip())
print("\n=== HASIL RINGKASAN ABSTRACTIVE ===")
print(summary[0]["summary_text"])
```

### 3. Evaluasi Ringkasan dengan ROUGE Score

```python
import evaluate

rouge = evaluate.load("rouge")

predictions = [summary[0]["summary_text"]]
references = [
    "Pemerintah mempercepat pembangunan BTS 4G di wilayah 3T untuk pemerataan internet dan mendukung ekonomi lokal."
]

scores = rouge.compute(predictions=predictions, references=references)
print("=== SKOR ROUGE ===")
for k, v in scores.items():
    print(f"{k:>10}: {v:.4f}")
```

### 4. Praktikum 2: Extractive Question Answering

Kita gunakan model QA multibahasa berbasis Transformer:

```python
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/gelectra-base-germanquad", # Atau model QA multibahasa bert-base-multilingual-cased
    device=device
)

konteks_qa = """
Kecerdasan Buatan (Artificial Intelligence) didirikan sebagai disiplin ilmu akademik pada konferensi musim 
panas di Dartmouth College pada tahun 1956. John McCarthy, Marvin Minsky, Nathaniel Rochester, dan Claude Shannon 
menjadi penggagas utama lokakarya tersebut. John McCarthy pertama kali memperkenalkan istilah Artificial Intelligence 
dalam proposal riset yang ia tulis setahun sebelumnya pada tahun 1955.
"""

pertanyaan_list = [
    "Kapan istilah Artificial Intelligence pertama kali diperkenalkan?",
    "Di mana konferensi pendirian kecerdasan buatan diselenggarakan?",
    "Siapakah tokoh yang menggagas lokakarya tersebut?"
]

print("=== QUESTION ANSWERING SYSTEM ===")
for query in pertanyaan_list:
    hasil = qa_pipeline(question=query, context=konteks_qa)
    print(f"Pertanyaan : {query}")
    print(f"Jawaban    : {hasil['answer']}")
    print(f"Score      : {hasil['score']:.4f}")
    print(f"Span Index : ({hasil['start']}, {hasil['end']})\n")
```

### 5. Praktikum 3: Dense Retrieval & Semantic Search

Membangun sistem mesin pencari semantik menggunakan `sentence-transformers`:

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Muat embedding model multibahasa
bi_encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Korpus pengetahuan (Knowledge Base)
knowledge_base = [
    "Mahasiswa wajib menyelesaikan minimal 144 SKS untuk memperoleh gelar Sarjana Teknologi Informasi.",
    "Prosedur pengajuan cuti akademik dapat dilakukan secara online melalui portal akademik kampus.",
    "Perpustakaan pusat buka setiap hari Senin hingga Jumat pukul 08.00 sampai 16.30 WIB.",
    "Beasiswa prestasi akademik dibuka setiap awal semester genap bagi mahasiswa dengan IPK di atas 3.5.",
    "Praktikum laboratorium dilaksanakan secara tatap muka dengan kehadiran minimal 80 persen.",
    "Sidang skripsi mensyaratkan skor TOEFL minimal 450 dan sertifikat bebas tanggungan perpustakaan."
]

# Encode semua dokumen menjadi vektor representasi
doc_embeddings = bi_encoder.encode(knowledge_base, convert_to_tensor=True)
print(f"Dimensi vektor korpus: {doc_embeddings.shape}")
```

Fungsi pencarian query terhadap korpus:

```python
def search_knowledge_base(query, top_k=2):
    query_emb = bi_encoder.encode(query, convert_to_tensor=True)
    # Hitung cosine similarity
    scores = util.cos_sim(query_emb, doc_embeddings)[0]
    
    # Ambil index dengan similarity tertinggi
    top_results = torch.topk(scores, k=top_k)
    
    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        results.append({
            "doc": knowledge_base[idx],
            "similarity": float(score)
        })
    return results

# Uji semantic query yang tidak memiliki overlap kata kunci langsung
query_test = "Bagaimana cara istirahat kuliah sementara?"
matches = search_knowledge_base(query_test)

print(f"Query: '{query_test}'\n")
for i, m in enumerate(matches, 1):
    print(f"Peringkat {i} (Similarity: {m['similarity']:.4f}):")
    print(f"  {m['doc']}\n")
```

### 6. Praktikum 4: Membangun Antarmuka Interaktif Sederhana (Gradio Demo)

```python
import gradio as gr

def nlp_showcase(user_query):
    results = search_knowledge_base(user_query, top_k=3)
    output = f"Hasil Pencarian Semantik untuk: '{user_query}'\n\n"
    for i, res in enumerate(results, 1):
        output += f"{i}. [{res['similarity']:.3f}] {res['doc']}\n\n"
    return output

# Membuat UI sederhana
demo = gr.Interface(
    fn=nlp_showcase,
    inputs=gr.Textbox(lines=2, placeholder="Ketik pertanyaan atau kata kunci di sini...", label="Query Mahasiswa"),
    outputs=gr.Textbox(lines=8, label="Dokumen Terkait yang Ditemukan"),
    title="Sistem FAQ Semantik Kampus",
    description="Demo pencarian semantik menggunakan Pretrained Sentence-Transformers.",
    examples=[
        ["Bagaimana syarat kelulusan sarjana?"],
        ["Berapa batas ketidakhadiran praktikum?"],
        ["Apakah ada bantuan biaya kuliah untuk mahasiswa berprestasi?"]
    ]
)

# Jalankan server lokal (share=True jika di Colab)
# demo.launch(share=False)
print("Aplikasi Gradio siap diluncurkan dengan demo.launch()")
```

---

## E. Tugas Analisis

1. Bandingkan ringkasan abstractive yang dihasilkan model dengan ringkasan manual (*human summary*). Apakah ditemukan gejala *hallucination* (model menambahkan fakta yang tidak tertulis pada teks asli)?
2. Ujilah sistem *Dense Retrieval* dengan memberikan query yang mengandung kata gaul (*slang*) atau sinonim. Bandingkan hasilnya jika menggunakan pencarian kata kunci berbasis BoW/TF-IDF dari Modul 2.
3. Analisis mengapa model Question Answering gagal memberikan jawaban jika pertanyaan membutuhkan penalaran bertingkat (*multi-hop reasoning*).

---

## F. Pertanyaan Diskusi

1. Mengapa evaluasi generasi teks (Summarization / MT) jauh lebih sulit dibandingkan evaluasi klasifikasi teks? Apa keterbatasan metrik ROUGE dan BLEU?
2. Jelaskan bagaimana mekanisme *Retrieval-Augmented Generation (RAG)* menggabungkan kekuatan Dense Retrieval dan Pretrained Generative Model untuk mengatasi halusinasi!
3. Apa perbedaan arsitektur antara *Bi-Encoder* (seperti Sentence-BERT) dan *Cross-Encoder* dalam hal akurasi vs kecepatan pemrosesan?
4. Bagaimana pengaruh pemilihan `top_k` dan nilai ambang batas kesamaan (*similarity threshold*) pada keandalan sistem pencarian informasi?

---

## G. Output Pertemuan 15

File yang dikumpulkan:

```text
advanced_nlp_tasks.ipynb
```

> Output: Notebook terintegrasi yang mencakup demonstrasi summarization beserta evaluasi ROUGE, extractive QA dengan konteks kustom, semantic search engine berbasis dense embedding, dan prototype UI interaktif Gradio.

---

> **Pertemuan sebelumnya:** [Pertemuan 14 — NER dan Information Extraction](praktikum14.md)
>
> **Pertemuan berikutnya:** [Pertemuan 16 — Final NLP Project](praktikum16.md)
