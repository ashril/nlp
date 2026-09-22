# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 3 — TRANSFORMER, PRETRAINED MODEL, DAN APLIKASI NLP

---

## A. Identitas Modul

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Program Studi | S1 Teknologi Informasi |
| Modul | Modul 3 |
| Topik | Transformer, Pretrained Model, dan Aplikasi NLP |
| Platform | VS Code / Google Colab |
| Environment | Jupyter Notebook |
| Bahasa | Python |
| Pertemuan | 11–16 |

---

## B. Deskripsi Modul

Modul 3 adalah puncak dari mata kuliah NLP. Mahasiswa mempelajari arsitektur Transformer yang menjadi fondasi semua model NLP modern, kemudian menggunakan pretrained language model (BERT, IndoBERT, GPT) untuk menyelesaikan berbagai task NLP nyata.

Alur Modul 3:

```text
Attention (dari Modul 2)
        ↓
Transformer Architecture
        ↓
Pretrained Language Model (BERT / IndoBERT)
        ↓
Fine-Tuning
        ↓
NLP Tasks (NER, Classification, Summarization, QA)
        ↓
NLP Application
```

---

## C. Tujuan Modul

Setelah menyelesaikan Modul 3, mahasiswa diharapkan mampu:

1. Memahami arsitektur Transformer secara menyeluruh.
2. Memahami self-attention, multi-head attention, dan positional encoding.
3. Menggunakan Hugging Face Transformers untuk load pretrained model.
4. Melakukan tokenisasi subword dengan tokenizer Transformer.
5. Menjalankan inference dengan pretrained model.
6. Melakukan fine-tuning model untuk task klasifikasi.
7. Mengimplementasikan NER Bahasa Indonesia.
8. Menggunakan model untuk sentiment analysis, summarization, atau QA.
9. Membangun dan mempresentasikan prototype aplikasi NLP end-to-end.

---

## D. Pembagian Praktikum

| Pertemuan | File | Materi | Output |
|-----------|------|--------|--------|
| 11 | [praktikum11.md](praktikum11.md) | Transformer | Eksperimen arsitektur Transformer |
| 12 | [praktikum12.md](praktikum12.md) | Pretrained Language Model | Notebook eksplorasi BERT/IndoBERT |
| 13 | [praktikum13.md](praktikum13.md) | Fine-Tuning dan NLP Tasks | Model NLP hasil fine-tuning |
| 14 | [praktikum14.md](praktikum14.md) | NER dan Information Extraction | Prototype information extraction |
| 15 | [praktikum15.md](praktikum15.md) | Advanced NLP Tasks | Prototype aplikasi NLP |
| 16 | [praktikum16.md](praktikum16.md) | Final NLP Project | Demo & presentasi |

---

## E. Library Modul 3

Install library berikut:

```bash
pip install transformers datasets accelerate sentencepiece
```

Untuk GPU (opsional namun direkomendasikan):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verifikasi:

```python
import transformers
import datasets
import torch

print("Transformers:", transformers.__version__)
print("Datasets    :", datasets.__version__)
print("PyTorch     :", torch.__version__)
print("GPU tersedia:", torch.cuda.is_available())
```

---

## F. Rekomendasi Platform

| Platform | Keterangan |
|----------|-----------|
| VS Code + Jupyter | Untuk eksplorasi lokal |
| Google Colab (Free) | Untuk fine-tuning dengan GPU gratis |
| Google Colab Pro | Untuk fine-tuning model besar |
| Kaggle Notebooks | Alternatif GPU gratis |

> Untuk pertemuan 13–16 (fine-tuning), **sangat disarankan** menggunakan Google Colab atau Kaggle agar tersedia GPU.

---

## G. Struktur Folder

```text
NLP/
│
├── modul-3/
│   │
│   ├── pertemuan-11/
│   │   └── transformer.ipynb
│   │
│   ├── pertemuan-12/
│   │   └── pretrained_lm.ipynb
│   │
│   ├── pertemuan-13/
│   │   └── fine_tuning.ipynb
│   │
│   ├── pertemuan-14/
│   │   └── ner_information_extraction.ipynb
│   │
│   ├── pertemuan-15/
│   │   └── advanced_nlp_tasks.ipynb
│   │
│   └── pertemuan-16/
│       ├── final_project.ipynb
│       └── README.md
```

---

## H. Posisi Modul 3 dalam Roadmap NLP

```text
[Modul 1] Text Processing
    ↓
[Modul 2] Representasi & Neural NLP
    ↓
[Modul 3] Transformer & Pretrained Model  ← Anda di sini
```

Modul 3 menjawab pertanyaan:

> *Bagaimana kita menggunakan model bahasa yang sudah dilatih pada data besar untuk menyelesaikan permasalahan NLP nyata dengan hasil yang sangat baik?*
