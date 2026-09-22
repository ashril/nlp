# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 3 — PERTEMUAN 16
## Final NLP Project: Integrasi, Evaluasi Komprehensif, dan Deployment Prototype

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 3 — Transformer, Pretrained Model, dan Aplikasi NLP |
| Pertemuan | 16 |
| Topik | Final NLP Capstone Project & Evaluasi Akhir |
| Output | `final_project.ipynb` & Repository/Demo Project |

---

## B. Tujuan

Mahasiswa mampu:

* merancang solusi masalah berbasis Natural Language Processing secara *end-to-end* (mulai dari formulasi masalah hingga prototipe aplikasi);
* mengintegrasikan seluruh materi yang dipelajari dari Modul 1 (Text Processing), Modul 2 (Representasi & Neural NLP), dan Modul 3 (Transformer & Fine-Tuning);
* melakukan eksperimen komparatif terstruktur (*baseline model* vs *advanced transformer model*);
* melakukan analisis error (*error analysis*) mendalam untuk mengevaluasi batasan model;
* membungkus model hasil pelatihan menjadi aplikasi prototipe interaktif (Streamlit / Gradio / Web API);
* menyusun dokumentasi teknis yang profesional dalam bentuk notebook reproducible dan repository GitHub.

---

## C. Panduan Final Project NLP

Final Project adalah puncak dari praktikum NLP yang menguji kemampuan mahasiswa menyelesaikan studi kasus nyata (*real-world case study*).

### 1. Siklus Hidup Project NLP (End-to-End Workflow)

```text
┌────────────────────────────────────────────────────────┐
│ 1. Problem Formulation & Scope Definition              │
│    - Pilih domain (e-commerce, kesehatan, hukum, dsb.) │
│    - Definisikan task (klasifikasi, NER, QA, summary)  │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│ 2. Data Collection & Preprocessing (Modul 1)           │
│    - Dataset acquisition (Hugging Face / Kaggle / Crawl│
│    - Cleaning, case folding, normalisasi, tokenisasi   │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│ 3. Eksperimen Komparatif & Modeling (Modul 2 & 3)      │
│    - Baseline: TF-IDF + Klasik (SVM / Naive Bayes / LR)│
│    - Neural Baseline: LSTM / BiLSTM / GRU              │
│    - State-of-the-Art: Fine-tuned IndoBERT / RoBERTa   │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│ 4. Evaluasi Ketat & Error Analysis                     │
│    - Metrik: Macro F1, Precision, Recall, Confusion Mat│
│    - Analisis kelemahan model & false predictions      │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│ 5. Prototype Deployment & Interactive Showcase         │
│    - Antarmuka interaktif: Gradio / Streamlit          │
│    - Dokumentasi teknis & README github                │
└────────────────────────────────────────────────────────┘
```

---

## D. Struktur Standar Notebook `final_project.ipynb`

Mahasiswa wajib menyusun notebook final project dengan sistematika terstandarisasi sebagai berikut:

### Bagian 1: Judul dan Identitas Tim

```python
# ==============================================================================
# FINAL PROJECT NATURAL LANGUAGE PROCESSING
# Judul Proyek : Klasifikasi Sentimen Aspek Multi-Domain Ulasan Konsumen
# Anggota Tim  :
# 1. [Nama Mahasiswa 1] - [NIM 1]
# 2. [Nama Mahasiswa 2] - [NIM 2]
# ==============================================================================
```

### Bagian 2: Data Acquisition & EDA (Exploratory Data Analysis)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
# Mahasiswa dapat menggunakan dataset lokal atau dari Hugging Face Hub
df = pd.read_csv("dataset_final_project.csv")
print(f"Total baris data: {len(df)}")
print("Distribusi Target Label:")
print(df["label"].value_counts(normalize=True))

# Visualisasi distribusi panjang kalimat
df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(8, 4))
sns.histplot(df["word_count"], kde=True, bins=30, color="teal")
plt.title("Distribusi Panjang Kata dalam Dokumen")
plt.xlabel("Jumlah Kata")
plt.ylabel("Frekuensi")
plt.show()
```

### Bagian 3: Text Preprocessing Pipeline (Modul 1)

```python
import re
import string

def preprocess_pipeline(text):
    # Case folding
    text = str(text).lower()
    # Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Hapus mention dan hashtag
    text = re.sub(r"[@#]\w+", "", text)
    # Hapus karakter tanda baca dan angka
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", "", text)
    # Hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(preprocess_pipeline)
```

### Bagian 4: Model Baseline 1 — TF-IDF + Machine Learning (Modul 2)

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Ekstraksi fitur TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Latih Baseline Classifier
baseline_clf = LogisticRegression(max_iter=1000)
baseline_clf.fit(X_train_tfidf, y_train)

y_pred_baseline = baseline_clf.predict(X_test_tfidf)
print("=== BASELINE 1 (TF-IDF + LOGISTIC REGRESSION) ===")
print(classification_report(y_test, y_pred_baseline))
```

### Bagian 5: Model Utama — Fine-Tuning IndoBERT (Modul 3)

```python
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHECKPOINT = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

train_dataset = Dataset.from_pandas(pd.DataFrame({"text": X_train, "label": y_train}).reset_index(drop=True))
test_dataset = Dataset.from_pandas(pd.DataFrame({"text": X_test, "label": y_test}).reset_index(drop=True))

def tokenize_batch(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_tokenized = train_dataset.map(tokenize_batch, batched=True)
test_tokenized = test_dataset.map(tokenize_batch, batched=True)

f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

num_classes = len(df["label"].unique())
model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=num_classes).to(device)

training_args = TrainingArguments(
    output_dir="./final_project_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    seed=42,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
```

### Bagian 6: Tabel Perbandingan dan Error Analysis

```python
# Evaluasi komparatif
test_preds = trainer.predict(test_tokenized)
y_pred_transformer = np.argmax(test_preds.predictions, axis=1)

print("=== PERBANDINGAN PERFORMA MODEL ===")
# Cetak tabel perbandingan metrik Baseline vs Transformer
# Lakukan analisis terhadap 5 sampel di mana baseline salah tapi Transformer benar (dan sebaliknya)
```

### Bagian 7: Interactive Demo Deployment (Gradio)

```python
import gradio as gr
from transformers import pipeline

classifier_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def predict_nlp(text):
    clean = preprocess_pipeline(text)
    res = classifier_pipe(clean)[0]
    return f"Label: {res['label']} (Confidence: {res['score']:.4f})"

demo = gr.Interface(
    fn=predict_nlp,
    inputs=gr.Textbox(lines=3, placeholder="Masukkan teks uji coba..."),
    outputs="text",
    title="Demonstrasi Prototype Final Project NLP"
)
# demo.launch()
```

---

## E. Template README Repository GitHub

Mahasiswa diwajibkan menyertakan `README.md` pada repository GitHub proyek dengan struktur:

```markdown
# [Judul Project NLP Anda]

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 1. Ringkasan Proyek
Penjelasan latar belakang masalah, tujuan, dan urgensi solusi NLP yang dibangun.

## 2. Dataset
- Sumber data: (Kaggle / Crawling / dsb.)
- Jumlah sampel: (e.g. 10.000 kalimat)
- Jumlah kelas / skema anotasi: ...

## 3. Arsitektur Model & Metodologi
Penjelasan eksperimen model yang dibandingkan (Baseline vs Transformer IndoBERT).

## 4. Hasil Eksperimen & Perbandingan
| Model | Akurasi | Precision | Recall | F1-Score |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | ... | ... | ... | ... |
| LSTM / BiLSTM | ... | ... | ... | ... |
| IndoBERT (Fine-Tuned) | ... | ... | ... | ... |

## 5. Cara Menjalankan
```bash
git clone <url-repo>
pip install -r requirements.txt
python app.py
```

## 6. Anggota Tim & Kontribusi
- [Nama Mahasiswa 1] - Preprocessing, Baseline Modeling, Dokumentasi
- [Nama Mahasiswa 2] - Fine-Tuning Transformer, Deployment UI Gradio
```

---

## F. Rubrik Penilaian Capstone Project

| Komponen Penilaian | Bobot | Kriteria Evaluasi |
|---|---|---|
| **Problem Definition & EDA** | 15% | Kejelasan rumusan masalah, kualitas visualisasi EDA, dan pemahaman karakteristik data. |
| **Pipeline Preprocessing** | 15% | Kebersihan kode, teknik normalisasi teks yang tepat, dan penanganan tokenisasi yang benar. |
| **Eksperimen Komparatif** | 25% | Mengimplementasikan minimal 2 jenis model (Baseline Klasik/RNN vs Transformer) dengan perbandingan yang adil (*fair benchmark*). |
| **Evaluasi & Error Analysis** | 20% | Kedalaman analisis kesalahan prediksi, pembahasan faktor linguistik, dan interpretasi metrik (bukan hanya skor angka). |
| **Interactive Prototype / UI** | 15% | Fungsionalitas aplikasi prototipe (Gradio/Streamlit), kemudahan penggunaan, dan respon sistem. |
| **Dokumentasi & Reproducibility**| 10% | Kerapian notebook, kejelasan markdown, dan kelengkapan README pada repository GitHub. |

---

## G. Output Pertemuan 16

1. **Jupyter Notebook**: `final_project.ipynb` yang dapat dijalankan dari awal hingga akhir (*clean run*).
2. **Repository GitHub**: Berisi source code, dataset (atau link download jika ukuran besar), dan file `README.md` proyek.
3. **Demo Showcase**: Tautan deployment publik (Hugging Face Spaces / Streamlit Cloud) atau rekaman video demonstrasi singkat (2–3 menit).

---

> **Pertemuan sebelumnya:** [Pertemuan 15 — Advanced NLP Tasks](praktikum15.md)
>
> 🏁 **Selamat! Anda telah menyelesaikan seluruh rangkaian Modul Praktikum Natural Language Processing (Pertemuan 1–16).**
