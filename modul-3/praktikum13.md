# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 3 — PERTEMUAN 13
## Fine-Tuning dan NLP Tasks

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 3 — Transformer, Pretrained Model, dan Aplikasi NLP |
| Pertemuan | 13 |
| Topik | Fine-Tuning Pretrained Model untuk Text Classification |
| Output | `fine_tuning.ipynb` |

---

## B. Tujuan

Mahasiswa mampu:

* memahami perbedaan *feature extraction* (frozen backbone) dan *end-to-end fine-tuning*;
* memahami mekanisme *classification head* pada model berbasis Transformer;
* menyiapkan dan memformat dataset menggunakan Hugging Face `datasets`;
* mengonfigurasi `TrainingArguments` dan menjalankan proses training menggunakan Hugging Face `Trainer`;
* menghitung dan menginterpretasikan metrik evaluasi (Accuracy, Precision, Recall, Macro/Weighted F1-Score);
* menyimpan (*save*) model hasil fine-tuning dan melakukan *inference* pada kalimat baru;
* menganalisis performa model serta efek dari *hyperparameter tuning* (learning rate, epoch, batch size).

---

## C. Materi Teori

### 1. Konsep Fine-Tuning vs Feature Extraction

Ketika memanfaatkan model pretrained seperti IndoBERT atau BERT, terdapat dua pendekatan utama:

```text
Pendekatan 1: Feature Extraction
[Teks Input] → [BERT Backbone (FROZEN)] → [Embedding [CLS]] → [Classifier Baru (Dilatih)]
- Bobot BERT tidak diupdate.
- Cepat dan hemat komputasi, namun adaptasi representasi terbatas.

Pendekatan 2: End-to-End Fine-Tuning (Standar Modern)
[Teks Input] → [BERT Backbone (UPDATED)] → [Embedding [CLS]] → [Classifier Baru (Dilatih)]
- Bobot BERT dan Classifier diupdate bersamaan dengan learning rate kecil (e.g., 2e-5).
- Memberikan performa jauh lebih tinggi karena representasi internal beradaptasi dengan domain tugas.
```

### 2. Arsitektur Sequence Classification

Untuk tugas klasifikasi kalimat (seperti Analisis Sentimen atau Deteksi Hoaks), kita menambahkan sebuah linear layer (*classification head*) di atas representasi token khusus `[CLS]`:

$$\mathbf{h}_{\text{CLS}} = \text{Encoder}(\mathbf{x})[0]$$

$$\hat{\mathbf{y}} = \text{Softmax}(\mathbf{W} \mathbf{h}_{\text{CLS}} + \mathbf{b})$$

Di mana:
* $\mathbf{h}_{\text{CLS}} \in \mathbb{R}^{d_{\text{model}}}$ (misalnya $d = 768$ untuk `indobert-base-p1`).
* $\mathbf{W} \in \mathbb{R}^{C \times d}$ dan $\mathbf{b} \in \mathbb{R}^{C}$ adalah bobot dan bias classification head ($C$ adalah jumlah kelas).
* Loss function yang digunakan adalah Cross-Entropy Loss:

$$\mathcal{L} = - \sum_{c=1}^C y_c \log \hat{y}_c$$

### 3. Komponen Utama Hugging Face Ecosystem

```text
Hugging Face Workflow:
1. AutoTokenizer          → Tokenisasi subword, padding, truncation
2. Dataset / DatasetDict  → Pengelolaan data tabular / teks efisien
3. AutoModelForSequenceClassification → Pretrained transformer + classification head
4. TrainingArguments      → Konfigurasi training (lr, batch size, epochs, logging)
5. Trainer API            → Training loop, validation, checkpointing otomatis
6. Pipeline               → Inference end-to-end yang mudah digunakan
```

### 4. Tantangan Fine-Tuning: Catastrophic Forgetting

Jika learning rate terlalu besar (misal $10^{-3}$ seperti pada training dari nol), bobot pretrained model yang sudah kaya pengetahuan linguistik akan rusak drastis (*catastrophic forgetting*). Oleh karena itu:
* Gunakan learning rate sangat kecil: $10^{-5}$ hingga $5 \times 10^{-5}$ (misal $2 \times 10^{-5}$ atau $3 \times 10^{-5}$).
* Gunakan *warmup steps* dan *linear learning rate decay*.
* Cukup latih dalam $3$ hingga $5$ epoch.

---

## D. Praktikum

### 1. Persiapan Environment

Jalankan instalasi library di notebook:

```python
!pip install -q transformers datasets evaluate accelerate scikit-learn matplotlib seaborn
```

Verifikasi GPU:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device yang digunakan: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Tersedia: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 2. Memuat dan Menyiapkan Dataset

Kita akan menggunakan dataset sentimen ulasan dalam Bahasa Indonesia (contoh dataset sentimen publik atau dataset lokal):

```python
import pandas as pd
import numpy as np

# Contoh dataset sentimen ulasan produk / layanan Bahasa Indonesia
sample_data = {
    "text": [
        "Pelayanan sangat ramah dan pengiriman sangat cepat, saya puas sekali!",
        "Barang rusak saat sampai, kemasan penyok tidak ada bubble wrap.",
        "Kualitas produk biasa saja, sesuai dengan harga yang murah.",
        "Aplikasi ini sering crash setelah update terbaru, tolong diperbaiki.",
        "Sangat recommended! Desain elegan dan performanya mantap.",
        "Kecewa banget, respon admin lambat dan tidak solutif sama sekali.",
        "Cukup bagus untuk pemula, panduannya mudah dipahami.",
        "Pengiriman super lama, seminggu baru sampai padahal satu kota.",
        "Fitur-fiturnya sangat lengkap dan mempermudah pekerjaan sehari-hari.",
        "Harga kemahalan untuk kualitas barang yang seperti ini.",
        "Makanannya lezat sekali, bumbunya pas dan porsinya mengenyangkan.",
        "Pesanan tidak sesuai dengan yang di foto, sangat menyesal beli di sini.",
        "Sistem pembayaran lancar tanpa kendala sama sekali.",
        "Jaringan sering putus-putus, tidak bisa dipakai meeting online.",
        "CS ramah dan langsung membantu menyelesaikan kendala dalam 5 menit."
    ] * 20,  # Digandakan untuk simulasi dataset 300 data
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 20 # 1: Positif, 0: Negatif
}

df = pd.DataFrame(sample_data)
print(f"Total data: {len(df)}")
print(df["label"].value_counts())
df.head()
```

### 3. Konversi ke Hugging Face Dataset & Train-Val-Test Split

```python
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Split stratified 70% train, 15% validation, 15% test
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True))
})

print(dataset)
```

### 4. Tokenisasi Dataset dengan Pretrained Tokenizer

Gunakan pretrained tokenizer `indobenchmark/indobert-base-p1`:

```python
from transformers import AutoTokenizer

MODEL_NAME = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    # Truncation dan padding dinamis
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )

tokenized_datasets = dataset.map(preprocess_function, batched=True)
print("Fitur dataset setelah tokenisasi:", tokenized_datasets["train"].column_names)
```

### 5. Memuat Pretrained Model dengan Classification Head

```python
from transformers import AutoModelForSequenceClassification

id2label = {0: "NEGATIF", 1: "POSITIF"}
label2id = {"NEGATIF": 0, "POSITIF": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)
model.to(device)
```

### 6. Menentukan Fungsi Evaluasi Metrik

```python
import evaluate

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    prec = precision_metric.compute(predictions=preds, references=labels, average="macro")["precision"]
    rec = recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"]
    
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": prec,
        "recall_macro": rec
    }
```

### 7. Konfigurasi TrainingArguments & Inisialisasi Trainer

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results_sentiment",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_dir="./logs",
    logging_steps=10,
    seed=42,
    report_to="none" # Nonaktifkan wandb/telemetry
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```

### 8. Menjalankan Training (Fine-Tuning)

```python
# Latih model
train_result = trainer.train()

# Simpan model terbaik
model.save_pretrained("./best_indobert_sentiment")
tokenizer.save_pretrained("./best_indobert_sentiment")
print("Model berhasil disimpan ke folder ./best_indobert_sentiment")
```

### 9. Evaluasi pada Test Set & Analisis Error

```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluasi pada data test yang belum pernah dilihat model
test_predictions = trainer.predict(tokenized_datasets["test"])
y_true = test_predictions.label_ids
y_pred = np.argmax(test_predictions.predictions, axis=1)

# Print Classification Report
target_names = ["Negatif", "Positif"]
print("\n=== CLASSIFICATION REPORT PADA TEST SET ===")
print(classification_report(y_true, y_pred, target_names=target_names))

# Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix — Test Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
```

### 10. Pipeline Inference pada Teks Baru

```python
from transformers import pipeline

# Muat model menggunakan Hugging Face pipeline
sentiment_classifier = pipeline(
    "text-classification",
    model="./best_indobert_sentiment",
    tokenizer="./best_indobert_sentiment",
    device=0 if torch.cuda.is_available() else -1
)

test_sentences = [
    "Barang original, sellernya amanah dan fast respon!",
    "Parah banget, barang tidak datang dan dana tidak dikembalikan!",
    "Bagus sih, tapi warnanya sedikit beda dengan yang di katalog foto."
]

print("=== PENGUJIAN INFERENCE MODEL BARU ===")
for sent in test_sentences:
    res = sentiment_classifier(sent)[0]
    print(f"Kalimat : {sent}")
    print(f"Prediksi: {res['label']} (Confidence: {res['score']:.4f})\n")
```

---

## E. Tugas Analisis

1. Lakukan eksperimen perbandingan dengan mengubah *learning rate*:
   * $5 \times 10^{-4}$ (terlalu besar)
   * $2 \times 10^{-5}$ (standar yang direkomendasikan)
   * $1 \times 10^{-6}$ (sangat kecil)
   Gambarkan grafik loss dan bandingkan performa Macro F1-score pada validasi.
2. Bandingkan performa fine-tuning antara model `indobenchmark/indobert-base-p1` dengan model multilingual `bert-base-multilingual-cased` pada test set yang sama.
3. Analisis minimal 3 contoh kalimat yang salah diprediksi (*false positive* atau *false negative*). Jelaskan faktor linguistik apa yang menyebabkan model salah mengklasifikasikannya (misal: kalimat sarkasme, negasi ganda, atau kata serapan baru).

---

## F. Pertanyaan Diskusi

1. Mengapa fine-tuning seluruh bobot Transformer biasanya membutuhkan learning rate yang jauh lebih kecil ($10^{-5}$) dibandingkan melatih model neural network dari nol ($10^{-3}$)?
2. Jelaskan fungsi dari token khusus `[CLS]` dalam arsitektur BERT untuk tugas klasifikasi kalimat!
3. Apa risiko yang dihadapi jika melakukan fine-tuning pada dataset yang sangat kecil (<100 sampel)? Bagaimana cara mengatasinya?
4. Apa perbedaan antara *Macro Average* dan *Weighted Average* F1-score, dan mengapa Macro F1 lebih disukai pada dataset yang tidak seimbang (*imbalanced*)?
5. Kapan sebaiknya kita memilih pendekatan *Feature Extraction* (frozen backbone) daripada *End-to-End Fine-Tuning*?

---

## G. Output Pertemuan 13

File yang dikumpulkan:

```text
fine_tuning.ipynb
```

> Output: Notebook lengkap berisi pipeline preprocessing data, training loop/Trainer HuggingFace, evaluasi metrik (Confusion Matrix & Classification Report), serta pengujian inference kalimat baru.

---

> **Pertemuan sebelumnya:** [Pertemuan 12 — Pretrained Language Model](praktikum12.md)
>
> **Pertemuan berikutnya:** [Pertemuan 14 — NER dan Information Extraction](praktikum14.md)
