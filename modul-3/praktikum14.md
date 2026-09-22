# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 3 — PERTEMUAN 14
## Named Entity Recognition dan Information Extraction

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 3 — Transformer, Pretrained Model, dan Aplikasi NLP |
| Pertemuan | 14 |
| Topik | Named Entity Recognition (NER) dan Information Extraction |
| Output | `ner_information_extraction.ipynb` |

---

## B. Tujuan

Mahasiswa mampu:

* memahami konsep Information Extraction (IE) dan Named Entity Recognition (NER) sebagai tugas *Token Classification*;
* memahami skema pelabelan entitas BIO (*Beginning, Inside, Outside*) / IOB2;
* menyelesaikan tantangan *subword token alignment* antara tokenizer Transformer dan label tingkat kata;
* memuat model token classification pretrained atau melakukan fine-tuning pada dataset NER Bahasa Indonesia;
* mengevaluasi model NER menggunakan metrik standar industri (*seqeval*: Precision, Recall, F1-score tingkat entitas);
* membangun pipeline Information Extraction untuk mengekstrak entitas terstruktur (Nama Tokoh, Organisasi, Lokasi, Tanggal/Waktu) dari teks tidak terstruktur.

---

## C. Materi Teori

### 1. Information Extraction (IE) dan Token Classification

Information Extraction bertujuan mengubah teks bebas yang tidak terstruktur (*unstructured text*) menjadi data relasional yang terstruktur (*structured records*). Fondasi utama IE adalah **Named Entity Recognition (NER)**.

Berbeda dengan klasifikasi kalimat (di mana satu label diberikan untuk seluruh kalimat), NER adalah tugas **Token Classification**, di mana setiap token diprediksi label kategorinya:

$$\mathbf{x} = (x_1, x_2, \dots, x_N) \longrightarrow \mathbf{y} = (y_1, y_2, \dots, y_N)$$

### 2. Skema Pelabelan BIO (IOB2)

Untuk mengenali entitas multi-kata (seperti *"Joko Widodo"* atau *"Universitas Gadjah Mada"*), digunakan skema BIO:

* **B-TAG (Beginning)**: Token pertama dari sebuah entitas.
* **I-TAG (Inside)**: Token lanjutan yang masih berada dalam entitas yang sama.
* **O (Outside)**: Token yang bukan bagian dari entitas bernama manapun.

Contoh:

| Kata | Token | Label BIO |
|---|---|---|
| Presiden | Presiden | O |
| Joko | Joko | B-PER |
| Widodo | Widodo | I-PER |
| mengunjungi | mengunjungi | O |
| Ibu | Ibu | B-LOC |
| Kota | Kota | I-LOC |
| Nusantara | Nusantara | I-LOC |
| kemarin | kemarin | O |

Tipe entitas standar:
* `PER` / `PERSON`: Nama orang / tokoh
* `ORG` / `ORGANIZATION`: Organisasi, perusahaan, institusi
* `LOC` / `LOCATION`: Lokasi geografis, kota, negara
* `TIME` / `DATE`: Waktu, tanggal, durasi

### 3. Masalah Subword Alignment

Tokenizer modern (WordPiece, BPE) memecah kata majemuk atau berimbuhan menjadi beberapa subword token:

```text
Kata asli :  "Banyuwangi"       "mengunjungi"
Subwords  :  ["Banyu", "##wangi"]  ["meng", "##unjung", "##i"]
Label kata:  B-LOC              O
```

Strategi pelabelan subword:
1. **First-subword labeling**: Token subword pertama mewarisi label asli kata, sedangkan subword berikutnya (`##wangi`) diberi label khusus `-100` agar diabaikan oleh CrossEntropyLoss di PyTorch.
2. **All-subword labeling**: Semua subword diberi label yang sama (atau `I-LOC` untuk kelanjutannya).

Strategi standar Hugging Face menggunakan `label = -100` untuk subword lanjutan dan token khusus (`[CLS]`, `[SEP]`, `[PAD]`).

---

## D. Praktikum

### 1. Persiapan Environment

Instalasi library khusus sequence labeling:

```python
!pip install -q transformers datasets evaluate seqeval accelerate scikit-learn
```

Verifikasi GPU dan library:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
```

### 2. Memahami Struktur Dataset NER Bahasa Indonesia

Kita buat simulasi representasi dataset berformat standar CoNLL/BIO:

```python
import pandas as pd

# Contoh dataset anotasi NER Bahasa Indonesia
raw_ner_data = [
    {
        "tokens": ["Presiden", "Joko", "Widodo", "meresmikan", "kantor", "baru", "di", "Jakarta", "."],
        "ner_tags": ["O", "B-PER", "I-PER", "O", "O", "O", "O", "B-LOC", "O"]
    },
    {
        "tokens": ["Menteri", "Keuangan", "Sri", "Mulyani", "menghadiri", "rapat", "tahunan", "Bank", "Indonesia", "."],
        "ner_tags": ["O", "O", "B-PER", "I-PER", "O", "O", "O", "B-ORG", "I-ORG", "O"]
    },
    {
        "tokens": ["Tim", "peneliti", "dari", "Institut", "Teknologi", "Bandung", "mengembangkan", "vaksin", "baru", "."],
        "ner_tags": ["O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "O", "O"]
    },
    {
        "tokens": ["Gojek", "dan", "Tokopedia", "melakukan", "merger", "membentuk", "GoTo", "."],
        "ner_tags": ["B-ORG", "O", "B-ORG", "O", "O", "O", "B-ORG", "O"]
    },
    {
        "tokens": ["Kunjungan", "wisatawan", "ke", "Pulau", "Bali", "dan", "Lombok", "meningkat", "drastis", "."],
        "ner_tags": ["O", "O", "O", "B-LOC", "I-LOC", "O", "B-LOC", "O", "O", "O"]
    }
]

# Daftar label unik
label_list = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC"
]

label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}
print("Label mapping:", label2id)
```

### 3. Mengonversi ke Hugging Face Dataset & Token Alignment

```python
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

MODEL_NAME = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Siapkan data dengan ID numerik
formatted_data = []
for item in raw_ner_data * 20: # Duplikasi data untuk latihan pipeline
    formatted_data.append({
        "tokens": item["tokens"],
        "ner_tags": [label2id[tag] for tag in item["ner_tags"]]
    })

hf_dataset = Dataset.from_list(formatted_data)
# Split 80% train, 20% validation
dataset_split = hf_dataset.train_test_split(test_size=0.2, seed=42)
print(dataset_split)
```

Fungsi penyelarasan label dengan token subword:

```python
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens ([CLS], [SEP]) dipetakan ke None -> label -100
            if word_idx is None:
                label_ids.append(-100)
            # Token subword pertama dari sebuah kata
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # Token subword lanjutan dari kata yang sama -> diabaikan loss-nya
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_ner = dataset_split.map(tokenize_and_align_labels, batched=True)
print("Contoh input_ids [0]:", tokenized_ner["train"][0]["input_ids"][:8])
print("Contoh labels    [0]:", tokenized_ner["train"][0]["labels"][:8])
```

### 4. Data Collator untuk Token Classification

Pada token classification, padding pada kalimat juga harus menyertakan padding pada label dengan nilai `-100`:

```python
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

### 5. Metrik Evaluasi Seqeval

Seqeval menghitung Precision, Recall, dan F1 pada level entitas utuh (*entity chunk*), bukan sekadar akurasi per token:

```python
import evaluate
import numpy as np

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Hilangkan token spesial (-100) dan kembalikan ke string label
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
```

### 6. Memuat Model Token Classification & Training

```python
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
).to(device)

training_args = TrainingArguments(
    output_dir="./results_ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    seed=42,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ner["train"],
    eval_dataset=tokenized_ner["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Jalankan training
trainer.train()
```

### 7. Inference Menggunakan Pretrained Pipeline NER Indonesia

Di Hugging Face Hub, tersedia berbagai model NER Bahasa Indonesia yang sudah terlatih pada ribuan korpus (misalnya `cahya/bert-base-indonesian-NER`). Mari kita uji inference langsung:

```python
from transformers import pipeline

# Pipeline NER dengan agregasi token subword secara otomatis
ner_pipeline = pipeline(
    "ner",
    model="cahya/bert-base-indonesian-NER",
    tokenizer="cahya/bert-base-indonesian-NER",
    aggregation_strategy="simple", # Menggabungkan subword menjadi satu entitas
    device=0 if torch.cuda.is_available() else -1
)

artikel_berita = """
Presiden Joko Widodo didampingi Menteri BUMN Erick Thohir meresmikan proyek kereta cepat Whoosh 
di Stasiun Halim, Jakarta Timur. Kereta ini menghubungkan Jakarta dan Bandung dalam waktu 45 menit. 
Proyek ini dikembangkan melalui kerja sama konsorsium PT Kereta Cepat Indonesia China (KCIC).
"""

entities = ner_pipeline(artikel_berita)
df_entities = pd.DataFrame(entities)
df_entities[["word", "entity_group", "score", "start", "end"]]
```

### 8. Membangun Pipeline Information Extraction Terstruktur

Kita buat fungsi untuk memparsing teks berita menjadi JSON terstruktur:

```python
def extract_structured_info(text, ner_pipe):
    extracted_entities = ner_pipe(text)
    
    result = {
        "PERSON": [],
        "ORGANIZATION": [],
        "LOCATION": [],
        "RAW_ENTITIES": []
    }
    
    for ent in extracted_entities:
        word = ent["word"].strip()
        group = ent["entity_group"].upper()
        confidence = round(float(ent["score"]), 4)
        
        result["RAW_ENTITIES"].append({"entity": word, "type": group, "confidence": confidence})
        
        if "PER" in group and word not in result["PERSON"]:
            result["PERSON"].append(word)
        elif "ORG" in group and word not in result["ORGANIZATION"]:
            result["ORGANIZATION"].append(word)
        elif "LOC" in group and word not in result["LOCATION"]:
            result["LOCATION"].append(word)
            
    return result

import json
info = extract_structured_info(artikel_berita, ner_pipeline)
print(json.dumps(info, indent=2, ensure_ascii=False))
```

### 9. Visualisasi Entitas Teks (Highlighter)

```python
from IPython.core.display import display, HTML

def render_ner_html(text, entities):
    # Sort entities descending berdasarkan posisi start
    sorted_ents = sorted(entities, key=lambda x: x["start"], reverse=True)
    
    color_map = {
        "PER": "#ffebee", # Merah muda
        "ORG": "#e8f5e9", # Hijau muda
        "LOC": "#e3f2fd", # Biru muda
        "MISC": "#fff3e0" # Oranye muda
    }
    
    html_text = text
    for ent in sorted_ents:
        start = ent["start"]
        end = ent["end"]
        label = ent["entity_group"]
        word = text[start:end]
        bg_color = color_map.get(label, "#f5f5f5")
        
        replacement = f'<span style="background-color: {bg_color}; padding: 2px 6px; border-radius: 4px; border: 1px solid #ccc;"><b>{word}</b> <small style="color: #666;">[{label}]</small></span>'
        html_text = html_text[:start] + replacement + html_text[end:]
        
    display(HTML(f"<div style='line-height: 2.2; font-size: 16px;'>{html_text}</div>"))

render_ner_html(artikel_berita, entities)
```

---

## E. Tugas Analisis

1. Ujilah model NER pada minimal 5 kalimat berita kontemporer yang mengandung:
   * Nama tokoh asing (e.g. *"Elon Musk"*, *"Sam Altman"*)
   * Nama lokasi baru (e.g. *"Ibu Kota Nusantara"*)
   * Singkatan organisasi (e.g. *"IKN"*, *"BRIN"*, *"BMKG"*)
   Evaluasi apakah model berhasil mengekstrak entitas tersebut dengan benar.
2. Analisis bagaimana penanganan token subword memengaruhi hasil ekstraksi nama yang jarang muncul (*Out-of-Vocabulary*).
3. Mengapa metrik akurasi per token sangat menyesatkan pada tugas NER? Jelaskan mengapa metrik *seqeval* (chunk-level evaluation) adalah standar yang wajib digunakan.

---

## F. Pertanyaan Diskusi

1. Apa perbedaan mendasar antara skema pelabelan IOB1, IOB2 (BIO), dan BIOES/BILOU? Mengapa BIOES kadang menghasilkan skor F1 yang sedikit lebih baik?
2. Mengapa kita memberi label `-100` pada token subword lanjutan di PyTorch CrossEntropyLoss?
3. Bagaimana cara mengatasi keterbatasan panjang konteks Transformer (512 token) ketika melakukan ekstraksi entitas pada dokumen hukum atau artikel panjang?
4. Bagaimana peran modul Named Entity Recognition (NER) dalam arsitektur modern seperti *Knowledge Graph* dan *Retrieval-Augmented Generation (RAG)*?

---

## G. Output Pertemuan 14

File yang dikumpulkan:

```text
ner_information_extraction.ipynb
```

> Output: Notebook interaktif yang mendemonstrasikan alignment label subword, fine-tuning model Token Classification, metrik seqeval, visualisasi entitas teks dengan HTML highlighter, dan fungsi ekstraksi data terstruktur JSON.

---

> **Pertemuan sebelumnya:** [Pertemuan 13 — Fine-Tuning dan NLP Tasks](praktikum13.md)
>
> **Pertemuan berikutnya:** [Pertemuan 15 — Advanced NLP Tasks](praktikum15.md)
