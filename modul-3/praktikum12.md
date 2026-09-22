# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 3 — PERTEMUAN 12
## Pretrained Language Model

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 3 — Transformer, Pretrained Model, dan Aplikasi NLP |
| Pertemuan | 12 |
| Topik | Pretrained Language Model |
| Output | `pretrained_lm.ipynb` |

---

## B. Tujuan

Mahasiswa mampu:

* memahami konsep pretraining dan transfer learning;
* memahami perbedaan BERT, RoBERTa, ALBERT, dan GPT;
* memahami Masked Language Modeling (MLM) dan Causal Language Modeling (CLM);
* menggunakan Hugging Face Transformers untuk load pretrained model;
* melakukan tokenisasi subword dengan tokenizer pretrained;
* menjalankan inference dengan pretrained model;
* menganalisis embedding yang dihasilkan pretrained model.

---

## C. Materi Teori

### 1. Pretraining

Pretrained language model adalah model yang telah dilatih pada corpus teks sangat besar sebelum digunakan pada task spesifik.

```text
Pretraining:
  Corpus besar (Wikipedia, web, buku) → Latih model dari nol
  → Model memahami bahasa secara umum

Fine-tuning:
  Dataset task spesifik (sentiment, NER, dll) → Sesuaikan model
  → Model beradaptasi ke task tertentu
```

### 2. Transfer Learning

Transfer learning memungkinkan pengetahuan yang dipelajari dari satu tugas digunakan untuk tugas lain:

```text
Pretraining (task umum)
   ↓
Pretrained Model (pengetahuan bahasa)
   ↓
Fine-tuning (task spesifik)
   ↓
Model siap pakai
```

Keuntungan: tidak perlu data berlabel dalam jumlah besar untuk fine-tuning.

### 3. BERT (Bidirectional Encoder Representations from Transformers)

BERT (Devlin et al., 2018) menggunakan **encoder Transformer** yang dilatih dengan dua objective:

**Masked Language Modeling (MLM):**
```text
Input : "Saya [MASK] NLP dengan giat."
Target: "belajar"
```

BERT melihat konteks dari **dua arah** (kiri dan kanan) untuk memprediksi token yang tersembunyi.

**Next Sentence Prediction (NSP):**
```text
Kalimat A: "Saya belajar NLP."
Kalimat B: "Mata kuliah ini sangat menarik."
Label: IsNext
```

### 4. Varian Model

| Model | Kelebihan | Kekurangan |
|-------|-----------|------------|
| BERT | Bidirectional, cocok untuk pemahaman | Tidak bisa generate text |
| RoBERTa | Lebih baik dari BERT (lebih banyak data, tanpa NSP) | Lebih besar |
| ALBERT | Lebih ringan dari BERT (parameter sharing) | Lebih lambat (relatif) |
| DistilBERT | 40% lebih kecil, 60% lebih cepat dari BERT | Akurasi sedikit lebih rendah |
| GPT | Bagus untuk generasi teks | Hanya melihat ke kiri (causal) |

### 5. Model Bahasa Indonesia

| Model | Base | Fitur |
|-------|------|-------|
| IndoBERT | BERT | Dilatih pada corpus Bahasa Indonesia |
| IndoBERT-large | BERT | Versi besar IndoBERT |
| IndoRoBERTa | RoBERTa | Berbasis RoBERTa |
| mBERT | BERT | Multilingual, mendukung 104 bahasa |

### 6. Subword Tokenization

Tokenizer Transformer menggunakan subword, sehingga dapat menangani kata yang tidak ada dalam vocabulary (OOV):

```text
BERT Multilingual:
"mempelajari" → ["me", "##m", "##pe", "##laja", "##ri"]

IndoBERT:
"mempelajari" → ["mem", "##pelajar", "##i"]
```

---

## D. Praktikum

### 1. Membuat Notebook

Buat file:

```text
pertemuan-12/pretrained_lm.ipynb
```

### 2. Import Library

```python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    pipeline
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
```

### 3. Load IndoBERT

```python
MODEL_NAME = "indobenchmark/indobert-base-p1"
# Alternatif: "bert-base-multilingual-cased"

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
model.to(device)

print("Model berhasil dimuat.")
print(f"Hidden size  : {model.config.hidden_size}")
print(f"Num layers   : {model.config.num_hidden_layers}")
print(f"Num heads    : {model.config.num_attention_heads}")
print(f"Vocab size   : {tokenizer.vocab_size:,}")
print(f"Total params : {sum(p.numel() for p in model.parameters()):,}")
```

### 4. Tokenisasi Detail

```python
texts = [
    "Saya belajar Natural Language Processing.",
    "mempelajarinya sangat menyenangkan",
    "Ketidakberhasilan program itu mengecewakan.",
    "IndoBERT model bahasa Indonesia terbaik!",
]

print("Analisis Tokenisasi:")
print("=" * 70)
for text in texts:
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(f"Teks   : {text}")
    print(f"Tokens : {tokens}")
    print(f"IDs    : {ids}")
    print(f"Jumlah token: {len(tokens)}")
    print()
```

### 5. Mendapatkan Embedding Kontekstual

```python
def get_contextual_embeddings(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output  # representasi [CLS]

    return last_hidden_state, pooler_output, inputs

# Contoh
text = "Saya belajar NLP dengan giat."
last_hs, pooler, inputs = get_contextual_embeddings(text, tokenizer, model, device)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

print(f"Teks   : {text}")
print(f"Tokens : {tokens}")
print(f"Shape last_hidden_state: {last_hs.shape}")
print(f"Shape pooler_output    : {pooler.shape}")
```

### 6. Embedding per Token

```python
token_embeddings = last_hs[0].cpu().numpy()  # (seq_len, hidden_size)

print("Embedding per token (10 dim pertama):")
print("-" * 50)
for i, token in enumerate(tokens):
    print(f"  {token:<15} : {token_embeddings[i, :10].round(4)}")
```

### 7. Masked Language Modeling

```python
# Load model untuk MLM
mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
mlm_model.eval()
mlm_model.to(device)

# Pipeline MLM
fill_mask = pipeline("fill-mask", model=mlm_model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Contoh
masked_texts = [
    f"Saya {tokenizer.mask_token} NLP.",
    f"Model bahasa {tokenizer.mask_token} sangat berguna.",
    f"Transformers adalah {tokenizer.mask_token} penting dalam NLP.",
]

print("Masked Language Modeling (Fill-the-Blank):")
print("=" * 60)
for text in masked_texts:
    print(f"\nInput: {text}")
    try:
        results = fill_mask(text, top_k=5)
        for res in results:
            print(f"  [{res['score']:.4f}] {res['token_str']:<20} → {res['sequence']}")
    except Exception as e:
        print(f"  Error: {e}")
```

### 8. Semantic Similarity dengan BERT Embedding

```python
def get_sentence_embedding(text, tokenizer, model, device):
    """Mean pooling dari last hidden state."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling (selain [PAD] token)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    embedding = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
    return embedding[0].cpu().numpy()

from sklearn.metrics.pairwise import cosine_similarity

sentence_pairs = [
    ("Hotel ini sangat bagus.", "Penginapan tersebut sangat baik."),
    ("Hotel ini sangat bagus.", "Makanan di sini enak sekali."),
    ("Saya belajar NLP.", "Saya mempelajari pemrosesan bahasa alami."),
    ("Saya suka kucing.", "Harga saham naik tajam."),
]

print("Semantic Similarity dengan BERT:")
print("=" * 70)
for s1, s2 in sentence_pairs:
    e1 = get_sentence_embedding(s1, tokenizer, model, device)
    e2 = get_sentence_embedding(s2, tokenizer, model, device)
    sim = cosine_similarity([e1], [e2])[0][0]
    print(f"\nK1: {s1}")
    print(f"K2: {s2}")
    print(f"Similarity: {sim:.4f}")
```

### 9. Analisis Embedding Layer per Layer

```python
text = "Bank itu letaknya di tepi kali."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

hidden_states = outputs.hidden_states  # tuple: (embedding + 12 layers)
print(f"Jumlah layer (termasuk embedding): {len(hidden_states)}")
print(f"Shape setiap layer: {hidden_states[0].shape}")

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(f"\nToken: {tokens}")

# Bandingkan embedding token tertentu di layer berbeda
target_token = "bank"
if target_token in tokens:
    token_idx = tokens.index(target_token)
    print(f"\nEmbedding token '{target_token}' di berbagai layer (10 dim pertama):")
    for layer_idx in [0, 3, 6, 9, 12]:
        emb = hidden_states[layer_idx][0, token_idx, :10].cpu().numpy()
        print(f"  Layer {layer_idx:>2}: {emb.round(3)}")
```

### 10. Visualisasi Kemiripan Antar Kalimat

```python
sentences = df[text_column].sample(20, random_state=42).astype(str).tolist()

# Encode semua kalimat
embeddings = np.array([
    get_sentence_embedding(s, tokenizer, model, device)
    for s in sentences
])

sim_matrix = cosine_similarity(embeddings)

plt.figure(figsize=(12, 10))
sns.heatmap(
    sim_matrix,
    cmap="RdYlGn",
    vmin=0, vmax=1,
    square=True,
    xticklabels=[f"K{i+1}" for i in range(len(sentences))],
    yticklabels=[f"K{i+1}" for i in range(len(sentences))]
)
plt.title("Semantic Similarity Matrix — IndoBERT Embedding")
plt.tight_layout()
plt.show()

# Tampilkan kalimat
for i, sent in enumerate(sentences):
    print(f"K{i+1}: {sent[:80]}...")
```

---

## E. Tugas Analisis

1. Bandingkan MLM output IndoBERT vs BERT Multilingual untuk kalimat Bahasa Indonesia. Mana yang lebih akurat?
2. Uji semantic similarity pada 5 pasang kalimat. Apakah hasilnya sesuai intuisi?
3. Analisis embedding token yang sama dalam dua konteks berbeda (polisemi). Apakah vektornya berbeda?
4. Hitung berapa jumlah token subword yang dihasilkan untuk 10 kata berimbuhan Bahasa Indonesia.

---

## F. Pertanyaan Diskusi

1. Apa keuntungan menggunakan IndoBERT dibandingkan BERT Multilingual untuk teks Bahasa Indonesia?
2. Mengapa MLM (Masked Language Modeling) efektif untuk pretraining?
3. Apa perbedaan antara pooler output ([CLS]) dan mean pooling pada sentence embedding?
4. Mengapa kita menggunakan model yang sudah di-pretrain daripada melatih dari nol?
5. Apa trade-off antara model besar (lebih akurat) dan model kecil (lebih cepat)?

---

## G. Output Pertemuan 12

File yang dikumpulkan:

```text
pretrained_lm.ipynb
```

> Output: Notebook eksplorasi pretrained language model mencakup tokenisasi, embedding kontekstual, MLM, dan semantic similarity.

---

> **Pertemuan sebelumnya:** [Pertemuan 11 — Transformer](praktikum11.md)
>
> **Pertemuan berikutnya:** [Pertemuan 13 — Fine-Tuning dan NLP Tasks](praktikum13.md)
