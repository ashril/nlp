# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 3 — PERTEMUAN 11
## Transformer

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 3 — Transformer, Pretrained Model, dan Aplikasi NLP |
| Pertemuan | 11 |
| Topik | Transformer Architecture |
| Output | `transformer.ipynb` |

---

## B. Tujuan

Mahasiswa mampu:

* memahami keterbatasan RNN yang mendorong lahirnya Transformer;
* memahami komponen utama arsitektur Transformer;
* memahami multi-head self-attention;
* memahami positional encoding;
* memahami encoder dan decoder Transformer;
* menggunakan library Hugging Face untuk eksplorasi tokenizer Transformer;
* memvisualisasikan attention pada model Transformer pretrained.

---

## C. Materi Teori

### 1. Keterbatasan RNN dan Motivasi Transformer

| Keterbatasan RNN | Solusi pada Transformer |
|-----------------|------------------------|
| Komputasi sekuensial (tidak bisa diparalelkan) | Semua token diproses sekaligus (paralel) |
| Vanishing gradient pada sequence panjang | Attention langsung antar semua token |
| Dependensi jarak jauh terbatas | Self-attention menghubungkan semua pasangan token |

### 2. Arsitektur Transformer

Transformer (Vaswani et al., *"Attention Is All You Need"*, 2017) terdiri dari:

```text
INPUT
  ↓
Token Embedding + Positional Encoding
  ↓
┌─────────────────────────────┐
│   ENCODER (N layers)        │
│  ┌────────────────────────┐ │
│  │ Multi-Head Self-Attention│ │
│  │ Add & Norm              │ │
│  │ Feed-Forward Network    │ │
│  │ Add & Norm              │ │
│  └────────────────────────┘ │
└─────────────────────────────┘
  ↓
┌─────────────────────────────┐
│   DECODER (N layers)        │
│  ┌────────────────────────┐ │
│  │ Masked Self-Attention   │ │
│  │ Add & Norm              │ │
│  │ Cross-Attention         │ │
│  │ Add & Norm              │ │
│  │ Feed-Forward Network    │ │
│  │ Add & Norm              │ │
│  └────────────────────────┘ │
└─────────────────────────────┘
  ↓
Linear + Softmax
  ↓
OUTPUT
```

### 3. Positional Encoding

Karena Transformer tidak memproses token secara sekuensial, informasi posisi harus ditambahkan secara eksplisit menggunakan positional encoding:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### 4. Multi-Head Attention

Multi-head attention menjalankan beberapa attention secara paralel dengan proyeksi berbeda, memungkinkan model menangkap berbagai jenis relasi sekaligus:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

di mana setiap $\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$

### 5. Encoder vs Decoder

| | Encoder | Decoder |
|--|---------|---------|
| Self-Attention | Full (semua token bisa melihat semua) | Masked (hanya bisa melihat token sebelumnya) |
| Cross-Attention | Tidak ada | Ya (melihat output encoder) |
| Digunakan di | BERT (encoder-only) | GPT (decoder-only) |
| Keduanya | T5, BART | T5, BART |

### 6. Variant Transformer

| Model | Arsitektur | Kegunaan |
|-------|-----------|---------|
| BERT | Encoder only | Pemahaman teks (classification, NER, QA) |
| GPT | Decoder only | Generasi teks |
| T5 | Encoder-Decoder | Translation, summarization, QA |
| BART | Encoder-Decoder | Summarization, translation |

---

## D. Praktikum

### 1. Membuat Notebook

Buat file:

```text
pertemuan-11/transformer.ipynb
```

### 2. Install Library

```bash
pip install transformers torch
```

### 3. Import Library

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel

print("Transformers:", __import__("transformers").__version__)
print("PyTorch     :", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device      :", device)
```

### 4. Positional Encoding — Implementasi dan Visualisasi

```python
def get_positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))
    positions = np.arange(max_len).reshape(-1, 1)
    div_term = np.power(10000, np.arange(0, d_model, 2) / d_model)

    PE[:, 0::2] = np.sin(positions / div_term)
    PE[:, 1::2] = np.cos(positions / div_term)
    return PE

# Visualisasi
PE = get_positional_encoding(max_len=50, d_model=128)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.imshow(PE.T, cmap="RdBu", aspect="auto")
plt.colorbar()
plt.title("Positional Encoding Matrix")
plt.xlabel("Position")
plt.ylabel("Dimension")

plt.subplot(1, 2, 2)
for i in [0, 5, 10, 20]:
    plt.plot(PE[:, i], label=f"dim {i}")
plt.title("Positional Encoding — Beberapa Dimensi")
plt.xlabel("Position")
plt.legend()
plt.tight_layout()
plt.show()
```

### 5. Eksplorasi Tokenizer Transformer

Hugging Face menyediakan tokenizer subword yang digunakan oleh model Transformer.

```python
# Load tokenizer BERT multilingual
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
print("Tokenizer berhasil dimuat.")
print("Ukuran vocabulary:", tokenizer.vocab_size)
```

Tokenisasi teks:

```python
texts = [
    "Saya belajar Natural Language Processing.",
    "IndoBERT adalah model bahasa Indonesia.",
    "Mempelajari NLP sangat menarik!",
    "Ketidakberhasilan program itu mengecewakan.",
]

print("Eksplorasi Tokenizer Transformer:")
print("=" * 60)
for text in texts:
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)

    print(f"\nTeks: {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs   : {token_ids}")
    print(f"Jumlah token: {len(tokens)}")
```

### 6. Perbandingan Tokenizer

```python
from transformers import AutoTokenizer

tokenizers = {
    "BERT Multilingual": "bert-base-multilingual-cased",
    "IndoBERT"         : "indobenchmark/indobert-base-p1",
}

test_text = "Pembelajaran mesin dan NLP berkembang sangat pesat."

print("Perbandingan Tokenizer:")
print("=" * 60)
print(f"Teks: {test_text}")
print()

for name, model_name in tokenizers.items():
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        tokens = tok.tokenize(test_text)
        print(f"{name}:")
        print(f"  Tokens: {tokens}")
        print(f"  Jumlah: {len(tokens)}")
        print()
    except Exception as e:
        print(f"{name}: Error — {e}")
        print()
```

### 7. Load Model Transformer Pretrained

```python
# Load IndoBERT atau BERT multilingual
model_name = "indobenchmark/indobert-base-p1"
# Alternatif: "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {model_name}")
print(f"Total parameter: {total_params:,}")
print(f"Konfigurasi:\n{model.config}")
```

### 8. Mendapatkan Output Transformer

```python
text = "Saya belajar Natural Language Processing dengan giat."

# Tokenisasi
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

print("Struktur input:")
for key, val in inputs.items():
    print(f"  {key}: {val.shape}")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

last_hidden_state = outputs.last_hidden_state
print(f"\nShape last_hidden_state: {last_hidden_state.shape}")
print(f"  [batch_size, seq_len, hidden_size]")
print(f"  = [{last_hidden_state.shape[0]}, {last_hidden_state.shape[1]}, {last_hidden_state.shape[2]}]")
```

### 9. Analisis [CLS] Token

```python
# [CLS] token (index 0) merepresentasikan seluruh kalimat
cls_embedding = last_hidden_state[:, 0, :]
print(f"[CLS] embedding shape: {cls_embedding.shape}")
print(f"[CLS] embedding (10 dim pertama): {cls_embedding[0, :10].cpu().numpy().round(4)}")
```

### 10. Visualisasi Attention

```python
# Ambil attention dari layer terakhir, head pertama
attention_weights = outputs.attentions  # tuple of (batch, heads, seq, seq)

layer_idx = -1  # layer terakhir
head_idx = 0    # head pertama
attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()

# Dapatkan token labels
token_labels = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

plt.figure(figsize=(12, 10))
sns.heatmap(
    attn,
    xticklabels=token_labels,
    yticklabels=token_labels,
    cmap="Blues",
    square=True
)
plt.title(f"Attention Weights — Layer {len(attention_weights)} Head {head_idx + 1}")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.show()
```

### 11. Attention dari Beberapa Head

```python
num_heads = min(4, attention_weights[layer_idx].shape[1])
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for i in range(num_heads):
    attn_head = attention_weights[layer_idx][0, i].cpu().numpy()
    sns.heatmap(
        attn_head,
        xticklabels=token_labels,
        yticklabels=token_labels,
        cmap="Blues",
        square=True,
        ax=axes[i]
    )
    axes[i].set_title(f"Head {i+1}")
    axes[i].tick_params(axis="x", rotation=45, labelsize=8)
    axes[i].tick_params(axis="y", rotation=0, labelsize=8)

plt.suptitle(f"Multi-Head Attention — Layer Terakhir", fontsize=14)
plt.tight_layout()
plt.show()
```

---

## E. Tugas Analisis

1. Bandingkan tokenisasi `"ketidakberhasilan"` antara BERT Multilingual dan IndoBERT. Berapa subword yang dihasilkan? Mengapa berbeda?
2. Visualisasikan attention dari beberapa layer. Apakah attention berubah dari layer awal ke layer akhir?
3. Bandingkan panjang token (dari tokenizer Transformer) dengan panjang token dari NLTK. Apa perbedaannya?
4. Temukan token dalam kalimat yang mendapat attention tinggi dari token `[CLS]`. Apa artinya?

---

## F. Pertanyaan Diskusi

1. Mengapa Transformer membutuhkan positional encoding, sedangkan RNN tidak?
2. Apa yang dilakukan oleh setiap "head" dalam multi-head attention? Mengapa menggunakan beberapa head?
3. Apa perbedaan self-attention pada encoder dan masked self-attention pada decoder?
4. Mengapa BERT hanya menggunakan bagian encoder, sedangkan GPT hanya bagian decoder?
5. Berapa jumlah parameter BERT-base? Mengapa butuh banyak data untuk melatihnya dari nol?

---

## G. Output Pertemuan 11

File yang dikumpulkan:

```text
transformer.ipynb
```

> Output: Visualisasi positional encoding, eksplorasi tokenizer Transformer, analisis output model, dan visualisasi multi-head attention.

---

> **Pertemuan sebelumnya:** [Modul 2 — Pertemuan 10](../modul-2/praktikum10.md)
>
> **Pertemuan berikutnya:** [Pertemuan 12 — Pretrained Language Model](praktikum12.md)
