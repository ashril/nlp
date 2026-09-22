# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 2 — PERTEMUAN 9
## RNN, LSTM, dan GRU

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 2 — Representasi Teks dan Neural NLP |
| Pertemuan | 9 |
| Topik | RNN, LSTM, dan GRU |
| Output | `rnn_lstm_gru.ipynb` |

---

## B. Tujuan

Mahasiswa mampu:

* memahami mengapa sequential data membutuhkan model khusus;
* memahami arsitektur Recurrent Neural Network (RNN);
* memahami masalah vanishing gradient pada RNN;
* memahami arsitektur LSTM dan mekanisme gate-nya;
* memahami arsitektur GRU sebagai varian LSTM yang lebih efisien;
* mengimplementasikan RNN, LSTM, dan GRU dengan PyTorch;
* membandingkan kemampuan ketiga model dalam menangkap konteks.

---

## C. Materi Teori

### 1. Sequential Data dalam NLP

Teks adalah data sequential — urutan kata sangat menentukan makna:

```text
"Saya tidak suka makanan ini."  → sentimen negatif
"Saya suka makanan ini."        → sentimen positif
```

Model seperti BoW dan TF-IDF kehilangan informasi urutan ini.

### 2. Recurrent Neural Network (RNN)

RNN dirancang untuk memproses data sequential dengan menyimpan **hidden state** yang merupakan "memori" dari input sebelumnya.

```text
Input:  x₁   x₂   x₃   x₄
         ↓    ↓    ↓    ↓
        [RNN]→[RNN]→[RNN]→[RNN]→ output
         h₁   h₂   h₃   h₄
```

Setiap cell RNN menerima input saat ini (xₜ) dan hidden state sebelumnya (hₜ₋₁):

$$h_t = \tanh(W_h \cdot h_{t-1} + W_x \cdot x_t + b)$$

### 3. Masalah Vanishing Gradient

Pada sequence panjang, gradien yang di-backpropagate melalui banyak timestep akan menjadi sangat kecil (vanishing) atau sangat besar (exploding).

```text
Kalimat: "Ikan ... (50 kata) ... berenang di kolam."

RNN sulit mengingat bahwa subjek awal ("Ikan") penting
untuk memahami kata kerja di akhir ("berenang").
```

### 4. Long Short-Term Memory (LSTM)

LSTM (Hochreiter & Schmidhuber, 1997) mengatasi vanishing gradient dengan mekanisme **gate**:

| Gate | Fungsi |
|------|--------|
| **Forget Gate** | Memutuskan informasi mana dari cell state yang dilupakan |
| **Input Gate** | Memutuskan informasi baru mana yang ditambahkan ke cell state |
| **Output Gate** | Memutuskan bagian cell state yang digunakan sebagai output |

LSTM memiliki dua state:
* **Cell state (c)** — memori jangka panjang
* **Hidden state (h)** — output pada setiap timestep

### 5. Gated Recurrent Unit (GRU)

GRU (Cho et al., 2014) adalah penyederhanaan LSTM dengan hanya dua gate:

| Gate | Fungsi |
|------|--------|
| **Reset Gate** | Menentukan berapa banyak info masa lalu yang diabaikan |
| **Update Gate** | Menentukan berapa banyak info masa lalu yang dipertahankan |

GRU memiliki parameter lebih sedikit dari LSTM dan sering kali memiliki performa sebanding.

### 6. Perbandingan RNN, LSTM, GRU

| Aspek | RNN | LSTM | GRU |
|-------|-----|------|-----|
| State | Hidden state (h) | Cell state (c) + Hidden state (h) | Hidden state (h) |
| Gate | Tidak ada | 3 gate | 2 gate |
| Mengatasi vanishing gradient | Tidak | Ya | Ya |
| Parameter | Paling sedikit | Paling banyak | Menengah |
| Kecepatan training | Cepat | Lambat | Sedang |
| Kemampuan memori jangka panjang | Terbatas | Baik | Baik |

### 7. Bidirectional RNN

Bidirectional RNN memproses sequence dari dua arah:

```text
→ Forward RNN  (kiri ke kanan)
← Backward RNN (kanan ke kiri)
```

Cocok untuk task yang memerlukan konteks dari kedua arah, seperti POS tagging dan NER.

---

## D. Praktikum

### 1. Membuat Notebook

Buat file:

```text
pertemuan-09/rnn_lstm_gru.ipynb
```

### 2. Import Library

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

print("PyTorch version:", torch.__version__)
print("GPU tersedia    :", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device yang digunakan:", device)
```

### 3. Load dan Persiapkan Dataset

```python
df = pd.read_csv("../dataset/dataset_preprocessed.csv")

# Sesuaikan nama kolom dengan dataset masing-masing
text_column = "processed_text"
label_column = "label"  # sesuaikan

df = df.dropna(subset=[text_column, label_column])
df = df.reset_index(drop=True)

print(f"Jumlah data: {len(df)}")
print(f"Distribusi label:\n{df[label_column].value_counts()}")
```

### 4. Membangun Vocabulary

```python
def build_vocab(texts, max_vocab=10000, min_freq=2):
    counter = Counter()
    for text in texts:
        tokens = str(text).split()
        counter.update(tokens)

    # Tambahkan token khusus
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in counter.most_common(max_vocab):
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(df[text_column])
print(f"Ukuran vocabulary: {len(vocab)}")
```

### 5. Encode Teks dan Label

```python
def text_to_indices(text, vocab, max_len=100):
    tokens = str(text).split()[:max_len]
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    # Padding
    if len(indices) < max_len:
        indices += [vocab["<PAD>"]] * (max_len - len(indices))
    return indices

# Encode label
le = LabelEncoder()
labels = le.fit_transform(df[label_column])
num_classes = len(le.classes_)
print(f"Kelas label: {le.classes_}")
print(f"Jumlah kelas: {num_classes}")

MAX_LEN = 100
X = [text_to_indices(text, vocab, MAX_LEN) for text in df[text_column]]
y = labels.tolist()
```

### 6. Dataset dan DataLoader

```python
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.LongTensor(texts)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")
```

### 7. Arsitektur RNN

```python
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=0):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded)
        # Gunakan hidden state terakhir
        out = self.fc(self.dropout(hidden[-1]))
        return out
```

### 8. Arsitektur LSTM

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=0):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            batch_first=True,
            num_layers=2,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded)
        out = self.fc(self.dropout(hidden[-1]))
        return out
```

### 9. Arsitektur GRU

```python
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=0):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            batch_first=True,
            num_layers=2,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded)
        out = self.fc(self.dropout(hidden[-1]))
        return out
```

### 10. Fungsi Training dan Evaluasi

```python
def train_model(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0, 0
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = total_correct / len(loader.dataset)
    return avg_loss, accuracy

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = total_correct / len(loader.dataset)
    return avg_loss, accuracy
```

### 11. Training dan Perbandingan

```python
VOCAB_SIZE = len(vocab)
EMBED_DIM  = 128
HIDDEN_DIM = 256
NUM_EPOCHS = 5

models = {
    "RNN" : RNNClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, num_classes).to(device),
    "LSTM": LSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, num_classes).to(device),
    "GRU" : GRUClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, num_classes).to(device),
}

criterion = nn.CrossEntropyLoss()
results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    results[model_name] = history
    print(f"  {model_name} Test Accuracy: {val_acc:.4f}")
```

### 12. Visualisasi Perbandingan

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for model_name, history in results.items():
    axes[0].plot(history["train_acc"], label=f"{model_name} Train")
    axes[0].plot(history["val_acc"], "--", label=f"{model_name} Val")

axes[0].set_title("Accuracy per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

for model_name, history in results.items():
    axes[1].plot(history["train_loss"], label=f"{model_name} Train")
    axes[1].plot(history["val_loss"], "--", label=f"{model_name} Val")

axes[1].set_title("Loss per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.show()
```

### 13. Tabel Hasil Perbandingan

```python
print("\nRingkasan Hasil:")
print("=" * 60)
print(f"{'Model':<10} | {'Val Accuracy':<15} | {'Jumlah Parameter'}")
print("-" * 60)
for model_name, model in models.items():
    num_params = sum(p.numel() for p in model.parameters())
    final_acc = results[model_name]["val_acc"][-1]
    print(f"{model_name:<10} | {final_acc:<15.4f} | {num_params:,}")
```

---

## E. Tugas Analisis

1. Jalankan training dengan jumlah epoch yang lebih besar (10–15). Apakah model mengalami overfitting?
2. Coba ubah `hidden_dim` menjadi 64 dan 512. Bagaimana pengaruhnya terhadap akurasi dan waktu training?
3. Tambahkan `bidirectional=True` pada LSTM. Apakah akurasi meningkat?
4. Buat tabel perbandingan RNN, LSTM, GRU berdasarkan akurasi, loss, dan jumlah parameter.
5. Pada kalimat panjang (>50 kata), model mana yang lebih baik menangkap konteks?

---

## F. Pertanyaan Diskusi

1. Mengapa RNN sederhana sulit menangani dependensi jangka panjang?
2. Bagaimana forget gate pada LSTM membantu mengatasi vanishing gradient?
3. Mengapa GRU lebih cepat dilatih dibandingkan LSTM?
4. Kapan sebaiknya menggunakan Bidirectional RNN?
5. Apa keterbatasan model RNN/LSTM dibandingkan Transformer?
6. Mengapa model sekuensial (RNN) sulit diparalelkan?

---

## G. Output Pertemuan 9

File yang dikumpulkan:

```text
rnn_lstm_gru.ipynb
```

> Output: Model sequence processing sederhana berbasis RNN, LSTM, dan GRU dengan perbandingan performa pada task klasifikasi teks.

---

> **Pertemuan sebelumnya:** [Pertemuan 8 — Contextual Representation](praktikum8.md)
>
> **Pertemuan berikutnya:** [Pertemuan 10 — Seq2Seq & Attention](praktikum10.md)
