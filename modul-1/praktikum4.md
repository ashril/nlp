# PRAKTIKUM NATURAL LANGUAGE PROCESSING
# MODUL 1 — PERTEMUAN 4
## Linguistic Processing

---

## A. Identitas Praktikum

| Komponen | Keterangan |
|---|---|
| Mata Kuliah | Natural Language Processing |
| Modul | Modul 1 — Fundamental dan Text Processing |
| Pertemuan | 4 |
| Topik | Linguistic Processing |
| Output | `linguistic_processing.ipynb` |

---

## B. Tujuan

Mahasiswa mampu:

* memahami lexical analysis;
* memahami morphology Bahasa Indonesia;
* memahami konsep Part-of-Speech (POS);
* melakukan POS tagging;
* memahami syntax dan dependency parsing;
* membaca struktur kalimat;
* memvisualisasikan dependency tree;
* menganalisis kesalahan linguistic processing.

---

## C. Materi Teori

### 1. Lexical Analysis

Lexical analysis berkaitan dengan unit-unit kata dalam bahasa. Dalam NLP, setiap kata dalam kalimat dianalisis secara individual.

Contoh:

```text
Kalimat : "Mahasiswa belajar NLP."
Token   : ["Mahasiswa", "belajar", "NLP", "."]
```

### 2. Morphology

Morphology mempelajari struktur pembentukan kata. Bahasa Indonesia kaya akan imbuhan (afiks).

Contoh pembentukan kata:

```text
ajar
 ↓ + be- + -ar
belajar
 ↓ + mem-pe- + -i
mempelajari
 ↓ + pe- + -an
pembelajaran
```

Contoh lain:

```text
main → bermain → permainan → memainkan → dimainkan
hadir → kehadiran → ketidakhadiran
```

Stemmer (seperti Sastrawi) mencoba menemukan bentuk dasar/stem dari kata berimbuhan.

### 3. Part-of-Speech (POS)

POS menunjukkan kelas kata (word class) dalam kalimat.

Contoh:

```text
Saya      → PRON  (Pronoun / Kata ganti)
membaca   → VERB  (Verb / Kata kerja)
buku      → NOUN  (Noun / Kata benda)
menarik   → ADJ   (Adjective / Kata sifat)
sangat    → ADV   (Adverb / Kata keterangan)
```

Tag POS yang umum digunakan (Universal POS Tags):

| Tag | Kelas Kata | Contoh |
|-----|-----------|--------|
| NOUN | Kata benda | buku, universitas |
| VERB | Kata kerja | belajar, membaca |
| ADJ | Kata sifat | bagus, menarik |
| ADV | Kata keterangan | sangat, sudah |
| PRON | Kata ganti | saya, dia, mereka |
| DET | Kata penentu | ini, itu |
| ADP | Preposisi | di, ke, dari, untuk |
| NUM | Angka | satu, pertama |
| CONJ | Konjungsi | dan, tetapi, karena |
| PUNCT | Tanda baca | . , ! ? |

### 4. POS Tagging

POS Tagging adalah proses memberikan label kelas kata kepada setiap token dalam kalimat.

```text
Saya/PRON membaca/VERB buku/NOUN yang/DET menarik/ADJ .
```

### 5. Dependency Parsing

Dependency parsing mencoba menemukan hubungan gramatikal antar token.

Contoh:

```text
Mahasiswa → membaca → buku
```

Hubungan:
* `Mahasiswa` adalah **subjek** dari `membaca`
* `buku` adalah **objek** dari `membaca`

Representasi dependency tree:

```text
         membaca (ROOT)
         /       \
   Mahasiswa     buku
   (nsubj)       (obj)
```

---

## D. Praktikum

### 1. Membuat Notebook

Buat file:

```text
pertemuan-04/linguistic_processing.ipynb
```

### 2. Install dan Load spaCy

```python
import spacy
```

Jika menggunakan model multilingual (xx):

```bash
python -m spacy download xx_ent_wiki_sm
```

Load model:

```python
nlp = spacy.load("xx_ent_wiki_sm")
```

Atau jika tersedia model Bahasa Indonesia khusus:

```bash
pip install https://huggingface.co/indonesian-nlp/spacy-id-ud-gsd/resolve/main/id_ud_gsd-any-py3-none-any.whl
```

```python
nlp = spacy.load("id_ud_gsd")
```

### 3. Memproses Kalimat

```python
doc = nlp("Mahasiswa belajar Natural Language Processing.")

for token in doc:
    print(f"{token.text:<15} {token.pos_:<10} {token.dep_}")
```

### 4. Analisis POS Token

```python
doc = nlp("Mahasiswa belajar Natural Language Processing.")

print(f"{'Token':<15} {'POS':<10} {'Lemma'}")
print("-" * 40)
for token in doc:
    print(f"{token.text:<15} {token.pos_:<10} {token.lemma_}")
```

### 5. Membuat Tabel Analisis POS

```python
import pandas as pd

def analyze_pos(text, nlp_model):
    doc = nlp_model(text)
    results = []
    for token in doc:
        results.append({
            "Token": token.text,
            "POS": token.pos_,
            "POS Tag": token.tag_,
            "Lemma": token.lemma_,
            "Dependency": token.dep_,
            "Head": token.head.text
        })
    return pd.DataFrame(results)

df_pos = analyze_pos("Mahasiswa belajar NLP.", nlp)
print(df_pos.to_string(index=False))
```

### 6. Analisis Beberapa Kalimat

```python
sentences = [
    "Mahasiswa belajar NLP.",
    "Dosen mengajar Natural Language Processing.",
    "Mahasiswa membaca buku.",
    "Sistem memproses data teks.",
    "Model bahasa memprediksi kata berikutnya."
]

for sentence in sentences:
    print(f"\nKalimat: {sentence}")
    df_result = analyze_pos(sentence, nlp)
    print(df_result[["Token", "POS", "Dependency", "Head"]].to_string(index=False))
    print()
```

### 7. Dependency Parsing

```python
doc = nlp("Mahasiswa belajar NLP.")

print(f"{'Token':<15} {'Dep':<12} {'Head'}")
print("-" * 40)
for token in doc:
    print(f"{token.text:<15} {token.dep_:<12} {token.head.text}")
```

### 8. Visualisasi Dependency

spaCy menyediakan visualisasi dependency tree pada Jupyter Notebook:

```python
from spacy import displacy

doc = nlp("Mahasiswa belajar Natural Language Processing.")

displacy.render(
    doc,
    style="dep",
    jupyter=True,
    options={"distance": 90}
)
```

Untuk simpan sebagai HTML:

```python
html = displacy.render(doc, style="dep")
with open("dependency_tree.html", "w", encoding="utf-8") as f:
    f.write(html)
```

### 9. Analisis Dataset

Menerapkan POS tagging ke dataset:

```python
import pandas as pd

df = pd.read_csv("../dataset/dataset_preprocessed.csv")
text_column = "text"  # sesuaikan

# Ambil sample untuk analisis
sample = df.sample(10, random_state=42)

for _, row in sample.iterrows():
    text = str(row[text_column])[:200]  # batasi panjang
    doc = nlp(text)
    pos_counts = {}
    for token in doc:
        pos = token.pos_
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    print(f"Teks: {text[:80]}...")
    print(f"POS Distribution: {pos_counts}")
    print()
```

### 10. Analisis Kesalahan

Linguistic processing tidak selalu sempurna. Uji dengan teks informal:

```python
informal_texts = [
    "aku lg belajar NLP nih",
    "gue udah baca buku itu",
    "modelnya bagus bgt sih",
    "nggak ngerti sama sekali",
    "udah instal belum?"
]

print("Analisis Teks Informal:")
print("=" * 60)
for text in informal_texts:
    doc = nlp(text)
    print(f"\nTeks: {text}")
    for token in doc:
        print(f"  {token.text:<15} {token.pos_}")
```

Kesalahan dapat disebabkan oleh:
* typo dan singkatan
* slang dan bahasa informal
* campuran bahasa (code-mixing)
* nama orang atau organisasi
* domain khusus yang tidak ada dalam data latih model

---

## E. Tugas Analisis Linguistik

Pilih minimal **10 kalimat** dari dataset.

Untuk setiap kalimat:

1. Lakukan tokenization.
2. Lakukan POS tagging.
3. Lakukan dependency parsing.
4. Visualisasikan dependency tree.
5. Identifikasi kesalahan yang ditemukan.

Buat tabel hasil analisis:

| Kalimat | Token | POS | Dependency | Kesalahan? |
|---------|-------|-----|-----------|------------|
| ... | ... | ... | ... | ... |

---

## F. Pertanyaan Diskusi

1. Apakah POS Tagging selalu menghasilkan label yang benar?
2. Apa penyebab utama kesalahan POS Tagging pada teks Bahasa Indonesia informal?
3. Bagaimana slang mempengaruhi POS Tagging?
4. Bagaimana nama organisasi (`GOTO`, `Tokopedia`) mempengaruhi POS Tagging?
5. Mengapa parsing penting dalam NLP? Berikan contoh aplikasi yang memerlukannya.
6. Apa hubungan morphology dengan stemming?
7. Apakah stemming dan POS Tagging menyelesaikan permasalahan yang sama?

---

## G. Output Pertemuan 4

File yang dikumpulkan:

```text
linguistic_processing.ipynb
```

> Output: Analisis struktur linguistik corpus yang mencakup token, POS, dependency, visualisasi, dan analisis kesalahan.

Notebook berisi:

```text
1. Import Library
2. Load spaCy Model
3. POS Tagging — Kalimat Formal
4. POS Tagging — Kalimat Informal
5. Dependency Parsing
6. Visualisasi Dependency Tree
7. Dataset Analysis
8. Error Analysis
9. Kesimpulan
```

---

> **Pertemuan sebelumnya:** [Pertemuan 3 — Text Normalization](praktikum3.md)
>
> **Pertemuan berikutnya:** [Pertemuan 5 — N-Gram & Language Model](praktikum5.md)
