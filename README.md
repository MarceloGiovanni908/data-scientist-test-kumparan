# 📰 Kumparan Topic Extractor Model  
### *FastText + Logistic Regression Version*  
**Author:** Heydar Emir Alvaro  
**Compatible with:** `kumparanian ds verify`

---

## 📘 Deskripsi Proyek
Proyek ini merupakan implementasi model **ekstraksi topik artikel (topic extractor)** yang dikembangkan untuk platform **Kumparan**.  
Model ini memanfaatkan kombinasi antara:
- **FastText** sebagai *word embedding generator*, dan  
- **Logistic Regression** sebagai *topic classifier*.

Tujuan utama dari model ini adalah untuk mengklasifikasikan artikel berdasarkan topik secara otomatis menggunakan pembelajaran mesin berbasis teks berbahasa Indonesia.

---

## ⚙️ Fitur Utama
✅ Preprocessing teks otomatis (normalisasi, tokenisasi, *stopword removal*, dan *light stemming*)  
✅ Pelatihan model FastText dari awal (*from scratch*)  
✅ Ekstraksi vektor kalimat berbasis rata-rata embedding kata  
✅ Pelatihan model klasifikasi dengan Logistic Regression  
✅ Evaluasi model dengan *classification report* dan akurasi validasi  
✅ Penyimpanan model dan laporan hasil pelatihan secara otomatis  
✅ Kompatibel dengan pipeline `kumparanian ds verify`

---

## 🧠 Arsitektur Model
```bash
Article Text
                │
                ▼
Preprocessing (tokenization, stopwords, stemming)
                |
                ▼
FastText Embedding (vector_size=300)
                │
                ▼
Sentence Vectorization (mean pooling)
                │
                ▼
Logistic Regression Classifier
                │
                ▼
Predicted Topic
```

---

## 🧩 Struktur File
```bash
project/
│
├── model.py # Skrip utama model FastText + Logistic Regression
├── data.csv # Dataset artikel (wajib ada kolom 'article_content' & 'article_topic')
├── model.pickle # Model hasil pelatihan (FastText + classifier)
├── training_report.txt # Laporan hasil training (akurasi & laporan klasifikasi)
└── requirements.txt # Dependensi Python
```
Catatan:
Beberapa pustaka NLTK akan diunduh secara otomatis saat pertama kali menjalankan model.

---
## 🧾 Cara Menjalankan
1. Siapkan Dataset

Dataset harus berupa file CSV dengan minimal dua kolom berikut:

`article_content` → berisi teks artikel

`article_topic` → berisi label topik artikel

Contoh format data.csv:
```csv
article_content,article_topic
"Presiden meresmikan jalan tol baru di Jawa Tengah","news"
"Tips menjaga kesehatan mental di tempat kerja","lifestyle"
```
2. Latih Model

Jalankan perintah berikut di terminal:
```bash
python model.py
```
Model akan:

Membersihkan teks,

Melatih embedding FastText,

Melatih model Logistic Regression,

Mengevaluasi kinerja model,

Menyimpan hasil pelatihan dan model ke file `model.pickle`.

3. Gunakan Model untuk Prediksi

Kamu bisa memuat model dan melakukan prediksi dengan contoh kode berikut:

```python
from model import Model

model = Model()
model.load_pickle("model.pickle")

sample_text = "Pemerintah meluncurkan program bantuan sosial untuk masyarakat"
predicted_topic = model.predict(sample_text)

print("Topik yang diprediksi:", predicted_topic)
```
Output
```bash
Topik yang diprediksi: news
```
---
## 📊 Hasil Pelatihan
Setelah pelatihan, hasil evaluasi akan disimpan otomatis ke file `training_report.txt`isi laporan ini mencakup :
```bash
Tanggal pelatihan

Ukuran dataset

Jumlah topik unik

Akurasi validasi

Laporan klasifikasi per topik

Durasi pelatihan
```
contoh:
```bash
FASTTEXT + LOGISTIC REGRESSION TOPIC EXTRACTOR REPORT
============================================================
training_date: 2025-10-26 12:30:45
dataset_size: (10000, 3)
unique_topics: 8
validation_accuracy: 0.9123
training_duration_sec: 124.56

=== Classification Report ===
              precision    recall  f1-score   support
news          0.91        0.92      0.91       200
sports        0.93        0.90      0.91       180
...
```
---
## 💾 Penyimpanan Model
Model disimpan dalam format pickle yang berisi:

`fasttext_model` → model embedding kata

`classifier` → model Logistic Regression

`predict()` → fungsi prediksi topik

file disimpan dengan nama 
```bash
model.pickle
```
model bisa digunakan dengan cara:
```bash
model.load_pickle("model.pickle")
```
---
## 📚 Teknologi yang Digunakan
```python
Python 3.8+

Gensim (FastText)

Scikit-learn

NLTK

NumPy

Pandas

Kumparanian SDK
```
---
## 🧹 Fungsi Utama dalam Kode
| Fungsi | Deskripsi |
|--------|------------|
| `clean_and_tokenize(text)` | Membersihkan teks dari URL, angka, tanda baca, dan melakukan tokenisasi serta stemming ringan. |
| `get_sentence_vector(tokens)` | Menghasilkan representasi vektor kalimat dengan rata-rata embedding kata. |
| `train(data_path)` | Melatih model FastText dan Logistic Regression menggunakan dataset CSV. |
| `_save_report(path)` | Menyimpan laporan hasil pelatihan ke file teks. |
| `predict(text)` | Mengklasifikasikan topik artikel baru. |
| `save_pickle(path)` | Menyimpan seluruh model ke dalam satu file pickle. |
| `load_pickle(path)` | Memuat ulang model dari file pickle. |