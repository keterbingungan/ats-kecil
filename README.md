# 🧠 Intelligent ATS: Resume Parser & Ranker with NER Transformers

**Intelligent ATS** adalah sistem end-to-end NLP project yang dirancang untuk mengotomatisasi penyaringan CV. Project ini berfokus pada ekstraksi entitas (Named Entity Recognition) menggunakan **Fine-tuned DistilBERT** untuk mengidentifikasi *Technical Skills* dan melakukan ranking kandidat menggunakan pendekatan **Hybrid Scoring** (Semantic + Keyword Coverage).

🔗 **Live Demo:** [https://ats-kecil.vercel.app](https://ats-kecil.vercel.app)

---

## 🔬 Data Science Methodology

Project ini tidak hanya sekadar memanggil API, tetapi melalui proses eksperimen model deep learning untuk mendapatkan akurasi ekstraksi terbaik.

### 1. Model Selection & Experimentation
Kami membandingkan tiga arsitektur model untuk task Named Entity Recognition (NER) pada dataset resume:

| Model Architecture | F1-Score | Accuracy | Status |
|--------------------|----------|----------|--------|
| **DistilBERT (Transformer)** | **0.9086** | **0.9730** | ✅ **Selected** |
| BiLSTM | 0.8686 | 0.9582 | ❌ Discarded |
| BiGRU | 0.8632 | 0.9562 | ❌ Discarded |

> **Insight:** DistilBERT dipilih karena menawarkan performa terbaik dengan F1-Score **90.86%** dan akurasi **97.30%**. Arsitektur Transformer dengan mekanisme self-attention mampu menangkap relasi semantik dalam dokumen secara lebih akurat dibanding model RNN tradisional.

### 2. Preprocessing Pipeline
Rangkaian preprocessing yang diterapkan untuk menjaga kualitas ekstraksi skill:

1. **Tokenization**: Memecah teks CV menjadi unit-unit kata/frasa menggunakan tokenizer DistilBERT.
2. **Selective Cleaning**: Menghapus noise namun mempertahankan simbol teknis penting seperti "C++", "C#", "TensorFlow 2.0".
3. **Fuzzy String Matching**: Menggunakan Levenshtein Distance untuk menstandarisasi variasi penulisan (misalnya: "Javascript" vs "JavaScript", "machine learning" vs "machine-learning").

### 3. Algoritma Scoring (Hybrid Approach)
Sistem menggunakan **Hybrid Scoring Mechanism** untuk ranking kandidat:

1.  **NER Extraction:** Mengekstrak entitas skill dari PDF menggunakan model DistilBERT yang telah di-fine-tune.
2.  **Coverage Ratio:** Menghitung persentase skill kandidat yang memenuhi *hard-requirement*.
    $$Score_{coverage} = \frac{\text{Detected Skills} \cap \text{Required Skills}}{\text{Required Skills}} \times 100\%$$
3.  **Cosine Similarity:** Mengukur kesesuaian semantik antara representasi vektor skill kandidat dengan deskripsi pekerjaan target.

---

## 🛠️ Technical Architecture

Sistem dibangun dengan arsitektur **Microservices** untuk memisahkan beban komputasi ML dengan antarmuka pengguna.

### Backend & ML Pipeline
- **Framework**: FastAPI (Python) untuk high-performance asynchronous serving.
- **ML Engine**: Hugging Face Transformers & TensorFlow.
- **Preprocessing**: PDF text extraction, Tokenization, Fuzzy Matching.
- **Model Serving**: Model diload ke memory saat startup untuk low-latency inference.
- **Deployment**: AWS EC2 + Systemd untuk fault-tolerant daemon process, Ngrok untuk secure HTTPS tunneling.

### Frontend
- **Framework**: Next.js 14 & Tailwind CSS.
- **Hosting**: Vercel dengan CDN (Content Delivery Network).
- **Role**: Mengirim file PDF ke backend dan memvisualisasikan hasil scoring (Skills, Score, Missing Skills).

---

## � Dataset

Dataset yang digunakan berasal dari platform **HuggingFace** yang berisi kumpulan resume/CV dalam format:
- **JSON**: Teks mentah dengan annotations label entitas (SKILL) beserta koordinat karakter.
- **PDF**: Format tidak terstruktur untuk testing real-world scenario.

---

## 📝 Future Improvements
- Mengeksplorasi varian model Transformer lain seperti **RoBERTa** atau **Longformer** untuk menangani dokumen panjang.
- Memperluas cakupan NER ke entitas lain seperti *Pengalaman Kerja*, *Pendidikan*, dan *Sertifikasi*.
- Menambah variasi dataset CV termasuk **Bahasa Indonesia** untuk meningkatkan kemampuan generalisasi model.