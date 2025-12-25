# 🚀 Intelligent ATS (AI-Powered Recruitment Tool)

**Intelligent ATS** adalah aplikasi web modern yang dirancang untuk streamline proses rekrutmen menggunakan **Artificial Intelligence**. Aplikasi ini secara otomatis mengekstrak skill dari CV pelamar (PDF) dan mencocokkannya dengan requirement pekerjaan secara real-time.

Proyek ini mendemonstrasikan integrasi antara **Machine Learning (NER/NLP)** dengan **Modern Web Development**.

🔗 **Live Demo:** [https://ats-kecil.vercel.app](https://ats-kecil.vercel.app)

---

## ✨ Fitur Utama

- **📄 Smart PDF Parsing**: Upload banyak CV sekaligus dan ekstrak teks secara otomatis.
- **🤖 AI-Driven Extraction**: Menggunakan model **DistilBERT** (Fine-tuned) untuk mendeteksi technical skills dari teks yang tidak terstruktur.
- **📊 Automated Matching**: Algoritma scoring cerdas yang menghitung kecocokan kandidat vs. lowongan.
- **🔍 Detailed Insights**: Visualisasi skill yang *Matched*, *Missing*, dan *Extra* untuk setiap kandidat.
- **⚡ Modern UI**: Interface futuristik dengan Glassmorphism, dibangun menggunakan Next.js & Tailwind CSS.

---

## 🛠️ Tech Stack

### Frontend (Repository Ini)
- **Framework**: [Next.js 14](https://nextjs.org/) (App Router)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **Components**: React Icons, standar UI modern
- **Deployment**: Vercel

### Backend (Sistem Private)
> *Catatan: Kode backend (API & Model ML) disimpan di repository terpisah untuk alasan privasi/lisensi.*

- **API**: FastAPI (Python)
- **ML Engine**: TensorFlow, Hugging Face Transformers
- **Model**: DistilBERT (Token Classification)
- **Infrastructure**: GPU-accelerated cloud instance

---

## 📸 Preview

*Visualisasi antarmuka aplikasi yang menampilkan hasil ranking kandidat dan detail skill.*

---

## 🚀 Cara Menjalankan (Frontend)

1. **Clone Repository**
   ```bash
   git clone https://github.com/keterbingungan/ats-kecil.git
   cd ats-kecil/frontend
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Jalankan Development Server**
   ```bash
   npm run dev
   ```

4. **Buka Browser**
   Buka [http://localhost:3000](http://localhost:3000) untuk melihat aplikasi.

---

## 📝 License

Project ini dibuat untuk tujuan edukasi dan portofolio.
