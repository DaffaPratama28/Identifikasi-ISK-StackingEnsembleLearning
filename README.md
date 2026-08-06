# Urinary Tract Infection Classification using Stacking Ensemble Learning

> Undergraduate Thesis Project | Universitas Muhammadiyah Pontianak

A Streamlit-based web application for predicting **Urinary Tract Infection (UTI)** using a **Stacking Ensemble Learning** model trained on urinalysis data.

---

## 🇮🇩 Bahasa Indonesia

### Tentang Proyek

Repositori ini merupakan implementasi dari skripsi berjudul:

> **Penerapan Teknik Stacking Ensemble Learning untuk Klasifikasi Penyakit Infeksi Saluran Kemih Berdasarkan Hasil Urinalisis**

Aplikasi ini dibangun menggunakan **Streamlit** sebagai antarmuka web dan model Machine Learning yang telah dilatih menggunakan teknik **Stacking Ensemble Learning**.

Model memanfaatkan tiga algoritma dasar:

- Naïve Bayes
- Random Forest
- XGBoost

dengan **Logistic Regression** sebagai meta-classifier.

---

### Dataset

Dataset berasal dari Kaggle dan berisi hasil pemeriksaan urinalisis pasien.

- Total data: **1,436**
- Features: **16**
- Target:
  - Positive UTI
  - Negative UTI

---

### Hasil Model

| Metric | Score |
|--------|-------|
| Accuracy | **96.49%** |
| Precision | **71.43%** |
| Recall | **62.50%** |
| F1 Score | **66.67%** |
| ROC-AUC | **89.31%** |

---

### Teknologi

- Python
- Streamlit
- Scikit-learn
- XGBoost
- Pandas
- NumPy

---

### Menjalankan Aplikasi

Clone repository

```bash
git clone https://github.com/username/repository-name.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Jalankan aplikasi

```bash
streamlit run app.py
```

---

## 🇺🇸 English

### About

This repository contains the implementation of my undergraduate thesis:

> **Applying Stacking Ensemble Learning for Urinary Tract Infection Classification Based on Urinalysis Results**

The application is developed using **Streamlit** and utilizes a **Stacking Ensemble Learning** model for predicting Urinary Tract Infection (UTI).

The ensemble consists of:

- Naïve Bayes
- Random Forest
- XGBoost

with **Logistic Regression** as the meta-classifier.

---

### Dataset

The dataset was obtained from Kaggle and contains urinalysis records.

- Total samples: **1,436**
- Features: **16**
- Binary classification:
  - Positive UTI
  - Negative UTI

---

### Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **96.49%** |
| Precision | **71.43%** |
| Recall | **62.50%** |
| F1 Score | **66.67%** |
| ROC-AUC | **89.31%** |

---

### Tech Stack

- Python
- Streamlit
- Scikit-learn
- XGBoost
- Pandas
- NumPy

---

### Run Locally

Clone the repository

```bash
git clone https://github.com/username/repository-name.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

## Project Structure

```
.
├── app.py
├── requirements.txt
├── model/
├── assets/
├── notebooks/
└── README.md
```

---

## Author

**Daffa Pratama**

Bachelor of Informatics Engineering  
Universitas Muhammadiyah Pontianak

---

## Disclaimer

This project was developed for academic purposes as part of an undergraduate thesis. It is intended to demonstrate the application of Machine Learning techniques for educational and research purposes and should not be used as a substitute for professional medical diagnosis.
