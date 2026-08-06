# Urinary Tract Infection Classification using Stacking Ensemble Learning

This repository contains the implementation of my undergraduate thesis, developed as an interactive web application using Streamlit for Urinary Tract Infection (UTI) classification based on urinalysis results.

The application predicts whether a patient is likely to have a urinary tract infection using a **Stacking Ensemble Learning** model.

## Thesis

**Title**

Penerapan Teknik *Stacking Ensemble Learning* untuk Klasifikasi Penyakit Infeksi Saluran Kemih Berdasarkan Hasil Urinalisis

This research applies a stacking ensemble approach by combining multiple machine learning algorithms to improve classification performance on urinalysis data.

### Base Models

- Naïve Bayes
- Random Forest
- XGBoost

### Meta Model

- Logistic Regression

## Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **96.49%** |
| Precision | **71.43%** |
| Recall | **62.50%** |
| F1-Score | **66.67%** |
| ROC-AUC | **89.31%** |

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Joblib

## Dataset

- Source: Kaggle Urinalysis Dataset
- Samples: **1,436**
- Features: **16**

## Running the Application

Clone the repository

```bash
git clone https://github.com/DaffaPratama28/Identifikasi-ISK-StackingEnsembleLearning.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

## Repository Structure

```
├── app.py
├── model/
├── assets/
├── requirements.txt
└── README.md
```

## About

This project was developed as part of my undergraduate thesis at **Universitas Muhammadiyah Pontianak**.

The purpose of this repository is to demonstrate the implementation of machine learning techniques for educational and research purposes. It is **not intended for clinical diagnosis or medical decision-making**.

## Live Demo

https://identifikasi-isk-stackingensemblelearning-rjowaqwsnghnxmqhrrqr.streamlit.app/

---

## Bahasa Indonesia

Repository ini merupakan implementasi skripsi saya yang membahas penerapan **Stacking Ensemble Learning** untuk klasifikasi penyakit **Infeksi Saluran Kemih (ISK)** berdasarkan hasil urinalisis. Aplikasi dikembangkan menggunakan **Streamlit** sehingga model dapat digunakan melalui antarmuka web yang sederhana. Penelitian ini menggabungkan algoritma **Naïve Bayes, Random Forest, dan XGBoost** dengan **Logistic Regression** sebagai *meta-classifier* untuk meningkatkan performa klasifikasi.
