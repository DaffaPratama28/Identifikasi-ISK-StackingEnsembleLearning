import streamlit as st
import pandas as pd

# ===========================
# Halaman Informasi Model Identifikasi
# ===========================

st.set_page_config(page_title="Informasi Model Identifikasi")

# ====== CSS agar layout lebih rapi ======
st.markdown("""
<style>
    .model-title {
        font-size: 28px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 0px;
    }
    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 25px;
    }
    .metric-box {
        background-color: #F8F9FA;
        padding: 12px 15px;
        border-radius: 8px;
        border-left: 5px solid #4A6CF7;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ===========================
# Sidebar Model Selector
# ===========================

st.sidebar.header("Pilih Model")
model_choice = st.sidebar.selectbox(
    "Model yang ingin ditampilkan",
    [
        "Stacking Ensemble Learning",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
        "Logistic Regression"
    ]
)

# ===========================
# Data Metrik Setiap Model
# ===========================

model_metrics = {
    "Naive Bayes": {
        "Akurasi": "84.21%",
        "Presisi": "19.14%",
        "Recall": "56.25%",
        "F1-Score": "28.57%",
        "ROC-AUC": "67.72%",
        "cm": "img/conf_matrix_gnb.png",
        "cr": "img/cr_gnb.PNG"
    },

    "Random Forest": {
        "Akurasi": "96.14%",
        "Presisi": "72.73%",
        "Recall": "50.00%",
        "F1-Score": "59.26%",
        "ROC-AUC": "86.27%",
        "cm": "img/conf_matrix_rf.png",
        "cr": "img/cr_rf.PNG"
    },

    "XGBoost": {
        "Akurasi": "95.79%",
        "Presisi": "66.67%",
        "Recall": "50.00%",
        "F1-Score": "57.14%",
        "ROC-AUC": "86.32%",
        "cm": "img/conf_matrix_xgb.png",
        "cr": "img/cr_xgb.PNG"
    },

    "Logistic Regression": {
        "Akurasi": "85.61%",
        "Presisi": "23.40%",
        "Recall": "68.75%",
        "F1-Score": "34.92%",
        "ROC-AUC": "77.86%",
        "cm": "img/conf_matrix_lr.png",
        "cr": "img/cr_lr.PNG"
    },

    "Stacking Ensemble Learning": {
        "Akurasi": "96.49%",
        "Presisi": "71.43%",
        "Recall": "62.50%",
        "F1-Score": "66.67%",
        "ROC-AUC": "89.31%",
        "cm": "img/conf_matrix_stacking.png",
        "cr": "img/cr_stacking.PNG"
    },
}


# ===========================
# Deskripsi Model
# ===========================

model_descriptions = {
    "Naive Bayes": """
Naive Bayes adalah model probabilistik berbasis Teorema Bayes yang mengasumsikan bahwa setiap fitur bersifat independen satu sama lain. 
Meskipun sederhana, model ini efektif pada dataset dengan ukuran kecil dan memiliki performa yang stabil dalam mendeteksi pola berbasis probabilitas. 
Pada konteks identifikasi ISK, Naive Bayes memberikan gambaran awal bagaimana distribusi fitur urin dapat mempengaruhi kemungkinan terjadinya infeksi.
""",

    "Random Forest": """
Random Forest adalah model ensemble yang terdiri dari banyak decision tree. 
Setiap pohon dilatih pada subset data, dan prediksi akhir ditentukan melalui voting atau rata-rata. 
Model ini mampu menangani interaksi antar fitur dan cenderung memiliki akurasi tinggi serta lebih tahan terhadap overfitting. 
Dalam kasus ISK, Random Forest bekerja sangat baik dalam mengidentifikasi kombinasi fitur urinalisis yang menunjukkan adanya infeksi.
""",

    "XGBoost": """
XGBoost adalah algoritma boosting yang sangat kuat, cepat, dan efisien. 
Model ini membangun pohon secara bertahap dengan fokus pada perbaikan kesalahan dari model sebelumnya. 
XGBoost populer untuk data tabular karena kemampuannya menangani fitur kompleks dan memberikan performa tinggi. 
Untuk deteksi ISK, XGBoost berhasil menangkap pola-pola halus dalam data urinalisis.
""",

    "Logistic Regression": """
Logistic Regression adalah model linear yang memetakan hubungan antara fitur dan probabilitas kejadian. 
Meskipun sederhana, model ini interpretatif dan dapat menjelaskan kontribusi masing-masing fitur terhadap prediksi. 
Dalam analisis ISK, Logistic Regression memberikan baseline model yang solid untuk membandingkan metode yang lebih kompleks.
""",

    "Stacking Ensemble Learning": """
Stacking menggabungkan beberapa model dasar (Naive Bayes, Random Forest, dan XGBoost) dengan meta-learner Logistic Regression. 
Pendekatan ini memanfaatkan kelebihan masing-masing model sehingga meningkatkan performa secara keseluruhan. 
Model stacking terbukti memberikan hasil terbaik pada penelitian ini, terutama dalam meningkatkan recall dan F1-score pada prediksi ISK.
"""
}

# ===========================
# SECTION TITLE
# ===========================
st.title("Informasi Model Identifikasi ISK")

st.markdown(f"<div class='model-title'>{model_choice}</div>", unsafe_allow_html=True)
st.markdown("---")

# ===========================
# TWO COLUMNS FOR IMAGES
# ===========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Confusion Matrix")
    st.image(model_metrics[model_choice]["cm"], use_container_width=True)

with col2:
    st.markdown("### Classification Report")
    st.image(model_metrics[model_choice]["cr"], use_container_width=True)

# ===========================
# MODEL DESCRIPTION
# ===========================
st.markdown("### Penjelasan Model")
st.write(model_descriptions[model_choice])

# ===========================
# METRIC TABLE
# ===========================
st.markdown("### 📊 Performa Model")

metrics_df = pd.DataFrame({
    "Metrik": list(model_metrics[model_choice].keys())[:5],
    "Nilai": list(model_metrics[model_choice].values())[:5]
})

st.dataframe(metrics_df, use_container_width=True)
