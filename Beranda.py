import streamlit as st
import joblib
import numpy as np
import pandas as pd
from preprocessor import UrinalysisPreprocessor

import re

def validate_wbc_rbc(value):
    val = value.strip().upper()  # normalisasi uppercase

    # ===============================
    # 1. Konversi bahasa Indonesia → LOADED / TNTC
    # ===============================
    if val in ["BANYAK"]:
        return True, "LOADED"
    if val in ["SANGAT BANYAK"]:
        return True, "TNTC"

    # ===============================
    # 2. Regex format valid
    # ===============================
    pattern = r"^(LOADED|TNTC|[0-9]+|[0-9]+-[0-9]+|>[0-9]+)$"

    if not re.match(pattern, val):
        return False, "Format tidak valid. Gunakan angka (0, 5), rentang (1-5), >50, atau input 'Banyak' / 'Sangat Banyak'."

    # ===============================
    # 3. Validasi rentang (harus low < high)
    # ===============================
    if "-" in val:
        low, high = val.split("-")
        low = int(low)
        high = int(high)
        if low > high:
            return False, "Rentang tidak valid (contoh benar: 3-5, bukan 5-3)."

    # ===============================
    # 4. Return jika valid
    # ===============================
    return True, val


# ================================
# LOAD PREPROCESSOR & MODELS
# ================================
pre = joblib.load("preprocessor.pkl")

# ================================
# HOTFIX MAPPING (TETAP DIPAKAI)
# ================================
pre.transparency_map = {
    'CLEAR':0,
    'SLIGHTLY HAZY':1,
    'HAZY':2,
    'CLOUDY':3,
    'TURBID':4
}

pre.protein_map = {
    'NEGATIVE':0,
    'TRACE':1,
    '1+':2,
    '2+':3,
    '3+':4
}

pre.glucose_map = {
    'NEGATIVE':0,
    'TRACE':1,
    '1+':2,
    '2+':3,
    '3+':4,
    '4+':5
}

pre.epith_map = {
    'NONE SEEN':0,
    'RARE':1,
    'FEW':2,
    'OCCASIONAL':3,
    'MODERATE':4,
    'LOADED':5,
    'PLENTY':6
}

pre.mucous_map = {
    'NONE SEEN':0,
    'RARE':1,
    'FEW':2,
    'OCCASIONAL':3,
    'MODERATE':4,
    'PLENTY':5
}

pre.amorphous_map = {
    'NONE SEEN':0,
    'RARE':1,
    'FEW':2,
    'OCCASIONAL':3,
    'MODERATE':4,
    'PLENTY':5
}

pre.bacteria_map = {
    'NONE SEEN':0,
    'RARE':1,
    'FEW':2,
    'OCCASIONAL':3,
    'MODERATE':4,
    'LOADED':5,
    'PLENTY':6
}

pre.color_map = {
    "LIGHT YELLOW":0,
    "STRAW":1,
    "AMBER":2,
    "BROWN":3,
    "DARK YELLOW":4,
    "YELLOW":5,
    "REDDISH YELLOW":6,
    "REDDISH":7,
    "LIGHT RED":8,
    "RED":9
}

pre.final_columns = [
    'Age', 'Color', 'Transparency', 'Glucose', 'Protein',
    'pH', 'Specific Gravity', 'WBC', 'RBC',
    'Epithelial Cells', 'Mucous Threads', 'Amorphous Urates',
    'Bacteria', 'Gender_MALE'
]

# ================================
# LOAD MODELS STACKING
# ================================
nb_model   = joblib.load("models/full_model_gnb.pkl")
rf_model   = joblib.load("models/full_model_rf.pkl")
xgb_model  = joblib.load("models/full_model_xgb.pkl")
lr_model = joblib.load("models/full_model_lr.pkl")
meta_model = joblib.load("models/meta_stacking_model.pkl")


# ================================
# STREAMLIT UI
# ================================
st.title("Identifikasi Penyakit Infeksi Saluran Kemih (ISK)")
st.write("Aplikasi ini menggunakan model Machine Learning untuk mengidentifikasi Infeksi Saluran Kemih (ISK), dengan teknik Stacking Ensemble Learning sebagai metode utama.")

st.header("Input Data Urinalisis Pasien")

st.sidebar.header("Pengaturan Identifikasi")

model_choice = st.sidebar.selectbox(
    "Pilih Model Prediksi",
    [
        "Stacking Ensemble Learning",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
        "Logistic Regression"
    ]
)

st.sidebar.markdown("### Informasi Model")

# --- CUSTOM CSS KHUSUS UNTUK SIDEBAR INFO ---
st.markdown(
    """
    <style>
    /* 1. Menargetkan kotak info (stAlert) */
    /* 2. Yang berada di dalam Sidebar (data-testid="stSidebar") */
    
    [data-testid="stSidebar"] .stAlert {
        background-color: #37bf8633; /* Warna Latar Belakang Custom (misal: Orange Lembut) */
        color: #FAFAFA;           /* Warna Teks Custom (misal: Orange Tua) */
        padding: 0px;
        border-radius: 6px;
    }

    [data-testid="stSidebar"] .st-al {
        color: #FAFAFA;
    }

    [data-testid="stSidebar"] .st-at {
        background-color: #00000000;
    }
    
    /* Opsional: Ubah warna ikon di sidebar info */
    [data-testid="stSidebar"] .stAlert svg {
        fill: #e65100; /* Mengubah warna ikon (misal: ikon "i" di st.info) */
    }
    </style>
    """,
    unsafe_allow_html=True
)

if model_choice == "Stacking Ensemble Learning":
    st.sidebar.info("Menggunakan Ensemble (Naive Bayes, Random Forest, XGBoost) → Meta Logistic Regression")
elif model_choice == "Naive Bayes":
    st.sidebar.info("Model probabilistik berbasis Teorema Bayes.")
elif model_choice == "Random Forest":
    st.sidebar.info("Model ensemble berbasis banyak decision tree.")
elif model_choice == "XGBoost":
    st.sidebar.info("Model boosting yang sangat powerful untuk tabular.")
elif model_choice == "Logistic Regression":
    st.sidebar.info("Model regresi logistik yang dilatih langsung dari dataset.")



# MAPPING UI → MODEL (Bahasa Indonesia → Inggris)
gender_ui = {
    "Laki-Laki": "MALE",
    "Perempuan": "FEMALE"
}

color_ui = {
    "Kuning Muda": "LIGHT YELLOW",
    "Jerami": "STRAW",
    "Amber": "AMBER",
    "Coklat": "BROWN",
    "Kuning Gelap": "DARK YELLOW",
    "Kuning": "YELLOW",
    "Kuning Kemerahan": "REDDISH YELLOW",
    "Kemerahan": "REDDISH",
    "Merah Muda": "LIGHT RED",
    "Merah": "RED"
}

transparency_ui = {
    "Jernih": "CLEAR",
    "Sedikit Keruh": "SLIGHTLY HAZY",
    "Keruh": "HAZY",
    "Keruh Pekat": "CLOUDY",
    "Sangat Keruh": "TURBID"
}

glucose_ui = {
    "Negatif": "NEGATIVE",
    "Jejak": "TRACE",
    "1+": "1+",
    "2+": "2+",
    "3+": "3+",
    "4+": "4+"
}

protein_ui = {
    "Negatif": "NEGATIVE",
    "Jejak": "TRACE",
    "1+": "1+",
    "2+": "2+",
    "3+": "3+"
}

epith_ui = {
    "Tidak Terlihat": "NONE SEEN",
    "Jarang": "RARE",
    "Sedikit": "FEW",
    "Kadang-Kadang": "OCCASIONAL",
    "Sedang": "MODERATE",
    "Banyak": "LOADED",
    "Sangat Banyak": "PLENTY"
}

mucous_ui = {
    "Tidak Terlihat": "NONE SEEN",
    "Jarang": "RARE",
    "Sedikit": "FEW",
    "Kadang-Kadang": "OCCASIONAL",
    "Sedang": "MODERATE",
    "Banyak": "PLENTY"
}

amorphous_ui = {
    "Tidak Terlihat": "NONE SEEN",
    "Jarang": "RARE",
    "Sedikit": "FEW",
    "Kadang-Kadang": "OCCASIONAL",
    "Sedang": "MODERATE",
    "Banyak": "PLENTY"
}

bacteria_ui = {
    "Tidak Terlihat": "NONE SEEN",
    "Jarang": "RARE",
    "Sedikit": "FEW",
    "Kadang-Kadang": "OCCASIONAL",
    "Sedang": "MODERATE",
    "Banyak": "LOADED",
    "Sangat Banyak": "PLENTY"
}


# ================================
# FORM INPUT FIELDS
# ================================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur", min_value=0, max_value=120, value=25)
    gender = st.selectbox("Jenis Kelamin", list(gender_ui.keys()))
with col2:
    color = st.selectbox("Warna Urin", list(color_ui.keys()))
    transparency = st.selectbox("Kekeruhan (Transparansi)", list(transparency_ui.keys()))


# Slider pH
ph = st.slider(
    "pH",
    min_value=0.0,
    max_value=14.0,
    value=6.0,
    step=0.1,
    format="%.1f"
)

# Slider SG (x.xxx)
sg = st.slider(
    "Berat Jenis (Specific Gravity)",
    min_value=1.000,
    max_value=1.050,
    value=1.015,
    step=0.001,
    format="%.3f"
)

glucose = st.selectbox("Glukosa", list(glucose_ui.keys()))
protein = st.selectbox("Protein", list(protein_ui.keys()))
wbc_input = st.text_input("WBC (contoh: 0–2, >100, Banyak, Sangat Banyak)", "0")
rbc_input = st.text_input("RBC (contoh: 0–2, >100, Banyak, Sangat Banyak)", "0")

epith = st.selectbox("Sel Epitel", list(epith_ui.keys()))
mucous = st.selectbox("Benang Lendir", list(mucous_ui.keys()))
amorphous = st.selectbox("Kristal Amorf", list(amorphous_ui.keys()))
bacteria = st.selectbox("Bakteri", list(bacteria_ui.keys()))



# ================================
# PREDIKSI
# ================================

st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
submit = st.button("Identifikasi Pasien", use_container_width=True)

if submit:

    # ============================
    # VALIDASI WBC & RBC
    # ============================
    valid_wbc, wbc_msg = validate_wbc_rbc(wbc_input)
    if not valid_wbc:
        st.error(f"WBC Error: {wbc_msg}")
        st.stop()

    valid_rbc, rbc_msg = validate_wbc_rbc(rbc_input)
    if not valid_rbc:
        st.error(f"RBC Error: {rbc_msg}")
        st.stop()

    # hasil valid
    wbc = wbc_msg
    rbc = rbc_msg

    # ============================
    # SIAPKAN DICTIONARY INPUT
    # ============================
    input_dict = {
        "Age": age,
        "pH": ph,
        "Specific Gravity": sg,
        "Gender": gender_ui[gender],
        "Color": color_ui[color],
        "Transparency": transparency_ui[transparency],
        "Glucose": glucose_ui[glucose],
        "Protein": protein_ui[protein],
        "WBC": wbc,
        "RBC": rbc,
        "Epithelial Cells": epith_ui[epith],
        "Mucous Threads": mucous_ui[mucous],
        "Amorphous Urates": amorphous_ui[amorphous],
        "Bacteria": bacteria_ui[bacteria]
    }

    # ============================
    # PREPROCESSING
    # ============================
    try:
        X_processed = pre.transform(input_dict)
    except Exception as e:
        st.error(f"Terjadi error saat preprocessing: {e}")
        st.stop()

    # ============================
    # PREDIKSI SESUAI MODEL PILIHAN
    # ============================

    if model_choice == "Naive Bayes":
        final_pred = nb_model.predict(X_processed)[0]
        final_proba = nb_model.predict_proba(X_processed)[0][1]

    elif model_choice == "Random Forest":
        final_pred = rf_model.predict(X_processed)[0]
        final_proba = rf_model.predict_proba(X_processed)[0][1]

    elif model_choice == "XGBoost":
        final_pred = xgb_model.predict(X_processed)[0]
        final_proba = xgb_model.predict_proba(X_processed)[0][1]

    elif model_choice == "Logistic Regression":
        final_pred = lr_model.predict(X_processed)[0]
        final_proba = lr_model.predict_proba(X_processed)[0][1]

    else:
        # ============================
        # STACKING PROBA BASE MODELS
        # ============================
        try:
            p_nb  = nb_model.predict_proba(X_processed)[0][1]
            p_rf  = rf_model.predict_proba(X_processed)[0][1]
            p_xgb = xgb_model.predict_proba(X_processed)[0][1]
        except Exception as e:
            st.error(f"Error pada prediksi base model: {e}")
            st.stop()

        # INPUT KE META MODEL
        meta_input = np.array([[p_nb, p_rf, p_xgb]])
        final_pred = meta_model.predict(meta_input)[0]
        final_proba = meta_model.predict_proba(meta_input)[0][1]


    # ============================
    # OUTPUT
    # ============================
    st.subheader("Hasil Prediksi")

    if final_pred == 1:
        st.success("**Hasil: POSITIF ISK**")
    else:
        st.info("**Hasil: NEGATIF ISK**")

    # tampilkan probabilitas dalam persen
    st.write(f"Probabilitas prediksi: **{final_proba * 100:.2f}%**")

    # hanya tampilkan probabilitas base model jika memilih stacking
    if model_choice == "Stacking Ensemble Learning":
        st.subheader("Probabilitas Base Models")
        st.write(f"- Naive Bayes: `{p_nb * 100:.2f}%`")
        st.write(f"- Random Forest: `{p_rf * 100:.2f}%`")
        st.write(f"- XGBoost: `{p_xgb * 100:.2f}%`")

        

    # ============================
    # PENJELASAN PROBABILITAS
    # ============================
    st.markdown(
        """
        **Keterangan Probabilitas:**  
        Nilai probabilitas menunjukkan tingkat keyakinan model terhadap hasil prediksi kelas positif (ISK). 
        Nilai probabilitas yang mendekati 100% menunjukkan bahwa model semakin yakin terhadap prediksi 
        positif ISK, sedangkan nilai probabilitas yang mendekati 0% menunjukkan keyakinan model yang tinggi 
        terhadap prediksi negatif ISK. Sementara itu, nilai probabilitas yang berada di sekitar 50% 
        mengindikasikan bahwa model masih memiliki tingkat ketidakpastian yang relatif tinggi, sehingga 
        hasil prediksi perlu dipertimbangkan secara hati-hati dan tidak digunakan sebagai dasar keputusan 
        diagnosis utama.
        """
    )

