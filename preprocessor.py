import numpy as np
import pandas as pd

class UrinalysisPreprocessor:

    def __init__(self):

        # =============================
        #  MAPPING ORDINAL & KATEGORIKAL
        # =============================

        self.transparency_map = {
            'CLEAR':0,
            'SLIGHTLY HAZY':1,
            'HAZY':2,
            'CLOUDY':3,
            'TURBID':4
        }

        self.protein_map = {
            'NEGATIVE':0,
            'TRACE':1,
            '1+':2,
            '2+':3,
            '3+':4
        }

        self.glucose_map = {
            'NEGATIVE':0,
            'TRACE':1,
            '1+':2,
            '2+':3,
            '3+':4,
            '4+':5
        }

        self.epith_map = {
            'NONE SEEN':0,
            'RARE':1,
            'FEW':2,
            'OCCASIONAL':3,
            'MODERATE':4,
            'LOADED':5,
            'PLENTY':6
        }

        self.mucous_map = {
            'NONE SEEN':0,
            'RARE':1,
            'FEW':2,
            'OCCASIONAL':3,
            'MODERATE':4,
            'PLENTY':5
        }

        self.amorphous_map = {
            'NONE SEEN':0,
            'RARE':1,
            'FEW':2,
            'OCCASIONAL':3,
            'MODERATE':4,
            'PLENTY':5
        }

        self.bacteria_map = {
            'NONE SEEN':0,
            'RARE':1,
            'FEW':2,
            'OCCASIONAL':3,
            'MODERATE':4,
            'LOADED':5,
            'PLENTY':6
        }

        self.color_map = {
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

        # =============================
        # Column order EXACTLY as training
        # =============================
        self.final_columns = [
            'Age', 'Color', 'Transparency', 'Glucose', 'Protein',
            'pH', 'Specific Gravity', 'WBC', 'RBC',
            'Epithelial Cells', 'Mucous Threads', 'Amorphous Urates',
            'Bacteria', 'Gender_MALE'
        ]

        # scaler dari training (akan ditimpa saat load pkl)
        self.scaler = None

        # WBC & RBC binning
        self.wbc_bins = None
        self.rbc_bins = None
        self.max_wbc = None
        self.max_rbc = None


    # ============================================================
    # Convert nilai WBC/RBC mentah
    # ============================================================
    def convert_temp(self, value):
        value = str(value).strip()
        if ">" in value:
            try:
                return int(value.replace(">", "").strip())
            except:
                return np.nan
        if value.upper() in ["LOADED", "TNTC"]:
            return None
        if "-" in value:
            try:
                s, e = map(int, value.split('-'))
                return s + (e - s) / 2
            except:
                return np.nan
        try:
            return float(value)
        except:
            return np.nan


    # ============================================================
    # Simpan binning values dari df_num (training)
    # ============================================================
    def set_binning(self, df_num):
        self.max_wbc = df_num["WBC"].dropna().max()
        self.max_rbc = df_num["RBC"].dropna().max()
        self.wbc_bins = sorted(df_num["WBC"].astype(float).unique())
        self.rbc_bins = sorted(df_num["RBC"].astype(float).unique())


    # ============================================================
    # Terapkan ordinal binning
    # ============================================================
    def apply_binning(self, val, unique_list, num_bins=13):
        splitted = np.array_split(unique_list, num_bins)
        mapping = {}
        for i, arr in enumerate(splitted, start=1):
            for v in arr:
                mapping[float(v)] = i
        nearest = min(unique_list, key=lambda x: abs(x - float(val)))
        return mapping[float(nearest)]


    # ============================================================
    # Fungsi utama transform input user → DataFrame fitur final
    # ============================================================
    def transform(self, inp):

        # Numeric
        age = inp["Age"]
        ph = inp["pH"]
        sg = inp["Specific Gravity"]

        # WBC
        w = self.convert_temp(inp["WBC"])
        if w is None:
            w = self.max_wbc + 1 if inp["WBC"].upper() == "LOADED" else self.max_wbc + 2
        wbinned = self.apply_binning(w, self.wbc_bins)

        # RBC
        r = self.convert_temp(inp["RBC"])
        if r is None:
            r = self.max_rbc + 1 if inp["RBC"].upper() == "LOADED" else self.max_rbc + 2
        rbinned = self.apply_binning(r, self.rbc_bins)

        # Bangun dict row
        row = {
            "Age": age,
            "Color": self.color_map[inp["Color"]],
            "Transparency": self.transparency_map[inp["Transparency"]],
            "Glucose": self.glucose_map[inp["Glucose"]],
            "Protein": self.protein_map[inp["Protein"]],
            "pH": ph,
            "Specific Gravity": sg,
            "WBC": wbinned,
            "RBC": rbinned,
            "Epithelial Cells": self.epith_map[inp["Epithelial Cells"]],
            "Mucous Threads": self.mucous_map[inp["Mucous Threads"]],
            "Amorphous Urates": self.amorphous_map[inp["Amorphous Urates"]],
            "Bacteria": self.bacteria_map[inp["Bacteria"]],
            "Gender_MALE": 1 if inp["Gender"] == "MALE" else 0
        }

        df = pd.DataFrame([row])

        # ===============================
        # SCALE SEMUA KOLOM SESUAI TRAINING
        # ===============================
        df[self.final_columns] = self.scaler.transform(
            df[self.final_columns]
        )

        # Column order
        df = df[self.final_columns]

        return df
