# EduTrack - Student Lifestyle Analytics Dashboard

Analisis data gaya hidup mahasiswa untuk memprediksi GPA dan stress level menggunakan machine learning.

**Dataset**: 2000 mahasiswa, 8 fitur lifestyle dari Kaggle  
**Models**: Regresi (GPA) + Klasifikasi (Stress Level)  
**Framework**: Streamlit Dashboard + Jupyter Notebook

[Fitur](#fitur) • [Instalasi](#instalasi-cepat) • [Penggunaan](#penggunaan) • [Struktur](#struktur-project)

---

## Fitur

- **Dashboard Interaktif**: 3 tabs (Overview, Analyzer, What If)
- **Model Regresi**: Random Forest untuk prediksi GPA (R² 0.5165)
- **Model Klasifikasi**: Logistic Regression untuk prediksi stress level (Akurasi 84.75%)
- **Hyperparameter Tuning**: RandomizedSearchCV & GridSearchCV
- **EDA Lengkap**: Visualisasi data, correlation heatmap, feature importance

---

## Instalasi Cepat

### 1. Setup Environment

**Dengan Anaconda (Recommended):**
```bash
conda create -n edutrack python=3.10
conda activate edutrack
```

**Atau dengan venv:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Kaggle API (Opsional)
```bash
# Download kaggle.json dari https://www.kaggle.com/settings/account
# Simpan di: C:\Users\<username>\.kaggle\kaggle.json (Windows)
# atau ~/.kaggle/kaggle.json (Mac/Linux)
```

---

## Penggunaan

### Training Model (Notebook)
```bash
conda activate edutrack  # Jika pakai Anaconda
jupyter notebook
# Buka notebook.ipynb dan jalankan semua cells
```

**Output**: Model akan di-save di folder `models/`

### Jalankan Dashboard
```bash
conda activate edutrack  # Jika pakai Anaconda
streamlit run dashboard.py
# Akses di http://localhost:8501
```

---

## Struktur Project

```
edutrack_student_lifestyle_analytic/
├── README.md                 # Dokumentasi ini
├── requirements.txt          # Dependencies
├── notebook.ipynb           # EDA & Training Model
├── dashboard.py             # Dashboard Streamlit
└── models/                  # Folder trained models
    ├── rf_reg_model.pkl
    ├── rf_clf_model.pkl
    ├── label_encoder.pkl
    ├── feature_cols.pkl
    └── model_metadata.pkl
```

---

## Dataset & Models

### Input Features (5)
- Study Hours Per Day
- Sleep Hours Per Day
- Physical Activity Hours Per Day
- Social Hours Per Day
- Extracurricular Hours Per Day

### Model 1: Prediksi GPA (Regresi)
- **Algorithm**: Random Forest Regressor (Tuned)
- **RMSE**: 0.2125
- **R² Score**: 0.5165
- **Feature Paling Penting**: Study Hours (75.95%)

### Model 2: Prediksi Stress Level (Klasifikasi)
- **Algorithm**: Logistic Regression
- **Accuracy**: 84.75% (Test Set)
- **Classes**: Low, Moderate, High
- **CV Method**: 5-fold cross-validation

---

## Dashboard Tabs

| Tab | Fungsi |
|-----|--------|
| **Overview** | Ringkasan data, distribusi GPA, stress level |
| **Analyzer** | Heatmap korelasi, box plots, statistik |
| **What If** | Simulator skenario & prediksi personal |

---

## Quick Tips

- Pastikan folder `models/` ada sebelum jalankan dashboard
- Jalankan notebook dulu untuk training & save models
- Untuk setup kaggle API, ikuti step di instalasi
- Port default dashboard: 8501 (ubah dengan `--server.port 8502`)

---

## Troubleshooting

**Error: "models folder not found"**
→ Jalankan notebook.ipynb terlebih dahulu

**Error: "ModuleNotFoundError"**
→ Install requirements: `pip install -r requirements.txt`

**Error: "kagglehub dataset_load failed"**
→ Setup kaggle.json di folder .kaggle

**Dashboard tidak terbuka**
→ Cek port 8501 tidak terpakai, atau ubah dengan `--server.port`

---

## Requirements

- Python 3.14
- Anaconda (recommended) atau venv
- RAM minimal 2GB
- Internet (untuk download dataset dari Kaggle)

---

## File Penting

- `notebook.ipynb` - EDA, model training, dan model saving
- `dashboard.py` - Streamlit dashboard dengan 3 tabs
- `requirements.txt` - Semua dependencies
- `models/` - Pre-trained models (dibuat saat menjalankan notebook)

---

## Lisensi

MIT License - Lihat file LICENSE untuk detail

---

## Author

Created for Dicoding Bootcamp Final Project - January 2026

Dataset: [Kaggle Student Lifestyle Dataset](https://www.kaggle.com/datasets/akmaldwiputra/student-lifestyle-dataset)
