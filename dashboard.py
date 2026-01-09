import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import kagglehub
from kagglehub import KaggleDatasetAdapter
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Dashboard Analisis Gaya Hidup Mahasiswa",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== MUAT DATA & MODEL ==========
@st.cache_resource
def load_data():
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "akmaldwiputra/student-lifestyle-dataset",
        "student_lifestyle_dataset.csv"
    )
    return df

@st.cache_resource
def load_models():
    """Load pre-trained models dari pkl files"""
    models_dir = 'models'
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        st.error("❌ Folder 'models' tidak ditemukan!")
        st.error("Silakan jalankan notebook terlebih dahulu untuk melatih dan menyimpan models.")
        st.stop()
    
    try:
        # Load models
        rf_reg_model = joblib.load(os.path.join(models_dir, 'rf_reg_model.pkl'))
        rf_clf_model = joblib.load(os.path.join(models_dir, 'rf_clf_model.pkl'))
        le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))
        
        return rf_reg_model, rf_clf_model, le, feature_cols
    
    except FileNotFoundError as e:
        st.error(f"❌ Salah satu file model tidak ditemukan: {str(e)}")
        st.error("Silakan jalankan notebook untuk melatih dan menyimpan models.")
        st.stop()

# Load data dan models
df = load_data()
rf_reg_model, rf_clf_model, le, feature_cols = load_models()

# ========== JUDUL & SIDEBAR ==========
st.title("Dashboard Analisis Gaya Hidup Mahasiswa")
st.markdown("---")

# ========== TAB-TAB ==========
tab1, tab2, tab3 = st.tabs(["EDA & Visualisasi", "Prediksi Model", "Simulator What-If"])

# ========== TAB 1: EDA & VISUALISASI ==========
with tab1:
    st.header("Analisis Data Eksploratori & Visualisasi Interaktif")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scatter Plot: Jam Belajar vs GPA")
        scatter_fig = px.scatter(
            df, 
            x='Study_Hours_Per_Day', 
            y='GPA',
            color='Stress_Level',
            size='Sleep_Hours_Per_Day',
            hover_data=['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'GPA', 'Stress_Level'],
            title='Jam Belajar vs GPA (Ukuran: Jam Tidur)',
            color_discrete_map={'Low': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c'},
            labels={
                'Study_Hours_Per_Day': 'Jam Belajar Per Hari',
                'GPA': 'GPA',
                'Stress_Level': 'Level Stress'
            }
        )
        scatter_fig.update_layout(height=500)
        st.plotly_chart(scatter_fig, use_container_width=True)
    
    with col2:
        st.subheader("Scatter Plot: Jam Tidur vs GPA")
        scatter_fig2 = px.scatter(
            df, 
            x='Sleep_Hours_Per_Day', 
            y='GPA',
            color='Stress_Level',
            size='Study_Hours_Per_Day',
            hover_data=['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'GPA', 'Stress_Level'],
            title='Jam Tidur vs GPA (Ukuran: Jam Belajar)',
            color_discrete_map={'Low': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c'},
            labels={
                'Sleep_Hours_Per_Day': 'Jam Tidur Per Hari',
                'GPA': 'GPA',
                'Stress_Level': 'Level Stress'
            }
        )
        scatter_fig2.update_layout(height=500)
        st.plotly_chart(scatter_fig2, use_container_width=True)
    
    st.subheader("Box Plot: Fitur Berdasarkan Level Stress")
    
    feature_select = st.selectbox(
        "Pilih Fitur untuk Divisualisasikan:",
        ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 
         'Social_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 'GPA']
    )
    
    box_fig = px.box(
        df,
        x='Stress_Level',
        y=feature_select,
        color='Stress_Level',
        points='outliers',
        color_discrete_map={'Low': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c'},
        labels={
            'Stress_Level': 'Level Stress',
            feature_select: feature_select.replace('_', ' ')
        },
        title=f'Distribusi {feature_select.replace("_", " ")} Berdasarkan Level Stress'
    )
    box_fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(box_fig, use_container_width=True)
    
    # Statistik
    st.subheader("Statistik Data Berdasarkan Level Stress")
    stress_stats = df.groupby('Stress_Level')[feature_cols + ['GPA']].agg(['mean', 'std', 'min', 'max'])
    st.dataframe(stress_stats, use_container_width=True)


# ========== TAB 2: PREDIKSI MODEL ==========
with tab2:
    st.header("Prediksi Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Prediksi GPA")
        st.info("Random Forest Regressor")
        
        st.markdown("### Input Data Gaya Hidup Mahasiswa:")
        study_hours = st.slider("Jam Belajar Per Hari", 0.0, 12.0, 5.0, step=0.5)
        sleep_hours = st.slider("Jam Tidur Per Hari", 0.0, 12.0, 7.0, step=0.5)
        physical_activity = st.slider("Jam Aktivitas Fisik Per Hari", 0.0, 10.0, 2.0, step=0.5)
        social_hours = st.slider("Jam Bersosialisasi Per Hari", 0.0, 10.0, 2.0, step=0.5)
        extracurricular = st.slider("Jam Ekstrakurikuler Per Hari", 0.0, 10.0, 2.0, step=0.5)
        
        # Predict GPA
        input_data_reg = np.array([[study_hours, sleep_hours, physical_activity, social_hours, extracurricular]])
        predicted_gpa = rf_reg_model.predict(input_data_reg)[0]
        
        # Display prediction
        st.metric("GPA Terprediksi", f"{predicted_gpa:.2f}", delta=f"(Rentang: 0.0 - 4.0)")
        
        # Add interpretation
        if predicted_gpa >= 3.5:
            st.success("Performa Sangat Baik!")
        elif predicted_gpa >= 3.0:
            st.info("Performa Baik")
        elif predicted_gpa >= 2.5:
            st.warning("Performa Sedang")
        else:
            st.error("Performa Kurang Baik")
    
    with col2:
        st.subheader("Model Prediksi Level Stress")
        st.info("Random Forest Classifier")
        
        st.markdown("### Menggunakan GPA dari Prediksi Sebelumnya")
        
        # Predict Stress Level
        input_data_clf = np.array([[study_hours, sleep_hours, physical_activity, social_hours, extracurricular, predicted_gpa]])
        stress_pred_encoded = rf_clf_model.predict(input_data_clf)[0]
        stress_pred_proba = rf_clf_model.predict_proba(input_data_clf)[0]
        
        stress_level = le.inverse_transform([int(stress_pred_encoded)])[0]
        
        # Display prediction
        st.metric("Level Stress Terprediksi", stress_level)
        
        # Probability for each stress level
        st.markdown("### Kepercayaan Prediksi:")
        for idx, label in enumerate(le.classes_):
            st.write(f"{label}: {stress_pred_proba[idx]:.2%}")
        
        # Add color-coded message
        if stress_level == 'Low':
            st.success("Manajemen stress baik!")
        elif stress_level == 'Moderate':
            st.warning("Stress sedang - pertimbangkan keseimbangan gaya hidup")
        else:
            st.error("Stress tinggi - intervensi mendesak direkomendasikan")
    
    # Perbandingan dengan dataset
    st.markdown("---")
    st.subheader("Perbandingan dengan Distribusi Dataset")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_gpa = df['GPA'].mean()
        st.metric("Rata-rata GPA (Dataset)", f"{avg_gpa:.2f}", delta=f"{predicted_gpa - avg_gpa:+.2f}")
    
    with col2:
        stress_dist = df['Stress_Level'].value_counts()
        st.metric("Level Stress Paling Umum", stress_dist.idxmax(), delta=f"{stress_dist.max()} mahasiswa")
    
    with col3:
        avg_study = df['Study_Hours_Per_Day'].mean()
        st.metric("Rata-rata Jam Belajar", f"{avg_study:.1f}", delta=f"{study_hours - avg_study:+.1f}")
# ========== TAB 3: SIMULATOR WHAT-IF ==========
with tab3:
    st.header("Simulator What-If")
    st.markdown("Simulasikan berbagai skenario gaya hidup dan lihat dampaknya terhadap GPA dan Level Stress!")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pembuat Skenario")
        
        scenario_name = st.text_input("Nama Skenario", "Skenario Saya")
        
        st.markdown("### Sesuaikan Parameter Gaya Hidup:")
        
        slider_col1, slider_col2 = st.columns(2)
        
        with slider_col1:
            sim_study = st.slider("Jam Belajar", 0.0, 12.0, 5.0, step=0.5, key="sim_study")
            sim_activity = st.slider("Aktivitas Fisik", 0.0, 10.0, 2.0, step=0.5, key="sim_activity")
            sim_social = st.slider("Jam Bersosialisasi", 0.0, 10.0, 2.0, step=0.5, key="sim_social")
        
        with slider_col2:
            sim_sleep = st.slider("Jam Tidur", 0.0, 12.0, 7.0, step=0.5, key="sim_sleep")
            sim_extra = st.slider("Jam Ekstrakurikuler", 0.0, 10.0, 2.0, step=0.5, key="sim_extra")
        
        # Predict for scenario
        sim_input_reg = np.array([[sim_study, sim_sleep, sim_activity, sim_social, sim_extra]])
        sim_gpa = rf_reg_model.predict(sim_input_reg)[0]
        
        sim_input_clf = np.array([[sim_study, sim_sleep, sim_activity, sim_social, sim_extra, sim_gpa]])
        sim_stress_encoded = rf_clf_model.predict(sim_input_clf)[0]
        sim_stress = le.inverse_transform([int(sim_stress_encoded)])[0]
    
    with col2:
        st.subheader("Hasil Simulasi")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.metric("GPA Terprediksi", f"{sim_gpa:.2f}")
        
        with result_col2:
            st.metric("Level Stress Terprediksi", sim_stress)
        
        # Buat grafik perbandingan
        comparison_data = pd.DataFrame({
            'Parameter': ['Jam Belajar', 'Jam Tidur', 'Aktivitas Fisik', 'Jam Bersosialisasi', 'Jam Ekstrakurikuler'],
            'Input Anda': [sim_study, sim_sleep, sim_activity, sim_social, sim_extra],
            'Rata-rata Dataset': [
                df['Study_Hours_Per_Day'].mean(),
                df['Sleep_Hours_Per_Day'].mean(),
                df['Physical_Activity_Hours_Per_Day'].mean(),
                df['Social_Hours_Per_Day'].mean(),
                df['Extracurricular_Hours_Per_Day'].mean()
            ]
        })
        
        comparison_fig = px.bar(
            comparison_data,
            x='Parameter',
            y=['Input Anda', 'Rata-rata Dataset'],
            barmode='group',
            title='Skenario Anda vs Rata-rata Dataset',
            color_discrete_map={'Input Anda': '#3498db', 'Rata-rata Dataset': '#95a5a6'}
        )
        st.plotly_chart(comparison_fig, use_container_width=True)
    
    st.markdown("---")
    
    # Perbandingan banyak skenario
    st.subheader("Bandingkan Berbagai Skenario")
    
    # Predefined scenarios
    scenarios = {
        "Mahasiswa Seimbang": {
            'Study_Hours': 5.0, 'Sleep_Hours': 8.0, 'Physical_Activity': 2.0,
            'Social_Hours': 3.0, 'Extracurricular': 2.0
        },
        "Mahasiswa Workaholic": {
            'Study_Hours': 10.0, 'Sleep_Hours': 5.0, 'Physical_Activity': 0.5,
            'Social_Hours': 1.0, 'Extracurricular': 1.0
        },
        "Mahasiswa Fokus Kesehatan": {
            'Study_Hours': 4.0, 'Sleep_Hours': 9.0, 'Physical_Activity': 4.0,
            'Social_Hours': 4.0, 'Extracurricular': 3.0
        },
        "Mahasiswa Rata-rata": {
            'Study_Hours': df['Study_Hours_Per_Day'].mean(),
            'Sleep_Hours': df['Sleep_Hours_Per_Day'].mean(),
            'Physical_Activity': df['Physical_Activity_Hours_Per_Day'].mean(),
            'Social_Hours': df['Social_Hours_Per_Day'].mean(),
            'Extracurricular': df['Extracurricular_Hours_Per_Day'].mean()
        },
        "Skenario Anda": {
            'Study_Hours': sim_study, 'Sleep_Hours': sim_sleep, 'Physical_Activity': sim_activity,
            'Social_Hours': sim_social, 'Extracurricular': sim_extra
        }
    }
    
    # Hitung prediksi untuk semua skenario
    scenario_results = []
    for scenario_name, params in scenarios.items():
        input_reg = np.array([[params['Study_Hours'], params['Sleep_Hours'], params['Physical_Activity'],
                              params['Social_Hours'], params['Extracurricular']]])
        gpa = rf_reg_model.predict(input_reg)[0]
        
        input_clf = np.array([[params['Study_Hours'], params['Sleep_Hours'], params['Physical_Activity'],
                              params['Social_Hours'], params['Extracurricular'], gpa]])
        stress_enc = rf_clf_model.predict(input_clf)[0]
        stress = le.inverse_transform([int(stress_enc)])[0]
        
        scenario_results.append({
            'Skenario': scenario_name,
            'Jam Belajar': params['Study_Hours'],
            'Jam Tidur': params['Sleep_Hours'],
            'GPA': gpa,
            'Level Stress': stress
        })
    
    results_df = pd.DataFrame(scenario_results)
    st.dataframe(results_df, use_container_width=True)
    
    # Visualisasi: GPA vs Jam Belajar untuk semua skenario
    scenario_viz = px.scatter(
        results_df,
        x='GPA',
        y='Jam Belajar',
        color='Level Stress',
        size='Jam Tidur',
        hover_data=['Skenario', 'GPA', 'Jam Belajar', 'Jam Tidur', 'Level Stress'],
        title='Skenario: GPA vs Jam Belajar (Ukuran: Jam Tidur)',
        color_discrete_map={'Low': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c'},
        text='Skenario'
    )
    scenario_viz.update_traces(textposition='top center')
    st.plotly_chart(scenario_viz, use_container_width=True)
    
    # Rekomendasi
    st.markdown("---")
    st.subheader("Rekomendasi")
    
    if sim_gpa >= 3.5 and sim_stress == 'Low':
        st.success("Skenario sangat baik! Anda mencapai GPA tinggi dengan stress rendah. Ini ideal!")
    elif sim_gpa >= 3.5 and sim_stress in ['Moderate', 'High']:
        st.warning("GPA baik tapi stress tinggi. Pertimbangkan mengurangi beban kerja atau meningkatkan jam tidur.")
    elif sim_stress == 'High':
        st.error("Stress tinggi terdeteksi. Rekomendasi:\n- Tingkatkan jam tidur\n- Kurangi jam belajar atau tingkatkan efisiensi\n- Tingkatkan aktivitas sosial/fisik untuk mengurangi stress")
    elif sim_sleep < 6:
        st.warning("Jam tidur terlalu rendah! Tidur cukup sangat penting untuk GPA dan manajemen stress.")
    elif sim_activity < 1:
        st.info("Pertimbangkan meningkatkan aktivitas fisik untuk kesehatan mental dan manajemen stress yang lebih baik.")
    else:
        st.success("Skenario Anda terlihat seimbang! Lanjutkan menjaga gaya hidup ini.")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: #7f8c8d;'>
        Dashboard Analisis Gaya Hidup Mahasiswa | Wawasan Berbasis Data untuk Kesuksesan Akademik
    </p>
</div>
""", unsafe_allow_html=True)
