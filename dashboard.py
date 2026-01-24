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
        
        # Load metadata jika ada
        metadata = None
        if os.path.exists(os.path.join(models_dir, 'model_metadata.pkl')):
            metadata = joblib.load(os.path.join(models_dir, 'model_metadata.pkl'))
        
        return rf_reg_model, rf_clf_model, le, feature_cols, metadata
    
    except FileNotFoundError as e:
        st.error(f"❌ Salah satu file model tidak ditemukan: {str(e)}")
        st.error("Silakan jalankan notebook untuk melatih dan menyimpan models.")
        st.stop()

# Load data dan models
df = load_data()
rf_reg_model, rf_clf_model, le, feature_cols, metadata = load_models()

# ========== FRIENDLY VARIABLE NAMES MAPPING ==========
FEATURE_NAMES = {
    'Study_Hours_Per_Day': 'Jam Belajar Per Hari',
    'Sleep_Hours_Per_Day': 'Jam Tidur Per Hari',
    'Physical_Activity_Hours_Per_Day': 'Jam Aktivitas Fisik Per Hari',
    'Social_Hours_Per_Day': 'Jam Bersosialisasi Per Hari',
    'Extracurricular_Hours_Per_Day': 'Jam Ekstrakurikuler Per Hari',
    'GPA': 'GPA',
    'Stress_Level': 'Level Stress'
}

STRESS_COLORS = {
    'Low': '#2ecc71',      # Hijau - Stress Rendah
    'Moderate': '#f39c12',  # Oranye - Stress Sedang
    'High': '#e74c3c'       # Merah - Stress Tinggi
}

# ========== JUDUL & SIDEBAR ==========
st.title("Dashboard Analisis Gaya Hidup Mahasiswa")
st.markdown("---")

# Load metadata untuk info model
if metadata:
    model_info = f"Model Classifier: {metadata.get('best_classifier_model', 'N/A')} | Akurasi: {metadata.get('classifier_accuracy', 'N/A'):.2%}"
else:
    model_info = "Model Information: Available"

st.caption(f"ℹ️ {model_info}")

# ========== TAB-TAB ==========
tab1, tab2, tab3 = st.tabs(["Overview", "Analyzer", "What If"])

# ========== TAB 1: OVERVIEW ==========
with tab1:
    st.header("Overview Data Gaya Hidup Mahasiswa")
    
    # Model Performance Info
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.info("Model Regresi: Random Forest | Target: GPA")
    
    with col_info2:
        if metadata:
            st.success(f"Model Terpilih: {metadata.get('best_classifier_model', 'N/A')} | Akurasi: {metadata.get('classifier_accuracy', 'N/A'):.2%}")
        else:
            st.warning("Model metadata tidak tersedia")
    
    with col_info3:
        if metadata and 'all_models_comparison' in metadata:
            st.info(f"Total Model Dikomparasi: {len(metadata['all_models_comparison'])}")
    
    # Show detailed model comparison
    if metadata and 'all_models_comparison' in metadata:
        with st.expander("Perbandingan Semua Model Klasifikasi (Klik untuk expand)", expanded=False):
            comparison_df = pd.DataFrame([
                {'Model': model_name, 'CV Akurasi': f"{scores['cv']:.4f}", 'Val Akurasi': f"{scores['val']:.4f}", 'Test Akurasi': f"{scores['test']:.4f} ({scores['test']:.2%})"}
                for model_name, scores in sorted(metadata['all_models_comparison'].items(), key=lambda x: x[1]['test'], reverse=True)
            ])
            
            st.markdown("### Hasil Perbandingan 4 Model dengan Cross-Validation 5-Fold:")
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            st.markdown(f"""
            **Model Terbaik: {metadata.get('best_classifier_model', 'N/A')}**
            
            Model ini dipilih berdasarkan akurasi tertinggi dari perbandingan:
            - Logistic Regression dengan 5-Fold CV
            - Random Forest Classifier dengan 5-Fold CV
            - XGBoost dengan GridSearchCV (54 kombinasi parameter)
            - XGBoost dengan RandomizedSearchCV (20 iterasi random)
            
            """)
    
    st.markdown("---")
    
    # Summary Statistics Cards
    st.subheader("Ringkasan Data")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("Jumlah Mahasiswa", len(df))
    
    with col_stat2:
        st.metric("Rata-rata GPA", f"{df['GPA'].mean():.2f}")
    
    with col_stat3:
        stress_counts = df['Stress_Level'].value_counts()
        st.metric("Stress Rendah", stress_counts.get('Low', 0))
    
    with col_stat4:
        st.metric("Rata-rata Jam Tidur", f"{df['Sleep_Hours_Per_Day'].mean():.1f} jam")
    
    st.markdown("---")
    
    # Distribution Charts
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        st.subheader("Distribusi GPA")
        gpa_hist = px.histogram(
            df,
            x='GPA',
            nbins=20,
            title='Histogram GPA',
            color_discrete_sequence=['#3498db']
        )
        gpa_hist.update_layout(height=400, showlegend=False)
        st.plotly_chart(gpa_hist, use_container_width=True)
    
    with col_dist2:
        st.subheader("Distribusi Level Stress")
        stress_dist = df['Stress_Level'].value_counts()
        stress_pie = px.pie(
            values=stress_dist.values,
            names=stress_dist.index,
            title='Proporsi Level Stress',
            color_discrete_map=STRESS_COLORS
        )
        stress_pie.update_layout(height=400)
        st.plotly_chart(stress_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Scatter Plots
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
            color_discrete_map=STRESS_COLORS,
            labels={
                'Study_Hours_Per_Day': FEATURE_NAMES['Study_Hours_Per_Day'],
                'GPA': FEATURE_NAMES['GPA'],
                'Stress_Level': FEATURE_NAMES['Stress_Level'],
                'Sleep_Hours_Per_Day': FEATURE_NAMES['Sleep_Hours_Per_Day']
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
            color_discrete_map=STRESS_COLORS,
            labels={
                'Sleep_Hours_Per_Day': FEATURE_NAMES['Sleep_Hours_Per_Day'],
                'GPA': FEATURE_NAMES['GPA'],
                'Stress_Level': FEATURE_NAMES['Stress_Level'],
                'Study_Hours_Per_Day': FEATURE_NAMES['Study_Hours_Per_Day']
            }
        )
        scatter_fig2.update_layout(height=500)
        st.plotly_chart(scatter_fig2, use_container_width=True)


# ========== TAB 2: ANALYZER ==========
with tab2:
    st.header("Analisa Mendalam Fitur Data")
    st.markdown("Analisis distribusi fitur berdasarkan level stress")
    
    st.markdown("---")
    
    # Correlation Heatmap
    st.subheader("Correlation Matrix - Hubungan Antar Fitur")
    
    # Prepare data for correlation (exclude categorical columns)
    numeric_cols = feature_cols + ['GPA']
    corr_matrix = df[numeric_cols].corr()
    
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[FEATURE_NAMES.get(col, col) for col in corr_matrix.columns],
        y=[FEATURE_NAMES.get(col, col) for col in corr_matrix.columns],
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Korelasi")
    ))
    
    heatmap_fig.update_layout(
        title='Matriks Korelasi Fitur-Fitur',
        height=500,
        width=800
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========== EDA & VISUALISASI ==========
    st.subheader("Box Plot: Fitur Berdasarkan Level Stress")
    
    feature_select = st.selectbox(
        "Pilih Fitur untuk Divisualisasikan:",
        {
            'Jam Belajar Per Hari': 'Study_Hours_Per_Day',
            'Jam Tidur Per Hari': 'Sleep_Hours_Per_Day',
            'Jam Aktivitas Fisik Per Hari': 'Physical_Activity_Hours_Per_Day',
            'Jam Bersosialisasi Per Hari': 'Social_Hours_Per_Day',
            'Jam Ekstrakurikuler Per Hari': 'Extracurricular_Hours_Per_Day',
            'GPA': 'GPA'
        }
    )
    feature_select_actual = {
        'Jam Belajar Per Hari': 'Study_Hours_Per_Day',
        'Jam Tidur Per Hari': 'Sleep_Hours_Per_Day',
        'Jam Aktivitas Fisik Per Hari': 'Physical_Activity_Hours_Per_Day',
        'Jam Bersosialisasi Per Hari': 'Social_Hours_Per_Day',
        'Jam Ekstrakurikuler Per Hari': 'Extracurricular_Hours_Per_Day',
        'GPA': 'GPA'
    }[feature_select]
    
    box_fig = px.box(
        df,
        x='Stress_Level',
        y=feature_select_actual,
        color='Stress_Level',
        points='outliers',
        color_discrete_map=STRESS_COLORS,
        labels={
            'Stress_Level': FEATURE_NAMES['Stress_Level'],
            feature_select_actual: feature_select
        },
        title=f'Distribusi {feature_select} Berdasarkan Level Stress'
    )
    box_fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(box_fig, use_container_width=True)
    
    # Statistik
    st.subheader("Statistik Data Berdasarkan Level Stress")
    stress_stats = df.groupby('Stress_Level')[feature_cols + ['GPA']].agg(['mean', 'std', 'min', 'max'])
    st.dataframe(stress_stats, use_container_width=True)


# ========== TAB 3: WHAT IF ==========
with tab3:
    st.header("What If - Simulator Skenario")
    st.markdown("Bandingkan berbagai skenario gaya hidup dan lihat dampaknya terhadap GPA dan Level Stress!")
    
    st.markdown("---")
    
    # Custom Scenario
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Buat Skenario Anda:")
        scenario_name = st.text_input("Nama Skenario", "Skenario Saya")
        
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            sim_study = st.number_input("Jam Belajar (jam)", min_value=0.0, max_value=24.0, value=0.0, step=0.5, key="sim_study")
            sim_activity = st.number_input("Aktivitas Fisik (jam)", min_value=0.0, max_value=24.0, value=0.0, step=0.5, key="sim_activity")
            sim_social = st.number_input("Jam Bersosialisasi (jam)", min_value=0.0, max_value=24.0, value=0.0, step=0.5, key="sim_social")
        
        with input_col2:
            sim_sleep = st.number_input("Jam Tidur (jam)", min_value=0.0, max_value=24.0, value=0.0, step=0.5, key="sim_sleep")
            sim_extra = st.number_input("Jam Ekstrakurikuler (jam)", min_value=0.0, max_value=24.0, value=0.0, step=0.5, key="sim_extra")
        
        # Hitung total jam
        total_hours = sim_study + sim_sleep + sim_activity + sim_social + sim_extra
        
        st.markdown("---")
        
        # Display total jam dengan warna berdasarkan validitas
        if total_hours < 24:
            st.warning(f"Total harus tepat 24 jam per hari. Sisa: {24 - total_hours:.1f} jam")
            button_disabled = True
        elif total_hours > 24:
            st.error(f"⚠️ Total Jam: {total_hours:.1f} jam (Melebihi 24 jam!)")
            st.error(f"Total harus tepat 24 jam per hari. Kurangi: {total_hours - 24:.1f} jam")
            button_disabled = True
        else:
            st.success(f"✅ Total Jam: {total_hours:.1f} jam (Tepat 24 jam - Valid!)")
            button_disabled = False
        
        st.markdown("---")
        
        # Button untuk proses prediksi
        if st.button("🚀 Proses Prediksi Skenario", use_container_width=True, disabled=button_disabled):
            # Predict for custom scenario
            sim_input_reg = np.array([[sim_study, sim_sleep, sim_activity, sim_social, sim_extra]])
            sim_gpa = rf_reg_model.predict(sim_input_reg)[0]
            
            # Classifier menggunakan hanya 5 lifestyle features (tanpa GPA untuk menghindari data leakage)
            sim_input_clf = np.array([[sim_study, sim_sleep, sim_activity, sim_social, sim_extra]])
            sim_stress_encoded = rf_clf_model.predict(sim_input_clf)[0]
            sim_stress = le.inverse_transform([int(sim_stress_encoded)])[0]
            
            # Simpan hasil ke session state
            st.session_state.sim_gpa = sim_gpa
            st.session_state.sim_stress = sim_stress
            st.session_state.scenario_processed = True
            st.success("✅ Prediksi selesai! Lihat hasil di sebelah kanan.")
    
    with col2:
        st.markdown("### Hasil Skenario Anda:")
        
        # Tampilkan hasil hanya jika sudah diproses
        if 'scenario_processed' in st.session_state and st.session_state.scenario_processed:
            sim_gpa = st.session_state.sim_gpa
            sim_stress = st.session_state.sim_stress
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.metric("GPA Terprediksi", f"{sim_gpa:.2f}")
            
            with result_col2:
                st.metric("Level Stress Terprediksi", sim_stress)
            
            st.markdown("---")
            
            # Grafik perbandingan
            comparison_data = pd.DataFrame({
                'Parameter': [
                    FEATURE_NAMES['Study_Hours_Per_Day'],
                    FEATURE_NAMES['Sleep_Hours_Per_Day'],
                    FEATURE_NAMES['Physical_Activity_Hours_Per_Day'],
                    FEATURE_NAMES['Social_Hours_Per_Day'],
                    FEATURE_NAMES['Extracurricular_Hours_Per_Day']
                ],
                'Skenario Anda': [sim_study, sim_sleep, sim_activity, sim_social, sim_extra],
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
                y=['Skenario Anda', 'Rata-rata Dataset'],
                barmode='group',
                title='Skenario Anda vs Rata-rata Dataset',
                color_discrete_map={'Skenario Anda': '#3498db', 'Rata-rata Dataset': '#95a5a6'}
            )
            comparison_fig.update_layout(
                xaxis_title="Jenis Aktivitas",
                yaxis_title="Jumlah Jam",
                height=400
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
        else:
            st.info("📝 Isi form di sebelah kiri dan klik tombol 'Proses Prediksi Skenario' untuk melihat hasil")
    
    st.markdown("---")
    
    # Predefined Scenarios Comparison
    st.subheader("Perbandingan Skenario Terdefinisi")
    
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
        "Mahasiswa Rata-rata (Dataset)": {
            'Study_Hours': df['Study_Hours_Per_Day'].mean(),
            'Sleep_Hours': df['Sleep_Hours_Per_Day'].mean(),
            'Physical_Activity': df['Physical_Activity_Hours_Per_Day'].mean(),
            'Social_Hours': df['Social_Hours_Per_Day'].mean(),
            'Extracurricular': df['Extracurricular_Hours_Per_Day'].mean()
        }
    }
    
    # Tambahkan skenario custom hanya jika sudah diproses
    if 'scenario_processed' in st.session_state and st.session_state.scenario_processed:
        scenarios[scenario_name] = {
            'Study_Hours': sim_study, 'Sleep_Hours': sim_sleep, 'Physical_Activity': sim_activity,
            'Social_Hours': sim_social, 'Extracurricular': sim_extra
        }
    
    # Hitung prediksi untuk semua skenario
    scenario_results = []
    for scen_name, params in scenarios.items():
        input_reg = np.array([[params['Study_Hours'], params['Sleep_Hours'], params['Physical_Activity'],
                              params['Social_Hours'], params['Extracurricular']]])
        gpa = rf_reg_model.predict(input_reg)[0]
        
        # Classifier menggunakan hanya 5 lifestyle features (tanpa GPA untuk menghindari data leakage)
        input_clf = np.array([[params['Study_Hours'], params['Sleep_Hours'], params['Physical_Activity'],
                              params['Social_Hours'], params['Extracurricular']]])
        stress_enc = rf_clf_model.predict(input_clf)[0]
        stress = le.inverse_transform([int(stress_enc)])[0]
        
        scenario_results.append({
            'Skenario': scen_name,
            FEATURE_NAMES['Study_Hours_Per_Day']: params['Study_Hours'],
            FEATURE_NAMES['Sleep_Hours_Per_Day']: params['Sleep_Hours'],
            'GPA': gpa,
            'Level Stress': stress
        })
    
    results_df = pd.DataFrame(scenario_results)
    st.dataframe(results_df, use_container_width=True)
    
    # Visualisasi Scatter
    scenario_viz = px.scatter(
        results_df,
        x='GPA',
        y=FEATURE_NAMES['Study_Hours_Per_Day'],
        color='Level Stress',
        size=FEATURE_NAMES['Sleep_Hours_Per_Day'],
        hover_data=['Skenario', 'GPA', FEATURE_NAMES['Study_Hours_Per_Day'], FEATURE_NAMES['Sleep_Hours_Per_Day'], 'Level Stress'],
        title='Perbandingan Skenario: GPA vs Jam Belajar (Ukuran: Jam Tidur)',
        color_discrete_map=STRESS_COLORS,
        text='Skenario'
    )
    scenario_viz.update_traces(textposition='top center')
    st.plotly_chart(scenario_viz, use_container_width=True)
    
    # Rekomendasi
    st.markdown("---")
    st.subheader("Rekomendasi Berdasarkan Skenario Anda")
    
    if 'scenario_processed' in st.session_state and st.session_state.scenario_processed:
        sim_gpa = st.session_state.sim_gpa
        sim_stress = st.session_state.sim_stress
        
        if sim_gpa >= 3.5 and sim_stress == 'Low':
            st.success("Skenario sangat baik! Anda mencapai GPA tinggi dengan stress rendah. Ini adalah kondisi ideal!")
        elif sim_gpa >= 3.5 and sim_stress in ['Moderate', 'High']:
            st.warning("GPA baik tetapi stress tinggi. Rekomendasi:\n- Coba tingkatkan jam tidur\n- Pertimbangkan mengurangi beban ekstrakurikuler\n- Perbanyak waktu untuk relaksasi")
        elif sim_stress == 'High':
            st.error("Stress tinggi terdeteksi. Rekomendasi:\n- **Prioritaskan tidur cukup** (7-9 jam per hari)\n- Kurangi beban belajar atau gunakan teknik belajar yang lebih efisien\n- Tingkatkan aktivitas fisik dan sosial untuk mengurangi stress")
        elif sim_sleep < 6:
            st.warning("Perhatian! Jam tidur terlalu rendah. Tidur cukup sangat penting untuk:\n- Meningkatkan performa akademik (GPA)\n- Mengurangi tingkat stress\n- Menjaga kesehatan mental dan fisik")
        elif sim_activity < 1:
            st.info("Saran: Tingkatkan aktivitas fisik ke minimal 1-2 jam per hari.\nActivitas fisik terbukti membantu:\n- Meningkatkan kesehatan mental\n- Mengurangi stress\n- Meningkatkan fokus belajar")
        else:
            st.success("Skenario Anda terlihat seimbang dan sehat! Lanjutkan menjaga gaya hidup ini.")
    else:
        st.info("📝 Isi form di sebelah kiri dan klik tombol 'Proses Prediksi Skenario' untuk melihat rekomendasi")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: #7f8c8d;'>
        Dashboard Analisis Gaya Hidup Mahasiswa | Wawasan Berbasis Data untuk Kesuksesan Akademik
    </p>
</div>
""", unsafe_allow_html=True)
