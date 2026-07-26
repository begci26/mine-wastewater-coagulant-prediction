import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from flask import Blueprint, render_template, redirect, url_for, flash, jsonify, send_from_directory
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import Config
from utils.logger import app_logger
from utils.helpers import check_dataset_uploaded, get_dataset_path, REQUIRED_TARGET

# Inisialisasi blueprint
eda_bp = Blueprint('eda', __name__)

def get_processed_data():
    """Membaca dataset hasil prapemrosesan (df_active.csv)."""
    processed_dir = os.path.join(Config.UPLOAD_FOLDER, 'processed')
    df_active_path = os.path.join(processed_dir, 'df_active.csv')
    if os.path.exists(df_active_path):
        try:
            return pd.read_csv(df_active_path)
        except Exception as error:
            app_logger.error(f"Gagal memuat df_active.csv di EDA: {error}")
    return None

def calculate_vif(df_features):
    """Menghitung Variance Inflation Factor (VIF) untuk setiap fitur numerik."""
    vif_dict = {}
    cols = df_features.select_dtypes(include=['number']).columns.tolist()
    
    # Hapus kolom non-fitur jika ada
    for col_to_remove in ['No', REQUIRED_TARGET]:
        if col_to_remove in cols:
            cols.remove(col_to_remove)
            
    df_clean = df_features[cols].dropna()
    
    for col in cols:
        y = df_clean[col]
        X = df_clean.drop(columns=[col])
        if X.shape[1] == 0:
            vif_dict[col] = 1.0
            continue
        try:
            lr = LinearRegression()
            lr.fit(X, y)
            r2 = lr.score(X, y)
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
            vif_dict[col] = round(float(vif), 4)
        except Exception as e:
            app_logger.error(f"Gagal menghitung VIF untuk {col}: {e}")
            vif_dict[col] = 999.0
            
    return vif_dict

def generate_bab4_outputs(df):
    """Menghasilkan 10 grafik PNG resolusi tinggi dan 1 file Excel ringkasan statistik ke output/bab4/."""
    try:
        output_dir = os.path.join(Config.UPLOAD_FOLDER, 'output', 'bab4')
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Excel Ringkasan Statistik
        desc_df = df.describe(include=[np.number])
        excel_path = os.path.join(output_dir, 'summary_statistic.xlsx')
        desc_df.to_excel(excel_path, engine='openpyxl')
        
        # Setup style global matplotlib
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['figure.dpi'] = 300
        
        # 2. histogram_cl80.png
        plt.figure(figsize=(6, 4))
        plt.hist(df[REQUIRED_TARGET].dropna(), bins=20, color='#10b981', edgecolor='black', alpha=0.7)
        plt.title('Distribusi Target Dosis CL-80', fontsize=12, fontweight='bold')
        plt.xlabel('Dosis CL-80 (ppm)', fontsize=10)
        plt.ylabel('Frekuensi', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'histogram_cl80.png'), dpi=300)
        plt.close()
        
        # 3. boxplot_tss.png
        plt.figure(figsize=(6, 4))
        if 'Inlet TSS (mg/L)' in df.columns:
            plt.boxplot(df['Inlet TSS (mg/L)'].dropna(), patch_artist=True,
                        boxprops=dict(facecolor='#3b82f6', color='#2563eb'),
                        medianprops=dict(color='red'))
            plt.title('Boxplot Inlet TSS', fontsize=12, fontweight='bold')
            plt.ylabel('Inlet TSS (mg/L)', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'boxplot_tss.png'), dpi=300)
        plt.close()
        
        # 4. heatmap.png
        numeric_cols = [c for c in df.columns if c not in ['No', 'Date', 'Time']]
        numeric_cols = df[numeric_cols].select_dtypes(include=['number']).columns.tolist()
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(numeric_cols)))
        ax.set_yticks(np.arange(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(numeric_cols, fontsize=8)
        fig.colorbar(im, ax=ax, label='Koefisien Korelasi')
        plt.title('Heatmap Korelasi Pearson', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmap.png'), dpi=300)
        plt.close()
        
        # 5. korelasi.png
        plt.figure(figsize=(8, 4))
        target_corr = corr_matrix[REQUIRED_TARGET].drop(REQUIRED_TARGET).sort_values(ascending=False)
        target_corr.plot(kind='bar', color='#3b82f6', edgecolor='black')
        plt.title('Korelasi Fitur terhadap Target Dosis CL-80', fontsize=12, fontweight='bold')
        plt.ylabel('Koefisien Korelasi Pearson', fontsize=10)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'korelasi.png'), dpi=300)
        plt.close()
        
        # 6. trend_cl80.png
        plt.figure(figsize=(10, 4))
        plt.plot(df[REQUIRED_TARGET], color='#10b981', label='Dosis CL-80')
        plt.title('Trend Time Series Dosis CL-80', fontsize=12, fontweight='bold')
        plt.xlabel('Sampel (Urutan Waktu)', fontsize=10)
        plt.ylabel('CL-80 (ppm)', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trend_cl80.png'), dpi=300)
        plt.close()
        
        # 7. trend_tss.png
        plt.figure(figsize=(10, 4))
        if 'Inlet TSS (mg/L)' in df.columns:
            plt.plot(df['Inlet TSS (mg/L)'], color='#3b82f6', label='Inlet TSS')
        if 'Outlet TSS (mg/L)' in df.columns:
            plt.plot(df['Outlet TSS (mg/L)'], color='#ef4444', label='Outlet TSS')
        plt.title('Trend Time Series TSS', fontsize=12, fontweight='bold')
        plt.xlabel('Sampel', fontsize=10)
        plt.ylabel('TSS (mg/L)', fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trend_tss.png'), dpi=300)
        plt.close()
        
        # 8. trend_ph.png
        plt.figure(figsize=(10, 4))
        if 'Inlet pH' in df.columns:
            plt.plot(df['Inlet pH'], color='#f59e0b', label='Inlet pH')
        if 'Outlet pH' in df.columns:
            plt.plot(df['Outlet pH'], color='#8b5cf6', label='Outlet pH')
        plt.title('Trend Time Series pH', fontsize=12, fontweight='bold')
        plt.xlabel('Sampel', fontsize=10)
        plt.ylabel('pH', fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trend_ph.png'), dpi=300)
        plt.close()
        
        # 9. trend_debit.png
        plt.figure(figsize=(10, 4))
        if 'Inlet Disch (m3/s)' in df.columns:
            plt.plot(df['Inlet Disch (m3/s)'], color='#06b6d4', label='Debit')
        plt.title('Trend Time Series Debit Aliran', fontsize=12, fontweight='bold')
        plt.xlabel('Sampel', fontsize=10)
        plt.ylabel('Debit (m³/s)', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trend_debit.png'), dpi=300)
        plt.close()
        
        # 10. vif.png
        vifs = calculate_vif(df)
        plt.figure(figsize=(8, 4))
        vif_series = pd.Series(vifs).sort_values(ascending=False)
        vif_series.plot(kind='bar', color='#ef4444', edgecolor='black')
        plt.axhline(10, color='red', linestyle='--', label='Ambang Batas Kolilinearitas (>10)')
        plt.title('Nilai Variance Inflation Factor (VIF) Fitur', fontsize=12, fontweight='bold')
        plt.ylabel('Nilai VIF', fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vif.png'), dpi=300)
        plt.close()
        
        # 11. target_distribution.png
        plt.figure(figsize=(6, 4))
        sns_color = '#10b981'
        df[REQUIRED_TARGET].plot(kind='kde', color=sns_color, linewidth=2)
        plt.title('Estimasi Densitas Kernel (KDE) Target Dosis CL-80', fontsize=12, fontweight='bold')
        plt.xlabel('Dosis CL-80 (ppm)', fontsize=10)
        plt.ylabel('Densitas', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=300)
        plt.close()
        
        app_logger.info("BAB IV Tesis outputs (10 PNGs + 1 Excel) successfully generated and saved.")
    except Exception as e:
        app_logger.error(f"Gagal memproses otomatis grafik BAB IV: {e}", exc_info=True)

@eda_bp.route('/')
def index():
    """Merender halaman dashboard visualisasi Eksplorasi Data (EDA)."""
    if not check_dataset_uploaded():
        flash("Silakan unggah dataset terlebih dahulu.", "warning")
        return redirect(url_for('dataset.index'))
        
    df = get_processed_data()
    if df is None:
        flash("Silakan lakukan prapemrosesan data terlebih dahulu agar mendapatkan fitur lengkap.", "warning")
        return redirect(url_for('preprocessing.index'))
        
    # Picu pembuatan output BAB IV
    generate_bab4_outputs(df)
    
    # 1. Ringkasan Fitur
    summary_cards = {
        'rows': len(df),
        'cols': len(df.columns),
        'missing': int(df.isnull().sum().sum()),
        'duplicates': int(df.duplicated().sum()),
        'target': REQUIRED_TARGET
    }
    
    # Rentang Tanggal
    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date']).dropna()
        if not dates.empty:
            summary_cards['date_range'] = f"{dates.min().strftime('%d-%m-%Y')} s.d. {dates.max().strftime('%d-%m-%Y')}"
        else:
            summary_cards['date_range'] = "Tidak ada tanggal"
    else:
        summary_cards['date_range'] = "Kolom tanggal tidak ada"
        
    # Hitung outlier total harian
    from ml.preprocessor import WastewaterPreprocessor
    preprocessor = WastewaterPreprocessor()
    _, outlier_idx = preprocessor.detect_outliers_iqr(df)
    summary_cards['outliers'] = len(outlier_idx)
    
    # Hitung VIF & Korelasi
    vif_dict = calculate_vif(df)
    
    # Korelasi tabel
    numeric_cols = [c for c in df.columns if c not in ['No', 'Date', 'Time']]
    numeric_cols = df[numeric_cols].select_dtypes(include=['number']).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    
    # Korelasi ke target CL-80
    target_corr = corr_matrix[REQUIRED_TARGET].drop(REQUIRED_TARGET).to_dict()
    
    # Bangun daftar fitur
    features_list = [c for c in df.columns if c not in ['No', 'Date', 'Time', REQUIRED_TARGET]]
    
    # VIF Status & Rekomendasi
    vif_table = []
    for feat in features_list:
        v_val = vif_dict.get(feat, 1.0)
        c_val = target_corr.get(feat, 0.0)
        
        status = "Aman (VIF <= 10)"
        recommendation = "Fitur dipertahankan"
        
        if v_val > 10.0:
            status = "Multikolinearitas Tinggi (>10)"
            if abs(c_val) < 0.2:
                recommendation = "Dipertimbangkan untuk dihapus (VIF Tinggi, Korelasi Rendah)"
            else:
                recommendation = "Fitur dipertahankan untuk pemodelan non-linier"
        else:
            if abs(c_val) < 0.1:
                recommendation = "Dipertimbangkan untuk dihapus (Korelasi Sangat Lemah)"
                
        vif_table.append({
            'feature': feat,
            'vif': v_val,
            'status': status,
            'correlation': round(c_val, 4),
            'recommendation': recommendation
        })
        
    # Urutkan korelasi dari terbesar ke terkecil
    sorted_corr = sorted(target_corr.items(), key=lambda item: abs(item[1]), reverse=True)
    corr_ranking = []
    for feat, coef in sorted_corr:
        val_abs = abs(coef)
        if val_abs >= 0.8:
            kategori = "Sangat Kuat"
        elif val_abs >= 0.6:
            kategori = "Kuat"
        elif val_abs >= 0.4:
            kategori = "Sedang"
        elif val_abs >= 0.2:
            kategori = "Lemah"
        else:
            kategori = "Sangat Lemah"
            
        corr_ranking.append({
            'feature': feat,
            'coef': round(coef, 4),
            'category': kategori
        })
        
    return render_template(
        'eda.html',
        summary=summary_cards,
        features=features_list,
        vif_table=vif_table,
        corr_ranking=corr_ranking
    )

@eda_bp.route('/statistics')
def statistics():
    """Mengembalikan JSON data statistik deskriptif untuk DataTable."""
    df = get_processed_data()
    if df is None:
        return jsonify({'error': 'Dataset hasil preprocessing tidak ditemukan'}), 404
        
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if 'No' in numeric_cols:
            numeric_cols.remove('No')
            
        stats_list = []
        for col in numeric_cols:
            col_series = df[col].dropna()
            
            # Perhitungan kurtosis dan skewness manual/pandas
            skew = col_series.skew()
            kurt = col_series.kurt()
            
            # Median & Modus
            median = col_series.median()
            modes = col_series.mode()
            mode = modes[0] if not modes.empty else 0.0
            
            # IQR
            q1 = col_series.quantile(0.25)
            q3 = col_series.quantile(0.75)
            iqr = q3 - q1
            
            stats_list.append({
                'column': col,
                'count': int(col_series.count()),
                'mean': round(float(col_series.mean()), 4),
                'median': round(float(median), 4),
                'mode': round(float(mode), 4),
                'min': round(float(col_series.min()), 4),
                'max': round(float(col_series.max()), 4),
                'q1': round(float(q1), 4),
                'q3': round(float(q3), 4),
                'iqr': round(float(iqr), 4),
                'std': round(float(col_series.std()), 4),
                'var': round(float(col_series.var()), 4),
                'skewness': round(float(skew), 4) if not pd.isnull(skew) else 0.0,
                'kurtosis': round(float(kurt), 4) if not pd.isnull(kurt) else 0.0
            })
            
        return jsonify(stats_list)
    except Exception as e:
        app_logger.error(f"Gagal menghitung deskriptif stats: {e}")
        return jsonify({'error': str(e)}), 500

@eda_bp.route('/distribution/<path:feature>')
def distribution(feature):
    """Menghitung histogram, boxplot, density, violin untuk satu fitur."""
    df = get_processed_data()
    if df is None:
        return jsonify({'error': 'Dataset tidak ditemukan'}), 404
    if feature not in df.columns:
        return jsonify({'error': 'Fitur tidak valid'}), 400
        
    try:
        col_data = df[feature].dropna()
        
        # 1. Histogram
        fig_hist = px.histogram(df, x=feature, color_discrete_sequence=['#3b82f6'], opacity=0.7)
        fig_hist.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        
        # 2. Boxplot
        fig_box = px.box(df, y=feature, color_discrete_sequence=['#10b981'])
        fig_box.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        
        # 3. Density Plot (KDE hampiran lewat histogram densitas ter-smooth)
        counts, bins = np.histogram(col_data, bins=50, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        fig_density = go.Figure()
        fig_density.add_trace(go.Scatter(x=bin_centers, y=counts, mode='lines', line=dict(color='#8b5cf6', width=2), fill='tozeroy'))
        fig_density.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title=feature, yaxis_title="Kepadatan (Density)")
        
        # 4. Violin Plot
        fig_violin = px.violin(df, y=feature, box=True, points="all", color_discrete_sequence=['#f59e0b'])
        fig_violin.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        
        return jsonify({
            'histogram': json.loads(fig_hist.to_json()),
            'boxplot': json.loads(fig_box.to_json()),
            'density': json.loads(fig_density.to_json()),
            'violin': json.loads(fig_violin.to_json())
        })
    except Exception as e:
        app_logger.error(f"Error pada grafik distribusi: {e}")
        return jsonify({'error': str(e)}), 500

@eda_bp.route('/correlation_heatmap')
def correlation_heatmap():
    """Mengembalikan korelasi heatmap JSON."""
    df = get_processed_data()
    if df is None:
        return jsonify({'error': 'Data tidak ditemukan'}), 404
        
    try:
        numeric_cols = [c for c in df.columns if c not in ['No', 'Date', 'Time']]
        numeric_cols = df[numeric_cols].select_dtypes(include=['number']).columns.tolist()
        corr = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr,
            labels=dict(color="Korelasi"),
            x=corr.columns,
            y=corr.index,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            text_auto=".2f"
        )
        fig.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        return jsonify(json.loads(fig.to_json()))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@eda_bp.route('/target_analysis')
def target_analysis():
    """Menghitung QQ Plot, Histogram, Boxplot, dan Trend untuk Dosis CL-80."""
    df = get_processed_data()
    if df is None:
        return jsonify({'error': 'Dataset tidak ditemukan'}), 404
        
    try:
        y_data = df[REQUIRED_TARGET].dropna().values
        sorted_y = np.sort(y_data)
        n = len(sorted_y)
        
        # 1. Theoretical normal quantiles
        from scipy.stats import norm
        theoretical = norm.ppf((np.arange(n) + 0.5) / n)
        
        # Skalakan theoretical ke rata-rata dan std data agar presisi garis QQ
        mean_y = np.mean(y_data)
        std_y = np.std(y_data)
        line_x = np.linspace(np.min(theoretical), np.max(theoretical), 100)
        line_y = line_x * std_y + mean_y
        
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=theoretical, y=sorted_y, mode='markers', name='Titik Data', marker=dict(color='#10b981', size=4)))
        fig_qq.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Garis Referensi', line=dict(color='red', width=1.5, dash='dash')))
        fig_qq.update_layout(
            margin=dict(l=40, r=40, t=10, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Kuantil Teoretis Normal",
            yaxis_title="Kuantil Sampel Dosis CL-80"
        )
        
        # 2. Histogram Target
        fig_hist = px.histogram(df, x=REQUIRED_TARGET, color_discrete_sequence=['#10b981'], opacity=0.7)
        fig_hist.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        
        # 3. Boxplot Target
        fig_box = px.box(df, y=REQUIRED_TARGET, color_discrete_sequence=['#10b981'])
        fig_box.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        
        # 4. Trend Target
        fig_trend = px.line(df, y=REQUIRED_TARGET, color_discrete_sequence=['#10b981'])
        fig_trend.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        
        return jsonify({
            'qq': json.loads(fig_qq.to_json()),
            'histogram': json.loads(fig_hist.to_json()),
            'boxplot': json.loads(fig_box.to_json()),
            'trend': json.loads(fig_trend.to_json())
        })
    except Exception as e:
        app_logger.error(f"Gagal melakukan Analisis Target: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@eda_bp.route('/timeseries_analysis')
def timeseries_analysis():
    """Mengembalikan trend multi-variabel time series."""
    df = get_processed_data()
    if df is None:
        return jsonify({'error': 'Data tidak ditemukan'}), 404
        
    try:
        # Buat kolom Datetime jika Date dan Time tersedia
        if 'Date' in df.columns and 'Time' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
            x_col = 'Datetime'
        else:
            x_col = df.index
            
        # 1. Trend CL-80
        fig_cl80 = px.line(df, x=x_col, y=REQUIRED_TARGET, color_discrete_sequence=['#10b981'])
        fig_cl80.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis_title="CL-80 (ppm)")
        
        # 2. Trend TSS
        fig_tss = go.Figure()
        if 'Inlet TSS (mg/L)' in df.columns:
            fig_tss.add_trace(go.Scatter(x=df[x_col], y=df['Inlet TSS (mg/L)'], mode='lines', name='Inlet TSS', line=dict(color='#3b82f6')))
        if 'Outlet TSS (mg/L)' in df.columns:
            fig_tss.add_trace(go.Scatter(x=df[x_col], y=df['Outlet TSS (mg/L)'], mode='lines', name='Outlet TSS', line=dict(color='#ef4444')))
        fig_tss.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis_title="TSS (mg/L)")
        
        # 3. Trend pH
        fig_ph = go.Figure()
        if 'Inlet pH' in df.columns:
            fig_ph.add_trace(go.Scatter(x=df[x_col], y=df['Inlet pH'], mode='lines', name='Inlet pH', line=dict(color='#f59e0b')))
        if 'Outlet pH' in df.columns:
            fig_ph.add_trace(go.Scatter(x=df[x_col], y=df['Outlet pH'], mode='lines', name='Outlet pH', line=dict(color='#8b5cf6')))
        fig_ph.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis_title="pH")
        
        # 4. Trend Debit
        fig_debit = go.Figure()
        if 'Inlet Disch (m3/s)' in df.columns:
            fig_debit.add_trace(go.Scatter(x=df[x_col], y=df['Inlet Disch (m3/s)'], mode='lines', name='Debit', line=dict(color='#06b6d4')))
        fig_debit.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis_title="Debit (m³/s)")
        
        # 5. Trend Alum & Lime
        fig_chem = go.Figure()
        if 'Dosis Alum (ppm)' in df.columns:
            fig_chem.add_trace(go.Scatter(x=df[x_col], y=df['Dosis Alum (ppm)'], mode='lines', name='Dosis Alum', line=dict(color='#2563eb')))
        if 'Dosis Lime (ppm)' in df.columns:
            fig_chem.add_trace(go.Scatter(x=df[x_col], y=df['Dosis Lime (ppm)'], mode='lines', name='Dosis Lime', line=dict(color='#a855f7')))
        fig_chem.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis_title="Dosis (ppm)")
        
        return jsonify({
            'cl80': json.loads(fig_cl80.to_json()),
            'tss': json.loads(fig_tss.to_json()),
            'ph': json.loads(fig_ph.to_json()),
            'debit': json.loads(fig_debit.to_json()),
            'chem': json.loads(fig_chem.to_json())
        })
    except Exception as e:
        app_logger.error(f"Gagal melakukan time series analysis: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@eda_bp.route('/download_bab4/<filename>')
def download_bab4(filename):
    """Menyajikan file BAB IV yang diekspor untuk diunduh (grafik/excel)."""
    output_dir = os.path.join(Config.UPLOAD_FOLDER, 'output', 'bab4')
    if not os.path.exists(os.path.join(output_dir, filename)):
        flash(f"File {filename} belum dibentuk. Akses menu EDA terlebih dahulu untuk membentuk visualisasi.", "danger")
        return redirect(url_for('eda.index'))
    return send_from_directory(
        directory=output_dir,
        path=filename,
        as_attachment=True
    )
