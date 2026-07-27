import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from flask import Blueprint, render_template, redirect, url_for, flash, jsonify, send_from_directory
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as ReportImage,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import Config
from utils.logger import app_logger
from utils.helpers import (
    MODEL_FEATURES,
    check_dataset_uploaded,
    get_research_columns,
    REQUIRED_TARGET,
)

# Inisialisasi blueprint
eda_bp = Blueprint('eda', __name__)
NON_ANALYTIC_COLUMNS = {"Date", "Time", "Timestamp", "Datetime"}

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

def _coerced_numeric_frame(df, columns):
    frame = pd.DataFrame(index=df.index)
    for column in columns:
        if column in df.columns:
            frame[column] = pd.to_numeric(df[column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan)


def distribution_columns(df):
    """Return model-scope columns containing at least one valid numeric value."""
    candidates = [column for column in MODEL_FEATURES if column in df.columns]
    return [
        column
        for column in candidates
        if pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().any()
    ]


def prepare_correlation_matrix(df):
    """Build the matrix shared by the web heatmap and PNG export."""
    candidates = [
        column
        for column in MODEL_FEATURES + [REQUIRED_TARGET]
        if column in df.columns and column not in NON_ANALYTIC_COLUMNS
    ]
    numeric = _coerced_numeric_frame(df, candidates)
    excluded = {}
    usable = []
    for column in numeric.columns:
        valid = numeric[column].dropna()
        if len(valid) < 2:
            excluded[column] = "Data numerik valid kurang dari dua observasi"
        elif valid.nunique() <= 1:
            excluded[column] = "Kolom konstan"
        else:
            usable.append(column)
    matrix = numeric[usable].corr(min_periods=2) if usable else pd.DataFrame()
    return numeric[usable], matrix, excluded


def calculate_vif(df_features):
    """Calculate VIF safely; constants/unusable columns are explicitly unavailable."""
    columns = [column for column in MODEL_FEATURES if column in df_features.columns]
    numeric = _coerced_numeric_frame(df_features, columns)
    result = {}
    usable = []
    for column in columns:
        valid = numeric[column].dropna()
        if len(valid) < 2 or valid.nunique() <= 1:
            result[column] = None
        else:
            usable.append(column)

    if usable:
        complete = numeric[usable].copy()
        complete = complete.fillna(complete.median(numeric_only=True))
        complete = complete.dropna(axis=1, how="any")
        for column in usable:
            if column not in complete.columns:
                result[column] = None
                continue
            y = complete[column]
            X = complete.drop(columns=[column])
            if X.shape[1] == 0:
                result[column] = 1.0
                continue
            try:
                model = LinearRegression().fit(X, y)
                r2 = float(model.score(X, y))
                if not np.isfinite(r2):
                    result[column] = None
                elif r2 >= 1.0 - 1e-12:
                    result[column] = float("inf")
                else:
                    result[column] = round(max(1.0, 1.0 / (1.0 - r2)), 4)
            except (ValueError, np.linalg.LinAlgError) as error:
                app_logger.warning("VIF tidak tersedia untuk %s: %s", column, error)
                result[column] = None
    return {column: result.get(column) for column in columns}


def _plotly_json(figure):
    return json.loads(figure.to_json())


def _vif_figure(vif_values):
    available = [(name, value) for name, value in vif_values.items() if value is not None]
    figure = go.Figure()
    if not available:
        figure.add_annotation(text="VIF tidak tersedia untuk data ini.", showarrow=False)
    else:
        finite = [value for _, value in available if np.isfinite(value)]
        cap = max([10.0, *finite]) * 1.1
        x = [name for name, _ in available]
        y = [value if np.isfinite(value) else cap for _, value in available]
        labels = [f"{value:.2f}" if np.isfinite(value) else "∞" for _, value in available]
        figure.add_trace(
            go.Bar(
                x=x,
                y=y,
                text=labels,
                textposition="outside",
                marker_color=["#ef4444" if value > 10 else "#10b981" for _, value in available],
            )
        )
        figure.add_hline(y=10, line_dash="dash", line_color="#dc2626")
    figure.update_layout(
        height=420,
        margin=dict(l=45, r=25, t=20, b=120),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Nilai VIF",
    )
    return figure


def _write_eda_pdf(df, output_dir):
    """Create a landscape, multi-page EDA report without stretched or clipped assets."""
    path = os.path.join(output_dir, "laporan_eda.pdf")
    document = SimpleDocTemplate(
        path,
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="Laporan Exploratory Data Analysis CL-80",
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Laporan Exploratory Data Analysis (EDA) CL-80", styles["Title"]),
        Spacer(1, 5 * mm),
        Paragraph(
            "Dataset aktif tersanitasi untuk kondisi tanpa Alum dan Lime. "
            "Visualisasi menggunakan data hasil prapemrosesan dan 11 prediktor final.",
            styles["BodyText"],
        ),
        Spacer(1, 5 * mm),
    ]

    numeric = _coerced_numeric_frame(
        df, [column for column in MODEL_FEATURES + [REQUIRED_TARGET] if column in df.columns]
    )
    description = numeric.describe().T
    headers = ["Variabel", "Count", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"]
    rows = [headers]
    for name, values in description.iterrows():
        rows.append(
            [
                Paragraph(str(name), styles["BodyText"]),
                f"{values['count']:.0f}",
                f"{values['mean']:.4g}",
                f"{values['std']:.4g}",
                f"{values['min']:.4g}",
                f"{values['25%']:.4g}",
                f"{values['50%']:.4g}",
                f"{values['75%']:.4g}",
                f"{values['max']:.4g}",
            ]
        )
    available_width = landscape(A4)[0] - 24 * mm
    first_width = 48 * mm
    other_width = (available_width - first_width) / (len(headers) - 1)
    table = LongTable(
        rows,
        colWidths=[first_width] + [other_width] * (len(headers) - 1),
        repeatRows=1,
        splitByRow=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#94a3b8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.extend([Paragraph("Statistik Deskriptif", styles["Heading2"]), table])

    chart_sections = [
        ("Distribusi Target", "histogram_cl80.png"),
        ("Boxplot Inlet TSS", "boxplot_tss.png"),
        ("Heatmap Korelasi", "heatmap.png"),
        ("Korelasi terhadap Target", "korelasi.png"),
        ("Trend Dosis CL-80", "trend_cl80.png"),
        ("Trend TSS", "trend_tss.png"),
        ("Trend pH", "trend_ph.png"),
        ("Trend Debit", "trend_debit.png"),
        ("Variance Inflation Factor", "vif.png"),
        ("Distribusi KDE Target", "target_distribution.png"),
    ]
    max_width = available_width
    max_height = landscape(A4)[1] - 42 * mm
    for title, filename in chart_sections:
        image_path = os.path.join(output_dir, filename)
        if not os.path.exists(image_path):
            continue
        image = ReportImage(image_path)
        image._restrictSize(max_width, max_height)
        story.extend([PageBreak(), Paragraph(title, styles["Heading2"]), Spacer(1, 2 * mm), image])
    document.build(story)

def generate_bab4_outputs(df):
    """Menghasilkan 10 grafik PNG resolusi tinggi dan 1 file Excel ringkasan statistik ke output/bab4/."""
    try:
        output_dir = os.path.join(Config.UPLOAD_FOLDER, 'output', 'bab4')
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Excel Ringkasan Statistik
        research_df = df.loc[:, get_research_columns(df)]
        desc_df = research_df.describe(include=[np.number])
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
        plt.savefig(os.path.join(output_dir, 'histogram_cl80.png'), dpi=300, bbox_inches="tight")
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
        plt.savefig(os.path.join(output_dir, 'boxplot_tss.png'), dpi=300, bbox_inches="tight")
        plt.close()
        
        # 4. heatmap.png
        _, corr_matrix, _ = prepare_correlation_matrix(df)
        numeric_cols = corr_matrix.columns.tolist()
        if corr_matrix.empty:
            raise ValueError("Matriks korelasi tidak memiliki kolom numerik non-konstan.")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(numeric_cols)))
        ax.set_yticks(np.arange(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(numeric_cols, fontsize=8)
        fig.colorbar(im, ax=ax, label='Koefisien Korelasi')
        plt.title('Heatmap Korelasi Pearson', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmap.png'), dpi=300, bbox_inches="tight")
        plt.close()
        
        # 5. korelasi.png
        plt.figure(figsize=(8, 4))
        target_corr = corr_matrix[REQUIRED_TARGET].drop(REQUIRED_TARGET).sort_values(ascending=False)
        target_corr.plot(kind='bar', color='#3b82f6', edgecolor='black')
        plt.title('Korelasi Fitur terhadap Target Dosis CL-80', fontsize=12, fontweight='bold')
        plt.ylabel('Koefisien Korelasi Pearson', fontsize=10)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'korelasi.png'), dpi=300, bbox_inches="tight")
        plt.close()
        
        # 6. trend_cl80.png
        plt.figure(figsize=(10, 4))
        plt.plot(df[REQUIRED_TARGET], color='#10b981', label='Dosis CL-80')
        plt.title('Trend Time Series Dosis CL-80', fontsize=12, fontweight='bold')
        plt.xlabel('Sampel (Urutan Waktu)', fontsize=10)
        plt.ylabel('CL-80 (ppm)', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trend_cl80.png'), dpi=300, bbox_inches="tight")
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
        plt.savefig(os.path.join(output_dir, 'trend_tss.png'), dpi=300, bbox_inches="tight")
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
        plt.savefig(os.path.join(output_dir, 'trend_ph.png'), dpi=300, bbox_inches="tight")
        plt.close()
        
        # 9. trend_debit.png
        plt.figure(figsize=(10, 4))
        if 'Inlet Disch (m3/s)' in df.columns:
            plt.plot(df['Inlet Disch (m3/s)'], color='#06b6d4', label='Debit')
        plt.title('Trend Time Series Debit Aliran', fontsize=12, fontweight='bold')
        plt.xlabel('Sampel', fontsize=10)
        plt.ylabel('Debit (m³/s)', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trend_debit.png'), dpi=300, bbox_inches="tight")
        plt.close()
        
        # 10. vif.png
        vifs = calculate_vif(df)
        plt.figure(figsize=(8, 4))
        available_vifs = {name: value for name, value in vifs.items() if value is not None}
        finite_vifs = [value for value in available_vifs.values() if np.isfinite(value)]
        vif_cap = max([10.0, *finite_vifs]) * 1.1
        vif_series = pd.Series(
            {
                name: value if np.isfinite(value) else vif_cap
                for name, value in available_vifs.items()
            },
            dtype=float,
        ).sort_values(ascending=False)
        if vif_series.empty:
            plt.text(0.5, 0.5, "VIF tidak tersedia", ha="center", va="center")
            plt.xticks([])
        else:
            vif_series.plot(kind='bar', color='#ef4444', edgecolor='black')
            for index, (name, plotted_value) in enumerate(vif_series.items()):
                if np.isinf(available_vifs[name]):
                    plt.text(index, plotted_value, "∞", ha="center", va="bottom", fontsize=9)
        plt.axhline(10, color='red', linestyle='--', label='Ambang Batas Kolilinearitas (>10)')
        plt.title('Nilai Variance Inflation Factor (VIF) Fitur', fontsize=12, fontweight='bold')
        plt.ylabel('Nilai VIF', fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vif.png'), dpi=300, bbox_inches="tight")
        plt.close()
        
        # 11. target_distribution.png
        plt.figure(figsize=(6, 4))
        sns_color = '#10b981'
        target_values = pd.to_numeric(df[REQUIRED_TARGET], errors="coerce").dropna()
        if target_values.nunique() > 1:
            target_values.plot(kind='kde', color=sns_color, linewidth=2)
        elif not target_values.empty:
            plt.axvline(target_values.iloc[0], color=sns_color, linewidth=2)
        plt.title('Estimasi Densitas Kernel (KDE) Target Dosis CL-80', fontsize=12, fontweight='bold')
        plt.xlabel('Dosis CL-80 (ppm)', fontsize=10)
        plt.ylabel('Densitas', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=300, bbox_inches="tight")
        plt.close()

        _write_eda_pdf(df, output_dir)
        app_logger.info("BAB IV Tesis outputs (10 PNGs, 1 Excel, 1 PDF) successfully generated.")
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
    research_df = df.loc[:, get_research_columns(df)]
    summary_cards = {
        'rows': len(df),
        'cols': len(research_df.columns),
        'missing': int(research_df.isnull().sum().sum()),
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
    _, corr_matrix, corr_excluded = prepare_correlation_matrix(df)

    # Korelasi ke target CL-80
    target_corr = (
        corr_matrix[REQUIRED_TARGET].drop(REQUIRED_TARGET).dropna().to_dict()
        if REQUIRED_TARGET in corr_matrix.columns
        else {}
    )

    # Dropdown distribusi hanya menerima fitur numerik dalam ruang lingkup model.
    features_list = distribution_columns(df)
    
    # VIF Status & Rekomendasi
    vif_table = []
    for feat in [feature for feature in MODEL_FEATURES if feature in df.columns]:
        v_val = vif_dict.get(feat)
        c_val = target_corr.get(feat, 0.0)

        is_available = v_val is not None
        is_high = is_available and v_val > 10.0
        status = "Tidak tersedia" if not is_available else "Aman (VIF <= 10)"
        recommendation = (
            corr_excluded.get(feat, "VIF tidak dapat dihitung dari data valid yang tersedia")
            if not is_available
            else "Fitur dipertahankan"
        )

        if is_high:
            status = "Multikolinearitas Tinggi (>10)"
            if abs(c_val) < 0.2:
                recommendation = "Dipertimbangkan untuk dihapus (VIF Tinggi, Korelasi Rendah)"
            else:
                recommendation = "Fitur dipertahankan untuk pemodelan non-linier"
        elif is_available:
            if abs(c_val) < 0.1:
                recommendation = "Dipertimbangkan untuk dihapus (Korelasi Sangat Lemah)"

        vif_table.append({
            'feature': feat,
            'vif': v_val,
            'vif_display': "Tidak tersedia" if v_val is None else ("∞" if np.isinf(v_val) else v_val),
            'is_available': is_available,
            'is_high': is_high,
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
        
    heatmap_path = os.path.join(Config.UPLOAD_FOLDER, "output", "bab4", "heatmap.png")
    bab4_version = int(os.path.getmtime(heatmap_path)) if os.path.exists(heatmap_path) else 0
    return render_template(
        'eda.html',
        summary=summary_cards,
        features=features_list,
        vif_table=vif_table,
        corr_ranking=corr_ranking,
        bab4_version=bab4_version,
    )

@eda_bp.route('/statistics')
def statistics():
    """Mengembalikan JSON data statistik deskriptif untuk DataTable."""
    df = get_processed_data()
    if df is None:
        return jsonify({'error': 'Dataset hasil preprocessing tidak ditemukan'}), 404
        
    try:
        research_df = df.loc[:, get_research_columns(df)]
        numeric_cols = research_df.select_dtypes(include=['number']).columns.tolist()
            
        stats_list = []
        for col in numeric_cols:
            col_series = research_df[col].dropna()
            
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
    allowed = distribution_columns(df)
    if feature not in allowed:
        return jsonify(
            {
                'error': 'Fitur distribusi tidak valid atau tidak memiliki data numerik.',
                'allowed_features': allowed,
            }
        ), 400

    try:
        col_data = (
            pd.to_numeric(df[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if col_data.empty:
            return jsonify(
                {
                    'error': f'Tidak ada data numerik valid untuk {feature}.',
                    'empty_state': True,
                }
            ), 422

        chart_frame = pd.DataFrame({feature: col_data})
        unique_count = int(col_data.nunique())
        is_constant = unique_count == 1
        low_cardinality = unique_count <= 10

        # 1. Histogram
        fig_hist = px.histogram(
            chart_frame,
            x=feature,
            nbins=max(1, min(50, unique_count if low_cardinality else 30)),
            color_discrete_sequence=['#3b82f6'],
            opacity=0.7,
        )
        fig_hist.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        # 2. Boxplot
        fig_box = px.box(chart_frame, y=feature, color_discrete_sequence=['#10b981'])
        fig_box.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        # 3. Density Plot (histogram density); constants receive an explicit marker.
        fig_density = go.Figure()
        if is_constant:
            value = float(col_data.iloc[0])
            fig_density.add_trace(
                go.Scatter(
                    x=[value],
                    y=[1],
                    mode='markers',
                    marker=dict(color='#8b5cf6', size=14),
                    name='Nilai konstan',
                )
            )
            fig_density.add_annotation(
                text=f"Semua {len(col_data)} observasi bernilai {value:g}; KDE tidak bermakna.",
                x=value,
                y=1,
                showarrow=True,
                ay=-45,
            )
        else:
            bins_count = min(50, max(5, int(np.sqrt(len(col_data)))))
            counts, bins = np.histogram(col_data.to_numpy(dtype=float), bins=bins_count, density=True)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            fig_density.add_trace(go.Scatter(x=bin_centers, y=counts, mode='lines', line=dict(color='#8b5cf6', width=2), fill='tozeroy'))
        fig_density.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title=feature, yaxis_title="Kepadatan (Density)")

        # 4. Violin Plot
        fig_violin = px.violin(chart_frame, y=feature, box=True, points="outliers", color_discrete_sequence=['#f59e0b'])
        fig_violin.update_layout(margin=dict(l=40, r=40, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        return jsonify({
            'histogram': _plotly_json(fig_hist),
            'boxplot': _plotly_json(fig_box),
            'density': _plotly_json(fig_density),
            'violin': _plotly_json(fig_violin),
            'meta': {
                'feature': feature,
                'valid_count': int(len(col_data)),
                'missing_or_invalid_count': int(len(df) - len(col_data)),
                'unique_count': unique_count,
                'is_constant': is_constant,
                'low_cardinality': low_cardinality,
            },
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
        _, corr, excluded = prepare_correlation_matrix(df)
        if corr.empty:
            return jsonify(
                {
                    'error': 'Heatmap tidak tersedia: diperlukan minimal dua kolom numerik non-konstan.',
                    'excluded_columns': excluded,
                    'empty_state': True,
                }
            ), 422

        fig = px.imshow(
            corr,
            labels=dict(color="Korelasi"),
            x=corr.columns,
            y=corr.index,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            text_auto=".2f"
        )
        fig.update_layout(
            height=max(480, 32 * len(corr.columns)),
            margin=dict(l=130, r=30, t=20, b=130),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        heatmap = _plotly_json(fig)
        vif_chart = _plotly_json(_vif_figure(calculate_vif(df)))
        # Keep data/layout at the top level for backward compatibility.
        return jsonify(
            {
                **heatmap,
                'heatmap': heatmap,
                'vif_chart': vif_chart,
                'excluded_columns': excluded,
            }
        )
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

def _build_timeseries_response(df):
    series_data = df.copy()
    if "Timestamp" in series_data.columns:
        series_data["Datetime"] = pd.to_datetime(series_data["Timestamp"], errors="coerce")
    elif "Date" in series_data.columns and "Time" in series_data.columns:
        series_data["Datetime"] = pd.to_datetime(
            series_data["Date"].astype(str).str.strip()
            + " "
            + series_data["Time"].astype(str).str.strip(),
            errors="coerce",
        )
    else:
        return jsonify(
            {"error": "Time series tidak tersedia karena kolom timestamp tidak ditemukan.", "empty_state": True}
        ), 422

    invalid_timestamps = int(series_data["Datetime"].isna().sum())
    series_data = (
        series_data.dropna(subset=["Datetime"])
        .sort_values("Datetime", kind="mergesort")
        .reset_index(drop=True)
    )
    if series_data.empty:
        return jsonify(
            {"error": "Time series tidak tersedia karena seluruh timestamp tidak valid.", "empty_state": True}
        ), 422

    def scope_figure(columns, yaxis_title, colors_for_scope):
        figure = go.Figure()
        available = []
        for column, color in zip(columns, colors_for_scope):
            if column not in series_data.columns:
                continue
            values = pd.to_numeric(series_data[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            valid = values.notna()
            if not valid.any():
                continue
            available.append(column)
            figure.add_trace(
                go.Scatter(
                    x=series_data.loc[valid, "Datetime"],
                    y=values.loc[valid],
                    mode="lines",
                    name=column,
                    line=dict(color=color),
                )
            )
        if not available:
            figure.add_annotation(
                text="Tidak ada data numerik valid untuk ruang lingkup ini.", showarrow=False
            )
        figure.update_layout(
            height=320,
            margin=dict(l=55, r=25, t=15, b=45),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title=yaxis_title,
            xaxis_title="Timestamp",
        )
        return figure, available

    fig_cl80, cl80_columns = scope_figure([REQUIRED_TARGET], "CL-80 (ppm)", ["#10b981"])
    fig_tss, tss_columns = scope_figure(
        ["Inlet TSS (mg/L)", "Outlet TSS (mg/L)"], "TSS (mg/L)", ["#3b82f6", "#ef4444"]
    )
    fig_ph, ph_columns = scope_figure(
        ["Inlet pH", "Outlet pH"], "pH", ["#f59e0b", "#8b5cf6"]
    )
    fig_debit, debit_columns = scope_figure(
        ["Inlet Disch (m3/s)"], "Debit (m³/s)", ["#06b6d4"]
    )
    fig_lags, lag_columns = scope_figure(
        ["Kolom T1", "Kolom T2", "Kolom T3"],
        "Dosis historis CL-80 (ppm)",
        ["#2563eb", "#8b5cf6", "#f59e0b"],
    )
    return jsonify(
        {
            "cl80": _plotly_json(fig_cl80),
            "tss": _plotly_json(fig_tss),
            "ph": _plotly_json(fig_ph),
            "debit": _plotly_json(fig_debit),
            "lags": _plotly_json(fig_lags),
            "scopes": {
                "cl80": cl80_columns,
                "tss": tss_columns,
                "ph": ph_columns,
                "debit": debit_columns,
                "lags": lag_columns,
            },
            "invalid_timestamp_count": invalid_timestamps,
        }
    )


@eda_bp.route('/timeseries_analysis')
def timeseries_analysis():
    """Mengembalikan trend multi-variabel time series."""
    df = get_processed_data()
    if df is None:
        return jsonify({'error': 'Data tidak ditemukan'}), 404
    return _build_timeseries_response(df)

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
