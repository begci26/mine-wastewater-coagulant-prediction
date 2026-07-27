import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import Blueprint, render_template, session
from config import Config
from utils.logger import app_logger
from utils.helpers import (
    check_dataset_uploaded,
    get_dataset_path,
    get_research_columns,
    REQUIRED_TARGET,
    REQUIRED_FEATURES,
)
import plotly.express as px
import plotly.graph_objects as go

# Inisialisasi blueprint
dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
def index():
    """Merender dashboard utama dengan 10 kartu statistik, 5 grafik Plotly, dan deskripsi tesis."""
    dataset_status = check_dataset_uploaded()
    preprocessing_status = session.get('preprocessing_complete', False)
    training_status = session.get('training_complete', False)
    evaluation_status = session.get('evaluation_complete', False)
    
    # Inisialisasi default statistik
    row_count = 0
    feature_count = 0
    missing_value_count = 0
    outlier_count = 0
    date_range = "- s.d. -"
    
    best_model_name = "Belum Ada"
    best_model_rmse = "-"
    best_model_r2 = "-"
    best_model_mae = "-"
    model_status = "Belum Dilatih"
    
    # Inisialisasi Chart JSON
    chart_dosis_json = None
    chart_ts_json = None
    chart_tss_json = None
    chart_ph_json = None
    chart_debit_json = None
    
    if dataset_status:
        try:
            # Baca dataset aktif
            df = pd.read_csv(get_dataset_path())
            row_count = len(df)
            
            # Hitung feature (kolom numeric selain No dan target)
            feature_cols = [
                column
                for column in get_research_columns(df)
                if column not in ["Date", "Time", REQUIRED_TARGET]
            ]
            feature_count = len(feature_cols)
            
            # Hitung missing values
            missing_value_count = int(df.isnull().sum().sum())
            
            # Hitung outlier menggunakan IQR untuk semua kolom numerik
            outlier_count = 0
            numeric_cols = df.loc[:, get_research_columns(df)].select_dtypes(
                include=[np.number]
            ).columns
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count += int(((df[col] < lower) | (df[col] > upper)).sum())
                
            # Hitung rentang tanggal
            if 'Date' in df.columns:
                dates = df['Date'].dropna()
                if not dates.empty:
                    date_range = f"{dates.iloc[0]} s.d. {dates.iloc[-1]}"
                    
            # --- RENDER 5 PLOTLY CHARTS ---
            
            # Layout Helper
            def update_chart_layout(fig, title, xtitle, ytitle, color):
                fig.update_layout(
                    title={
                        'text': title,
                        'y': 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'family': 'Inter', 'size': 14, 'color': '#1e293b', 'weight': 'bold'}
                    },
                    margin=dict(l=40, r=40, t=50, b=45),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', size=11, color='#64748b'),
                    height=280,
                    xaxis=dict(
                        title=xtitle, 
                        showgrid=True, 
                        gridcolor='#e2e8f0', 
                        linecolor='#cbd5e1'
                    ),
                    yaxis=dict(
                        title=ytitle, 
                        showgrid=True, 
                        gridcolor='#e2e8f0', 
                        linecolor='#cbd5e1'
                    ),
                    showlegend=False
                )
                return json.dumps(json.loads(fig.to_json()))

            # 1. Distribusi Dosis CL80 (Histogram)
            if REQUIRED_TARGET in df.columns:
                fig_dosis = px.histogram(
                    df, 
                    x=REQUIRED_TARGET, 
                    nbins=40, 
                    color_discrete_sequence=['#2563eb']
                )
                chart_dosis_json = update_chart_layout(
                    fig_dosis, 
                    'Distribusi Dosis CL-80', 
                    'Dosis CL-80 (ppm)', 
                    'Frekuensi', 
                    '#2563eb'
                )

            # 2. Time Series Dosis CL80 (Line)
            if REQUIRED_TARGET in df.columns:
                # Ambil sampel max 500 baris agar grafik tetap responsif jika data sangat besar
                step_size = max(1, len(df) // 500)
                df_ts_sample = df.iloc[::step_size].reset_index()
                
                x_axis = df_ts_sample['Date'] if 'Date' in df_ts_sample.columns else df_ts_sample['index']
                fig_ts = px.line(
                    df_ts_sample, 
                    x=x_axis, 
                    y=REQUIRED_TARGET, 
                    color_discrete_sequence=['#10b981']
                )
                chart_ts_json = update_chart_layout(
                    fig_ts, 
                    'Time Series Dosis CL-80', 
                    'Tanggal/Urutan Waktu', 
                    'Dosis CL-80 (ppm)', 
                    '#10b981'
                )

            # 3. Distribusi Inlet TSS (Histogram)
            if 'Inlet TSS (mg/L)' in df.columns:
                fig_tss = px.histogram(
                    df, 
                    x='Inlet TSS (mg/L)', 
                    nbins=40, 
                    color_discrete_sequence=['#f59e0b']
                )
                chart_tss_json = update_chart_layout(
                    fig_tss, 
                    'Distribusi Inlet TSS', 
                    'Inlet TSS (mg/L)', 
                    'Frekuensi', 
                    '#f59e0b'
                )

            # 4. Distribusi pH (Histogram)
            if 'Inlet pH' in df.columns:
                fig_ph = px.histogram(
                    df, 
                    x='Inlet pH', 
                    nbins=30, 
                    color_discrete_sequence=['#ef4444']
                )
                chart_ph_json = update_chart_layout(
                    fig_ph, 
                    'Distribusi Inlet pH', 
                    'Inlet pH', 
                    'Frekuensi', 
                    '#ef4444'
                )

            # 5. Distribusi Debit (Histogram)
            if 'Inlet Disch (m3/s)' in df.columns:
                fig_debit = px.histogram(
                    df, 
                    x='Inlet Disch (m3/s)', 
                    nbins=30, 
                    color_discrete_sequence=['#8b5cf6']
                )
                chart_debit_json = update_chart_layout(
                    fig_debit, 
                    'Distribusi Debit Aliran', 
                    'Inlet Disch (m3/s)', 
                    'Frekuensi', 
                    '#8b5cf6'
                )
                
        except Exception as error:
            app_logger.error(f"Error loading dataset stats on dashboard: {error}", exc_info=True)
            
    # --- RETAIN MODEL EVALUATION STATS ---
    eval_results_path = os.path.join(Config.MODEL_FOLDER, 'evaluation_results.joblib')
    if os.path.exists(eval_results_path):
        try:
            stats = joblib.load(eval_results_path)
            best_model_name = stats.get('best_model')
            best_metrics = stats.get('best_metrics')
            
            best_model_r2 = f"{best_metrics.get('r2'):.4f}"
            best_model_rmse = f"{best_metrics.get('rmse'):.4f} ppm"
            best_model_mae = f"{best_metrics.get('mae'):.4f} ppm"
            model_status = "Siap Digunakan"
        except Exception as error:
            app_logger.error(f"Error loading evaluation stats on dashboard: {error}")
    elif training_status:
        model_status = "Sedang Dilatih"
        
    return render_template(
        'dashboard.html',
        dataset_status=dataset_status,
        preprocessing_status=preprocessing_status,
        training_status=training_status,
        evaluation_status=evaluation_status,
        
        row_count=row_count,
        feature_count=feature_count,
        missing_value_count=missing_value_count,
        outlier_count=outlier_count,
        date_range=date_range,
        
        best_model_name=best_model_name,
        best_model_rmse=best_model_rmse,
        best_model_r2=best_model_r2,
        best_model_mae=best_model_mae,
        model_status=model_status,
        
        chart_dosis_json=chart_dosis_json,
        chart_ts_json=chart_ts_json,
        chart_tss_json=chart_tss_json,
        chart_ph_json=chart_ph_json,
        chart_debit_json=chart_debit_json
    )
