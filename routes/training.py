import os
import time
import json
import threading
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
import joblib
from config import Config
from utils.logger import app_logger
from utils.helpers import check_dataset_uploaded, get_dataset_path
from ml.trainer import WastewaterTrainer
import datetime

# Inisialisasi blueprint
training_bp = Blueprint('training', __name__)

@training_bp.route('/')
def index():
    """Merender halaman dashboard pelatihan model regresi dan panel konfigurasi read-only."""
    if not check_dataset_uploaded():
        flash("Silakan unggah dataset terlebih dahulu.", "warning")
        return redirect(url_for('dataset.index'))
        
    if not session.get('preprocessing_complete'):
        flash("Silakan lakukan Prapemrosesan Data terlebih dahulu sebelum melatih model.", "warning")
        return redirect(url_for('preprocessing.index'))
        
    is_complete = session.get('training_complete', False)
    recap = None
    metadata = None
    
    eval_results_path = os.path.join(Config.MODEL_FOLDER, 'evaluation_results.joblib')
    metadata_path = os.path.join(Config.MODEL_FOLDER, 'model_metadata.json')
    
    if os.path.exists(eval_results_path):
        try:
            recap = joblib.load(eval_results_path)
            is_complete = True
        except Exception as error:
            app_logger.error(f"Gagal memuat hasil evaluasi biner: {error}")
            
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as error:
            app_logger.error(f"Gagal memuat metadata pelatihan: {error}")
            
    # Susun konfigurasi pelatihan (Read-Only)
    config_panel = {}
    
    test_ratio = 20.0
    train_ratio = 80.0
    rows_count = 0
    if 'preprocessing_results' in session:
        test_ratio = float(session['preprocessing_results'].get('test_size_percentage', 20.0))
        train_ratio = 100.0 - test_ratio
        rows_count = session['preprocessing_results'].get('train_rows', 0) + session['preprocessing_results'].get('test_rows', 0)
        
    config_panel['dataset_name'] = os.path.basename(get_dataset_path())
    config_panel['rows_count'] = metadata.get('rows_count', rows_count) if metadata else rows_count
    config_panel['cols_count'] = metadata.get('cols_count', 11) if metadata else 11
    config_panel['target_col'] = "Dosis CL-80 (ppm)"
    config_panel['split_method'] = "Chronological 80:20 (tanpa shuffle)"
    config_panel['train_ratio'] = train_ratio
    config_panel['test_ratio'] = test_ratio
    config_panel['random_state'] = "Tidak digunakan untuk split"
    config_panel['cv_folds'] = 5
    config_panel['algorithms'] = ["Linear Regression", "XGBoost Regressor", "LightGBM Regressor"]
    config_panel['training_date'] = metadata.get('training_date', "Belum Pernah Dilatih") if metadata else "Belum Pernah Dilatih"
    
    # Parameter tetap dari Master Prompt
    config_panel['params'] = {
        'Linear Regression': "fit_intercept=True, copy_X=True, positive=False",
        'XGBoost': "n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=3",
        'LightGBM': "n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, min_child_samples=20, subsample=0.8, colsample_bytree=0.8"
    }
    
    return render_template(
        'training.html', 
        is_complete=is_complete, 
        recap=recap, 
        config_panel=config_panel,
        metadata=metadata
    )

@training_bp.route('/run_ajax', methods=['POST'])
def run_ajax():
    """Memicu alur pelatihan di background thread agar non-blocking."""
    if not check_dataset_uploaded() or not session.get('preprocessing_complete'):
        return jsonify({'success': False, 'error': 'Prasyarat prapemrosesan belum terpenuhi.'}), 400
        
    progress_path = os.path.join(Config.UPLOAD_FOLDER, 'processed', 'training_progress.json')
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    
    start_time = time.time()
    initial_progress = {
        'status': "Menginisialisasi alur pelatihan...",
        'percentage': 0,
        'elapsed_time': 0.0,
        'estimated_time': 8.0,
        'start_time': datetime.date.fromtimestamp(start_time).strftime('%H:%M:%S'),
        'end_time': ""
    }
    
    with open(progress_path, 'w') as f:
        json.dump(initial_progress, f)
        
    def worker_job():
        try:
            trainer = WastewaterTrainer()
            result = trainer.train_and_evaluate_all()
            if not result.get('success'):
                raise RuntimeError(result.get('error', 'Training gagal'))
        except Exception as error:
            app_logger.error(f"Error pada background training thread: {error}", exc_info=True)
            err_state = {
                'status': f"Error: {str(error)}",
                'percentage': 0,
                'elapsed_time': 0,
                'estimated_time': 0,
                'error': str(error)
            }
            with open(progress_path, 'w') as f:
                json.dump(err_state, f)
                
    thread = threading.Thread(target=worker_job)
    thread.start()
    return jsonify({'success': True, 'message': 'Pelatihan dimulai di background.'})

@training_bp.route('/progress')
def progress():
    """Mengembalikan status progres saat ini dalam format JSON."""
    progress_path = os.path.join(Config.UPLOAD_FOLDER, 'processed', 'training_progress.json')
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r') as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({'status': f"Error membaca status: {e}", 'percentage': 0})
    return jsonify({'status': 'Menunggu alur...', 'percentage': 0})

@training_bp.route('/complete')
def complete():
    """Menyimpan status kelayakan model di sesi dan mengarahkan kembali ke index."""
    session['training_complete'] = True
    session['evaluation_complete'] = True
    flash("Seluruh model berhasil dilatih dan dievaluasi!", "success")
    return redirect(url_for('training.index'))

@training_bp.route('/download/<file_type>')
def download(file_type):
    """Menyajikan pengunduhan berkas laporan/metadata hasil training."""
    output_dir = os.path.join(Config.UPLOAD_FOLDER, 'output', 'training')
    
    if file_type == 'csv':
        filename = 'perbandingan_model.csv'
    elif file_type == 'xlsx':
        filename = 'perbandingan_model.xlsx'
    elif file_type == 'pdf':
        filename = 'perbandingan_model.pdf'
    elif file_type == 'json':
        filename = 'model_metadata.json'
    else:
        flash("Format unduhan tidak dikenali.", "danger")
        return redirect(url_for('training.index'))
        
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path):
        flash("File belum tersedia. Silakan jalankan training terlebih dahulu.", "danger")
        return redirect(url_for('training.index'))
        
    return send_from_directory(
        directory=output_dir,
        path=filename,
        as_attachment=True
    )
