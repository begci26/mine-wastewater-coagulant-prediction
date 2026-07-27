import json
import os
import shutil
from datetime import datetime
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from config import Config
from utils.logger import app_logger
from utils.helpers import (
    allowed_file, 
    check_dataset_uploaded, 
    get_dataset_path, 
    delete_active_dataset, 
    get_dataset_summary,
    invalidate_generated_artifacts,
    sanitize_dataframe,
    save_sanitization_summary,
    set_dataset_filename,
)

# Inisialisasi blueprint
dataset_bp = Blueprint('dataset', __name__)

@dataset_bp.route('/')
def index():
    """Merender halaman unggah dataset atau pratinjau data berdasarkan status unggahan."""
    is_uploaded = check_dataset_uploaded()
    summary = None
    
    if is_uploaded:
        dataset_path = get_dataset_path()
        summary = get_dataset_summary(dataset_path)
        if not summary.get('success'):
            flash(f"Gagal memuat dataset: {summary.get('error')}", 'danger')
            is_uploaded = False
            session['dataset_uploaded'] = False
        else:
            session['dataset_uploaded'] = True
    else:
        session['dataset_uploaded'] = False
        
    return render_template('dataset.html', is_uploaded=is_uploaded, summary=summary)

@dataset_bp.route('/upload', methods=['POST'])
def upload():
    """Validate, select zero Alum/Lime observations, and activate a sanitized dataset."""
    if 'dataset_file' not in request.files:
        flash("Tidak ada bagian file dalam request.", "danger")
        return redirect(url_for('dataset.index'))
        
    file = request.files['dataset_file']
    
    if file.filename == '':
        flash("Tidak ada file yang dipilih.", "danger")
        return redirect(url_for('dataset.index'))
        
    if file and allowed_file(file.filename):
        temp_path = None
        try:
            extension = file.filename.rsplit(".", 1)[1].lower()
            temp_dir = os.path.join(Config.UPLOAD_FOLDER, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"incoming.{extension}")
            file.save(temp_path)
            df = pd.read_excel(temp_path, sheet_name=0) if extension == "xlsx" else pd.read_csv(temp_path)
            sanitized, sanitization = sanitize_dataframe(df)
            if sanitized.empty:
                raise ValueError("Tidak ada observasi valid tanpa Alum dan Lime setelah sanitasi.")

            archive_dir = os.path.join(Config.UPLOAD_FOLDER, "source_archive")
            os.makedirs(archive_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            shutil.copy2(temp_path, os.path.join(archive_dir, f"original_{stamp}.{extension}"))

            previous_path = get_dataset_path()
            invalidate_generated_artifacts()
            original_filename = secure_filename(file.filename)
            if not original_filename:
                raise ValueError("Nama file tidak valid.")
            save_path = os.path.join(Config.UPLOAD_FOLDER, original_filename)
            if previous_path != save_path and os.path.isfile(previous_path):
                os.remove(previous_path)
            sanitized.to_csv(save_path, index=False)
            set_dataset_filename(original_filename)
            save_sanitization_summary(sanitization)

            session['dataset_uploaded'] = True
            session.pop('preprocessing_complete', None)
            session.pop('preprocessing_results', None)
            session.pop('training_complete', None)
            session.pop('best_model_name', None)
            session.pop('best_model_rmse', None)
            session.pop('evaluation_complete', None)
            session.pop('single_prediction_result', None)
            session.pop('single_prediction_inputs', None)
            session.pop('batch_prediction_result', None)
            session.pop('prediction_history', None)

            app_logger.info("Dataset uploaded and sanitized: %s", sanitization)
            flash(
                "Dataset aktif tersanitasi: "
                f"{sanitization['final_active_row_count']} dari "
                f"{sanitization['original_row_count']} baris dipertahankan; "
                f"{sanitization['excluded_alum_lime_row_count']} baris Alum/Lime dan "
                f"{sanitization['ambiguous_chemical_row_count']} baris ambigu dikeluarkan.",
                "success",
            )
        except Exception as error:
            app_logger.error(f"Kesalahan ketika mengunggah dataset: {error}", exc_info=True)
            flash(f"Gagal mengunggah file: {str(error)}", "danger")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        flash("Ekstensi file tidak valid. Hanya file .csv dan .xlsx yang didukung.", "danger")
        
    return redirect(url_for('dataset.index'))

@dataset_bp.route('/reset', methods=['POST'])
def reset():
    """Menghapus dataset aktif saat ini dan mereset status session terkait."""
    if delete_active_dataset():
        # Bersihkan semua status session
        session.pop('dataset_uploaded', None)
        session.pop('preprocessing_complete', None)
        session.pop('training_complete', None)
        session.pop('best_model_name', None)
        session.pop('best_model_rmse', None)
        session.pop('evaluation_complete', None)
        
        flash("Dataset berhasil dibersihkan.", "success")
    else:
        flash("Gagal menghapus dataset atau dataset tidak aktif.", "danger")
        
    return redirect(url_for('dataset.index'))
