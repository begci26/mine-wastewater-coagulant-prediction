import json
import os
import shutil
from io import BytesIO
from pathlib import Path
from datetime import datetime
import pandas as pd
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_file,
)
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from werkzeug.utils import secure_filename
from config import Config
from utils.logger import app_logger
from utils.helpers import (
    allowed_file,
    check_dataset_uploaded,
    get_dataset_path,
    delete_active_dataset,
    get_dataset_summary,
    get_source_dataset_path,
    invalidate_generated_artifacts,
    build_sanitization_report,
    load_dataset_metadata,
    load_sanitization_summary,
    read_source_dataset,
    sanitize_dataframe,
    sanitization_counts,
    save_sanitization_summary,
    set_dataset_filename,
)

# Inisialisasi blueprint
dataset_bp = Blueprint("dataset", __name__)


@dataset_bp.route("/")
def index():
    """Merender halaman unggah dataset atau pratinjau data berdasarkan status unggahan."""
    is_uploaded = check_dataset_uploaded()
    summary = None

    if is_uploaded:
        dataset_path = get_dataset_path()
        summary = get_dataset_summary(dataset_path)
        if not summary.get("success"):
            flash(f"Gagal memuat dataset: {summary.get('error')}", "danger")
            is_uploaded = False
            session["dataset_uploaded"] = False
        else:
            session["dataset_uploaded"] = True
    else:
        session["dataset_uploaded"] = False

    return render_template("dataset.html", is_uploaded=is_uploaded, summary=summary)


@dataset_bp.route("/upload", methods=["POST"])
def upload():
    """Validate, select zero Alum/Lime observations, and activate a sanitized dataset."""
    if "dataset_file" not in request.files:
        flash("Tidak ada bagian file dalam request.", "danger")
        return redirect(url_for("dataset.index"))

    file = request.files["dataset_file"]

    if file.filename == "":
        flash("Tidak ada file yang dipilih.", "danger")
        return redirect(url_for("dataset.index"))

    if file and allowed_file(file.filename):
        temp_path = None
        try:
            extension = file.filename.rsplit(".", 1)[1].lower()
            temp_dir = os.path.join(Config.UPLOAD_FOLDER, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"incoming.{extension}")
            file.save(temp_path)
            df = (
                pd.read_excel(temp_path, sheet_name=0)
                if extension == "xlsx"
                else pd.read_csv(temp_path)
            )
            sanitized, sanitization = sanitize_dataframe(df)
            if sanitized.empty:
                raise ValueError(
                    "Tidak ada observasi valid tanpa Alum dan Lime setelah sanitasi."
                )

            archive_dir = os.path.join(Config.UPLOAD_FOLDER, "source_archive")
            os.makedirs(archive_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            original_filename = secure_filename(file.filename)
            if not original_filename:
                raise ValueError("Nama file tidak valid.")
            archive_filename = f"original_{stamp}_{original_filename}"
            shutil.copy2(temp_path, os.path.join(archive_dir, archive_filename))

            previous_path = get_dataset_path()
            invalidate_generated_artifacts()
            save_path = os.path.join(Config.UPLOAD_FOLDER, original_filename)
            if previous_path != save_path and os.path.isfile(previous_path):
                os.remove(previous_path)
            sanitized.to_csv(save_path, index=False)
            set_dataset_filename(
                original_filename,
                source_archive_filename=archive_filename,
                original_filename=original_filename,
            )
            save_sanitization_summary(sanitization)

            # A new upload starts a new workflow. Generated files were removed
            # above; clear all cookie-backed workflow flags as well so no state
            # from the previous dataset can be restored in this request.
            session.clear()
            session["dataset_uploaded"] = True

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
            app_logger.error(
                f"Kesalahan ketika mengunggah dataset: {error}", exc_info=True
            )
            flash(f"Gagal mengunggah file: {str(error)}", "danger")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        flash(
            "Ekstensi file tidak valid. Hanya file .csv dan .xlsx yang didukung.",
            "danger",
        )

    return redirect(url_for("dataset.index"))


def _validated_sanitization_report():
    source_path = get_source_dataset_path()
    if not source_path or not os.path.exists(source_path):
        raise FileNotFoundError(
            "Dataset sumber asli tidak tersedia. Unggah ulang dataset untuk membuat laporan."
        )
    report = build_sanitization_report(read_source_dataset(source_path))
    counts = sanitization_counts(report)
    summary = load_sanitization_summary()
    expected = {
        "original": int(summary.get("original_row_count", -1)),
        "retained": int(summary.get("final_active_row_count", -1)),
        "chemical_removed": int(summary.get("excluded_alum_lime_row_count", -1)),
        "ambiguous_removed": int(summary.get("ambiguous_chemical_row_count", -1)),
    }
    actual = {key: counts[key] for key in expected}
    if actual != expected:
        raise ValueError(
            f"Laporan sanitasi tidak konsisten dengan ringkasan halaman: "
            f"aktual={actual}, ringkasan={expected}."
        )
    return report


def _sanitization_download_name(extension):
    metadata = load_dataset_metadata()
    original = metadata.get("original_filename", metadata.get("filename", "dataset"))
    return f"laporan_sanitasi_{Path(original).stem}.{extension}"


@dataset_bp.route("/sanitization-report/csv")
def download_sanitization_csv():
    try:
        report = _validated_sanitization_report()
        output = BytesIO()
        output.write(report.to_csv(index=False).encode("utf-8-sig"))
        output.seek(0)
        return send_file(
            output,
            mimetype="text/csv; charset=utf-8",
            as_attachment=True,
            download_name=_sanitization_download_name("csv"),
        )
    except Exception as error:
        app_logger.error("Gagal membuat laporan sanitasi CSV: %s", error, exc_info=True)
        flash(f"Gagal membuat laporan sanitasi CSV: {error}", "danger")
        return redirect(url_for("dataset.index"))


@dataset_bp.route("/sanitization-report/xlsx")
def download_sanitization_xlsx():
    try:
        report = _validated_sanitization_report()
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            report.to_excel(writer, index=False, sheet_name="Laporan Sanitasi")
            worksheet = writer.sheets["Laporan Sanitasi"]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = Font(bold=True)
            for index, column in enumerate(report.columns, start=1):
                values = [
                    str(column),
                    *report[column].fillna("").astype(str).head(200).tolist(),
                ]
                worksheet.column_dimensions[get_column_letter(index)].width = min(
                    max(len(value) for value in values) + 2,
                    40,
                )
        output.seek(0)
        return send_file(
            output,
            mimetype=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            as_attachment=True,
            download_name=_sanitization_download_name("xlsx"),
        )
    except Exception as error:
        app_logger.error(
            "Gagal membuat laporan sanitasi XLSX: %s", error, exc_info=True
        )
        flash(f"Gagal membuat laporan sanitasi XLSX: {error}", "danger")
        return redirect(url_for("dataset.index"))


@dataset_bp.route("/reset", methods=["POST"])
def reset():
    """Menghapus dataset aktif saat ini dan mereset status session terkait."""
    if delete_active_dataset():
        # Bersihkan semua status session
        session.pop("dataset_uploaded", None)
        session.pop("preprocessing_complete", None)
        session.pop("training_complete", None)
        session.pop("best_model_name", None)
        session.pop("best_model_rmse", None)
        session.pop("evaluation_complete", None)

        flash("Dataset berhasil dibersihkan.", "success")
    else:
        flash("Gagal menghapus dataset atau dataset tidak aktif.", "danger")

    return redirect(url_for("dataset.index"))
