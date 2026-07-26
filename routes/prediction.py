import io
import json
import os
from datetime import datetime

import pandas as pd
from flask import Blueprint, flash, jsonify, redirect, render_template, request, send_file, send_from_directory, session, url_for
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import Config
from ml.predictor import WastewaterPredictor
from utils.helpers import BASE_INPUT_FEATURES, MODEL_FEATURES, forbidden_chemical_columns, get_dataset_path
from utils.logger import app_logger

prediction_bp = Blueprint("prediction", __name__)


def get_feature_boundaries():
    path = os.path.join(Config.UPLOAD_FOLDER, "processed", "x_train_raw.csv")
    if os.path.exists(path):
        frame = pd.read_csv(path)
        return {
            feature: {
                "mean": round(float(frame[feature].median()), 2),
                "min": round(float(frame[feature].min()), 2),
                "max": round(float(frame[feature].max()), 2),
            }
            for feature in MODEL_FEATURES
        }
    defaults = {
        "Inlet TSS (mg/L)": (1000, 0, 2000),
        "Inlet pH": (7, 0, 14),
        "Outlet TSS (mg/L)": (30, 0, 300),
        "Outlet pH": (7, 0, 14),
        "Inlet Disch (m3/s)": (1.5, 0, 10),
        "Kolom T1": (5, 0, 100),
        "Kolom T2": (5, 0, 100),
        "Kolom T3": (5, 0, 100),
    }
    return {
        feature: {"mean": values[0], "min": values[1], "max": values[2]}
        for feature, values in defaults.items()
    }


def get_last_lags(prediction_timestamp=None):
    values = WastewaterPredictor().get_lag_values(prediction_timestamp)
    return values["Kolom T1"], values["Kolom T2"], values["Kolom T3"]


def _output_dirs():
    directories = [
        os.path.join(Config.OUTPUT_FOLDER, "prediksi"),
        os.path.join(Config.OUTPUT_FOLDER, "bab4"),
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return directories


def _prediction_record(date, time_value, feature_row, prediction, model):
    record = {
        "Date": date,
        "Time": time_value,
        **{feature: feature_row[feature] for feature in MODEL_FEATURES},
        "Predicted_Dosis_CL80_ppm": prediction,
        "Model": model,
        "Prediction_Timestamp": f"{date} {time_value}",
    }
    forbidden = forbidden_chemical_columns(record.keys())
    if forbidden:
        raise ValueError("Output prediksi mengandung kolom kimia terlarang.")
    return record


def _write_prediction_pdf(record, result, path):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=letter)
    feature_rows = [["Fitur", "Nilai"]] + [
        [feature, f"{float(record[feature]):.4f}"] for feature in MODEL_FEATURES
    ]
    contribution_rows = [["Fitur", "Kontribusi LOCO (ppm)"]] + [
        [feature, f"{float(result['contributions'][feature]):.4f}"]
        for feature in MODEL_FEATURES
    ]
    story = [
        Paragraph("Laporan Prediksi Dosis CL-80", styles["Title"]),
        Paragraph(
            f"Waktu prediksi: {record['Prediction_Timestamp']} | Model: {record['Model']}",
            styles["BodyText"],
        ),
        Paragraph(
            f"Prediksi dosis CL-80: {record['Predicted_Dosis_CL80_ppm']:.4f} ppm",
            styles["Heading2"],
        ),
        Spacer(1, 8),
        Table(feature_rows, repeatRows=1),
        Spacer(1, 8),
        Paragraph(
            "Interpretasi lokal: LOCO-based reference-value perturbation. Nilai ini "
            "merupakan perubahan prediksi diagnostik, bukan efek kausal.",
            styles["BodyText"],
        ),
        Table(contribution_rows, repeatRows=1),
    ]
    for item in [story[3], story[-1]]:
        if isinstance(item, Table):
            item.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ]
                )
            )
    doc.build(story)


def save_prediction_outputs(date, time_value, inputs, result_value, best_model, result=None):
    predictor = WastewaterPredictor()
    feature_row = predictor.build_feature_row(inputs).iloc[0].to_dict()
    record = _prediction_record(date, time_value, feature_row, result_value, best_model)
    result = result or {"contributions": {feature: 0.0 for feature in MODEL_FEATURES}}
    for directory in _output_dirs():
        frame = pd.DataFrame([record])
        frame.to_csv(os.path.join(directory, "hasil_prediksi_terakhir.csv"), index=False)
        frame.to_excel(os.path.join(directory, "hasil_prediksi_terakhir.xlsx"), index=False)
        with open(
            os.path.join(directory, "hasil_prediksi_terakhir.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "prediction": record,
                    "local_interpretation": {
                        "method": "LOCO-based reference-value perturbation",
                        "contribution_ppm": result["contributions"],
                        "causal_effect": False,
                    },
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        _write_prediction_pdf(
            record, result, os.path.join(directory, "laporan_prediksi_terakhir.pdf")
        )
        history_csv = os.path.join(directory, "riwayat_prediksi_akumulatif.csv")
        if os.path.exists(history_csv):
            old = pd.read_csv(history_csv)
            old = old.drop(columns=forbidden_chemical_columns(old.columns), errors="ignore")
            history = pd.concat([old, frame], ignore_index=True)
        else:
            history = frame
        history.to_csv(history_csv, index=False)
        history.to_excel(
            os.path.join(directory, "riwayat_prediksi_akumulatif.xlsx"), index=False
        )


@prediction_bp.route("/")
def index():
    predictor = WastewaterPredictor()
    if not predictor.check_model_exists():
        flash(
            "Model 11 fitur yang kompatibel belum tersedia. Jalankan training kembali.",
            "warning",
        )
        return redirect(url_for("training.index"))
    metadata = {}
    with open(predictor.metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    try:
        lag_values = predictor.get_lag_values()
    except Exception:
        lag_values = {}
    return render_template(
        "prediction.html",
        boundaries=get_feature_boundaries(),
        lag_values=lag_values,
        single_result=session.pop("single_prediction_result", None),
        single_inputs=session.pop("single_prediction_inputs", None),
        batch_result=session.pop("batch_prediction_result", None),
        history=session.get("prediction_history", []),
        metadata=metadata,
    )


@prediction_bp.route("/example")
def example():
    try:
        frame = pd.read_csv(get_dataset_path())
        if frame.empty:
            return jsonify({"error": "Dataset kosong"}), 404
        row = frame.iloc[-1].to_dict()
        return jsonify(
            {
                key: "" if pd.isna(value) else value.item() if hasattr(value, "item") else value
                for key, value in row.items()
                if key in ["Date", "Time"] + BASE_INPUT_FEATURES
            }
        )
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@prediction_bp.route("/single", methods=["POST"])
def single():
    predictor = WastewaterPredictor()
    if not predictor.check_model_exists():
        flash("Model kompatibel belum tersedia.", "danger")
        return redirect(url_for("prediction.index"))
    date = request.form.get("Tanggal", "").strip()
    time_value = request.form.get("Jam", "").strip()
    field_mapping = {
        "Inlet TSS (mg/L)": "Inlet TSS",
        "Inlet pH": "Inlet pH",
        "Outlet TSS (mg/L)": "Outlet TSS",
        "Outlet pH": "Outlet pH",
        "Inlet Disch (m3/s)": "Debit",
    }
    errors = []
    inputs = {}
    if not date or not time_value:
        errors.append("Date dan Time wajib diisi.")
    for feature, form_name in field_mapping.items():
        text = request.form.get(form_name, "").strip()
        try:
            inputs[feature] = float(text)
        except ValueError:
            errors.append(f"{form_name} harus berupa angka.")
    if not errors:
        if not 0 <= inputs["Inlet pH"] <= 14 or not 0 <= inputs["Outlet pH"] <= 14:
            errors.append("Nilai pH harus berada pada rentang 0 sampai 14.")
        if inputs["Inlet TSS (mg/L)"] < 0 or inputs["Outlet TSS (mg/L)"] < 0:
            errors.append("Nilai TSS tidak boleh negatif.")
        if inputs["Inlet Disch (m3/s)"] < 0:
            errors.append("Inlet Discharge tidak boleh negatif.")
    try:
        if not errors:
            timestamp = pd.to_datetime(f"{date} {time_value}", errors="raise")
            inputs.update(predictor.get_lag_values(timestamp))
    except Exception as error:
        errors.append(str(error))
    if errors:
        for error in errors:
            flash(error, "warning")
        return redirect(url_for("prediction.index"))

    result = predictor.predict_single(inputs)
    if not result.get("success"):
        flash(f"Prediksi gagal: {result.get('error')}", "danger")
        return redirect(url_for("prediction.index"))
    with open(predictor.metadata_path, encoding="utf-8") as handle:
        model_name = json.load(handle)["best_model"]
    result.update({"date": date, "time": time_value, "model": model_name})
    session["single_prediction_result"] = result
    session["single_prediction_inputs"] = {
        **inputs,
        "Date": date,
        "Time": time_value,
    }
    save_prediction_outputs(
        date, time_value, inputs, result["prediction"], model_name, result=result
    )
    history = session.get("prediction_history", [])
    history.insert(
        0,
        {
            "date": date,
            "time": time_value,
            "prediction": result["prediction"],
            "model": model_name,
            "inputs": inputs,
        },
    )
    session["prediction_history"] = history[:10]
    flash("Prediksi dan interpretasi LOCO berhasil dibuat.", "success")
    return redirect(url_for("prediction.index"))


@prediction_bp.route("/batch", methods=["POST"])
def batch():
    file = request.files.get("batch_file")
    if not file or not file.filename:
        flash("Pilih file CSV atau XLSX.", "danger")
        return redirect(url_for("prediction.index"))
    extension = file.filename.rsplit(".", 1)[-1].lower()
    if extension not in {"csv", "xlsx"}:
        flash("Format batch harus CSV atau XLSX.", "danger")
        return redirect(url_for("prediction.index"))
    temp_dir = os.path.join(Config.UPLOAD_FOLDER, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"batch_input.{extension}")
    try:
        file.save(temp_path)
        result = WastewaterPredictor().predict_batch(temp_path)
        if result.get("success"):
            session["batch_prediction_result"] = result
            flash("Prediksi batch selesai dan ekspornya telah disanitasi.", "success")
        else:
            flash(f"Prediksi batch gagal: {result.get('error')}", "danger")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return redirect(url_for("prediction.index"))


@prediction_bp.route("/download/<file_type>")
def download(file_type):
    filenames = {"csv": "batch_predictions.csv", "xlsx": "batch_predictions.xlsx"}
    filename = filenames.get(file_type)
    directory = os.path.join(Config.UPLOAD_FOLDER, "predictions")
    if not filename or not os.path.exists(os.path.join(directory, filename)):
        flash("File batch belum tersedia.", "danger")
        return redirect(url_for("prediction.index"))
    return send_from_directory(directory, filename, as_attachment=True)


@prediction_bp.route("/export/<file_type>")
def export(file_type):
    filenames = {
        "csv": "hasil_prediksi_terakhir.csv",
        "xlsx": "hasil_prediksi_terakhir.xlsx",
        "json": "hasil_prediksi_terakhir.json",
        "pdf": "laporan_prediksi_terakhir.pdf",
    }
    filename = filenames.get(file_type)
    directory = os.path.join(Config.OUTPUT_FOLDER, "prediksi")
    if not filename or not os.path.exists(os.path.join(directory, filename)):
        flash("Hasil prediksi belum tersedia.", "danger")
        return redirect(url_for("prediction.index"))
    return send_from_directory(directory, filename, as_attachment=True)
