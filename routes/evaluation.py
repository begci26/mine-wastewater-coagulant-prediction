import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Blueprint, flash, redirect, render_template, send_from_directory, session, url_for
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from utils.helpers import check_dataset_uploaded
from utils.logger import app_logger

evaluation_bp = Blueprint("evaluation", __name__)


def get_eval_paths():
    evaluation_results = os.path.join(Config.MODEL_FOLDER, "evaluation_results.joblib")
    metadata = os.path.join(Config.MODEL_FOLDER, "model_metadata.json")
    evaluation_dir = os.path.join(Config.UPLOAD_FOLDER, "output", "evaluasi")
    bab4_dir = os.path.join(Config.UPLOAD_FOLDER, "output", "bab4")
    os.makedirs(evaluation_dir, exist_ok=True)
    os.makedirs(bab4_dir, exist_ok=True)
    return evaluation_results, metadata, evaluation_dir, bab4_dir


def _evaluation_frame(metadata):
    rows = []
    for name, details in metadata["model_details"].items():
        rows.append(
            {
                "Model": name,
                "Train MAE": details["train"]["mae"],
                "Train RMSE": details["train"]["rmse"],
                "Train R2": details["train"]["r2"],
                "Test MAE": details["test"]["mae"],
                "Test RMSE": details["test"]["rmse"],
                "Test R2": details["test"]["r2"],
                "R2 Gap": details["r2_gap"],
                "CV RMSE Mean": details["cv"]["rmse_mean"],
                "CV RMSE Std": details["cv"]["rmse_std"],
                "Training Duration": details["duration"],
                "Prediction Duration": details["pred_duration"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Test RMSE", "Test MAE", "Test R2"], ascending=[True, True, False]
    )


def generate_static_eval_plots(recap, metadata):
    _, _, evaluation_dir, bab4_dir = get_eval_paths()
    frame = _evaluation_frame(metadata)
    actual = np.asarray(recap["y_test"])
    predicted = np.asarray(recap["y_pred"])
    residual = actual - predicted
    for target in [evaluation_dir, bab4_dir]:
        for metric, filename in [
            ("Test MAE", "bar_mae.png"),
            ("Test RMSE", "bar_rmse.png"),
            ("Test R2", "bar_r2.png"),
            ("Training Duration", "bar_duration.png"),
            ("Prediction Duration", "bar_pred_duration.png"),
        ]:
            plt.figure(figsize=(7, 4))
            plt.bar(frame["Model"], frame[metric], color="#3b82f6")
            plt.title(metric)
            plt.tight_layout()
            plt.savefig(os.path.join(target, filename), dpi=300)
            plt.close()
        plt.figure(figsize=(6, 6))
        plt.scatter(actual, predicted, alpha=0.6)
        minimum, maximum = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        plt.plot([minimum, maximum], [minimum, maximum], "r--")
        plt.xlabel("Actual Dosis CL-80 (ppm)")
        plt.ylabel("Predicted Dosis CL-80 (ppm)")
        plt.tight_layout()
        plt.savefig(os.path.join(target, "scatter_actual_vs_predicted.png"), dpi=300)
        plt.close()
        plt.figure(figsize=(7, 4))
        plt.scatter(predicted, residual, alpha=0.6, color="#ef4444")
        plt.axhline(0, color="black", linestyle="--")
        plt.xlabel("Predicted Dosis CL-80 (ppm)")
        plt.ylabel("Residual (actual - prediction)")
        plt.tight_layout()
        plt.savefig(os.path.join(target, "residual_plot.png"), dpi=300)
        plt.close()
        plt.figure(figsize=(7, 4))
        plt.hist(residual, bins=20, color="#f59e0b", edgecolor="black")
        plt.xlabel("Residual (actual - prediction)")
        plt.tight_layout()
        plt.savefig(os.path.join(target, "residual_distribution.png"), dpi=300)
        plt.close()


@evaluation_bp.route("/")
def index():
    if not check_dataset_uploaded():
        flash("Silakan unggah dataset terlebih dahulu.", "warning")
        return redirect(url_for("dataset.index"))
    if not session.get("preprocessing_complete"):
        flash("Silakan lakukan preprocessing terlebih dahulu.", "warning")
        return redirect(url_for("preprocessing.index"))
    evaluation_path, metadata_path, _, _ = get_eval_paths()
    if not os.path.exists(evaluation_path) or not os.path.exists(metadata_path):
        flash("Hasil training belum tersedia.", "warning")
        return redirect(url_for("training.index"))
    try:
        recap = joblib.load(evaluation_path)
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        generate_static_eval_plots(recap, metadata)
        actual = np.asarray(recap["y_test"])
        predicted = np.asarray(recap["y_pred"])
        residual = actual - predicted
        frame = _evaluation_frame(metadata)
        sorted_models = []
        for _, row in frame.iterrows():
            sorted_models.append(
                {
                    "name": row["Model"],
                    "mae": row["Test MAE"],
                    "rmse": row["Test RMSE"],
                    "r2": row["Test R2"],
                    "train_mae": row["Train MAE"],
                    "train_rmse": row["Train RMSE"],
                    "train_r2": row["Train R2"],
                    "r2_gap": row["R2 Gap"],
                    "cv": f"{row['CV RMSE Mean']:.4f} ± {row['CV RMSE Std']:.4f}",
                    "duration": row["Training Duration"],
                    "pred_time": row["Prediction Duration"],
                    "status": (
                        "Aktif (Model Terbaik)"
                        if row["Model"] == metadata["best_model"]
                        else "Model Pembanding"
                    ),
                }
            )
        figures = {
            "mae": px.bar(frame, x="Model", y="Test MAE"),
            "rmse": px.bar(frame, x="Model", y="Test RMSE"),
            "r2": px.bar(frame, x="Model", y="Test R2"),
            "dur": px.bar(frame, x="Model", y="Training Duration"),
            "pred_dur": px.bar(frame, x="Model", y="Prediction Duration"),
            "scatter": px.scatter(x=actual, y=predicted),
            "residual": px.scatter(x=predicted, y=residual),
            "res_hist": px.histogram(x=residual),
        }
        figures["scatter"].add_trace(
            go.Scatter(
                x=[min(actual.min(), predicted.min()), max(actual.max(), predicted.max())],
                y=[min(actual.min(), predicted.min()), max(actual.max(), predicted.max())],
                mode="lines",
                line={"dash": "dash", "color": "red"},
            )
        )
        figures["residual"].add_hline(y=0, line_dash="dash")
        charts = {name: json.loads(figure.to_json()) for name, figure in figures.items()}
        best = metadata["best_model"]
        best_metrics = metadata["best_metrics"]
        interpretations = [
            (
                f"Model <strong>{best}</strong> dipilih berdasarkan test RMSE terendah "
                f"({best_metrics['rmse']:.4f} ppm), dengan test MAE dan test R² sebagai tie-break."
            ),
            (
                "Validasi menggunakan Time Series Cross-Validation with 5 splits hanya pada "
                "periode training; setiap fold mempelajari preprocessing dari subset training fold."
            ),
            (
                f"Rata-rata residual actual − prediction adalah {residual.mean():.4f} ppm. "
                "Plot residual merupakan indikator diagnostik; inspeksi visual tidak membuktikan "
                "homoskedastisitas, normalitas, atau ketiadaan bias."
            ),
        ]
        conclusion = (
            f"{best} memberikan hasil terbaik pada periode testing berdasarkan kriteria yang "
            "ditetapkan. Kinerja perlu dipantau pada observasi operasional baru."
        )
        return render_template(
            "evaluation.html",
            best_model=best,
            best_metrics=best_metrics,
            eval_date=metadata["training_date"],
            sorted_models=sorted_models,
            res_min=round(float(residual.min()), 4),
            res_max=round(float(residual.max()), 4),
            res_mean=round(float(residual.mean()), 4),
            res_std=round(float(residual.std()), 4),
            charts=charts,
            interpretations=interpretations,
            conclusion=conclusion,
        )
    except Exception as error:
        app_logger.error("Evaluation page failed: %s", error, exc_info=True)
        flash(f"Gagal memuat evaluasi: {error}", "danger")
        return redirect(url_for("training.index"))


def _write_pdf(frame, recap, path):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=landscape(letter))
    data = [frame.columns.tolist()] + [
        [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        for row in frame.values.tolist()
    ]
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story = [
        Paragraph("Laporan Evaluasi Model Dosis CL-80", styles["Title"]),
        Paragraph(
            "Training/testing metrics and Time Series Cross-Validation with 5 splits.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        table,
        Spacer(1, 8),
        Paragraph(
            "Residual = actual − prediction. Residual plots are diagnostic indicators only.",
            styles["BodyText"],
        ),
    ]
    doc.build(story)


@evaluation_bp.route("/download/<file_type>")
def download(file_type):
    evaluation_path, metadata_path, output_dir, _ = get_eval_paths()
    if not os.path.exists(evaluation_path) or not os.path.exists(metadata_path):
        flash("Data evaluasi belum tersedia.", "danger")
        return redirect(url_for("training.index"))
    recap = joblib.load(evaluation_path)
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    frame = _evaluation_frame(metadata)
    filenames = {
        "csv": "perbandingan_evaluasi.csv",
        "xlsx": "perbandingan_evaluasi.xlsx",
        "pdf": "perbandingan_evaluasi.pdf",
    }
    filename = filenames.get(file_type)
    if not filename:
        flash("Format tidak didukung.", "danger")
        return redirect(url_for("evaluation.index"))
    path = os.path.join(output_dir, filename)
    if file_type == "csv":
        frame.to_csv(path, index=False)
    elif file_type == "xlsx":
        frame.to_excel(path, index=False)
    else:
        _write_pdf(frame, recap, path)
    return send_from_directory(output_dir, filename, as_attachment=True)
