import json
import os
import time
from datetime import datetime

import joblib
import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from ml.preprocessor import WastewaterPreprocessor, build_fold_safe_pipeline
from utils.helpers import ARTIFACT_SCHEMA_VERSION, MODEL_FEATURES
from utils.logger import app_logger


class WastewaterTrainer:
    """Train fixed model configurations with chronological, fold-safe validation."""

    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(
                fit_intercept=True, copy_X=True, positive=False
            ),
            "XGBoost": XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0,
                reg_alpha=0,
                reg_lambda=1,
                random_state=42,
                objective="reg:squarederror",
                n_jobs=1,
            ),
            "LightGBM": LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective="regression",
                verbose=-1,
                n_jobs=1,
            ),
        }

    @staticmethod
    def _processed_path(filename):
        return os.path.join(Config.UPLOAD_FOLDER, "processed", filename)

    def validate_dataset(self):
        required = [
            "x_train.csv",
            "x_test.csv",
            "x_train_raw.csv",
            "x_test_raw.csv",
            "y_train.csv",
            "y_test.csv",
            "train_timestamps.csv",
            "test_timestamps.csv",
        ]
        missing = [name for name in required if not os.path.exists(self._processed_path(name))]
        if missing:
            return False, "Split preprocessing tidak lengkap: " + ", ".join(missing)
        try:
            X_train, X_test, _, _ = WastewaterPreprocessor.load_processed_splits()
            if X_train.columns.tolist() != MODEL_FEATURES or X_test.columns.tolist() != MODEL_FEATURES:
                return False, "Artefak split bukan skema 11 fitur."
            train_ts = pd.to_datetime(pd.read_csv(self._processed_path("train_timestamps.csv"))["Timestamp"])
            test_ts = pd.to_datetime(pd.read_csv(self._processed_path("test_timestamps.csv"))["Timestamp"])
            if train_ts.max() >= test_ts.min():
                return False, "Split tidak kronologis: periode training bertumpang tindih dengan testing."
        except Exception as error:
            return False, str(error)
        return True, ""

    def update_progress(self, status, percentage, start_time, end_time=None):
        elapsed = round(time.time() - start_time, 2)
        progress = {
            "status": status,
            "percentage": percentage,
            "elapsed_time": elapsed,
            "estimated_time": round(max(0, elapsed * (100 - percentage) / percentage), 2)
            if percentage
            else 0,
            "start_time": datetime.fromtimestamp(start_time).strftime("%H:%M:%S"),
            "end_time": datetime.fromtimestamp(end_time).strftime("%H:%M:%S")
            if end_time
            else "",
        }
        os.makedirs(os.path.dirname(self._processed_path("training_progress.json")), exist_ok=True)
        with open(self._processed_path("training_progress.json"), "w", encoding="utf-8") as handle:
            json.dump(progress, handle, indent=2)

    @staticmethod
    def _metrics(actual, predicted):
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        return {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": rmse,
            "r2": float(r2_score(actual, predicted)),
        }

    def _cross_validate(self, estimator, X_train_raw, y_train):
        splitter = TimeSeriesSplit(n_splits=5)
        fold_metrics = []
        fold_ranges = []
        for fold, (train_indices, validation_indices) in enumerate(
            splitter.split(X_train_raw), start=1
        ):
            pipeline = build_fold_safe_pipeline(clone(estimator))
            pipeline.fit(X_train_raw.iloc[train_indices], y_train.iloc[train_indices])
            predictions = pipeline.predict(X_train_raw.iloc[validation_indices])
            metrics = self._metrics(y_train.iloc[validation_indices], predictions)
            fold_metrics.append(metrics)
            fold_ranges.append(
                {
                    "fold": fold,
                    "train_start_index": int(train_indices[0]),
                    "train_end_index": int(train_indices[-1]),
                    "validation_start_index": int(validation_indices[0]),
                    "validation_end_index": int(validation_indices[-1]),
                }
            )
        return {
            "rmse_mean": float(np.mean([fold["rmse"] for fold in fold_metrics])),
            "rmse_std": float(np.std([fold["rmse"] for fold in fold_metrics])),
            "mae_mean": float(np.mean([fold["mae"] for fold in fold_metrics])),
            "r2_mean": float(np.mean([fold["r2"] for fold in fold_metrics])),
            "n_splits": 5,
            "validation_method": "Time Series Cross-Validation with 5 splits",
            "fold_metrics": fold_metrics,
            "fold_ranges": fold_ranges,
        }

    def train_and_evaluate_all(self):
        started = time.time()
        valid, error = self.validate_dataset()
        if not valid:
            return {"success": False, "error": error}

        X_train, X_test, y_train, y_test = WastewaterPreprocessor.load_processed_splits()
        X_train_raw, _, _, _ = WastewaterPreprocessor.load_processed_splits(raw=True)
        with open(
            self._processed_path("preprocessing_state.json"), encoding="utf-8"
        ) as handle:
            preprocessing_state = json.load(handle)

        for stale in [
            "linear_regression.joblib",
            "xgboost.joblib",
            "lightgbm.joblib",
            "best_model.joblib",
            "saved_model.joblib",
            "evaluation_results.joblib",
            "model_metadata.json",
        ]:
            path = os.path.join(Config.MODEL_FOLDER, stale)
            if os.path.exists(path):
                os.remove(path)

        results = {}
        predictions_by_model = {}
        self.update_progress("Melatih tiga konfigurasi model...", 10, started)
        for index, (name, estimator) in enumerate(self.models.items(), start=1):
            fit_started = time.time()
            estimator.fit(X_train, y_train)
            duration = time.time() - fit_started
            train_prediction = estimator.predict(X_train)
            prediction_started = time.time()
            test_prediction = estimator.predict(X_test)
            prediction_duration = time.time() - prediction_started
            train_metrics = self._metrics(y_train, train_prediction)
            test_metrics = self._metrics(y_test, test_prediction)
            cv = self._cross_validate(estimator, X_train_raw, y_train)
            results[name] = {
                "train": train_metrics,
                "test": test_metrics,
                "r2_gap": train_metrics["r2"] - test_metrics["r2"],
                "rmse_gap": test_metrics["rmse"] - train_metrics["rmse"],
                "rmse_ratio": (
                    test_metrics["rmse"] / train_metrics["rmse"]
                    if train_metrics["rmse"] > 0
                    else None
                ),
                "cv": cv,
                "duration": float(duration),
                "pred_duration": float(prediction_duration),
                # Backward-compatible test aliases used by existing views.
                "mae": test_metrics["mae"],
                "rmse": test_metrics["rmse"],
                "r2": test_metrics["r2"],
                "cv_mean": cv["rmse_mean"],
                "cv_std": cv["rmse_std"],
            }
            predictions_by_model[name] = {
                "train": train_prediction.astype(float).tolist(),
                "test": test_prediction.astype(float).tolist(),
            }
            filename = name.lower().replace(" ", "_") + ".joblib"
            joblib.dump(estimator, os.path.join(Config.MODEL_FOLDER, filename))
            self.update_progress(f"{name} selesai dilatih dan divalidasi.", 15 + index * 20, started)

        best_name = min(
            results,
            key=lambda name: (
                results[name]["test"]["rmse"],
                results[name]["test"]["mae"],
                -results[name]["test"]["r2"],
            ),
        )
        joblib.dump(self.models[best_name], os.path.join(Config.MODEL_FOLDER, "best_model.joblib"))
        best_predictions = np.asarray(predictions_by_model[best_name]["test"])
        residuals = np.asarray(y_test) - best_predictions
        residual_statistics = {
            "definition": "actual - prediction",
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "min": float(residuals.min()),
            "max": float(residuals.max()),
        }
        finished = time.time()
        split_meta = preprocessing_state.get("preprocessing_metadata", {})
        metadata = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "target": "Dosis CL-80 (ppm)",
            "feature_names": MODEL_FEATURES.copy(),
            "feature_count": len(MODEL_FEATURES),
            "best_model": best_name,
            "selection_criterion": (
                "lowest test RMSE; tie-break by lowest test MAE, then highest test R2"
            ),
            "best_metrics": {
                **results[best_name]["test"],
                "cv_mean": results[best_name]["cv"]["rmse_mean"],
                "cv_std": results[best_name]["cv"]["rmse_std"],
            },
            "training_date": datetime.now().isoformat(timespec="seconds"),
            "rows_count": int(len(X_train) + len(X_test)),
            "cols_count": len(MODEL_FEATURES),
            "duration_seconds": float(finished - started),
            "split": split_meta,
            "validation_method": "Time Series Cross-Validation with 5 splits",
            "cv_n_splits": 5,
            "parameter_policy": "fixed model parameter configuration",
            "randomized_search_cv_used": False,
            "loco_method": "LOCO-based reference-value perturbation",
            "loco_causal_interpretation": False,
            "lag_definition": "previous observations (sampling interval not assumed hourly)",
            "residual_statistics": residual_statistics,
            "model_details": results,
        }
        os.makedirs(Config.MODEL_FOLDER, exist_ok=True)
        with open(
            os.path.join(Config.MODEL_FOLDER, "model_metadata.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

        recap = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "best_model": best_name,
            "best_metrics": metadata["best_metrics"],
            "training_date": metadata["training_date"],
            "y_train": y_train.astype(float).tolist(),
            "y_train_pred": predictions_by_model[best_name]["train"],
            "y_test": y_test.astype(float).tolist(),
            "y_pred": predictions_by_model[best_name]["test"],
            "residual_statistics": residual_statistics,
            "evaluations": {
                name: {
                    **details["test"],
                    "train_mae": details["train"]["mae"],
                    "train_rmse": details["train"]["rmse"],
                    "train_r2": details["train"]["r2"],
                    "cv_rmse_mean": details["cv"]["rmse_mean"],
                    "cv_rmse_std": details["cv"]["rmse_std"],
                    "training_time": details["duration"],
                    "pred_time": details["pred_duration"],
                }
                for name, details in results.items()
            },
            "detailed_evaluations": results,
            "num_features": len(MODEL_FEATURES),
            "num_dataset": int(len(X_train) + len(X_test)),
            "num_train": int(len(X_train)),
            "num_test": int(len(X_test)),
        }
        joblib.dump(recap, os.path.join(Config.MODEL_FOLDER, "evaluation_results.joblib"))
        self.export_reports(metadata)
        self.update_progress("Pelatihan dan evaluasi selesai.", 100, started, finished)
        return {
            "success": True,
            "best_model": best_name,
            "best_metrics": metadata["best_metrics"],
            "duration": metadata["duration_seconds"],
        }

    def export_reports(self, metadata):
        output_dir = os.path.join(Config.UPLOAD_FOLDER, "output", "training")
        os.makedirs(output_dir, exist_ok=True)
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
                    "Validation": details["cv"]["validation_method"],
                }
            )
        frame = pd.DataFrame(rows).sort_values(["Test RMSE", "Test MAE", "Test R2"], ascending=[True, True, False])
        frame.to_csv(os.path.join(output_dir, "perbandingan_model.csv"), index=False)
        frame.to_excel(os.path.join(output_dir, "perbandingan_model.xlsx"), index=False)
        with open(os.path.join(output_dir, "model_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

        for metric, filename, title in [
            ("Test RMSE", "bar_rmse.png", "Perbandingan Test RMSE"),
            ("Test MAE", "bar_mae.png", "Perbandingan Test MAE"),
            ("Test R2", "bar_r2.png", "Perbandingan Test R²"),
        ]:
            plt.figure(figsize=(7, 4))
            plt.bar(frame["Model"], frame[metric], color="#3b82f6")
            plt.title(title)
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()
        plt.figure(figsize=(7, 4))
        durations = [metadata["model_details"][name]["duration"] for name in frame["Model"]]
        plt.bar(frame["Model"], durations, color="#8b5cf6")
        plt.title("Durasi Training")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bar_duration.png"), dpi=300)
        plt.close()

        doc = SimpleDocTemplate(
            os.path.join(output_dir, "perbandingan_model.pdf"),
            pagesize=landscape(letter),
        )
        styles = getSampleStyleSheet()
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
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ]
            )
        )
        story = [
            Paragraph("Laporan Pelatihan dan Evaluasi Model Regresi", styles["Title"]),
            Paragraph(
                "Split kronologis 80:20; preprocessing di-fit hanya pada training; "
                "Time Series Cross-Validation with 5 splits.",
                styles["BodyText"],
            ),
            Spacer(1, 10),
            table,
            Spacer(1, 10),
            Paragraph(
                f"Model terpilih: {metadata['best_model']} berdasarkan {metadata['selection_criterion']}.",
                styles["BodyText"],
            ),
        ]
        doc.build(story)
