import json
import os

import joblib

from config import Config
from utils.helpers import ARTIFACT_SCHEMA_VERSION, MODEL_FEATURES
from utils.logger import app_logger


class WastewaterEvaluator:
    """Load and validate the train/test/CV evaluation produced by the trainer."""

    @staticmethod
    def evaluate_all(training_recap=None):
        try:
            metadata_path = os.path.join(Config.MODEL_FOLDER, "model_metadata.json")
            results_path = os.path.join(Config.MODEL_FOLDER, "evaluation_results.joblib")
            if not os.path.exists(metadata_path) or not os.path.exists(results_path):
                raise FileNotFoundError("Hasil training belum tersedia.")
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            if (
                metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION
                or metadata.get("feature_names") != MODEL_FEATURES
                or metadata.get("feature_count") != len(MODEL_FEATURES)
            ):
                raise ValueError("Artefak evaluasi lama tidak kompatibel dengan 11 fitur.")
            recap = joblib.load(results_path)
            required = {"evaluations", "best_model", "best_metrics", "y_train", "y_test", "y_pred"}
            if not required.issubset(recap):
                raise ValueError("Hasil evaluasi tidak memuat train, test, dan prediksi residual lengkap.")
            return {
                "success": True,
                "evaluations": recap["evaluations"],
                "best_model": recap["best_model"],
                "best_metrics": recap["best_metrics"],
                "validation_method": metadata["validation_method"],
            }
        except Exception as error:
            app_logger.error("Evaluation validation failed: %s", error, exc_info=True)
            return {"success": False, "error": str(error)}
