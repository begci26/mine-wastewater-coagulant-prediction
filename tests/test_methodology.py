import json
import os
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config import Config
from ml.predictor import WastewaterPredictor
from ml.preprocessor import LeakageSafePreprocessor, WastewaterPreprocessor
from ml.trainer import WastewaterTrainer
from utils.helpers import (
    MODEL_FEATURES,
    REQUIRED_TARGET,
    dataframe_has_forbidden_chemicals,
    sanitize_dataframe,
)


def synthetic_dataset(rows=90):
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="h")
    index = np.arange(rows, dtype=float)
    dose = 2.0 + 0.01 * index + 0.2 * np.sin(index / 4)
    return pd.DataFrame(
        {
            "Unnamed: 0": index,
            "No": index + 1,
            "Date": timestamps.strftime("%Y-%m-%d"),
            "Time": timestamps.strftime("%H:%M:%S"),
            "Inlet TSS (mg/L)": 500 + 2 * index,
            "Inlet pH": 7.2 + 0.01 * np.sin(index),
            "Outlet TSS (mg/L)": 100 + np.sin(index),
            "Outlet pH": 7.4 + 0.01 * np.cos(index),
            "Inlet Disch (m3/s)": 1.5 + index / 500,
            "Dosis Alum (ppm)": 0.0,
            "Dosis Lime (ppm)": 0.0,
            "Rasio Alum Lime": 99.0,
            REQUIRED_TARGET: dose,
        }
    )


class MethodologyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.original_paths = (
            Config.UPLOAD_FOLDER,
            Config.MODEL_FOLDER,
            Config.OUTPUT_FOLDER,
        )
        Config.UPLOAD_FOLDER = str(root / "uploads")
        Config.MODEL_FOLDER = str(root / "models")
        Config.OUTPUT_FOLDER = str(root / "output")
        for path in self.original_paths:
            pass
        for path in [Config.UPLOAD_FOLDER, Config.MODEL_FOLDER, Config.OUTPUT_FOLDER]:
            os.makedirs(path, exist_ok=True)

    def tearDown(self):
        Config.UPLOAD_FOLDER, Config.MODEL_FOLDER, Config.OUTPUT_FOLDER = self.original_paths
        self.temporary.cleanup()

    def _prepare(self):
        source = synthetic_dataset()
        sanitized, sanitization = sanitize_dataframe(source)
        sanitized.to_csv(Path(Config.UPLOAD_FOLDER) / "active_dataset.csv", index=False)
        engineered, cleaning = WastewaterPreprocessor().clean_and_engineer(sanitized)
        processor = WastewaterPreprocessor()
        train, test, metadata = processor.prepare_and_save(engineered)
        state_path = Path(Config.UPLOAD_FOLDER) / "processed" / "preprocessing_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "split": True,
                    "train_shape": [len(train), 11],
                    "test_shape": [len(test), 11],
                    "preprocessing_metadata": metadata,
                }
            ),
            encoding="utf-8",
        )
        return source, sanitized, engineered, train, test, metadata

    def test_upload_selection_and_ratio_removal(self):
        source = synthetic_dataset(20)
        source.loc[2, "Dosis Alum (ppm)"] = 3
        source.loc[3, "Dosis Lime (ppm)"] = 2
        source["Dosis Alum (ppm)"] = source["Dosis Alum (ppm)"].astype(object)
        source.loc[4, "Dosis Alum (ppm)"] = "unknown"
        sanitized, summary = sanitize_dataframe(source)
        self.assertEqual(summary["excluded_alum_lime_row_count"], 2)
        self.assertEqual(summary["ambiguous_chemical_row_count"], 1)
        self.assertEqual(len(sanitized), 17)
        self.assertFalse(dataframe_has_forbidden_chemicals(sanitized))
        self.assertNotIn("Unnamed: 0", sanitized.columns)

    def test_model_features_exact_order(self):
        self.assertEqual(
            MODEL_FEATURES,
            [
                "Inlet TSS (mg/L)",
                "Inlet pH",
                "Outlet TSS (mg/L)",
                "Outlet pH",
                "Inlet Disch (m3/s)",
                "Kolom T1",
                "Kolom T2",
                "Kolom T3",
                "Efisiensi_TSS",
                "Delta_pH",
                "Beban_TSS",
            ],
        )

    def test_chronological_split_and_lags(self):
        _, _, engineered, train, test, _ = self._prepare()
        self.assertLess(train["Timestamp"].max(), test["Timestamp"].min())
        self.assertEqual(engineered.loc[3, "Kolom T1"], engineered.loc[2, REQUIRED_TARGET])
        self.assertEqual(engineered.loc[3, "Kolom T3"], engineered.loc[0, REQUIRED_TARGET])

    def test_preprocessing_statistics_fit_train_only(self):
        frame = pd.DataFrame({feature: np.arange(10, dtype=float) for feature in MODEL_FEATURES})
        train = frame.iloc[:8].copy()
        test = frame.iloc[8:].copy()
        test.loc[:, MODEL_FEATURES[0]] = 10000
        processor = LeakageSafePreprocessor(feature_names=MODEL_FEATURES).fit(train)
        self.assertEqual(processor.imputer_.statistics_[0], np.median(train.iloc[:, 0]))
        self.assertEqual(processor.winsorizer_.q3_[0], np.percentile(train.iloc[:, 0], 75))
        self.assertEqual(processor.scaler_.mean_[0], np.mean(np.clip(np.arange(8), -3.5, 10.5)))
        processor.transform(test)

    def test_time_series_split_is_forward_only(self):
        X = np.arange(60).reshape(30, 2)
        for train_index, validation_index in TimeSeriesSplit(n_splits=5).split(X):
            self.assertLess(train_index.max(), validation_index.min())

    def test_all_models_metrics_prediction_exports_and_compatibility(self):
        _, _, _, _, _, _ = self._prepare()
        result = WastewaterTrainer().train_and_evaluate_all()
        self.assertTrue(result["success"], result)
        metadata = json.loads(
            (Path(Config.MODEL_FOLDER) / "model_metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(metadata["model_details"]), {"Linear Regression", "XGBoost", "LightGBM"})
        for details in metadata["model_details"].values():
            self.assertIn("train", details)
            self.assertIn("test", details)
            self.assertEqual(details["cv"]["n_splits"], 5)
        predictor = WastewaterPredictor()
        self.assertTrue(predictor.check_model_exists())
        lags = predictor.get_lag_values("2026-01-01")
        inputs = {
            "Inlet TSS (mg/L)": 650,
            "Inlet pH": 7.2,
            "Outlet TSS (mg/L)": 105,
            "Outlet pH": 7.4,
            "Inlet Disch (m3/s)": 1.7,
            **lags,
        }
        prediction = predictor.predict_single(inputs)
        self.assertTrue(prediction["success"], prediction)
        self.assertEqual(list(prediction["contributions"]), MODEL_FEATURES)

        batch = synthetic_dataset(3).drop(columns=[REQUIRED_TARGET])
        future = pd.date_range("2026-01-01", periods=3, freq="h")
        batch["Date"] = future.strftime("%Y-%m-%d")
        batch["Time"] = future.strftime("%H:%M:%S")
        batch_path = Path(Config.UPLOAD_FOLDER) / "batch.csv"
        batch.to_csv(batch_path, index=False)
        batch_result = predictor.predict_batch(str(batch_path))
        self.assertTrue(batch_result["success"], batch_result)
        exported = pd.read_csv(Path(Config.UPLOAD_FOLDER) / "predictions" / "batch_predictions.csv")
        self.assertFalse(dataframe_has_forbidden_chemicals(exported))

        metadata["feature_count"] = 14
        (Path(Config.MODEL_FOLDER) / "model_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        self.assertFalse(predictor.check_model_exists())


if __name__ == "__main__":
    unittest.main()
