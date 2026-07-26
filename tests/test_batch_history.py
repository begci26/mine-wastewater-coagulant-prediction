import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from ml.predictor import BATCH_HISTORY_ERROR, WastewaterPredictor
from utils.helpers import BASE_INPUT_FEATURES, REQUIRED_TARGET, dataframe_has_forbidden_chemicals


class IdentityPreprocessor:
    def transform(self, frame):
        return frame


class RecursiveLagModel:
    """Predict T1 + 1 so recursive lag use is directly observable."""

    def predict(self, frame):
        return np.array([float(frame.iloc[0]["Kolom T1"]) + 1.0])


def batch_frame(periods=5, start="2025-06-01", target=True):
    timestamps = pd.date_range(start, periods=periods, freq="h")
    frame = pd.DataFrame(
        {
            "Date": timestamps.strftime("%Y-%m-%d"),
            "Time": timestamps.strftime("%H:%M:%S"),
            "Inlet TSS (mg/L)": np.arange(periods) + 600.0,
            "Inlet pH": 7.2,
            "Outlet TSS (mg/L)": 100.0,
            "Outlet pH": 7.4,
            "Inlet Disch (m3/s)": 1.7,
        }
    )
    if target:
        frame[REQUIRED_TARGET] = np.arange(periods) * 10.0 + 10.0
    return frame


class BatchHistoryTests(unittest.TestCase):
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
        for path in (Config.UPLOAD_FOLDER, Config.MODEL_FOLDER, Config.OUTPUT_FOLDER):
            os.makedirs(path, exist_ok=True)
        self.predictor = WastewaterPredictor()
        self.predictor._load_artifacts = lambda: (
            RecursiveLagModel(),
            IdentityPreprocessor(),
            {"best_model": "Test Model"},
        )

    def tearDown(self):
        Config.UPLOAD_FOLDER, Config.MODEL_FOLDER, Config.OUTPUT_FOLDER = self.original_paths
        self.temporary.cleanup()

    def _write_batch(self, frame, name="batch.csv"):
        path = Path(Config.UPLOAD_FOLDER) / name
        frame.to_csv(path, index=False)
        return str(path)

    def _export(self):
        return pd.read_csv(
            Path(Config.UPLOAD_FOLDER) / "predictions" / "batch_predictions.csv"
        )

    def _write_active_history(self, timestamps, doses):
        frame = pd.DataFrame(
            {
                "Date": pd.DatetimeIndex(timestamps).strftime("%Y-%m-%d"),
                "Time": pd.DatetimeIndex(timestamps).strftime("%H:%M:%S"),
                REQUIRED_TARGET: doses,
            }
        )
        frame.to_csv(Path(Config.UPLOAD_FOLDER) / "active_dataset.csv", index=False)

    def test_external_history_predicts_first_row_and_recurses(self):
        self._write_active_history(
            pd.date_range("2025-05-31 21:00", periods=3, freq="h"),
            [1.0, 2.0, 3.0],
        )
        result = self.predictor.predict_batch(
            self._write_batch(batch_frame(periods=2, target=False))
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(result["summary"]["history_source"], "active dataset")
        self.assertEqual(result["summary"]["seed_rows_used"], 0)
        exported = self._export()
        self.assertEqual(exported["Prediction_Status"].tolist(), ["Predicted", "Predicted"])
        self.assertEqual(exported.loc[0, ["Kolom T1", "Kolom T2", "Kolom T3"]].tolist(), [3, 2, 1])
        self.assertEqual(exported.loc[1, ["Kolom T1", "Kolom T2", "Kolom T3"]].tolist(), [4, 3, 2])

    def test_compatible_saved_prediction_history_is_a_safe_fallback(self):
        history_dir = Path(Config.OUTPUT_FOLDER) / "prediksi"
        history_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "Prediction_Timestamp": pd.date_range(
                    "2025-05-31 21:00", periods=3, freq="h"
                ),
                "Predicted_Dosis_CL80_ppm": [1.0, 2.0, 3.0],
                "Model": ["Test Model"] * 3,
            }
        ).to_csv(history_dir / "riwayat_prediksi_akumulatif.csv", index=False)
        result = self.predictor.predict_batch(
            self._write_batch(batch_frame(periods=1, target=False))
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(result["summary"]["history_source"], "prediction history")
        exported = self._export()
        self.assertEqual(exported.loc[0, ["Kolom T1", "Kolom T2", "Kolom T3"]].tolist(), [3, 2, 1])

    def test_batch_seed_rows_are_not_predicted_and_actuals_do_not_leak(self):
        frame = batch_frame()
        frame.loc[3:, REQUIRED_TARGET] = [999.0, 888.0]
        result = self.predictor.predict_batch(self._write_batch(frame))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["summary"]["history_source"], "batch seed")
        self.assertEqual(result["summary"]["seed_rows_used"], 3)
        self.assertEqual(result["summary"]["predicted_rows"], 2)
        self.assertEqual(result["summary"]["first_predicted_timestamp"], "2025-06-01 03:00:00")
        exported = self._export()
        self.assertEqual(exported["Prediction_Status"].head(3).tolist(), ["Seed history"] * 3)
        self.assertTrue(exported["Predicted_Dosis_CL80_ppm"].head(3).isna().all())
        self.assertEqual(exported.loc[3, ["Kolom T1", "Kolom T2", "Kolom T3"]].tolist(), [30, 20, 10])
        self.assertEqual(exported.loc[3, "Predicted_Dosis_CL80_ppm"], 31)
        self.assertEqual(exported.loc[4, "Kolom T1"], 31)
        self.assertEqual(exported.loc[4, "Predicted_Dosis_CL80_ppm"], 32)
        self.assertTrue(exported["Historical_Dosis_CL80_ppm"].iloc[3:].isna().all())

    def test_missing_or_invalid_seed_fails_clearly(self):
        no_target = self.predictor.predict_batch(
            self._write_batch(batch_frame(target=False), "no-target.csv")
        )
        self.assertFalse(no_target["success"])
        self.assertEqual(no_target["error"], BATCH_HISTORY_ERROR)

        too_short = self.predictor.predict_batch(
            self._write_batch(batch_frame(periods=3), "too-short.csv")
        )
        self.assertFalse(too_short["success"])
        self.assertIn("sedikitnya satu baris untuk diprediksi", too_short["error"])

        invalid = batch_frame()
        invalid[REQUIRED_TARGET] = invalid[REQUIRED_TARGET].astype(object)
        invalid.loc[1, REQUIRED_TARGET] = "invalid"
        invalid_result = self.predictor.predict_batch(
            self._write_batch(invalid, "invalid-seed.csv")
        )
        self.assertFalse(invalid_result["success"])
        self.assertIn("numerik dan positif", invalid_result["error"])

    def test_unordered_rows_are_sorted_before_seed_and_prediction(self):
        frame = batch_frame().iloc[[3, 0, 2, 1, 4]].reset_index(drop=True)
        result = self.predictor.predict_batch(self._write_batch(frame))
        self.assertTrue(result["success"], result)
        self.assertTrue(result["summary"]["input_reordered"])
        exported = self._export()
        self.assertTrue(pd.to_datetime(exported["Timestamp"]).is_monotonic_increasing)
        self.assertEqual(exported.loc[3, ["Kolom T1", "Kolom T2", "Kolom T3"]].tolist(), [30, 20, 10])

    def test_duplicate_timestamps_are_rejected(self):
        frame = batch_frame()
        frame.loc[1, ["Date", "Time"]] = frame.loc[0, ["Date", "Time"]].values
        result = self.predictor.predict_batch(self._write_batch(frame))
        self.assertFalse(result["success"])
        self.assertIn("timestamp duplikat", result["error"])

    def test_invalid_operational_row_is_labeled_and_not_added_to_lags(self):
        frame = batch_frame(periods=6)
        frame.loc[4, "Inlet TSS (mg/L)"] = np.nan
        result = self.predictor.predict_batch(self._write_batch(frame))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["summary"]["invalid_rows"], 1)
        exported = self._export()
        self.assertEqual(exported.loc[4, "Prediction_Status"], "Invalid")
        self.assertTrue(pd.isna(exported.loc[4, "Predicted_Dosis_CL80_ppm"]))
        self.assertEqual(exported.loc[5, "Kolom T1"], 31)

    def test_manual_lags_are_ignored_and_exports_are_unambiguous(self):
        frame = batch_frame()
        frame["Dosis T1"] = 999
        frame["Dosis T2"] = 999
        frame["Dosis T3"] = 999
        result = self.predictor.predict_batch(self._write_batch(frame))
        self.assertTrue(result["success"], result)
        self.assertEqual(
            result["summary"]["manual_lag_columns_ignored"],
            ["Dosis T1", "Dosis T2", "Dosis T3"],
        )
        exported = self._export()
        self.assertFalse({"Dosis T1", "Dosis T2", "Dosis T3"} & set(exported.columns))
        self.assertIn("Historical_Dosis_CL80_ppm", exported.columns)
        self.assertIn("Predicted_Dosis_CL80_ppm", exported.columns)
        self.assertFalse(dataframe_has_forbidden_chemicals(exported))

    def test_safe_target_alias_is_recognized(self):
        frame = batch_frame().rename(columns={REQUIRED_TARGET: "CL80 Dose"})
        result = self.predictor.predict_batch(self._write_batch(frame))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["summary"]["seed_rows_used"], 3)


if __name__ == "__main__":
    unittest.main()
