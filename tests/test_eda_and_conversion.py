import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import numpy as np
import pandas as pd

from app import create_app
from config import Config
from ml.preprocessor import FatalConversionError, WastewaterPreprocessor
from routes.eda import (
    calculate_vif,
    generate_bab4_outputs,
    prepare_correlation_matrix,
)
from utils.helpers import BASE_INPUT_FEATURES, MODEL_FEATURES, REQUIRED_TARGET


def eda_frame(rows=40):
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="h")
    index = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "No": index + 1,
            "Date": timestamps.strftime("%Y-%m-%d"),
            "Time": timestamps.strftime("%H:%M:%S"),
            "Timestamp": timestamps.astype(str),
            "Inlet TSS (mg/L)": 400 + index * 3 + np.sin(index),
            "Inlet pH": 7.0 + np.sin(index / 3) / 10,
            "Outlet TSS (mg/L)": 90 + index / 4 + np.cos(index),
            "Outlet pH": 7.2 + np.cos(index / 5) / 10,
            "Inlet Disch (m3/s)": 1.2 + index / 100,
            REQUIRED_TARGET: 2.5 + index / 50 + np.sin(index / 4) / 10,
        }
    )
    frame["Kolom T1"] = frame[REQUIRED_TARGET].shift(1).bfill()
    frame["Kolom T2"] = frame[REQUIRED_TARGET].shift(2).bfill()
    frame["Kolom T3"] = frame[REQUIRED_TARGET].shift(3).bfill()
    frame["Efisiensi_TSS"] = (
        (frame["Inlet TSS (mg/L)"] - frame["Outlet TSS (mg/L)"])
        / frame["Inlet TSS (mg/L)"]
        * 100
    )
    frame["Delta_pH"] = frame["Outlet pH"] - frame["Inlet pH"]
    frame["Beban_TSS"] = frame["Inlet TSS (mg/L)"] * frame["Inlet Disch (m3/s)"]
    return frame


def conversion_frame(rows=12):
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="h")
    frame = pd.DataFrame(
        {
            "Date": timestamps.strftime("%Y-%m-%d"),
            "Time": timestamps.strftime("%H:%M:%S"),
            "Inlet TSS (mg/L)": np.linspace(500, 600, rows),
            "Inlet pH": np.repeat("7.4", rows),
            "Outlet TSS (mg/L)": np.linspace(100, 110, rows),
            "Outlet pH": np.linspace(7.2, 7.3, rows),
            "Inlet Disch (m3/s)": np.linspace(1.2, 1.4, rows),
            REQUIRED_TARGET: np.linspace(2.0, 3.0, rows),
        }
    )
    return frame


class EdaRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.testing = True
        cls.client = cls.app.test_client()

    def get_distribution(self, frame, feature):
        with patch("routes.eda.get_processed_data", return_value=frame):
            return self.client.get(f"/eda/distribution/{quote(feature, safe='')}")

    def test_distribution_normal_numeric_column(self):
        response = self.get_distribution(eda_frame(), "Inlet TSS (mg/L)")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["meta"]["valid_count"], 40)
        self.assertTrue(payload["histogram"]["data"])

    def test_distribution_drops_missing_only_for_selected_column(self):
        frame = eda_frame()
        frame.loc[[2, 5], "Inlet pH"] = np.nan
        response = self.get_distribution(frame, "Inlet pH")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["meta"]["missing_or_invalid_count"], 2)

    def test_distribution_constant_column_has_explicit_state(self):
        frame = eda_frame()
        frame["Delta_pH"] = 0.2
        response = self.get_distribution(frame, "Delta_pH")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["meta"]["is_constant"])

    def test_distribution_rejects_invalid_or_non_numeric_feature(self):
        response = self.get_distribution(eda_frame(), "Timestamp")
        self.assertEqual(response.status_code, 400)
        self.assertIn("allowed_features", response.get_json())

    def test_correlation_and_vif_render_payload(self):
        frame = eda_frame()
        with patch("routes.eda.get_processed_data", return_value=frame):
            response = self.client.get("/eda/correlation_heatmap")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["heatmap"]["data"])
        self.assertTrue(payload["vif_chart"]["data"])
        _, matrix, _ = prepare_correlation_matrix(frame)
        self.assertEqual(payload["heatmap"]["data"][0]["x"], list(matrix.columns))

    def test_vif_handles_singularity_and_constant(self):
        frame = pd.DataFrame({feature: np.arange(20, dtype=float) for feature in MODEL_FEATURES})
        frame["Beban_TSS"] = 1.0
        values = calculate_vif(frame)
        self.assertIsNone(values["Beban_TSS"])
        self.assertTrue(any(np.isinf(value) for value in values.values() if value is not None))

    def test_time_series_uses_valid_timestamp_and_lags_not_chemicals(self):
        with patch("routes.eda.get_processed_data", return_value=eda_frame()):
            response = self.client.get("/eda/timeseries_analysis")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertNotIn("chem", payload)
        self.assertEqual(payload["scopes"]["lags"], ["Kolom T1", "Kolom T2", "Kolom T3"])
        self.assertEqual(len(payload["lags"]["data"]), 3)


class OutputAndConversionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.original_upload = Config.UPLOAD_FOLDER
        Config.UPLOAD_FOLDER = self.temporary.name

    def tearDown(self):
        Config.UPLOAD_FOLDER = self.original_upload
        self.temporary.cleanup()

    def test_png_and_landscape_multipage_pdf_exports(self):
        output_dir = Path(Config.UPLOAD_FOLDER) / "output" / "bab4"
        generate_bab4_outputs(eda_frame())
        for filename in ["heatmap.png", "vif.png", "laporan_eda.pdf"]:
            self.assertTrue((output_dir / filename).exists(), filename)
            self.assertGreater((output_dir / filename).stat().st_size, 100)
        pdf = (output_dir / "laporan_eda.pdf").read_bytes()
        self.assertTrue(pdf.startswith(b"%PDF"))
        media_box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)", pdf)
        self.assertIsNotNone(media_box)
        self.assertGreater(float(media_box.group(1)), float(media_box.group(2)))
        self.assertGreaterEqual(len(re.findall(rb"/Type\s*/Page\b", pdf)), 2)

    def test_comma_decimal_and_invalid_text_are_distinguished(self):
        frame = conversion_frame()
        frame.loc[2, "Inlet pH"] = "7,5"
        frame.loc[3, "Inlet pH"] = "7..9"
        converted, report = WastewaterPreprocessor().convert_types(frame)
        self.assertEqual(converted.loc[2, "Inlet pH"], 7.5)
        self.assertTrue(np.isnan(converted.loc[3, "Inlet pH"]))
        statuses = {
            detail["row_index"]: detail["final_status"]
            for detail in report["details"]
            if detail["column"] == "Inlet pH"
        }
        self.assertEqual(statuses[2], "berhasil_dinormalisasi")
        self.assertEqual(statuses[3], "missing_akan_diimputasi")
        self.assertFalse(report["has_fatal_errors"])

    def test_recoverable_warning_versus_fatal_conversion(self):
        recoverable = conversion_frame()
        recoverable.loc[0, "Inlet pH"] = "invalid"
        engineered, summary = WastewaterPreprocessor().clean_and_engineer(recoverable)
        self.assertFalse(summary["has_fatal_conversion_errors"])
        self.assertTrue(summary["type_conversion_warnings"])
        self.assertFalse(engineered.empty)

        fatal = conversion_frame()
        fatal["Inlet pH"] = "invalid"
        with self.assertRaises(FatalConversionError) as context:
            WastewaterPreprocessor().clean_and_engineer(fatal)
        self.assertTrue(context.exception.report["has_fatal_errors"])

    def test_model_scope_remains_exactly_eleven_without_chemicals(self):
        self.assertEqual(len(MODEL_FEATURES), 11)
        normalized = " ".join(MODEL_FEATURES).casefold()
        self.assertNotIn("alum", normalized)
        self.assertNotIn("lime", normalized)
        self.assertEqual(MODEL_FEATURES[: len(BASE_INPUT_FEATURES)], BASE_INPUT_FEATURES)


if __name__ == "__main__":
    unittest.main()
