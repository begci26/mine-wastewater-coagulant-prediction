import json
import unittest
from pathlib import Path

from app import create_app
from routes.evaluation import _evaluation_frame


class UiPresentationTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()

    def test_metric_filter_formats_display_only(self):
        raw = 0.8854423314569766
        metric_filter = self.app.jinja_env.filters["metric4"]
        self.assertEqual(metric_filter(raw), "0.8854")
        self.assertEqual(raw, 0.8854423314569766)

    def test_evaluation_frame_preserves_full_precision(self):
        metrics = {
            "Linear Regression": {
                "train": {"mae": 0.1, "rmse": 0.2, "r2": 0.3},
                "test": {
                    "mae": 0.29049244770470256,
                    "rmse": 0.43766720437441314,
                    "r2": 0.8854423314569766,
                },
                "r2_gap": 0.01,
                "cv": {"rmse_mean": 0.4, "rmse_std": 0.05},
                "duration": 0.2,
                "pred_duration": 0.01,
            }
        }
        frame = _evaluation_frame({"model_details": metrics})
        self.assertEqual(frame.iloc[0]["Test MAE"], metrics["Linear Regression"]["test"]["mae"])
        self.assertEqual(frame.iloc[0]["Test RMSE"], metrics["Linear Regression"]["test"]["rmse"])
        self.assertEqual(frame.iloc[0]["Test R2"], metrics["Linear Regression"]["test"]["r2"])

    def test_templates_and_css_include_responsive_accessible_classes(self):
        root = Path(__file__).resolve().parents[1]
        evaluation = (root / "templates" / "evaluation.html").read_text(encoding="utf-8")
        prediction = (root / "templates" / "prediction.html").read_text(encoding="utf-8")
        css = (root / "static" / "css" / "custom.css").read_text(encoding="utf-8")

        self.assertEqual(evaluation.count("| metric4"), 3)
        self.assertIn("eval-metric-value", evaluation)
        self.assertIn("Nilai presisi penuh:", evaluation)
        self.assertIn("overflow-wrap: anywhere", evaluation)

        self.assertIn("contribution-ranking", prediction)
        self.assertIn("Menaikkan dosis", prediction)
        self.assertIn("Menurunkan dosis", prediction)
        self.assertNotIn("alert alert-secondary border-0 small text-muted", prediction)
        for selector in [
            ".contribution-positive",
            ".contribution-negative",
            ".contribution-neutral",
            ".contribution-item:hover",
        ]:
            self.assertIn(selector, css)

    def test_saved_metadata_still_contains_unrounded_metrics(self):
        path = Path("models/model_metadata.json")
        if not path.exists():
            self.skipTest("Model metadata is not available in this checkout.")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotEqual(
            metadata["best_metrics"]["rmse"],
            round(metadata["best_metrics"]["rmse"], 4),
        )


if __name__ == "__main__":
    unittest.main()
