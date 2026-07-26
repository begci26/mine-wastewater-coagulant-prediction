import json
import os

import joblib
import numpy as np
import pandas as pd

from config import Config
from utils.helpers import (
    ARTIFACT_SCHEMA_VERSION,
    BASE_INPUT_FEATURES,
    MODEL_FEATURES,
    OPTIONAL_CHEMICAL_COLUMNS,
    REQUIRED_TARGET,
    forbidden_chemical_columns,
    get_dataset_path,
    is_alum_lime_ratio_column,
    is_index_column,
)
from utils.logger import app_logger


class WastewaterPredictor:
    """Compatible 11-feature inference, chronological lags, batch prediction, and LOCO."""

    def __init__(self):
        self.model_path = os.path.join(Config.MODEL_FOLDER, "best_model.joblib")
        self.preprocessor_path = os.path.join(Config.MODEL_FOLDER, "preprocessor.joblib")
        self.metadata_path = os.path.join(Config.MODEL_FOLDER, "model_metadata.json")
        self.full_features = MODEL_FEATURES.copy()

    def check_model_exists(self):
        if not all(
            os.path.exists(path)
            for path in [self.model_path, self.preprocessor_path, self.metadata_path]
        ):
            return False
        try:
            self._load_artifacts()
            return True
        except Exception as error:
            app_logger.warning("Model artifact compatibility check failed: %s", error)
            return False

    def _load_artifacts(self):
        with open(self.metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Artefak model lama tidak kompatibel dengan skema saat ini.")
        if metadata.get("feature_names") != MODEL_FEATURES or metadata.get("feature_count") != 11:
            raise ValueError("Model bukan artefak 11 fitur dalam urutan yang diwajibkan.")
        preprocessor = joblib.load(self.preprocessor_path)
        if (
            getattr(preprocessor, "schema_version_", None) != ARTIFACT_SCHEMA_VERSION
            or getattr(preprocessor, "feature_names_in_", None) != MODEL_FEATURES
        ):
            raise ValueError("Preprocessor tidak kompatibel dengan skema 11 fitur.")
        model = joblib.load(self.model_path)
        if getattr(model, "n_features_in_", 11) != 11:
            raise ValueError("Model lama memiliki jumlah fitur selain 11.")
        return model, preprocessor, metadata

    @staticmethod
    def build_feature_row(inputs):
        values = {feature: float(inputs.get(feature, np.nan)) for feature in MODEL_FEATURES}
        inlet_tss = float(inputs.get("Inlet TSS (mg/L)", np.nan))
        outlet_tss = float(inputs.get("Outlet TSS (mg/L)", np.nan))
        inlet_ph = float(inputs.get("Inlet pH", np.nan))
        outlet_ph = float(inputs.get("Outlet pH", np.nan))
        discharge = float(inputs.get("Inlet Disch (m3/s)", np.nan))
        values["Efisiensi_TSS"] = (
            ((inlet_tss - outlet_tss) / inlet_tss) * 100.0
            if np.isfinite(inlet_tss) and inlet_tss != 0
            else np.nan
        )
        values["Delta_pH"] = outlet_ph - inlet_ph
        values["Beban_TSS"] = inlet_tss * discharge
        return pd.DataFrame([values], columns=MODEL_FEATURES).replace([np.inf, -np.inf], np.nan)

    def get_lag_values(self, prediction_timestamp=None):
        path = get_dataset_path()
        if not os.path.exists(path):
            raise ValueError("Dataset aktif tidak tersedia untuk mengambil riwayat CL-80.")
        history = pd.read_csv(path)
        if REQUIRED_TARGET not in history.columns or "Date" not in history or "Time" not in history:
            raise ValueError("Dataset aktif tidak memiliki riwayat Date, Time, dan Dosis CL-80.")
        history[REQUIRED_TARGET] = pd.to_numeric(history[REQUIRED_TARGET], errors="coerce")
        history["Timestamp"] = pd.to_datetime(
            history["Date"].astype(str).str.strip()
            + " "
            + history["Time"].astype(str).str.strip(),
            errors="coerce",
        )
        history = history.dropna(subset=["Timestamp", REQUIRED_TARGET])
        history = history.loc[history[REQUIRED_TARGET] > 0].sort_values("Timestamp")
        if prediction_timestamp is not None:
            timestamp = pd.to_datetime(prediction_timestamp, errors="coerce")
            if pd.isna(timestamp):
                raise ValueError("Tanggal dan waktu prediksi tidak valid.")
            history = history.loc[history["Timestamp"] < timestamp]
        doses = history[REQUIRED_TARGET].astype(float).tolist()
        if len(doses) < 3:
            raise ValueError(
                "Riwayat CL-80 sebelum waktu prediksi belum cukup; diperlukan sedikitnya "
                "tiga observasi valid untuk T1, T2, dan T3."
            )
        return {
            "Kolom T1": doses[-1],
            "Kolom T2": doses[-2],
            "Kolom T3": doses[-3],
        }

    def predict_single(self, input_features):
        try:
            model, preprocessor, _ = self._load_artifacts()
            row = self.build_feature_row(input_features)
            transformed = preprocessor.transform(row)
            prediction = max(0.0, float(model.predict(transformed)[0]))

            raw_train_path = os.path.join(
                Config.UPLOAD_FOLDER, "processed", "x_train_raw.csv"
            )
            if not os.path.exists(raw_train_path):
                raise ValueError("Data referensi training untuk LOCO tidak tersedia.")
            training = pd.read_csv(raw_train_path)
            if training.columns.tolist() != MODEL_FEATURES:
                raise ValueError("Data referensi LOCO tidak kompatibel dengan 11 fitur.")
            references = training.median(numeric_only=True)
            contributions = {}
            perturbed_predictions = {}
            for feature in MODEL_FEATURES:
                perturbed = row.copy()
                perturbed.loc[0, feature] = float(references[feature])
                perturbed_prediction = max(
                    0.0, float(model.predict(preprocessor.transform(perturbed))[0])
                )
                perturbed_predictions[feature] = perturbed_prediction
                contributions[feature] = prediction - perturbed_prediction

            return {
                "success": True,
                "prediction": round(prediction, 4),
                "base_value": round(float(training[MODEL_FEATURES].median().mean()), 4),
                "contributions": {
                    feature: round(value, 4) for feature, value in contributions.items()
                },
                "pct_contributions": {},
                "reference_values": {
                    feature: float(references[feature]) for feature in MODEL_FEATURES
                },
                "perturbed_predictions": {
                    feature: round(value, 4)
                    for feature, value in perturbed_predictions.items()
                },
                "interpretation_method": "LOCO-based reference-value perturbation",
                "contribution_definition": "baseline prediction - perturbed prediction (ppm)",
                "causal_effect": False,
                "feature_row": row.iloc[0].to_dict(),
            }
        except Exception as error:
            app_logger.error("Single prediction failed: %s", error, exc_info=True)
            return {"success": False, "error": str(error)}

    @staticmethod
    def _sanitize_batch(df):
        clean = df.copy()
        clean.columns = [str(column).strip() for column in clean.columns]
        alum, lime = OPTIONAL_CHEMICAL_COLUMNS
        presence = [column in clean.columns for column in OPTIONAL_CHEMICAL_COLUMNS]
        summary = {
            "original_row_count": int(len(clean)),
            "excluded_alum_lime_row_count": 0,
            "ambiguous_chemical_row_count": 0,
            "removed_column_names": [],
        }
        if any(presence) and not all(presence):
            raise ValueError("Batch memiliki hanya satu kolom Alum/Lime sehingga seleksi tidak aman.")
        if all(presence):
            alum_values = pd.to_numeric(clean[alum], errors="coerce")
            lime_values = pd.to_numeric(clean[lime], errors="coerce")
            ambiguous = (
                alum_values.isna()
                | lime_values.isna()
                | (alum_values < 0)
                | (lime_values < 0)
            )
            nonzero = (~ambiguous) & ((alum_values > 0) | (lime_values > 0))
            zero = (~ambiguous) & alum_values.eq(0) & lime_values.eq(0)
            summary["ambiguous_chemical_row_count"] = int(ambiguous.sum())
            summary["excluded_alum_lime_row_count"] = int(nonzero.sum())
            clean = clean.loc[zero].copy()
        removed = [
            column
            for column in clean.columns
            if is_index_column(column)
            or column in OPTIONAL_CHEMICAL_COLUMNS
            or is_alum_lime_ratio_column(column)
        ]
        clean = clean.drop(columns=removed, errors="ignore")
        summary["removed_column_names"] = removed
        summary["final_active_row_count"] = int(len(clean))
        return clean, summary

    def predict_batch(self, file_path):
        try:
            model, preprocessor, _ = self._load_artifacts()
            source = (
                pd.read_excel(file_path)
                if str(file_path).lower().endswith(".xlsx")
                else pd.read_csv(file_path)
            )
            clean, sanitization = self._sanitize_batch(source)
            required = ["Date", "Time"] + BASE_INPUT_FEATURES
            missing = [column for column in required if column not in clean.columns]
            if missing:
                raise ValueError("Kolom batch wajib tidak lengkap: " + ", ".join(missing))
            for column in BASE_INPUT_FEATURES:
                clean[column] = pd.to_numeric(clean[column], errors="coerce")
            clean["Timestamp"] = pd.to_datetime(
                clean["Date"].astype(str).str.strip()
                + " "
                + clean["Time"].astype(str).str.strip(),
                errors="coerce",
            )
            if clean["Timestamp"].isna().any():
                raise ValueError("Batch memiliki Date/Time yang tidak dapat diinterpretasi.")
            clean = clean.sort_values("Timestamp", kind="mergesort").drop_duplicates().reset_index(drop=True)

            active = pd.read_csv(get_dataset_path())
            active[REQUIRED_TARGET] = pd.to_numeric(active[REQUIRED_TARGET], errors="coerce")
            active_timestamp = pd.to_datetime(
                active["Date"].astype(str) + " " + active["Time"].astype(str), errors="coerce"
            )
            history = active.loc[
                active[REQUIRED_TARGET].notna() & (active[REQUIRED_TARGET] > 0),
                [REQUIRED_TARGET],
            ].copy()
            history["Timestamp"] = active_timestamp.loc[history.index]
            history = history.dropna(subset=["Timestamp"]).sort_values("Timestamp")

            predictions = []
            lag_records = []
            evolving_history = [
                (timestamp, float(dose))
                for timestamp, dose in zip(history["Timestamp"], history[REQUIRED_TARGET])
            ]
            for _, record in clean.iterrows():
                prior = [dose for timestamp, dose in evolving_history if timestamp < record["Timestamp"]]
                if len(prior) < 3:
                    raise ValueError(
                        f"Riwayat CL-80 tidak cukup sebelum {record['Timestamp']} untuk membentuk tiga lag."
                    )
                lags = {"Kolom T1": prior[-1], "Kolom T2": prior[-2], "Kolom T3": prior[-3]}
                inputs = {feature: record[feature] for feature in BASE_INPUT_FEATURES}
                inputs.update(lags)
                row = self.build_feature_row(inputs)
                prediction = max(0.0, float(model.predict(preprocessor.transform(row))[0]))
                predictions.append(prediction)
                lag_records.append(lags)
                evolving_history.append((record["Timestamp"], prediction))

            for lag in ["Kolom T1", "Kolom T2", "Kolom T3"]:
                clean[lag] = [values[lag] for values in lag_records]
            clean["Predicted_Dosis_CL80_ppm"] = np.round(predictions, 4)
            forbidden = forbidden_chemical_columns(clean.columns)
            if forbidden:
                raise ValueError("Ekspor batch masih mengandung kolom kimia.")

            prediction_dir = os.path.join(Config.UPLOAD_FOLDER, "predictions")
            os.makedirs(prediction_dir, exist_ok=True)
            clean.to_csv(os.path.join(prediction_dir, "batch_predictions.csv"), index=False)
            clean.to_excel(os.path.join(prediction_dir, "batch_predictions.xlsx"), index=False)
            return {
                "success": True,
                "csv_filename": "batch_predictions.csv",
                "xlsx_filename": "batch_predictions.xlsx",
                "preview_cols": list(clean.columns),
                "preview_data": clean.head(10).values.tolist(),
                "rows": len(clean),
                "sanitization": sanitization,
            }
        except Exception as error:
            app_logger.error("Batch prediction failed: %s", error, exc_info=True)
            return {"success": False, "error": str(error)}
