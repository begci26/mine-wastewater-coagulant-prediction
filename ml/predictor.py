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


BATCH_TARGET_ALIASES = (
    REQUIRED_TARGET,
    "Dosis CL-80",
    "CL-80",
    "CL80 Dose",
)
MANUAL_LAG_COLUMNS = (
    "Dosis T1",
    "Dosis T2",
    "Dosis T3",
    "Kolom T1",
    "Kolom T2",
    "Kolom T3",
)
BATCH_HISTORY_ERROR = (
    "Prediksi batch memerlukan tiga riwayat dosis CL-80 sebelum baris pertama "
    "yang diprediksi. Sediakan riwayat pada dataset aktif atau sertakan minimal "
    "tiga baris awal dengan nilai Dosis CL-80 sebagai seed history."
)


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
        if len(set(clean.columns)) != len(clean.columns):
            raise ValueError(
                "Nama kolom batch duplikat setelah spasi nama kolom dibersihkan."
            )
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

    @staticmethod
    def _parse_timestamps(frame):
        date_text = frame["Date"].astype(str).str.strip()
        time_text = frame["Time"].astype(str).str.strip()
        parsed_dates = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        for date_format in (
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%d-%b-%y",
            "%d-%b-%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
        ):
            unresolved = parsed_dates.isna()
            if not unresolved.any():
                break
            parsed_dates.loc[unresolved] = pd.to_datetime(
                date_text.loc[unresolved],
                format=date_format,
                errors="coerce",
            )

        normalized_time = time_text.where(
            ~time_text.str.fullmatch(r"\d{1,2}:\d{2}"),
            time_text + ":00",
        )
        parsed_times = pd.to_timedelta(normalized_time, errors="coerce")
        numeric_times = pd.to_numeric(time_text, errors="coerce")
        numeric_mask = parsed_times.isna() & numeric_times.between(0, 1, inclusive="left")
        parsed_times.loc[numeric_mask] = pd.to_timedelta(
            numeric_times.loc[numeric_mask], unit="D"
        )
        return parsed_dates.dt.normalize() + parsed_times

    @staticmethod
    def _exact_alias(columns, aliases):
        lookup = {str(column).strip().casefold(): column for column in columns}
        matches = []
        for alias in aliases:
            match = lookup.get(alias.casefold())
            if match is not None and match not in matches:
                matches.append(match)
        if len(matches) > 1:
            raise ValueError(
                "Batch memiliki lebih dari satu kolom riwayat CL-80 yang dikenali: "
                + ", ".join(matches)
            )
        return matches[0] if matches else None

    def _active_history_before(self, batch_start):
        path = get_dataset_path()
        if not os.path.exists(path):
            return []
        history = pd.read_csv(path)
        required = {"Date", "Time", REQUIRED_TARGET}
        if not required.issubset(history.columns):
            return []
        history["Timestamp"] = self._parse_timestamps(history)
        history["Dose"] = pd.to_numeric(history[REQUIRED_TARGET], errors="coerce")
        history = history.loc[
            history["Timestamp"].notna()
            & history["Dose"].notna()
            & history["Dose"].gt(0)
            & history["Timestamp"].lt(batch_start),
            ["Timestamp", "Dose"],
        ]
        history = history.loc[
            ~history["Timestamp"].duplicated(keep=False)
        ].sort_values("Timestamp", kind="mergesort")
        return history.tail(3).to_dict("records") if len(history) >= 3 else []

    def _prediction_history_before(self, batch_start, model_name):
        candidates = []
        paths = [
            os.path.join(
                Config.OUTPUT_FOLDER,
                "prediksi",
                "riwayat_prediksi_akumulatif.csv",
            ),
            os.path.join(
                Config.UPLOAD_FOLDER,
                "output",
                "prediksi",
                "riwayat_prediksi_akumulatif.csv",
            ),
        ]
        for path in paths:
            if not os.path.exists(path):
                continue
            history = pd.read_csv(path)
            required = {
                "Prediction_Timestamp",
                "Predicted_Dosis_CL80_ppm",
                "Model",
            }
            if not required.issubset(history.columns):
                continue
            timestamps = pd.to_datetime(
                history["Prediction_Timestamp"], errors="coerce"
            )
            doses = pd.to_numeric(
                history["Predicted_Dosis_CL80_ppm"], errors="coerce"
            )
            compatible = history["Model"].astype(str).eq(str(model_name))
            valid = (
                timestamps.notna()
                & timestamps.lt(batch_start)
                & doses.notna()
                & doses.gt(0)
                & compatible
            )
            candidates.extend(
                {
                    "Timestamp": timestamp,
                    "Dose": float(dose),
                }
                for timestamp, dose in zip(timestamps.loc[valid], doses.loc[valid])
            )
        if not candidates:
            return []
        history = (
            pd.DataFrame(candidates)
            .sort_values("Timestamp", kind="mergesort")
            .drop_duplicates(subset=["Timestamp"], keep=False)
        )
        return history.tail(3).to_dict("records") if len(history) >= 3 else []

    def _resolve_external_history(self, batch_start, model_name):
        active = self._active_history_before(batch_start)
        if len(active) == 3:
            return active, "active dataset"
        predictions = self._prediction_history_before(batch_start, model_name)
        if len(predictions) == 3:
            return predictions, "prediction history"
        return [], None

    def predict_batch(self, file_path):
        try:
            model, preprocessor, metadata = self._load_artifacts()
            source = (
                pd.read_excel(file_path)
                if str(file_path).lower().endswith(".xlsx")
                else pd.read_csv(file_path)
            )
            clean, sanitization = self._sanitize_batch(source)
            if clean.empty:
                raise ValueError("Batch tidak memiliki baris yang layak setelah sanitasi.")

            required = ["Date", "Time"] + BASE_INPUT_FEATURES
            missing = [column for column in required if column not in clean.columns]
            if missing:
                raise ValueError("Kolom batch wajib tidak lengkap: " + ", ".join(missing))

            target_column = self._exact_alias(clean.columns, BATCH_TARGET_ALIASES)
            manual_lag_columns = [
                column for column in MANUAL_LAG_COLUMNS if column in clean.columns
            ]
            clean = clean.drop(columns=manual_lag_columns, errors="ignore")
            warnings = []
            if manual_lag_columns:
                warnings.append(
                    "Kolom lag manual diabaikan; T1, T2, dan T3 dibentuk otomatis: "
                    + ", ".join(manual_lag_columns)
                )

            for column in BASE_INPUT_FEATURES:
                clean[column] = pd.to_numeric(clean[column], errors="coerce")
            invalid_numeric = clean[BASE_INPUT_FEATURES].isna().any(axis=1)
            clean["_Validation_Error"] = ""
            clean.loc[invalid_numeric, "_Validation_Error"] = (
                "Input operasional harus numerik dan lengkap."
            )
            invalid_ranges = (
                clean["Inlet TSS (mg/L)"].lt(0)
                | clean["Outlet TSS (mg/L)"].lt(0)
                | clean["Inlet Disch (m3/s)"].lt(0)
                | ~clean["Inlet pH"].between(0, 14)
                | ~clean["Outlet pH"].between(0, 14)
            ) & ~invalid_numeric
            if invalid_ranges.any():
                clean.loc[invalid_ranges, "_Validation_Error"] = (
                    "Input operasional berada di luar rentang yang valid."
                )
            invalid_input_count = int(
                clean["_Validation_Error"].astype(bool).sum()
            )
            if invalid_input_count:
                warnings.append(
                    f"{invalid_input_count} baris input invalid ditandai dan tidak "
                    "digunakan untuk prediksi maupun lag."
                )

            clean["Timestamp"] = self._parse_timestamps(clean)
            if clean["Timestamp"].isna().any():
                raise ValueError(
                    f"Batch memiliki {int(clean['Timestamp'].isna().sum())} Date/Time "
                    "yang tidak dapat diinterpretasi."
                )
            duplicate_timestamps = clean["Timestamp"].duplicated(keep=False)
            if duplicate_timestamps.any():
                duplicates = (
                    clean.loc[duplicate_timestamps, "Timestamp"]
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                    .drop_duplicates()
                    .head(5)
                    .tolist()
                )
                raise ValueError(
                    "Batch memiliki timestamp duplikat; perbaiki sebelum prediksi: "
                    + ", ".join(duplicates)
                )
            input_reordered = not clean["Timestamp"].is_monotonic_increasing
            clean = clean.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
            if input_reordered:
                warnings.append(
                    "Baris batch diurutkan naik berdasarkan timestamp sebelum lag dibentuk."
                )

            external_history, history_source = self._resolve_external_history(
                clean["Timestamp"].iloc[0],
                metadata.get("best_model"),
            )

            seed_count = 0
            first_prediction_index = 0
            if external_history:
                lag_buffer = [float(item["Dose"]) for item in external_history]
            else:
                if target_column is None:
                    raise ValueError(BATCH_HISTORY_ERROR)
                if len(clean) < 4:
                    raise ValueError(
                        BATCH_HISTORY_ERROR
                        + " File batch juga harus memiliki sedikitnya satu baris untuk diprediksi."
                    )
                seed_values = pd.to_numeric(
                    clean.loc[:2, target_column], errors="coerce"
                )
                if seed_values.isna().any() or seed_values.le(0).any():
                    raise ValueError(
                        "Tiga baris awal yang akan digunakan sebagai seed history harus "
                        "memiliki nilai Dosis CL-80 numerik dan positif."
                    )
                if clean.loc[:2, "_Validation_Error"].astype(bool).any():
                    raise ValueError(
                        "Tiga baris awal seed history harus memiliki input operasional "
                        "yang valid dan lengkap."
                    )
                lag_buffer = seed_values.astype(float).tolist()
                seed_count = 3
                first_prediction_index = 3
                history_source = "batch seed"
                warnings.append(
                    "Riwayat eksternal tidak mencukupi; tiga baris awal digunakan sebagai "
                    "seed history dan tidak diprediksi."
                )

            output_rows = []
            for index, record in clean.iterrows():
                inputs = {feature: record[feature] for feature in BASE_INPUT_FEATURES}
                feature_row = self.build_feature_row(inputs).iloc[0].to_dict()
                is_invalid = bool(record["_Validation_Error"])
                output = {
                    "Date": record["Timestamp"].strftime("%Y-%m-%d"),
                    "Time": record["Timestamp"].strftime("%H:%M:%S"),
                    "Timestamp": record["Timestamp"],
                    "Prediction_Status": (
                        "Seed history"
                        if index < first_prediction_index
                        else "Invalid"
                        if is_invalid
                        else "Predicted"
                    ),
                    "Validation_Message": record["_Validation_Error"] or "",
                    **{feature: float(record[feature]) for feature in BASE_INPUT_FEATURES},
                    "Kolom T1": np.nan,
                    "Kolom T2": np.nan,
                    "Kolom T3": np.nan,
                    "Efisiensi_TSS": feature_row["Efisiensi_TSS"],
                    "Delta_pH": feature_row["Delta_pH"],
                    "Beban_TSS": feature_row["Beban_TSS"],
                    "Historical_Dosis_CL80_ppm": np.nan,
                    "Predicted_Dosis_CL80_ppm": np.nan,
                    "Seed_Source": history_source,
                }
                if index < first_prediction_index:
                    output["Historical_Dosis_CL80_ppm"] = float(
                        clean.loc[index, target_column]
                    )
                    output_rows.append(output)
                    continue
                if is_invalid:
                    output_rows.append(output)
                    continue

                lags = {
                    "Kolom T1": lag_buffer[-1],
                    "Kolom T2": lag_buffer[-2],
                    "Kolom T3": lag_buffer[-3],
                }
                inputs.update(lags)
                row = self.build_feature_row(inputs)
                prediction = max(0.0, float(model.predict(preprocessor.transform(row))[0]))
                output.update(lags)
                output["Predicted_Dosis_CL80_ppm"] = prediction
                output_rows.append(output)
                lag_buffer.append(prediction)

            output = pd.DataFrame(output_rows)
            forbidden = forbidden_chemical_columns(output.columns)
            if forbidden:
                raise ValueError("Ekspor batch masih mengandung kolom kimia.")

            prediction_dir = os.path.join(Config.UPLOAD_FOLDER, "predictions")
            os.makedirs(prediction_dir, exist_ok=True)
            output.to_csv(
                os.path.join(prediction_dir, "batch_predictions.csv"), index=False
            )
            output.to_excel(
                os.path.join(prediction_dir, "batch_predictions.xlsx"), index=False
            )
            predicted_count = int(
                output["Prediction_Status"].eq("Predicted").sum()
            )
            invalid_rows = int(
                sanitization["ambiguous_chemical_row_count"] + invalid_input_count
            )
            skipped_rows = int(
                sanitization["original_row_count"] - sanitization["final_active_row_count"]
            )
            first_predicted = output.loc[
                output["Prediction_Status"].eq("Predicted"), "Timestamp"
            ].min()
            summary = {
                "total_uploaded_rows": int(sanitization["original_row_count"]),
                "sanitized_rows": int(len(clean)),
                "seed_rows_used": seed_count,
                "predicted_rows": predicted_count,
                "skipped_rows": skipped_rows,
                "invalid_rows": invalid_rows,
                "invalid_input_rows": invalid_input_count,
                "ambiguous_chemical_rows": int(
                    sanitization["ambiguous_chemical_row_count"]
                ),
                "excluded_alum_lime_rows": int(
                    sanitization["excluded_alum_lime_row_count"]
                ),
                "history_source": history_source,
                "first_predicted_timestamp": (
                    first_predicted.isoformat(sep=" ")
                    if pd.notna(first_predicted)
                    else None
                ),
                "input_reordered": input_reordered,
                "manual_lag_columns_ignored": manual_lag_columns,
            }
            preview = output.head(10).astype(object)
            preview = preview.where(pd.notna(preview), None)
            return {
                "success": True,
                "csv_filename": "batch_predictions.csv",
                "xlsx_filename": "batch_predictions.xlsx",
                "preview_cols": list(output.columns),
                "preview_data": preview.values.tolist(),
                "rows": len(output),
                "sanitization": sanitization,
                "summary": summary,
                "warnings": warnings,
            }
        except Exception as error:
            app_logger.error("Batch prediction failed: %s", error, exc_info=True)
            return {
                "success": False,
                "error": str(error),
                "error_category": "validation",
            }
