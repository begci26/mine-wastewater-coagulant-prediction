import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import Config
from utils.helpers import (
    ARTIFACT_SCHEMA_VERSION,
    BASE_INPUT_FEATURES,
    MODEL_FEATURES,
    REQUIRED_COLUMNS,
    REQUIRED_TARGET,
    forbidden_chemical_columns,
)

NULL_MARKERS = {"", "-", "na", "n/a", "null", "none", "nan"}


class FatalConversionError(ValueError):
    """Raised when a required numeric column has no usable observations."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _conversion_location(df, index):
    date = df.at[index, "Date"] if "Date" in df.columns else None
    time = df.at[index, "Time"] if "Time" in df.columns else None
    timestamp = " ".join(
        str(value).strip()
        for value in (date, time)
        if value is not None and not pd.isna(value)
    )
    try:
        row_index = int(index)
    except (TypeError, ValueError):
        row_index = str(index)
    return row_index, timestamp or None


def normalize_numeric_value(value):
    """Parse one numeric value without confusing decimal and grouping separators."""
    if pd.isna(value):
        return np.nan, "missing", "Nilai kosong", "Dipertahankan sebagai missing"
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        if np.isfinite(number):
            return number, "success", "Numerik valid", "Dikonversi langsung"
        return np.nan, "invalid", "Nilai tak hingga", "Dikonversi menjadi missing"

    original = str(value)
    text = original.replace("\u00a0", " ").strip()
    if text.casefold() in NULL_MARKERS:
        return np.nan, "missing", "Penanda null dikenal", "Dikonversi menjadi missing"

    compact = text
    normalized = compact != original

    # Spaces are removed only when they form conventional three-digit groups.
    if " " in compact:
        if re.fullmatch(r"[+-]?\d{1,3}(?: \d{3})+(?:[.,]\d+)?", compact):
            compact = compact.replace(" ", "")
            normalized = True
        else:
            return np.nan, "invalid", "Spasi bukan pemisah ribuan yang valid", "Dikonversi menjadi missing"

    if "," in compact and "." in compact:
        if re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+,\d+", compact):
            compact = compact.replace(".", "").replace(",", ".")
            normalized = True
        elif re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+\.\d+", compact):
            compact = compact.replace(",", "")
            normalized = True
        else:
            return np.nan, "invalid", "Format pemisah desimal/ribuan tidak konsisten", "Dikonversi menjadi missing"
    elif "," in compact:
        if re.fullmatch(r"[+-]?\d+,\d{1,2}", compact):
            compact = compact.replace(",", ".")
            normalized = True
        elif re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", compact):
            compact = compact.replace(",", "")
            normalized = True
        else:
            return np.nan, "invalid", "Pemisah koma ambigu atau tidak valid", "Dikonversi menjadi missing"
    elif compact.count(".") > 1:
        return np.nan, "invalid", "Terdapat lebih dari satu titik desimal", "Dikonversi menjadi missing"

    try:
        number = float(compact)
    except (TypeError, ValueError):
        return np.nan, "invalid", "Kontaminasi teks/non-numerik", "Dikonversi menjadi missing"
    if not np.isfinite(number):
        return np.nan, "invalid", "Nilai tak hingga", "Dikonversi menjadi missing"
    if normalized:
        return number, "normalized", "Format lokal/whitespace", "Dinormalisasi lalu dikonversi"
    return number, "success", "Numerik valid", "Dikonversi langsung"


class IQRWinsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature using IQR bounds learned by fit() only."""

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        self.q1_ = np.nanpercentile(array, 25, axis=0)
        self.q3_ = np.nanpercentile(array, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        self.lower_bounds_ = self.q1_ - self.factor * self.iqr_
        self.upper_bounds_ = self.q3_ + self.factor * self.iqr_
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=float)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("Jumlah fitur tidak cocok dengan IQRWinsorizer yang telah dilatih.")
        return np.clip(array, self.lower_bounds_, self.upper_bounds_)


class LeakageSafePreprocessor(BaseEstimator, TransformerMixin):
    """Median imputation -> train-fitted IQR clipping -> StandardScaler."""

    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        names = list(X.columns) if hasattr(X, "columns") else list(self.feature_names or [])
        if names and names != MODEL_FEATURES:
            raise ValueError(f"Urutan fitur tidak kompatibel. Diharapkan: {MODEL_FEATURES}")
        self.feature_names_in_ = MODEL_FEATURES.copy()
        self.schema_version_ = ARTIFACT_SCHEMA_VERSION
        self.pipeline_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("winsorizer", IQRWinsorizer(factor=1.5)),
                ("scaler", StandardScaler()),
            ]
        )
        self.pipeline_.fit(self._ordered(X), y)
        self.n_features_in_ = len(MODEL_FEATURES)
        return self

    def _ordered(self, X):
        if hasattr(X, "columns"):
            missing = [feature for feature in MODEL_FEATURES if feature not in X.columns]
            if missing:
                raise ValueError(f"Fitur model tidak lengkap: {', '.join(missing)}")
            return X.loc[:, MODEL_FEATURES]
        array = np.asarray(X)
        if array.ndim != 2 or array.shape[1] != len(MODEL_FEATURES):
            raise ValueError(f"Input harus memiliki tepat {len(MODEL_FEATURES)} fitur.")
        return array

    def transform(self, X):
        if getattr(self, "schema_version_", None) != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Artefak preprocessor lama tidak kompatibel dengan skema 11 fitur.")
        return self.pipeline_.transform(self._ordered(X))

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    @property
    def imputer_(self):
        return self.pipeline_.named_steps["imputer"]

    @property
    def winsorizer_(self):
        return self.pipeline_.named_steps["winsorizer"]

    @property
    def scaler_(self):
        return self.pipeline_.named_steps["scaler"]

    def statistics_metadata(self):
        return {
            "feature_names": MODEL_FEATURES.copy(),
            "medians": dict(zip(MODEL_FEATURES, self.imputer_.statistics_.astype(float))),
            "iqr_bounds": {
                feature: {
                    "q1": float(self.winsorizer_.q1_[index]),
                    "q3": float(self.winsorizer_.q3_[index]),
                    "iqr": float(self.winsorizer_.iqr_[index]),
                    "lower_bound": float(self.winsorizer_.lower_bounds_[index]),
                    "upper_bound": float(self.winsorizer_.upper_bounds_[index]),
                }
                for index, feature in enumerate(MODEL_FEATURES)
            },
            "scaler_mean": dict(zip(MODEL_FEATURES, self.scaler_.mean_.astype(float))),
            "scaler_scale": dict(zip(MODEL_FEATURES, self.scaler_.scale_.astype(float))),
            "fit_scope": "training_period_only",
        }

    def processing_status_metadata(self, X_train_raw, X_test_raw, completed_at=None):
        """Return auditable evidence for UI status reconstruction."""
        completed_at = completed_at or pd.Timestamp.now().isoformat()
        train_imputed = self.imputer_.transform(X_train_raw.loc[:, MODEL_FEATURES])
        test_imputed = self.imputer_.transform(X_test_raw.loc[:, MODEL_FEATURES])

        imputation_columns = {}
        outlier_bounds = {}
        total_missing_before = 0
        total_missing_after = 0
        total_train_clipped = 0
        total_test_clipped = 0

        for index, feature in enumerate(MODEL_FEATURES):
            train_missing_before = int(X_train_raw[feature].isna().sum())
            test_missing_before = int(X_test_raw[feature].isna().sum())
            train_missing_after = int(np.isnan(train_imputed[:, index]).sum())
            test_missing_after = int(np.isnan(test_imputed[:, index]).sum())
            total_missing_before += train_missing_before + test_missing_before
            total_missing_after += train_missing_after + test_missing_after
            imputation_columns[feature] = {
                "train_missing_before": train_missing_before,
                "test_missing_before": test_missing_before,
                "total_missing_before": train_missing_before + test_missing_before,
                "training_median": float(self.imputer_.statistics_[index]),
                "train_missing_after": train_missing_after,
                "test_missing_after": test_missing_after,
                "total_missing_after": train_missing_after + test_missing_after,
            }

            lower = float(self.winsorizer_.lower_bounds_[index])
            upper = float(self.winsorizer_.upper_bounds_[index])
            train_clipped = int(
                ((train_imputed[:, index] < lower) | (train_imputed[:, index] > upper)).sum()
            )
            test_clipped = int(
                ((test_imputed[:, index] < lower) | (test_imputed[:, index] > upper)).sum()
            )
            total_train_clipped += train_clipped
            total_test_clipped += test_clipped
            outlier_bounds[feature] = {
                "q1": float(self.winsorizer_.q1_[index]),
                "q3": float(self.winsorizer_.q3_[index]),
                "iqr": float(self.winsorizer_.iqr_[index]),
                "lower_bound": lower,
                "upper_bound": upper,
                "train_clipped_count": train_clipped,
                "test_clipped_count": test_clipped,
            }

        return {
            "split_completed": True,
            "split_method": "chronological_without_shuffle",
            "imputation": {
                "status": "completed",
                "method": "median",
                "fit_source": "training_only",
                "applied_to": ["training", "testing"],
                "columns": imputation_columns,
                "total_missing_before": total_missing_before,
                "total_missing_after": total_missing_after,
                "completed_at": completed_at,
            },
            "outlier_handling": {
                "status": "completed",
                "method": "IQR_clipping",
                "factor": 1.5,
                "fit_source": "training_only",
                "applied_to": ["training", "testing"],
                "bounds": outlier_bounds,
                "total_train_clipped": total_train_clipped,
                "total_test_clipped": total_test_clipped,
                "completed_at": completed_at,
            },
            "scaling": {
                "status": "completed",
                "method": "StandardScaler",
                "fit_source": "training_only",
                "applied_to": ["training", "testing"],
                "completed_at": completed_at,
            },
        }


def build_fold_safe_pipeline(estimator):
    return Pipeline(
        [
            ("preprocessor", LeakageSafePreprocessor(feature_names=MODEL_FEATURES)),
            ("estimator", estimator),
        ]
    )


class WastewaterPreprocessor:
    """Chronological cleaning, feature engineering, split, and train-only transforms."""

    def __init__(self):
        self.preprocessor = None

    def convert_types(self, df):
        clean = df.copy()
        details = []
        column_summary = {}
        warnings = []
        fatal_errors = []
        for column in BASE_INPUT_FEATURES + [REQUIRED_TARGET]:
            if column in clean.columns:
                counts = {"success": 0, "normalized": 0, "missing": 0, "invalid": 0}
                converted_values = []
                for index, original in clean[column].items():
                    converted, category, root_cause, action = normalize_numeric_value(original)
                    counts[category] += 1
                    converted_values.append(converted)
                    if category != "success":
                        row_index, timestamp = _conversion_location(df, index)
                        is_target = column == REQUIRED_TARGET
                        if category == "normalized":
                            final_status = "berhasil_dinormalisasi"
                        elif is_target:
                            action = f"{action}; baris dikeluarkan karena target tidak valid"
                            final_status = "dikeluarkan_invalid"
                        else:
                            action = f"{action}; akan diimputasi dengan median training"
                            final_status = "missing_akan_diimputasi"
                        details.append(
                            {
                                "column": column,
                                "row_index": row_index,
                                "timestamp": timestamp,
                                "original_value": None if pd.isna(original) else str(original),
                                "root_cause": root_cause,
                                "action": action,
                                "final_status": final_status,
                            }
                        )
                clean[column] = pd.Series(converted_values, index=clean.index, dtype=float)
                usable = int(clean[column].notna().sum())
                column_summary[column] = {**counts, "usable": usable, "total": int(len(clean))}
                if counts["normalized"] or counts["missing"] or counts["invalid"]:
                    warnings.append(
                        f"{column}: {counts['normalized']} dinormalisasi, "
                        f"{counts['missing'] + counts['invalid']} menjadi missing/ditangani"
                    )
                if usable == 0:
                    fatal_errors.append(f"{column}: tidak memiliki nilai numerik yang dapat digunakan")

        report = {
            "column_summary": column_summary,
            "details": details,
            "warnings": warnings,
            "fatal_errors": fatal_errors,
            "has_fatal_errors": bool(fatal_errors),
        }
        return clean, report

    @staticmethod
    def build_timestamp(df):
        date_text = df["Date"].astype(str).str.strip()
        time_text = df["Time"].astype(str).str.strip()
        return pd.to_datetime(date_text + " " + time_text, errors="coerce", dayfirst=False)

    def clean_and_engineer(self, df):
        forbidden = forbidden_chemical_columns(df.columns)
        if forbidden:
            raise ValueError(
                "Dataset aktif belum tersanitasi; ditemukan kolom kimia: " + ", ".join(forbidden)
            )
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")

        clean, conversion_report = self.convert_types(df)
        if conversion_report["has_fatal_errors"]:
            raise FatalConversionError(
                "Konversi numerik fatal: " + "; ".join(conversion_report["fatal_errors"]),
                conversion_report,
            )
        initial_rows = len(clean)
        target_missing_or_invalid = clean[REQUIRED_TARGET].isna()
        target_nonpositive = clean[REQUIRED_TARGET].notna() & (clean[REQUIRED_TARGET] <= 0)
        invalid_target_mask = target_missing_or_invalid | target_nonpositive
        invalid_target_rows = int(invalid_target_mask.sum())
        clean = clean.loc[~invalid_target_mask].copy()

        high_tss_mask = clean["Inlet TSS (mg/L)"] > 2000
        high_tss_rows = int(high_tss_mask.fillna(False).sum())
        clean = clean.loc[~high_tss_mask.fillna(False)].copy()

        clean["Timestamp"] = self.build_timestamp(clean)
        invalid_timestamp_rows = int(clean["Timestamp"].isna().sum())
        clean = clean.dropna(subset=["Timestamp"]).sort_values("Timestamp", kind="mergesort")

        before_duplicates = len(clean)
        duplicate_subset = [column for column in clean.columns if column != "Timestamp"]
        clean = clean.drop_duplicates(subset=duplicate_subset, keep="first")
        duplicate_rows = before_duplicates - len(clean)

        clean["Kolom T1"] = clean[REQUIRED_TARGET].shift(1)
        clean["Kolom T2"] = clean[REQUIRED_TARGET].shift(2)
        clean["Kolom T3"] = clean[REQUIRED_TARGET].shift(3)
        inlet_tss = clean["Inlet TSS (mg/L)"]
        clean["Efisiensi_TSS"] = np.where(
            inlet_tss.ne(0),
            ((inlet_tss - clean["Outlet TSS (mg/L)"]) / inlet_tss) * 100.0,
            np.nan,
        )
        clean["Delta_pH"] = clean["Outlet pH"] - clean["Inlet pH"]
        clean["Beban_TSS"] = inlet_tss * clean["Inlet Disch (m3/s)"]
        clean = clean.replace([np.inf, -np.inf], np.nan)

        lag_incomplete_mask = clean[["Kolom T1", "Kolom T2", "Kolom T3"]].isna().any(axis=1)
        lag_incomplete_rows = int(lag_incomplete_mask.sum())
        clean = clean.loc[~lag_incomplete_mask].reset_index(drop=True)
        clean = clean.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)

        summary = {
            "preprocessing_input_rows": int(initial_rows),
            "invalid_target_rows_removed": invalid_target_rows,
            "inlet_tss_over_2000_rows_removed": high_tss_rows,
            "invalid_timestamp_rows_removed": invalid_timestamp_rows,
            "duplicate_rows_removed": int(duplicate_rows),
            "lag_incomplete_rows_removed": lag_incomplete_rows,
            "final_row_count": int(len(clean)),
            "type_conversion_warnings": conversion_report["warnings"],
            "conversion_report": conversion_report,
            "has_fatal_conversion_errors": False,
        }
        return clean, summary

    @staticmethod
    def chronological_split(engineered, test_fraction=0.2):
        if len(engineered) < 12:
            raise ValueError("Dataset terlalu kecil untuk split kronologis dan TimeSeriesSplit 5 bagian.")
        if not 0 < test_fraction < 1:
            raise ValueError("Proporsi data uji harus berada di antara 0 dan 1.")
        data = engineered.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
        train_fraction = 1 - test_fraction
        nominal = int(len(data) * train_fraction)
        candidates = [
            index
            for index in range(1, len(data))
            if data.loc[index - 1, "Timestamp"] < data.loc[index, "Timestamp"]
        ]
        if not candidates:
            raise ValueError("Timestamp tidak menyediakan batas waktu unik untuk train-test split.")
        split_index = min(candidates, key=lambda index: abs(index - nominal))
        train = data.iloc[:split_index].copy()
        test = data.iloc[split_index:].copy()
        if train.empty or test.empty or train["Timestamp"].max() >= test["Timestamp"].min():
            raise ValueError("Validasi kronologis gagal: periode training harus mendahului testing.")
        return train, test

    def prepare_and_save(self, engineered, test_fraction=0.2):
        train, test = self.chronological_split(
            engineered,
            test_fraction=test_fraction,
        )
        X_train_raw = train.loc[:, MODEL_FEATURES].copy()
        X_test_raw = test.loc[:, MODEL_FEATURES].copy()
        y_train = train[REQUIRED_TARGET].copy()
        y_test = test[REQUIRED_TARGET].copy()

        self.preprocessor = LeakageSafePreprocessor(feature_names=MODEL_FEATURES)
        X_train = self.preprocessor.fit_transform(X_train_raw, y_train)
        X_test = self.preprocessor.transform(X_test_raw)
        processing_status = self.preprocessor.processing_status_metadata(
            X_train_raw, X_test_raw
        )

        processed_dir = os.path.join(Config.UPLOAD_FOLDER, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        pd.DataFrame(X_train, columns=MODEL_FEATURES).to_csv(
            os.path.join(processed_dir, "x_train.csv"), index=False
        )
        pd.DataFrame(X_test, columns=MODEL_FEATURES).to_csv(
            os.path.join(processed_dir, "x_test.csv"), index=False
        )
        X_train_raw.to_csv(os.path.join(processed_dir, "x_train_raw.csv"), index=False)
        X_test_raw.to_csv(os.path.join(processed_dir, "x_test_raw.csv"), index=False)
        y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
        train[["Timestamp"]].to_csv(os.path.join(processed_dir, "train_timestamps.csv"), index=False)
        test[["Timestamp"]].to_csv(os.path.join(processed_dir, "test_timestamps.csv"), index=False)

        os.makedirs(Config.MODEL_FOLDER, exist_ok=True)
        joblib.dump(self.preprocessor, os.path.join(Config.MODEL_FOLDER, "preprocessor.joblib"))
        metadata = {
            "total_final_observations": int(len(engineered)),
            "training_observations": int(len(train)),
            "testing_observations": int(len(test)),
            "training_date_range": [
                train["Timestamp"].min().isoformat(),
                train["Timestamp"].max().isoformat(),
            ],
            "testing_date_range": [
                test["Timestamp"].min().isoformat(),
                test["Timestamp"].max().isoformat(),
            ],
            "split_ratio": f"{len(train) / len(engineered):.4f}:{len(test) / len(engineered):.4f}",
            "requested_test_fraction": float(test_fraction),
            "requested_test_percentage": int(round(test_fraction * 100)),
            "split_method": "chronological_without_shuffle",
            "final_feature_names": MODEL_FEATURES.copy(),
            "feature_count": len(MODEL_FEATURES),
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "preprocessing": self.preprocessor.statistics_metadata(),
            **processing_status,
        }
        return train, test, metadata

    # Compatibility helpers used by chart routes.
    def check_missing_values(self, df):
        return {
            column: {
                "count": int(df[column].isna().sum()),
                "percentage": round(100 * df[column].isna().mean(), 2),
            }
            for column in df.columns
        }

    def check_duplicates(self, df):
        return int(df.duplicated().sum())

    def remove_duplicates(self, df):
        result = df.drop_duplicates()
        return result, len(df) - len(result)

    def detect_outliers_iqr(self, df):
        summary = {}
        indices = set()
        for column in [feature for feature in MODEL_FEATURES if feature in df.columns]:
            values = pd.to_numeric(df[column], errors="coerce")
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (values < lower) | (values > upper)
            indices.update(df.index[mask].tolist())
            summary[column] = {
                "outliers_count": int(mask.sum()),
                "q1": round(float(q1), 4),
                "q3": round(float(q3), 4),
                "iqr": round(float(iqr), 4),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
            }
        return summary, indices

    @staticmethod
    def load_processed_splits(raw=False):
        folder = os.path.join(Config.UPLOAD_FOLDER, "processed")
        suffix = "_raw" if raw else ""
        x_train = pd.read_csv(os.path.join(folder, f"x_train{suffix}.csv"))
        x_test = pd.read_csv(os.path.join(folder, f"x_test{suffix}.csv"))
        if x_train.columns.tolist() != MODEL_FEATURES or x_test.columns.tolist() != MODEL_FEATURES:
            raise ValueError("Split lama tidak kompatibel: model memerlukan tepat 11 fitur.")
        y_train = pd.read_csv(os.path.join(folder, "y_train.csv")).iloc[:, 0]
        y_test = pd.read_csv(os.path.join(folder, "y_test.csv")).iloc[:, 0]
        return x_train, x_test, y_train, y_test
