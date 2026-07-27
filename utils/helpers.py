import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from utils.logger import app_logger

REQUIRED_TARGET = "Dosis CL-80 (ppm)"
REQUIRED_COLUMNS = [
    "Date",
    "Time",
    "Inlet TSS (mg/L)",
    "Inlet pH",
    "Outlet TSS (mg/L)",
    "Outlet pH",
    "Inlet Disch (m3/s)",
    REQUIRED_TARGET,
]
OPTIONAL_CHEMICAL_COLUMNS = ["Dosis Alum (ppm)", "Dosis Lime (ppm)"]
BASE_INPUT_FEATURES = [
    "Inlet TSS (mg/L)",
    "Inlet pH",
    "Outlet TSS (mg/L)",
    "Outlet pH",
    "Inlet Disch (m3/s)",
]
MODEL_FEATURES = [
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
]
DATASET_PREVIEW_COLUMNS = [
    ("DATE", "Date"),
    ("TIME", "Time"),
    ("INLET TSS (MG/L)", "Inlet TSS (mg/L)"),
    ("DOSIS CL-80", REQUIRED_TARGET),
]
ROW_NUMBER_COLUMN_ALIASES = {
    "no",
    "nomor",
    "nomorurut",
    "rownumber",
}
# Backward-compatible name used by routes; it intentionally contains no chemicals.
REQUIRED_FEATURES = BASE_INPUT_FEATURES + ["Kolom T1", "Kolom T2", "Kolom T3"]
ARTIFACT_SCHEMA_VERSION = 2


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def check_dataset_uploaded():
    path = get_dataset_path()
    return os.path.exists(path) and os.path.getsize(path) > 0


def get_dataset_path():
    metadata_path = os.path.join(Config.UPLOAD_FOLDER, "active_dataset.json")
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            filename = os.path.basename(json.load(handle)["filename"])
        if filename:
            return os.path.join(Config.UPLOAD_FOLDER, filename)
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return os.path.join(Config.UPLOAD_FOLDER, "active_dataset.csv")


def set_dataset_filename(filename):
    """Remember the active upload without replacing its original filename."""
    metadata_path = os.path.join(Config.UPLOAD_FOLDER, "active_dataset.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump({"filename": os.path.basename(filename)}, handle, indent=2)


def sanitization_summary_path():
    return os.path.join(Config.UPLOAD_FOLDER, "sanitization_summary.json")


def _normalise_column_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).casefold())


def is_index_column(name):
    normalized = _normalise_column_name(name)
    return str(name).strip().casefold().startswith("unnamed:") or normalized in {
        "index",
        "level0",
    }


def is_sequential_row_number_column(name, values):
    """Identify a known row-number label only when its values form a sequence."""
    if _normalise_column_name(name) not in ROW_NUMBER_COLUMN_ALIASES:
        return False
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or numeric.empty or not np.equal(numeric, np.floor(numeric)).all():
        return False
    ordered = np.sort(numeric.astype(np.int64).to_numpy())
    return bool(
        ordered[0] in (0, 1)
        and np.array_equal(ordered, np.arange(ordered[0], ordered[0] + len(ordered)))
    )


def is_alum_lime_ratio_column(name):
    normalized = _normalise_column_name(name)
    return "alum" in normalized and "lime" in normalized


def forbidden_chemical_columns(columns):
    forbidden = []
    for column in columns:
        normalized = _normalise_column_name(column)
        if (
            normalized in {_normalise_column_name(c) for c in OPTIONAL_CHEMICAL_COLUMNS}
            or is_alum_lime_ratio_column(column)
        ):
            forbidden.append(column)
    return forbidden


def sanitize_dataframe(df):
    """Select CL-80-only observations and remove all chemical/index columns."""
    clean = df.copy()
    clean.columns = [str(column).strip() for column in clean.columns]
    if len(set(clean.columns)) != len(clean.columns):
        raise ValueError("Nama kolom duplikat terdeteksi setelah spasi nama kolom dibersihkan.")

    missing = [column for column in REQUIRED_COLUMNS if column not in clean.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")

    original_rows = len(clean)
    alum_col, lime_col = OPTIONAL_CHEMICAL_COLUMNS
    chemical_presence = [column in clean.columns for column in OPTIONAL_CHEMICAL_COLUMNS]
    if any(chemical_presence) and not all(chemical_presence):
        missing_chemical = lime_col if alum_col in clean.columns else alum_col
        raise ValueError(
            f"Kolom seleksi kimia tidak lengkap. {missing_chemical} diperlukan untuk "
            "menentukan observasi tanpa Alum dan Lime secara aman."
        )

    excluded_nonzero = 0
    ambiguous = 0
    if all(chemical_presence):
        alum = pd.to_numeric(clean[alum_col], errors="coerce")
        lime = pd.to_numeric(clean[lime_col], errors="coerce")
        ambiguous_mask = alum.isna() | lime.isna() | (alum < 0) | (lime < 0)
        nonzero_mask = (~ambiguous_mask) & ((alum > 0) | (lime > 0))
        valid_zero_mask = (~ambiguous_mask) & alum.eq(0) & lime.eq(0)
        ambiguous = int(ambiguous_mask.sum())
        excluded_nonzero = int(nonzero_mask.sum())
        clean = clean.loc[valid_zero_mask].copy()

    removed_columns = [
        column
        for column in clean.columns
        if is_index_column(column)
        or column in OPTIONAL_CHEMICAL_COLUMNS
        or is_alum_lime_ratio_column(column)
    ]
    clean = clean.drop(columns=removed_columns, errors="ignore")
    forbidden = forbidden_chemical_columns(clean.columns)
    if forbidden:
        raise ValueError(f"Kolom kimia terlarang masih tersisa: {', '.join(forbidden)}")

    summary = {
        "original_row_count": int(original_rows),
        "excluded_alum_lime_row_count": excluded_nonzero,
        "ambiguous_chemical_row_count": ambiguous,
        "final_active_row_count": int(len(clean)),
        "removed_column_names": removed_columns,
    }
    return clean.reset_index(drop=True), summary


def save_sanitization_summary(summary):
    with open(sanitization_summary_path(), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def load_sanitization_summary():
    path = sanitization_summary_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def invalidate_generated_artifacts():
    """Remove generated state that cannot be reused with a newly uploaded dataset."""
    targets = [
        Path(Config.UPLOAD_FOLDER) / "processed",
        Path(Config.UPLOAD_FOLDER) / "predictions",
        Path(Config.UPLOAD_FOLDER) / "output",
        Path(Config.MODEL_FOLDER),
        Path(Config.OUTPUT_FOLDER) / "prediksi",
        Path(Config.OUTPUT_FOLDER) / "bab4",
        Path(Config.OUTPUT_FOLDER) / "training",
        Path(Config.OUTPUT_FOLDER) / "evaluasi",
        Path(Config.OUTPUT_FOLDER) / "preprocessing",
    ]
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def delete_active_dataset():
    try:
        path = get_dataset_path()
        if os.path.exists(path):
            os.remove(path)
            invalidate_generated_artifacts()
            summary_path = sanitization_summary_path()
            if os.path.exists(summary_path):
                os.remove(summary_path)
            metadata_path = os.path.join(Config.UPLOAD_FOLDER, "active_dataset.json")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            app_logger.info("Active dataset and generated artifacts deleted.")
            return True
    except Exception as error:
        app_logger.error(f"Error deleting dataset: {error}")
    return False


def validate_csv_columns(file_path):
    try:
        columns = [str(col).strip() for col in pd.read_csv(file_path, nrows=5).columns]
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        return (False, f"Missing required columns: {', '.join(missing)}") if missing else (True, "")
    except Exception as error:
        app_logger.error(f"Error validating CSV columns: {error}")
        return False, f"Failed to read CSV headers: {error}"


def dataframe_has_forbidden_chemicals(df):
    return bool(forbidden_chemical_columns(df.columns))


def get_dataset_summary(file_path):
    try:
        df = pd.read_csv(file_path)
        row_number_columns = [
            column
            for column in df.columns
            if is_sequential_row_number_column(column, df[column])
        ]
        analytical_df = df.drop(columns=row_number_columns)
        preview_source_columns = [column for _, column in DATASET_PREVIEW_COLUMNS]
        missing_preview_columns = [
            column for column in preview_source_columns if column not in df.columns
        ]
        if missing_preview_columns:
            raise ValueError(
                "Kolom pratinjau tidak ditemukan: " + ", ".join(missing_preview_columns)
            )

        def preview_rows(frame):
            rows = frame.loc[:, preview_source_columns].replace({np.nan: None}).values.tolist()
            return [
                [int(index) + 1, *row]
                for index, row in zip(frame.index, rows)
            ]

        shape = analytical_df.shape
        size_bytes = os.path.getsize(file_path)
        file_size = (
            f"{size_bytes / (1024 * 1024):.2f} MB"
            if size_bytes >= 1024 * 1024
            else f"{size_bytes / 1024:.2f} KB"
        )
        numeric_cols = analytical_df.select_dtypes(include=["number"]).columns.tolist()
        columns_info = []
        for column in analytical_df.columns:
            missing_count = int(analytical_df[column].isnull().sum())
            columns_info.append(
                {
                    "name": column,
                    "type": str(analytical_df[column].dtype),
                    "missing_count": missing_count,
                    "missing_percentage": (
                        round(100 * missing_count / len(analytical_df), 2)
                        if len(analytical_df)
                        else 0
                    ),
                    "unique_count": int(analytical_df[column].nunique()),
                }
            )
        stats = []
        for column in numeric_cols:
            values = analytical_df[column].describe()
            stats.append(
                {
                    "name": column,
                    "count": int(values["count"]),
                    "mean": round(float(values["mean"]), 4) if pd.notna(values["mean"]) else 0,
                    "std": round(float(values["std"]), 4) if pd.notna(values["std"]) else 0,
                    "min": round(float(values["min"]), 4) if pd.notna(values["min"]) else 0,
                    "q1": round(float(values["25%"]), 4) if pd.notna(values["25%"]) else 0,
                    "median": round(float(values["50%"]), 4) if pd.notna(values["50%"]) else 0,
                    "q3": round(float(values["75%"]), 4) if pd.notna(values["75%"]) else 0,
                    "max": round(float(values["max"]), 4) if pd.notna(values["max"]) else 0,
                }
            )
        return {
            "success": True,
            "filename": os.path.basename(file_path),
            "filesize": file_size,
            "rows": len(df),
            "columns_count": shape[1],
            "numeric_vars_count": len(numeric_cols),
            "categorical_vars_count": shape[1] - len(numeric_cols),
            "total_missing": int(analytical_df.isnull().sum().sum()),
            "duplicate_count": int(df.duplicated().sum()),
            "row_number_columns": row_number_columns,
            "columns_info": columns_info,
            "stats": stats,
            "preview_cols": ["NO", *[label for label, _ in DATASET_PREVIEW_COLUMNS]],
            "head_data": preview_rows(df.head(10)),
            "tail_data": preview_rows(df.tail(10)),
            "sanitization": load_sanitization_summary(),
        }
    except Exception as error:
        app_logger.error(f"Error summarizing dataset: {error}")
        return {"success": False, "error": str(error)}
