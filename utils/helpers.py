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
}
SANITIZATION_STATUSES = {
    "retained": "Dipertahankan",
    "alum": "Dikeluarkan - Menggunakan Alum",
    "lime": "Dikeluarkan - Menggunakan Lime",
    "alum_lime": "Dikeluarkan - Menggunakan Alum & Lime",
    "ambiguous": "Dikeluarkan - Kimia Ambigu",
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


def load_dataset_metadata():
    metadata_path = os.path.join(Config.UPLOAD_FOLDER, "active_dataset.json")
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        return metadata if isinstance(metadata, dict) else {}
    except (OSError, TypeError, ValueError):
        return {}


def set_dataset_filename(filename, source_archive_filename=None, original_filename=None):
    """Remember active and raw-source filenames for the current upload."""
    metadata = {"filename": os.path.basename(filename)}
    if source_archive_filename:
        metadata["source_archive_filename"] = os.path.basename(source_archive_filename)
    if original_filename:
        metadata["original_filename"] = os.path.basename(original_filename)
    metadata_path = os.path.join(Config.UPLOAD_FOLDER, "active_dataset.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def get_source_dataset_path():
    filename = load_dataset_metadata().get("source_archive_filename")
    if not filename:
        return None
    return os.path.join(Config.UPLOAD_FOLDER, "source_archive", os.path.basename(filename))


def read_source_dataset(file_path):
    extension = Path(file_path).suffix.casefold()
    if extension == ".xlsx":
        return pd.read_excel(file_path, sheet_name=0)
    if extension == ".csv":
        return pd.read_csv(file_path)
    raise ValueError("Format sumber dataset tidak didukung.")


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


def is_unnamed_export_column(name):
    """Identify unnamed index artifacts created by prior CSV/XLSX exports."""
    return str(name).strip().casefold().startswith("unnamed:")


def is_row_number_column(name):
    """Match only the explicitly supported No/Nomor source metadata labels."""
    return str(name).strip().casefold() in ROW_NUMBER_COLUMN_ALIASES


def get_research_columns(df):
    return [column for column in df.columns if not is_row_number_column(column)]


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


def build_sanitization_report(df):
    """Append a per-row decision to an untouched copy of the source frame."""
    source = df.drop(
        columns=[
            column for column in df.columns if is_unnamed_export_column(column)
        ],
        errors="ignore",
    ).copy()
    stripped_columns = [str(column).strip() for column in source.columns]
    if len(set(stripped_columns)) != len(stripped_columns):
        raise ValueError("Nama kolom duplikat terdeteksi setelah spasi nama kolom dibersihkan.")
    lookup = dict(zip(stripped_columns, source.columns))
    chemical_presence = [
        chemical in lookup for chemical in OPTIONAL_CHEMICAL_COLUMNS
    ]
    if any(chemical_presence) and not all(chemical_presence):
        missing = next(
            chemical
            for chemical, present in zip(OPTIONAL_CHEMICAL_COLUMNS, chemical_presence)
            if not present
        )
        raise ValueError(
            f"Kolom seleksi kimia tidak lengkap. {missing} diperlukan untuk "
            "menentukan observasi tanpa Alum dan Lime secara aman."
        )

    if not any(chemical_presence):
        statuses = pd.Series(
            SANITIZATION_STATUSES["retained"], index=source.index, dtype=object
        )
    else:
        alum = pd.to_numeric(source[lookup[OPTIONAL_CHEMICAL_COLUMNS[0]]], errors="coerce")
        lime = pd.to_numeric(source[lookup[OPTIONAL_CHEMICAL_COLUMNS[1]]], errors="coerce")
        ambiguous = alum.isna() | lime.isna() | (alum < 0) | (lime < 0)
        statuses = pd.Series(SANITIZATION_STATUSES["retained"], index=source.index, dtype=object)
        statuses.loc[(~ambiguous) & alum.gt(0) & lime.eq(0)] = SANITIZATION_STATUSES["alum"]
        statuses.loc[(~ambiguous) & alum.eq(0) & lime.gt(0)] = SANITIZATION_STATUSES["lime"]
        statuses.loc[(~ambiguous) & alum.gt(0) & lime.gt(0)] = SANITIZATION_STATUSES["alum_lime"]
        statuses.loc[ambiguous] = SANITIZATION_STATUSES["ambiguous"]

    report = source.copy()
    report["Status"] = statuses
    return report


def sanitization_counts(report):
    counts = report["Status"].value_counts().to_dict()
    retained = int(counts.get(SANITIZATION_STATUSES["retained"], 0))
    ambiguous = int(counts.get(SANITIZATION_STATUSES["ambiguous"], 0))
    chemical = int(
        sum(
            counts.get(SANITIZATION_STATUSES[key], 0)
            for key in ("alum", "lime", "alum_lime")
        )
    )
    total = retained + chemical + ambiguous
    if total != len(report):
        raise ValueError(
            "Validasi laporan sanitasi gagal: jumlah status tidak sama dengan "
            "jumlah baris sumber."
        )
    return {
        "retained": retained,
        "chemical_removed": chemical,
        "ambiguous_removed": ambiguous,
        "original": int(len(report)),
        "by_status": {status: int(counts.get(status, 0)) for status in SANITIZATION_STATUSES.values()},
    }


def sanitize_dataframe(df):
    """Select CL-80-only observations and remove all chemical/index columns."""
    clean = df.copy()
    clean.columns = [str(column).strip() for column in clean.columns]
    if len(set(clean.columns)) != len(clean.columns):
        raise ValueError("Nama kolom duplikat terdeteksi setelah spasi nama kolom dibersihkan.")

    missing = [column for column in REQUIRED_COLUMNS if column not in clean.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")

    report = build_sanitization_report(df)
    counts = sanitization_counts(report)
    retained_mask = report["Status"].eq(SANITIZATION_STATUSES["retained"]).to_numpy()
    clean = clean.loc[retained_mask].copy()

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
        "original_row_count": counts["original"],
        "original_column_count": int(len(df.columns)),
        "excluded_alum_lime_row_count": counts["chemical_removed"],
        "ambiguous_chemical_row_count": counts["ambiguous_removed"],
        "final_active_row_count": int(len(clean)),
        "removed_column_names": removed_columns,
        "status_counts": counts["by_status"],
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
        active_df = pd.read_csv(file_path).reset_index(drop=True)
        row_number_columns = [
            column for column in active_df.columns if is_row_number_column(column)
        ]
        analytical_df = active_df.loc[:, get_research_columns(active_df)]
        preview_source_columns = [column for _, column in DATASET_PREVIEW_COLUMNS]
        missing_preview_columns = [
            column for column in preview_source_columns if column not in active_df.columns
        ]
        if missing_preview_columns:
            raise ValueError(
                "Kolom pratinjau tidak ditemukan: " + ", ".join(missing_preview_columns)
            )

        def preview_rows(frame):
            rows = frame.loc[:, preview_source_columns].replace({np.nan: None}).values.tolist()
            # The persisted No/Nomor column identifies the source row and can have
            # gaps after sanitization. The Dataset page must show active-row
            # sequence numbers so the final visible NO is always len(active_df).
            numbers = [int(index) + 1 for index in frame.index]
            return [[number, *row] for number, row in zip(numbers, rows)]

        active_shape = active_df.shape
        analytical_shape = analytical_df.shape
        active_row_count = int(len(active_df))
        sanitization = load_sanitization_summary()
        if sanitization:
            stored_active_count = int(
                sanitization.get("final_active_row_count", active_row_count)
            )
            removed_chemical = int(
                sanitization.get("excluded_alum_lime_row_count", 0)
            )
            removed_ambiguous = int(
                sanitization.get("ambiguous_chemical_row_count", 0)
            )
            raw_count = int(
                sanitization.get(
                    "original_row_count",
                    active_row_count + removed_chemical + removed_ambiguous,
                )
            )
            if stored_active_count != active_row_count:
                raise ValueError(
                    "Ringkasan sanitasi tidak konsisten dengan dataset aktif: "
                    f"metadata={stored_active_count}, len(active_df)={active_row_count}."
                )
            if raw_count != active_row_count + removed_chemical + removed_ambiguous:
                raise ValueError(
                    "Ringkasan sanitasi tidak konsisten: baris sumber tidak sama "
                    "dengan baris aktif ditambah seluruh baris yang dikeluarkan."
                )
            # Counts shown by the Dataset page derive the active count from the
            # loaded active DataFrame, never from a source row identifier.
            sanitization = {
                **sanitization,
                "final_active_row_count": active_row_count,
            }

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
            "rows": active_row_count,
            "shape": [active_row_count, int(active_shape[1])],
            "columns_count": analytical_shape[1],
            "source_columns_count": int(
                sanitization.get("original_column_count", len(active_df.columns))
            ),
            "numeric_vars_count": len(numeric_cols),
            "categorical_vars_count": analytical_shape[1] - len(numeric_cols),
            "total_missing": int(analytical_df.isnull().sum().sum()),
            "duplicate_count": int(active_df.duplicated().sum()),
            "row_number_columns": row_number_columns,
            "source_row_number_preserved": bool(row_number_columns),
            "columns_info": columns_info,
            "stats": stats,
            "preview_cols": ["NO", *[label for label, _ in DATASET_PREVIEW_COLUMNS]],
            "head_data": preview_rows(active_df.head(10)),
            "tail_data": preview_rows(active_df.tail(10)),
            "sanitization": sanitization,
            "sanitization_report_available": bool(
                get_source_dataset_path() and os.path.exists(get_source_dataset_path())
            ),
        }
    except Exception as error:
        app_logger.error(f"Error summarizing dataset: {error}")
        return {"success": False, "error": str(error)}
