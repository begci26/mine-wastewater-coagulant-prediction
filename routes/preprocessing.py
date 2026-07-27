import json
import os
import shutil

import joblib
import pandas as pd
import plotly.express as px
from flask import Blueprint, flash, jsonify, redirect, render_template, request, send_from_directory, session, url_for

from config import Config
from ml.preprocessor import FatalConversionError, WastewaterPreprocessor
from utils.helpers import (
    MODEL_FEATURES,
    REQUIRED_TARGET,
    check_dataset_uploaded,
    get_dataset_path,
    load_sanitization_summary,
)
from utils.logger import app_logger

preprocessing_bp = Blueprint("preprocessing", __name__)
MIN_TEST_SPLIT_PERCENT = 10
MAX_TEST_SPLIT_PERCENT = 40

STATUS_PRESENTATION = {
    "not_started": {
        "label": "Belum Diproses",
        "badge_class": "bg-secondary",
        "message": "Tahap preprocessing final belum dijalankan.",
    },
    "waiting_for_split": {
        "label": "Menunggu Split",
        "badge_class": "bg-info text-dark",
        "message": (
            "Median dan batas IQR akan dihitung setelah pembagian data agar hanya "
            "menggunakan data training."
        ),
    },
    "processing": {
        "label": "Sedang Diproses",
        "badge_class": "bg-primary",
        "message": "Proses preprocessing sedang dijalankan.",
    },
    "completed": {
        "label": "Selesai",
        "badge_class": "bg-success",
        "message": "Tahap preprocessing selesai.",
    },
    "completed_with_warning": {
        "label": "Selesai dengan Peringatan",
        "badge_class": "bg-warning text-dark",
        "message": "Tahap preprocessing selesai dengan rincian yang perlu diperhatikan.",
    },
    "failed": {
        "label": "Gagal",
        "badge_class": "bg-danger",
        "message": "Tahap preprocessing gagal.",
    },
    "unknown": {
        "label": "Status belum tersedia",
        "badge_class": "bg-secondary",
        "message": "Metadata lama belum memiliki bukti status yang cukup.",
    },
}


def get_preprocessing_paths():
    processed_dir = os.path.join(Config.UPLOAD_FOLDER, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    return (
        os.path.join(processed_dir, "df_active.csv"),
        os.path.join(processed_dir, "preprocessing_state.json"),
    )


def parse_test_split_percent(raw_value):
    """Validate the user-selected integer percentage for the test period."""
    if raw_value is None or not str(raw_value).strip():
        raise ValueError("Ukuran data uji wajib diisi.")
    try:
        percentage = int(str(raw_value).strip())
    except ValueError as error:
        raise ValueError("Ukuran data uji harus berupa bilangan bulat.") from error
    if not MIN_TEST_SPLIT_PERCENT <= percentage <= MAX_TEST_SPLIT_PERCENT:
        raise ValueError(
            f"Ukuran data uji harus antara {MIN_TEST_SPLIT_PERCENT}% dan "
            f"{MAX_TEST_SPLIT_PERCENT}%."
        )
    return percentage


def _initial_state():
    dataset = pd.read_csv(get_dataset_path()) if check_dataset_uploaded() else pd.DataFrame()
    sanitization = {
        key: value
        for key, value in load_sanitization_summary().items()
        if key != "removed_column_names"
    }
    return {
        "initial_shape": list(dataset.shape),
        "current_shape": list(dataset.shape),
        "sanitization": sanitization,
        "cleaning_applied": False,
        "cleaning_removed_count": 0,
        "cleaning_message": None,
        "type_conversion_errors": [],
        "conversion_report": {},
        "has_fatal_conversion_errors": False,
        "pipeline_blocked": False,
        "missing_checked": False,
        "missing_stats": {},
        "missing_handled": False,
        "missing_strategy": "training_median_only",
        "missing_message": None,
        "missing_handled_count": 0,
        "duplicates_checked": False,
        "duplicates_count": 0,
        "duplicates_handled": False,
        "duplicates_removed_count": 0,
        "outliers_detected": False,
        "outliers_stats": {},
        "outliers_count": 0,
        "outliers_handled": False,
        "outliers_strategy": "train_fitted_winsorization",
        "outliers_message": None,
        "outliers_handled_count": 0,
        "feature_engineering_applied": False,
        "feature_engineering_message": None,
        "feature_engineering_validation": {},
        "lag_rows_deleted": 0,
        "scaler_applied": False,
        "scaler_strategy": "standard",
        "scaler_message": None,
        "split": False,
        "split_ratio": 20,
        "split_method": "chronological_80_20",
        "train_shape": None,
        "test_shape": None,
        "comparison": {},
        "preprocessing_run": {
            "status": "not_started",
            "message": "Tahap preprocessing final belum dijalankan.",
        },
    }


def save_state(state):
    _, state_path = get_preprocessing_paths()
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)


def get_or_init_state():
    active_path, state_path = get_preprocessing_paths()
    if not os.path.exists(state_path):
        state = _initial_state()
        save_state(state)
        if check_dataset_uploaded():
            shutil.copy2(get_dataset_path(), active_path)
        return state
    try:
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
        if _synchronize_preprocessing_status(state):
            save_state(state)
        return state
    except Exception:
        state = _initial_state()
        save_state(state)
        return state


def _processed_file(name):
    return os.path.join(Config.UPLOAD_FOLDER, "processed", name)


def _has_completed_evidence(metadata):
    if not isinstance(metadata, dict):
        return False
    imputation = metadata.get("imputation", {})
    outlier = metadata.get("outlier_handling", {})
    scaling = metadata.get("scaling", {})
    return bool(
        metadata.get("split_completed")
        and metadata.get("split_method")
        in {"chronological_80_20", "chronological_without_shuffle"}
        and imputation.get("status") == "completed"
        and imputation.get("fit_source") == "training_only"
        and isinstance(imputation.get("columns"), dict)
        and outlier.get("status") == "completed"
        and outlier.get("fit_source") == "training_only"
        and isinstance(outlier.get("bounds"), dict)
        and scaling.get("status") == "completed"
        and scaling.get("fit_source") == "training_only"
    )


def _migrate_legacy_status(state):
    """Reconstruct legacy status only when split files and fitted object prove completion."""
    metadata = state.get("preprocessing_metadata", {})
    legacy_preprocessing = metadata.get("preprocessing", {})
    required_files = [
        _processed_file("x_train_raw.csv"),
        _processed_file("x_test_raw.csv"),
        _processed_file("x_train.csv"),
        _processed_file("x_test.csv"),
        os.path.join(Config.MODEL_FOLDER, "preprocessor.joblib"),
    ]
    if not (
        state.get("split")
        and metadata.get("split_method")
        in {"chronological_80_20", "chronological_without_shuffle"}
        and legacy_preprocessing.get("fit_scope") == "training_period_only"
        and all(os.path.exists(path) for path in required_files)
    ):
        return False
    try:
        preprocessor = joblib.load(os.path.join(Config.MODEL_FOLDER, "preprocessor.joblib"))
        X_train_raw = pd.read_csv(_processed_file("x_train_raw.csv"))
        X_test_raw = pd.read_csv(_processed_file("x_test_raw.csv"))
        evidence = preprocessor.processing_status_metadata(X_train_raw, X_test_raw)
        metadata.update(evidence)
        state["preprocessing_metadata"] = metadata
        state["status_migrated_from_legacy_evidence"] = True
        return True
    except Exception as error:
        app_logger.warning("Legacy preprocessing status could not be verified: %s", error)
        return False


def _synchronize_preprocessing_status(state):
    changed = False
    metadata = state.get("preprocessing_metadata", {})
    if state.get("split") and not _has_completed_evidence(metadata):
        changed = _migrate_legacy_status(state)
        metadata = state.get("preprocessing_metadata", {})

    run = state.get("preprocessing_run", {})
    if run.get("status") == "failed":
        status = "failed"
    elif run.get("status") == "processing":
        status = "processing"
    elif _has_completed_evidence(metadata):
        status = "completed"
    elif state.get("split"):
        status = "unknown"
    else:
        status = "waiting_for_split"

    if _has_completed_evidence(metadata):
        imputation = metadata["imputation"]
        outlier = metadata["outlier_handling"]
        missing_before = int(imputation.get("total_missing_before", 0))
        missing_after = int(imputation.get("total_missing_after", 0))
        imputation_status = (
            "completed_with_warning" if missing_before > 0 else "completed"
        )
        state.update(
            {
                "missing_checked": True,
                "missing_handled": True,
                "missing_strategy": "training_median_only",
                "missing_handled_count": missing_before,
                "missing_stats": imputation.get("columns", {}),
                "missing_message": (
                    f"Imputasi telah diterapkan pada {missing_before} nilai menggunakan "
                    "median dari data training."
                    if missing_before
                    else "Pemeriksaan selesai, tidak ditemukan nilai kosong yang perlu diimputasi."
                ),
                "outliers_detected": True,
                "outliers_handled": True,
                "outliers_strategy": "train_fitted_winsorization",
                "outliers_stats": outlier.get("bounds", {}),
                "outliers_count": int(outlier.get("total_train_clipped", 0))
                + int(outlier.get("total_test_clipped", 0)),
                "outliers_handled_count": int(outlier.get("total_train_clipped", 0))
                + int(outlier.get("total_test_clipped", 0)),
                "outliers_message": (
                    "Batas IQR telah dihitung dari data training dan diterapkan pada "
                    "data training dan testing."
                    if int(outlier.get("total_train_clipped", 0))
                    + int(outlier.get("total_test_clipped", 0))
                    else "Pemeriksaan selesai, tidak ada nilai yang memerlukan clipping."
                ),
                "automatic_step_status": {
                    "imputation": imputation_status,
                    "outlier_handling": "completed",
                    "scaling": "completed",
                },
            }
        )
        changed = True
    else:
        state["missing_handled"] = False
        state["outliers_handled"] = False
        state["automatic_step_status"] = {
            "imputation": status,
            "outlier_handling": status,
            "scaling": status,
        }
        changed = True
    return changed


def _status_view(state, key):
    code = state.get("automatic_step_status", {}).get(key, "unknown")
    view = dict(STATUS_PRESENTATION.get(code, STATUS_PRESENTATION["unknown"]))
    view["code"] = code
    if key == "imputation":
        if code in {"completed", "completed_with_warning"}:
            view["message"] = state.get("missing_message") or view["message"]
        elif code == "waiting_for_split":
            view["message"] = (
                "Imputasi akan dijalankan setelah pembagian data menggunakan median data training."
            )
    elif key == "outlier_handling":
        if code in {"completed", "completed_with_warning"}:
            view["message"] = state.get("outliers_message") or view["message"]
        elif code == "waiting_for_split":
            view["message"] = (
                "Batas IQR akan dihitung dari data training setelah pembagian data."
            )
    if code == "failed":
        view["message"] = state.get("preprocessing_run", {}).get(
            "message", view["message"]
        )
    return view


def _reset_final_processing_status(state):
    state.pop("preprocessing_metadata", None)
    state["preprocessing_run"] = {
        "status": "waiting_for_split",
        "message": "Menunggu final preprocessing dan chronological split.",
    }
    state["missing_handled"] = False
    state["outliers_handled"] = False
    state["automatic_step_status"] = {
        "imputation": "waiting_for_split",
        "outlier_handling": "waiting_for_split",
        "scaling": "waiting_for_split",
    }


def _engineer_from_sanitized_source(state):
    raw = pd.read_csv(get_dataset_path())
    try:
        engineered, cleaning = WastewaterPreprocessor().clean_and_engineer(raw)
    except FatalConversionError as error:
        state.update(
            {
                "cleaning_applied": False,
                "feature_engineering_applied": False,
                "scaler_applied": False,
                "split": False,
                "conversion_report": error.report,
                "type_conversion_errors": error.report.get("warnings", []),
                "has_fatal_conversion_errors": True,
                "pipeline_blocked": True,
                "cleaning_message": str(error),
            }
        )
        _write_conversion_report(error.report)
        save_state(state)
        raise
    active_path, _ = get_preprocessing_paths()
    engineered.to_csv(active_path, index=False)
    conversion_report = cleaning["conversion_report"]
    _write_conversion_report(conversion_report)

    state.update(
        {
            "cleaning_applied": True,
            "cleaning_removed_count": (
                cleaning["invalid_target_rows_removed"]
                + cleaning["inlet_tss_over_2000_rows_removed"]
                + cleaning["invalid_timestamp_rows_removed"]
            ),
            "cleaning_message": (
                "Target/rentang dibersihkan, timestamp dibentuk, data diurutkan kronologis, "
                "duplikat dihapus, lalu lag dan fitur turunan dibuat."
            ),
            "type_conversion_errors": cleaning["type_conversion_warnings"],
            "conversion_report": conversion_report,
            "has_fatal_conversion_errors": False,
            "pipeline_blocked": False,
            "duplicates_checked": True,
            "duplicates_count": cleaning["duplicate_rows_removed"],
            "duplicates_handled": True,
            "duplicates_removed_count": cleaning["duplicate_rows_removed"],
            "feature_engineering_applied": True,
            "feature_engineering_message": (
                "T1/T2/T3 dibuat dari tiga observasi sebelumnya; Efisiensi_TSS, "
                "Delta_pH, dan Beban_TSS dibuat tanpa fitur rasio kimia."
            ),
            "feature_engineering_validation": {feature: feature in engineered.columns for feature in MODEL_FEATURES},
            "lag_rows_deleted": cleaning["lag_incomplete_rows_removed"],
            "current_shape": list(engineered.shape),
            "cleaning_summary": cleaning,
            "split": False,
            "scaler_applied": False,
        }
    )
    return engineered, cleaning


def _write_conversion_report(report):
    """Persist row-level conversion diagnostics for CSV/JSON report exports."""
    output_dir = os.path.join(Config.UPLOAD_FOLDER, "output", "preprocessing")
    os.makedirs(output_dir, exist_ok=True)
    details = report.get("details", [])
    columns = [
        "column",
        "row_index",
        "timestamp",
        "original_value",
        "root_cause",
        "action",
        "final_status",
    ]
    pd.DataFrame(details, columns=columns).to_csv(
        os.path.join(output_dir, "conversion_report.csv"), index=False
    )
    with open(os.path.join(output_dir, "conversion_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


@preprocessing_bp.route("/")
def index():
    if not check_dataset_uploaded():
        flash("Silakan unggah dataset terlebih dahulu.", "warning")
        return redirect(url_for("dataset.index"))
    state = get_or_init_state()
    if _synchronize_preprocessing_status(state):
        save_state(state)
    active_path, _ = get_preprocessing_paths()
    raw = pd.read_csv(get_dataset_path())
    active = pd.read_csv(active_path) if os.path.exists(active_path) else raw
    return render_template(
        "preprocessing.html",
        state=state,
        missing_status=_status_view(state, "imputation"),
        outlier_status=_status_view(state, "outlier_handling"),
        preview_raw=raw.head(10).to_dict(orient="records"),
        preview_active=active.head(10).to_dict(orient="records"),
    )


@preprocessing_bp.route("/reset", methods=["POST"])
def reset():
    active_path, state_path = get_preprocessing_paths()
    for path in [active_path, state_path]:
        if os.path.exists(path):
            os.remove(path)
    for key in ["preprocessing_complete", "preprocessing_results", "training_complete", "evaluation_complete"]:
        session.pop(key, None)
    flash("Status prapemrosesan direset ke dataset aktif tersanitasi.", "success")
    return redirect(url_for("preprocessing.index"))


@preprocessing_bp.route("/step1_clean", methods=["POST"])
def step1_clean():
    state = get_or_init_state()
    try:
        _reset_final_processing_status(state)
        _, cleaning = _engineer_from_sanitized_source(state)
        save_state(state)
        flash(
            "Pembersihan kronologis selesai: "
            f"{cleaning['invalid_target_rows_removed']} target invalid, "
            f"{cleaning['inlet_tss_over_2000_rows_removed']} TSS > 2000, dan "
            f"{cleaning['invalid_timestamp_rows_removed']} timestamp invalid dihapus.",
            "success",
        )
    except Exception as error:
        app_logger.error("Preprocessing cleaning failed: %s", error, exc_info=True)
        flash(f"Gagal membersihkan data: {error}", "danger")
    return redirect(url_for("preprocessing.index") + "#step-1")


@preprocessing_bp.route("/step2_missing", methods=["POST"])
def step2_missing():
    state = get_or_init_state()
    try:
        if _has_completed_evidence(state.get("preprocessing_metadata", {})):
            _synchronize_preprocessing_status(state)
            save_state(state)
            flash(state["missing_message"], "success")
            return redirect(url_for("preprocessing.index") + "#step-2")
        if not state.get("feature_engineering_applied"):
            _engineer_from_sanitized_source(state)
        active = pd.read_csv(get_preprocessing_paths()[0])
        state["missing_checked"] = True
        state["missing_stats"] = WastewaterPreprocessor().check_missing_values(active)
        state["missing_handled"] = False
        state["missing_strategy"] = "training_median_only"
        state["missing_message"] = (
            "Nilai fitur kosong dipertahankan hingga split; median akan di-fit hanya pada periode training."
        )
        state["missing_handled_count"] = int(active[MODEL_FEATURES].isna().sum().sum())
        state["automatic_step_status"] = {
            **state.get("automatic_step_status", {}),
            "imputation": "waiting_for_split",
        }
        save_state(state)
        flash(state["missing_message"], "info")
    except Exception as error:
        state["preprocessing_run"] = {"status": "failed", "message": str(error)}
        state["automatic_step_status"] = {
            **state.get("automatic_step_status", {}),
            "imputation": "failed",
        }
        save_state(state)
        flash(f"Gagal memeriksa missing value: {error}", "danger")
    return redirect(url_for("preprocessing.index") + "#step-2")


@preprocessing_bp.route("/step3_duplicate", methods=["POST"])
def step3_duplicate():
    state = get_or_init_state()
    try:
        if not state.get("feature_engineering_applied"):
            _engineer_from_sanitized_source(state)
        save_state(state)
        flash(
            f"Duplikat telah dihapus setelah pengurutan timestamp: {state.get('duplicates_removed_count', 0)} baris.",
            "success",
        )
    except Exception as error:
        flash(f"Gagal memproses duplikat: {error}", "danger")
    return redirect(url_for("preprocessing.index") + "#step-3")


@preprocessing_bp.route("/step4_outlier", methods=["POST"])
def step4_outlier():
    state = get_or_init_state()
    try:
        if _has_completed_evidence(state.get("preprocessing_metadata", {})):
            _synchronize_preprocessing_status(state)
            save_state(state)
            flash(state["outliers_message"], "success")
            return redirect(url_for("preprocessing.index") + "#step-4")
        if not state.get("feature_engineering_applied"):
            _engineer_from_sanitized_source(state)
        active = pd.read_csv(get_preprocessing_paths()[0])
        stats, indices = WastewaterPreprocessor().detect_outliers_iqr(active)
        state.update(
            {
                "outliers_detected": True,
                "outliers_stats": stats,
                "outliers_count": len(indices),
                "outliers_handled": False,
                "outliers_strategy": "train_fitted_winsorization",
                "outliers_message": (
                    "Outlier ditandai untuk diagnostik. Batas final dihitung dari training saja "
                    "setelah split dan diterapkan ke training/testing."
                ),
                "outliers_handled_count": 0,
                "automatic_step_status": {
                    **state.get("automatic_step_status", {}),
                    "outlier_handling": "waiting_for_split",
                },
            }
        )
        save_state(state)
        flash(state["outliers_message"], "info")
    except Exception as error:
        state["preprocessing_run"] = {"status": "failed", "message": str(error)}
        state["automatic_step_status"] = {
            **state.get("automatic_step_status", {}),
            "outlier_handling": "failed",
        }
        save_state(state)
        flash(f"Gagal mendeteksi outlier: {error}", "danger")
    return redirect(url_for("preprocessing.index") + "#step-4")


@preprocessing_bp.route("/step5_feature_eng", methods=["POST"])
def step5_feature_eng():
    state = get_or_init_state()
    try:
        _reset_final_processing_status(state)
        _engineer_from_sanitized_source(state)
        save_state(state)
        flash(state["feature_engineering_message"], "success")
    except Exception as error:
        flash(f"Gagal melakukan rekayasa fitur: {error}", "danger")
    return redirect(url_for("preprocessing.index") + "#step-5")


@preprocessing_bp.route("/step6_scale_split", methods=["POST"])
def step6_scale_split():
    state = get_or_init_state()
    try:
        split_percentage = parse_test_split_percent(request.form.get("split_size"))
    except ValueError as error:
        flash(str(error), "danger")
        return redirect(url_for("preprocessing.index") + "#step-6")

    # Preserve valid input if a later preprocessing validation fails.
    state["split_ratio"] = split_percentage
    state.pop("preprocessing_metadata", None)
    state.update(
        {
            "preprocessing_run": {
                "status": "processing",
                "message": "Proses preprocessing sedang dijalankan.",
                "started_at": pd.Timestamp.now().isoformat(),
            },
            "automatic_step_status": {
                "imputation": "processing",
                "outlier_handling": "processing",
                "scaling": "processing",
            },
            "missing_handled": False,
            "outliers_handled": False,
            "split": False,
            "scaler_applied": False,
        }
    )
    save_state(state)
    try:
        engineered, cleaning = _engineer_from_sanitized_source(state)
        preprocessor = WastewaterPreprocessor()
        train, test, split_metadata = preprocessor.prepare_and_save(
            engineered,
            test_fraction=split_percentage / 100,
        )

        output_dir = os.path.join(Config.UPLOAD_FOLDER, "output", "preprocessing")
        os.makedirs(output_dir, exist_ok=True)
        export_columns = ["Timestamp", "Date", "Time"] + MODEL_FEATURES + [REQUIRED_TARGET]
        export = engineered.loc[:, [column for column in export_columns if column in engineered.columns]]
        export.to_csv(os.path.join(output_dir, "dataset_preprocessing.csv"), index=False)
        export.to_excel(os.path.join(output_dir, "dataset_preprocessing.xlsx"), index=False)

        sanitization = load_sanitization_summary()
        summary = {
            **{
                key: value
                for key, value in sanitization.items()
                if key != "removed_column_names"
            },
            **cleaning,
            **split_metadata,
            "waktu_eksekusi": pd.Timestamp.now().isoformat(),
            "jumlah_baris_akhir": len(engineered),
            "jumlah_kolom_akhir": len(export.columns),
            "missing_value_akhir_sebelum_train_imputation": int(engineered[MODEL_FEATURES].isna().sum().sum()),
            "duplikat_akhir": int(engineered.duplicated().sum()),
        }
        with open(os.path.join(output_dir, "summary_preprocessing.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        state.update(
            {
                "scaler_applied": True,
                "scaler_strategy": "standard",
                "scaler_message": "Median, IQR/Winsorization, dan StandardScaler di-fit pada training saja.",
                "split": True,
                "split_ratio": split_percentage,
                "split_method": (
                    f"chronological_{100 - split_percentage}_{split_percentage}"
                ),
                "train_shape": [len(train), len(MODEL_FEATURES)],
                "test_shape": [len(test), len(MODEL_FEATURES)],
                "preprocessing_metadata": split_metadata,
                "preprocessing_run": {
                    "status": "completed",
                    "message": "Preprocessing final selesai.",
                    "completed_at": split_metadata["scaling"]["completed_at"],
                },
                "current_shape": list(engineered.shape),
                "comparison": {
                    "before_rows": state["initial_shape"][0],
                    "after_rows": len(engineered),
                    "before_cols": state["initial_shape"][1],
                    "after_cols": len(export.columns),
                    "before_missing": 0,
                    "after_missing": int(engineered[MODEL_FEATURES].isna().sum().sum()),
                    "before_duplicates": 0,
                    "after_duplicates": 0,
                    "before_outliers": state.get("outliers_count", 0),
                    "after_outliers": state.get("outliers_count", 0),
                    "before_features": 5,
                    "after_features": len(MODEL_FEATURES),
                },
            }
        )
        _synchronize_preprocessing_status(state)
        save_state(state)
        session["preprocessing_complete"] = True
        session["preprocessing_results"] = {
            "imputation": "training_median_only",
            "scaler": "training_standard_scaler",
            "test_size_percentage": split_percentage,
            "train_rows": len(train),
            "train_cols": len(MODEL_FEATURES),
            "test_rows": len(test),
            "test_cols": len(MODEL_FEATURES),
        }
        for key in ["training_complete", "evaluation_complete"]:
            session.pop(key, None)
        flash("Preprocessing kronologis selesai tanpa kebocoran data.", "success")
    except Exception as error:
        state.update(
            {
                "preprocessing_run": {
                    "status": "failed",
                    "message": str(error),
                    "failed_at": pd.Timestamp.now().isoformat(),
                },
                "automatic_step_status": {
                    "imputation": "failed",
                    "outlier_handling": "failed",
                    "scaling": "failed",
                },
                "missing_handled": False,
                "outliers_handled": False,
                "split": False,
                "scaler_applied": False,
            }
        )
        save_state(state)
        app_logger.error("Final preprocessing failed: %s", error, exc_info=True)
        flash(f"Gagal menyelesaikan preprocessing: {error}", "danger")
    return redirect(url_for("preprocessing.index") + "#step-6")


@preprocessing_bp.route("/download/<file_type>")
def download(file_type):
    output_dir = os.path.join(Config.UPLOAD_FOLDER, "output", "preprocessing")
    filenames = {
        "csv": "dataset_preprocessing.csv",
        "xlsx": "dataset_preprocessing.xlsx",
        "conversion_csv": "conversion_report.csv",
        "conversion_json": "conversion_report.json",
    }
    filename = filenames.get(file_type)
    if not filename or not os.path.exists(os.path.join(output_dir, filename)):
        flash("Berkas ekspor belum tersedia.", "danger")
        return redirect(url_for("preprocessing.index"))
    return send_from_directory(output_dir, filename, as_attachment=True)


@preprocessing_bp.route("/outliers_chart")
def outliers_chart():
    active_path, _ = get_preprocessing_paths()
    if not os.path.exists(active_path):
        return jsonify({"error": "Data aktif tidak tersedia"}), 404
    try:
        df = pd.read_csv(active_path)
        features = [feature for feature in MODEL_FEATURES if feature in df.columns]
        melted = df[features].melt(var_name="Variabel", value_name="Nilai")
        figure = px.box(melted, x="Variabel", y="Nilai", color="Variabel")
        figure.update_layout(height=400, showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        return jsonify(json.loads(figure.to_json()))
    except Exception as error:
        return jsonify({"error": str(error)}), 500
