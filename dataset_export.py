# -*- coding: utf-8 -*-
"""Safe, deterministic Excel export for registered datasets."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import tempfile

import pandas as pd


_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def safe_filename_part(value, fallback="dataset"):
    text = _INVALID_FILENAME.sub("_", str(value or "")).strip(" ._-")
    text = re.sub(r"\s+", " ", text)
    return text[:120] or fallback


def creation_stamp(value=None):
    if value:
        try:
            parsed = datetime.fromisoformat(str(value))
            return parsed.strftime("%Y%m%d_%H%M%S")
        except (TypeError, ValueError):
            pass
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def dataset_export_name(source_path, source_name, dataset_name, created_at=None):
    source_value = source_name or (Path(source_path).name if source_path else "dataset")
    source_stem = safe_filename_part(Path(str(source_value)).stem, "dataset")
    dataset_part = safe_filename_part(dataset_name, "dataset")
    return f"{source_stem} - {dataset_part} - {creation_stamp(created_at)}.xlsx"


def model_export_name(source_path, source_name, experiment_name, created_at=None):
    source_value = source_name or (Path(source_path).name if source_path else "dataset")
    source_stem = safe_filename_part(Path(str(source_value)).stem, "dataset")
    experiment_part = safe_filename_part(experiment_name, "CatBoost")
    return f"{source_stem} - {experiment_part} - model - {creation_stamp(created_at)}.cbm"


def _available_path(path):
    if not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem} ({index}){path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def export_frame_to_excel(frame, *, source_path, source_name=None,
                          dataset_name=None, created_at=None):
    if frame is None or frame.empty:
        raise ValueError("Dataset пустой — выгружать нечего.")
    if not source_path:
        raise ValueError("Неизвестна директория исходного файла. Выберите файл заново.")
    directory = Path(source_path).expanduser().resolve().parent
    if not directory.is_dir():
        raise ValueError("Директория исходного файла недоступна.")
    if len(frame.index) > 1_048_576 or len(frame.columns) > 16_384:
        raise ValueError("Dataset превышает ограничение одного листа Excel.")
    target = _available_path(directory / dataset_export_name(
        source_path, source_name, dataset_name, created_at
    ))
    export_frame = frame.copy()
    for column in export_frame.columns:
        dtype = export_frame[column].dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            export_frame[column] = export_frame[column].dt.tz_localize(None)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".dashapp-export-", suffix=".xlsx", dir=directory, delete=False
        ) as temporary:
            temporary_name = temporary.name
        export_frame.to_excel(temporary_name, index=False, engine="openpyxl")
        Path(temporary_name).replace(target)
    except Exception:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return target


def export_catboost_model(model, *, source_path, source_name=None,
                          experiment_name=None, created_at=None):
    """Atomically save a native CatBoost artifact next to the source dataset."""
    if model is None:
        raise ValueError("Обученная модель недоступна.")
    if not source_path:
        raise ValueError("Неизвестна директория исходного файла. Выберите файл заново.")
    directory = Path(source_path).expanduser().resolve().parent
    if not directory.is_dir():
        raise ValueError("Директория исходного файла недоступна.")
    target = _available_path(directory / model_export_name(
        source_path, source_name, experiment_name, created_at
    ))
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".dashapp-model-", suffix=".cbm", dir=directory, delete=False
        ) as temporary:
            temporary_name = temporary.name
        model.save_model(temporary_name, format="cbm")
        Path(temporary_name).replace(target)
    except Exception:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return target
