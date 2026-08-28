# -*- coding: utf-8 -*-
"""
Callbacks: сохранение текущего датасета в Excel.
"""

import json
import re
import uuid
from pathlib import Path
import pandas as pd
from dash import callback, Output, Input, State, no_update
from dash.exceptions import PreventUpdate

from dash_app import app
from dataset_export import export_frame_to_excel, source_directory
from dataset_registry import SOURCE_DATASET_ID, get_record, payload_for_record
from utils import _make_error_notif, read_df_from_store


def _export_ok(path):
    return [{
        "id": str(uuid.uuid4()),
        "title": "Excel сохранён",
        "message": f"Файл сохранён: {path}",
        "color": "green",
        "loading": False,
        "action": "show",
        "autoClose": 7000,
    }]


@app.callback(
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("dataset-export-request", "value"),
    State("dataset-registry", "data"),
    State("source-file-path", "data"),
    State("source-file-name", "data"),
    prevent_initial_call=True,
)
def export_registered_dataset(request, registry, source_path, source_name):
    try:
        payload = json.loads(request or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        raise PreventUpdate
    dataset_id = str(payload.get("dataset_id") or "")
    record = get_record(registry, dataset_id)
    if not record:
        return _make_error_notif("Dataset для выгрузки не найден.")
    stored = payload_for_record(record)
    if not stored:
        return _make_error_notif("Содержимое dataset недоступно.")
    try:
        frame = read_df_from_store(stored, record.get("meta") or {})
        dataset_name = (
            "Исходный" if dataset_id == SOURCE_DATASET_ID
            else str(record.get("name") or dataset_id)
        )
        path = export_frame_to_excel(
            frame,
            source_path=source_path,
            source_name=source_name,
            dataset_name=dataset_name,
            created_at=record.get("created_at"),
        )
    except Exception as error:
        return _make_error_notif(f"Ошибка сохранения Excel: {error}")
    return _export_ok(path)


# Сохранить текущий датасет в Excel
@app.callback(
    Output("download-excel", "data"),
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),
    Input("download-excel-button", "n_clicks"),
    State("filtered-data", "data"),
    State("source-file-path", "data"),
    State("source-file-name", "data"),
    State("selected-sheet", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=True
)
def download_excel_dataset(n_clicks, filtered_json, source_path, source_name, sheet_name, meta):
    if not n_clicks:
        raise PreventUpdate

    if not filtered_json:
        return no_update, _make_error_notif(
            "Нет данных для сохранения. Загрузите файл и примените фильтры/кластеризацию."
        )

    if not source_path:
        return no_update, _make_error_notif(
            "Неизвестен путь исходного файла. Выберите файл заново через кнопку выбора файла."
        )

    try:
        df = read_df_from_store(filtered_json, meta)
    except Exception as e:
        return no_update, _make_error_notif(f"Не удалось прочитать текущий датасет: {e}")

    if df is None or df.empty:
        return no_update, _make_error_notif("Текущий датасет пустой — сохранять нечего.")

    stem = Path(source_name or source_path).stem if (source_name or source_path) else "dataset"
    sheet_suffix = f"_{sheet_name}" if sheet_name else ""
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{stem}{sheet_suffix}_filtered_{ts}.xlsx"

    out_name = re.sub(r'[<>:"/\\|?*]+', '_', out_name)
    out_path = source_directory(source_path) / out_name

    try:
        df.to_excel(out_path, index=False, engine="openpyxl")
        ok = [{
            "id": str(uuid.uuid4()),
            "title": "Excel сохранён",
            "message": f"Файл сохранён: {out_path}",
            "color": "green",
            "loading": False,
            "action": "show",
            "autoClose": 7000,
            "style": {"fontSize": 18},
        }]
        return no_update, ok
    except Exception as e:
        return no_update, _make_error_notif(f"Ошибка сохранения Excel: {e}")
