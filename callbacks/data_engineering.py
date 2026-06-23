# -*- coding: utf-8 -*-
"""
Callbacks: Data Engineering — создание текстовых копий столбцов (create_text_copies).
"""

import re
import pandas as pd
import numpy as np
from dash import callback, Output, Input, State, no_update
from dash.exceptions import PreventUpdate

from app import app
from utils import read_df_from_store, meta_from_df


@app.callback(
    Output("filtered-data", "data", allow_duplicate=True),
    Output("meta-columns", "data", allow_duplicate=True),
    Output("de-txt-status", "children"),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("btn-txtcopy", "n_clicks"),
    State("filtered-data", "data"),
    State("meta-columns", "data"),
    State("txtcopy-cols", "value"),
    State("txtcopy-suffix", "value"),
    State("txtcopy-strip", "checked"),
    prevent_initial_call=True
)
def create_text_copies(n_clicks, filtered_json, meta, selected_cols, suffix, do_strip):
    if not n_clicks:
        raise PreventUpdate

    notifications = []

    if not filtered_json:
        notifications.append({
            "id": "de-txtcopy-nodata",
            "title": "Data Engineering",
            "message": "Нет данных: сначала загрузите файл и/или примените фильтры.",
            "color": "red",
            "action": "show",
            "autoClose": 6000,
        })
        return no_update, no_update, "Нет данных для преобразования.", notifications

    try:
        df = read_df_from_store(filtered_json, meta)
    except Exception as e:
        notifications.append({
            "id": "de-txtcopy-badjson",
            "title": "Data Engineering",
            "message": f"Не удалось прочитать текущий датасет: {e}",
            "color": "red",
            "action": "show",
            "autoClose": 7000,
        })
        return no_update, no_update, "Ошибка чтения данных.", notifications

    if df is None or df.empty:
        notifications.append({
            "id": "de-txtcopy-empty",
            "title": "Data Engineering",
            "message": "Текущий датасет пустой — нечего преобразовывать.",
            "color": "yellow",
            "action": "show",
            "autoClose": 6000,
        })
        return no_update, no_update, "Текущий датасет пустой.", notifications

    selected_cols = selected_cols or []
    if not selected_cols:
        notifications.append({
            "id": "de-txtcopy-noselect",
            "title": "Data Engineering",
            "message": "Выберите хотя бы один столбец для создания текстовой копии.",
            "color": "yellow",
            "action": "show",
            "autoClose": 6000,
        })
        return no_update, no_update, "Выберите столбец(ы).", notifications

    suffix = (suffix or "_txt")
    suf_clean = str(suffix).strip()
    if suf_clean and not suf_clean.startswith("_"):
        suf_clean = "_" + suf_clean
    if not suf_clean:
        suf_clean = "_txt"

    def _norm_token_local(x) -> str:
        return re.sub(r"[\s\u00A0]+", "", str(x)).lower()

    col_map = {_norm_token_local(c): c for c in df.columns}

    created = []
    skipped = []

    for col in selected_cols:
        actual = col_map.get(_norm_token_local(col))
        if not actual:
            skipped.append(str(col))
            continue

        base_new = f"{actual}{suf_clean}"
        new_name = base_new
        k = 2
        while new_name in df.columns:
            new_name = f"{base_new}_{k}"
            k += 1

        s = df[actual]

        try:
            new_s = s.astype("string")
        except Exception:
            new_s = pd.Series([None if pd.isna(v) else str(v) for v in s], dtype="string")

        if do_strip:
            try:
                new_s = new_s.str.strip()
            except Exception:
                pass

        df[new_name] = new_s
        created.append(new_name)

    if not created:
        msg = "Не создано ни одной колонки (проверьте выбор столбцов)."
        if skipped:
            msg += f" Не найдены: {', '.join(skipped[:8])}" + ("..." if len(skipped) > 8 else "")
        notifications.append({
            "id": "de-txtcopy-none",
            "title": "Data Engineering",
            "message": msg,
            "color": "yellow",
            "action": "show",
            "autoClose": 8000,
        })
        return no_update, no_update, "Ничего не создано.", notifications

    meta = meta_from_df(df)
    out_json = df.to_json(date_format="iso", orient="split")

    notifications.append({
        "id": "de-txtcopy-ok",
        "title": "Data Engineering",
        "message": f"Созданы текстовые копии: {len(created)}",
        "color": "green",
        "action": "show",
        "autoClose": 4000,
    })

    status = f"Готово: создано {len(created)}: " + ", ".join(created[:6]) + ("..." if len(created) > 6 else "")
    if skipped:
        status += f" | не найдены: {', '.join(skipped[:4])}" + ("..." if len(skipped) > 4 else "")

    return out_json, meta, status, notifications