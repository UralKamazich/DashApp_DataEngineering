# -*- coding: utf-8 -*-
"""
Callbacks: скачивание HTML, Excel, clientside PNG.
"""

import re
import os
import uuid
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from dash import callback, Output, Input, State, no_update
from dash.exceptions import PreventUpdate

from dash_app import app
from utils import _make_error_notif, read_df_from_store


# Скачивание HTML
@app.callback(
    Output("download-file", "data"),
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),
    Input("download-button", "n_clicks"),
    State("graph", "figure"),
    State("dropdown_x", "value"),
    State("dropdown_y", "value"),
    State("segmented", "value"),
    prevent_initial_call=True
)
def download_html(n_clicks, figure, dropdown_x, dropdown_y, segmentedcontrol_value):
    if not n_clicks or not figure:
        raise PreventUpdate
    try:
        fig = go.Figure(figure)
        filename = (
            f'{dropdown_x} vs {dropdown_y} {segmentedcontrol_value}.html'
            if all([dropdown_x, dropdown_y, segmentedcontrol_value]) else "graph.html"
        )
        html_content = fig.to_html(include_plotlyjs='cdn')
        return {"content": html_content, "filename": filename, "type": "text/html"}, []
    except Exception as e:
        notif = _make_error_notif(f"Ошибка скачивания: {str(e)}")
        return no_update, notif


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
    out_path = Path(source_path).resolve().parent / out_name

    try:
        df.to_excel(out_path, index=False, engine="openpyxl")
        ok = [{
            "id": str(uuid.uuid4()),
            "title": "Excel сохранён",
            "message": f"Файл сохранён рядом с исходником: {out_path}",
            "color": "green",
            "loading": False,
            "action": "show",
            "autoClose": 7000,
            "style": {"fontSize": 18},
        }]
        return no_update, ok
    except Exception as e:
        return no_update, _make_error_notif(f"Ошибка сохранения Excel: {e}")


# Кнопка на графике: только копирование PNG в буфер, без скачивания файла.
app.clientside_callback(
    """
    function(n_clicks, figure) {
        if (!n_clicks || !figure) {
            throw window.dash_clientside.PreventUpdate;
        }

        function notification(title, message, color) {
            return [{
                id: crypto.randomUUID(),
                title: title,
                message: message,
                color: color,
                action: 'show',
                autoClose: 4500
            }];
        }

        if (!window.graphPng) {
            return notification('PNG не скопирован', 'Модуль экспорта не загружен.', 'red');
        }

        return window.graphPng.copyToClipboard().then(
            () => notification('PNG скопирован', 'Изображение помещено в буфер обмена.', 'green'),
            (error) => {
                console.error('Clipboard PNG error:', error);
                return notification('PNG не скопирован', error.message || 'Ошибка буфера обмена.', 'red');
            }
        );
    }
    """,
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("copy-png-button", "n_clicks"),
    State("graph", "figure"),
    prevent_initial_call=True
)


# Пункт контекстного меню: только сохранение PNG в файл, без буфера обмена.
app.clientside_callback(
    """
    function(n_clicks, figure) {
        if (!n_clicks || !figure) {
            throw window.dash_clientside.PreventUpdate;
        }

        function notification(title, message, color) {
            return [{
                id: crypto.randomUUID(),
                title: title,
                message: message,
                color: color,
                action: 'show',
                autoClose: 4500
            }];
        }

        if (!window.graphPng) {
            return notification('PNG не сохранён', 'Модуль экспорта не загружен.', 'red');
        }

        return window.graphPng.saveToFile().then(
            () => notification('PNG сохранён', 'Файл с графиком передан в загрузки.', 'green'),
            (error) => {
                console.error('Save PNG error:', error);
                return notification('PNG не сохранён', error.message || 'Ошибка сохранения файла.', 'red');
            }
        );
    }
    """,
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("save-png-button", "n_clicks"),
    State("graph", "figure"),
    prevent_initial_call=True
)
