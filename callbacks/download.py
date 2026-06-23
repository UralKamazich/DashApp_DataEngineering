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

from app import app
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


# Clientside callback для копирования PNG
app.clientside_callback(
    """
    function(n_clicks, figure) {
        if (!n_clicks || !figure) {
            throw window.dash_clientside.PreventUpdate;
        }

        // dataURL -> Blob без fetch (чтобы не терять user-gesture)
        function dataURLtoBlob(dataURL) {
            const [header, data] = dataURL.split(',');
            const mime = (header.match(/:(.*?);/) || [,'image/png'])[1];
            const binary = atob(data);
            const array = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) array[i] = binary.charCodeAt(i);
            return new Blob([array], { type: mime });
        }

        try {
            const host = document.getElementById('graph');
            const gd = host && host.getElementsByClassName('js-plotly-plot')[0];
            if (!gd) return 'График не найден в DOM.';

            // === WYSIWYG размеры: берём реальные видимые пиксели SVG ===
            const svg = gd.querySelector('svg.main-svg') || gd.querySelector('svg');
            let width, height;
            if (svg) {
                const r = svg.getBoundingClientRect(); // учитывает любые CSS-скейлы/zoom
                width  = Math.max(1, Math.round(r.width));
                height = Math.max(1, Math.round(r.height));
            } else {
                // фоллбэк, если svg не найден
                const r = gd.getBoundingClientRect();
                width  = Math.max(1, Math.round(r.width));
                height = Math.max(1, Math.round(r.height));
            }

            // Рисуем PNG ровно под видимые размеры (scale:1 для точного соответствия)
            return window.Plotly.toImage(gd, {
                format: 'png',
                width:  width,
                height: height,
                scale:  1
            })
            .then((dataUrl) => {
                // Пытаемся записать в буфер в рамках того же клика
                if (window.isSecureContext && window.ClipboardItem && navigator.clipboard && navigator.clipboard.write) {
                    try {
                        const blob = dataURLtoBlob(dataUrl);
                        const item = new ClipboardItem({ [blob.type]: blob });
                        return navigator.clipboard.write([item]).then(
                            () => {
                                // Параллельно сохраняем файл
                                const a = document.createElement('a');
                                a.href = dataUrl;
                                a.download = 'plotly_graph.png';
                                document.body.appendChild(a); a.click(); a.remove();
                                return 'PNG (как на экране) скопирован в буфер и сохранён как файл.';
                            },
                            (e) => {
                                console.warn('Clipboard write failed:', e);
                                const a = document.createElement('a');
                                a.href = dataUrl;
                                a.download = 'plotly_graph.png';
                                document.body.appendChild(a); a.click(); a.remove();
                                return 'Буфер недоступен — PNG (как на экране) сохранён как файл.';
                            }
                        );
                    } catch (e) {
                        console.warn('Clipboard exception:', e);
                    }
                }

                // Фоллбэк: только файл (например, не secure-контекст)
                const a = document.createElement('a');
                a.href = dataUrl;
                a.download = 'plotly_graph.png';
                document.body.appendChild(a); a.click(); a.remove();
                return 'Копирование в буфер недоступно — PNG (как на экране) сохранён как файл.';
            })
            .catch((err) => {
                console.error('toImage error:', err);
                return 'Ошибка генерации PNG. См. консоль.';
            });

        } catch (err) {
            console.error('Ошибка:', err);
            return 'Ошибка копирования/сохранения. См. консоль.';
        }
    }
    """,
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("copy-png-button", "n_clicks"),
    State("graph", "figure"),
    prevent_initial_call=True
)