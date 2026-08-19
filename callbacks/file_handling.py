# -*- coding: utf-8 -*-
"""
Callbacks: выбор файла, загрузка .xlsx/.pkl, выбор листа Excel.
"""

import os
import dash
from dash import callback, Output, Input, State, no_update, html, dcc, ALL, MATCH, clientside_callback
import dash_mantine_components as dmc
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify
import pandas as pd

from dash_app import app
from utils import meta_from_df, _make_error_notif, read_df_from_store
from config import STYLE_CARD


# ============ Локальный выбор файла (даёт полный путь) ============
@app.callback(
    Output("source-file-path", "data"),
    Output("source-file-name", "data"),
    Input("pick-file-btn", "n_clicks"),
    prevent_initial_call=True
)
def pick_local_file(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    try:
        import subprocess
        import sys
        from pathlib import Path

        # Запускаем filedialog в ОТДЕЛЬНОМ процессе, чтобы NSWindow был в главном потоке
        dialog_script = Path(__file__).resolve().parent.parent / "_filedialog.py"
        result = subprocess.run(
            [sys.executable, str(dialog_script)],
            capture_output=True, text=True, timeout=120,
        )
        path = result.stdout.strip()
        if not path:
            raise PreventUpdate
    except PreventUpdate:
        raise
    except Exception as e:
        return no_update, no_update

    return path, os.path.basename(path)


@app.callback(
    Output('status-message', 'children'),
    Output('sheet-menu-wrapper', 'children'),
    Output('stored-sheet-names', 'data'),
    Output('stored-data', 'data'),
    Output('selected-sheet', 'data'),
    Output('filtered-data', 'data', allow_duplicate=True),
    Output('meta-columns', 'data', allow_duplicate=True),
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),
    Input('source-file-path', 'data'),
    State('source-file-name', 'data'),
    prevent_initial_call=True
)
def on_excel_upload(local_path, local_name):
    """Загрузка данных ТОЛЬКО с локального диска через выбор файла (tkinter).
    Сохранение пути делается отдельным callback по кнопке pick-file-btn.
    """
    if not local_path:
        raise PreventUpdate

    filename = local_name or os.path.basename(local_path)
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == '.xlsx':
            # ВАЖНО: ExcelFile нужно закрывать, иначе на Windows исходный .xlsx может оставаться 'занятым'.
            with pd.ExcelFile(local_path, engine='openpyxl') as xl:
                sheets = xl.sheet_names
                # один лист — читаем сразу
                if len(sheets) == 1:
                    sheet_name = sheets[0]
                    df = xl.parse(sheet_name)

            # после выхода из with файл гарантированно закрыт
            if len(sheets) == 1:
                meta = meta_from_df(df)
                js = df.to_json(date_format='iso', orient='split')
                return (
                    "",
                    dash.no_update, sheets, js, sheet_name,
                    js, meta, []
                )

            # несколько листов — показываем модалку выбора
            modal = dmc.Modal(
                id="sheet-modal",
                title="Выберите лист Excel",
                opened=True,
                centered=True,
                zIndex=1000,
                children=[
                    dmc.Stack(
                        gap="sm",
                        children=[
                            dmc.Text("Выберите лист", c="blue", fw=500, size="sm"),
                            *[
                                dmc.Button(
                                    sheet,
                                    variant="light",
                                    fullWidth=True,
                                    id={"type": "sheet-select", "index": sheet},
                                    leftSection=DashIconify(icon="vscode-icons:file-type-excel2", width=20)
                                )
                                for sheet in sheets
                            ]
                        ]
                    )
                ]
            )
            return (
                "",
                modal, sheets, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, []
            )
        elif ext == '.pkl':
            df = pd.read_pickle(local_path)
            meta = meta_from_df(df)
            js = df.to_json(date_format='iso', orient='split')
            return (
                "",
                dash.no_update, None, js, None,
                js, meta, []
            )

    except Exception as e:
        notif = _make_error_notif(f"Ошибка загрузки файла: {str(e)}")
        return (
            html.Div(f"Ошибка: {e}", style={'color': 'red'}),
            dash.no_update, None, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, notif
        )

    notif = _make_error_notif("Неподдерживаемый формат (нужно .xlsx или .pkl)")
    return (
        html.Div("Неподдерживаемый формат", style={'color': 'red'}),
        dash.no_update, None, dash.no_update, dash.no_update,
        dash.no_update, dash.no_update, notif
    )


@app.callback(
    Output('sheet-modal', 'opened'),
    Input({'type': 'sheet-select', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def close_modal(n_clicks):
    if not any(n_clicks):
        raise PreventUpdate
    return False


@app.callback(
    Output('selected-sheet', 'data', allow_duplicate=True),
    Input({'type': 'sheet-select', 'index': ALL}, 'n_clicks'),
    State({'type': 'sheet-select', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def on_sheet_selected(n_clicks, ids):
    if not any(n_clicks):
        raise PreventUpdate
    clicked_idx = [i for i, n in enumerate(n_clicks) if n]
    if not clicked_idx:
        raise PreventUpdate
    selected = ids[clicked_idx[0]]['index']
    return selected


# ============ Информация о загруженном файле (внизу, серым) ============
@app.callback(
    Output('file-info-bar', 'children'),
    Output('file-info-bar', 'style'),
    Input('stored-data', 'data'),
    Input('meta-columns', 'data'),
    Input('selected-sheet', 'data'),
    State('source-file-name', 'data'),
    prevent_initial_call=True
)
def update_file_info(stored_json, meta, sheet_name, source_name):
    if not stored_json:
        raise PreventUpdate

    n_rows = "?"
    n_cols = "?"
    try:
        df = read_df_from_store(stored_json, meta)
        n_rows = len(df)
        n_cols = len(df.columns)
    except Exception:
        if isinstance(meta, dict):
            n_cols = len(meta.get("columns", []) or [])

    parts = []
    if source_name:
        parts.append(f"📄 {source_name}")
    if sheet_name:
        parts.append(f"Лист: {sheet_name}")
    parts.append(f"Строк: {n_rows}")
    parts.append(f"Столбцов: {n_cols}")

    info_text = "  |  ".join(parts)
    info_style = {
        "backgroundColor": "#1A1B1E",
        "borderTop": "1px solid #2C2E33",
        "padding": "16px",
        "display": "block",
        "color": "#888",
        "fontSize": "12px",
        "fontFamily": "monospace",
        "textAlign": "right",
        "minHeight": "36px",
    }
    return info_text, info_style


@app.callback(
    Output('stored-data', 'data', allow_duplicate=True),
    Output('status-message', 'children', allow_duplicate=True),
    Output('sheet-modal', 'opened', allow_duplicate=True),
    Output('filtered-data', 'data', allow_duplicate=True),
    Output('meta-columns', 'data', allow_duplicate=True),
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),
    Input('selected-sheet', 'data'),
    State('source-file-path', 'data'),
    prevent_initial_call=True
)
def load_selected_sheet(sheet_name, local_path):
    if not sheet_name:
        raise PreventUpdate

    if not local_path:
        notif = _make_error_notif("Нет пути к исходному файлу. Выберите файл заново.")
        return dash.no_update, html.Div("Ошибка: нет пути к файлу", style={'color': 'red'}), False, dash.no_update, dash.no_update, notif

    try:
        df = pd.read_excel(local_path, engine='openpyxl', sheet_name=sheet_name)
    except Exception as e:
        notif = _make_error_notif(f"Ошибка загрузки листа: {str(e)}")
        return dash.no_update, html.Div(f"Ошибка: {e}", style={'color': 'red'}), False, dash.no_update, dash.no_update, notif

    try:
        meta = meta_from_df(df)
        js = df.to_json(date_format='iso', orient='split')

        return js, "", False, js, meta, []
    except Exception as e:
        notif = _make_error_notif(f"Ошибка загрузки листа: {str(e)}")
        return dash.no_update, html.Div(f"Ошибка: {e}", style={'color': 'red'}), False, dash.no_update, dash.no_update, notif