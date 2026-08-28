# -*- coding: utf-8 -*-
"""
Callbacks: локальный/сетевой ввод данных и выбор листа Excel.
"""

import os
import dash
from dash import callback, Output, Input, State, no_update, html, dcc, ALL, MATCH, clientside_callback
import dash_mantine_components as dmc
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify
import pandas as pd

from dash_app import app
from data_import import (
    inspect_archive,
    popular_dataset_by_url,
    read_archive_table,
    read_delimited_dataset,
    source_name_from_location,
    validate_remote_url,
)
from utils import meta_from_df, _make_error_notif, read_df_from_store
from config import STYLE_CARD


def _clicked_sheet(n_clicks, ids, triggered_id=None):
    """Resolve the sheet that actually caused an ALL-pattern callback."""
    if (
        isinstance(triggered_id, dict)
        and triggered_id.get("type") in {"sheet-select", "archive-select"}
    ):
        return triggered_id.get("index")
    clicked = [
        index for index, count in enumerate(n_clicks or [])
        if count and index < len(ids or [])
    ]
    if not clicked:
        return None
    return ids[clicked[-1]].get("index")


def _human_size(value):
    size = float(value or 0)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024 or unit == "ГБ":
            return f"{size:.1f} {unit}" if unit != "Б" else f"{int(size)} Б"
        size /= 1024


def _archive_modal(tables):
    return dmc.Modal(
        id="sheet-modal",
        title="Выберите таблицу в ZIP",
        opened=True,
        centered=True,
        zIndex=1000,
        size="lg",
        children=dmc.ScrollArea(
            h=min(480, max(100, len(tables) * 42)),
            children=dmc.Stack(
                gap=5,
                children=[
                    dmc.Button(
                        dmc.Group(
                            [
                                dmc.Text(item["name"], size="xs", truncate="end"),
                                dmc.Text(
                                    _human_size(item["size"]), size="10px", c="dimmed"
                                ),
                            ],
                            justify="space-between", wrap="nowrap", w="100%",
                        ),
                        variant="subtle",
                        fullWidth=True,
                        id={"type": "archive-select", "index": item["name"]},
                    )
                    for item in tables
                ],
            ),
        ),
    )


# ============ Локальный выбор файла (даёт полный путь) ============
@app.callback(
    Output("source-file-path", "data"),
    Output("source-file-name", "data"),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("pick-file-btn", "n_clicks"),
    Input("dataset-file-drop-store", "data"),
    Input("online-dataset-load", "n_clicks"),
    State("online-dataset-catalog", "value"),
    State("online-dataset-url", "value"),
    prevent_initial_call=True
)
def pick_local_file(n_clicks, dropped_file, online_clicks, catalog_url, custom_url):
    if dash.ctx.triggered_id == "dataset-file-drop-store":
        path = str((dropped_file or {}).get("path") or "")
        if not path:
            raise PreventUpdate
        name = str((dropped_file or {}).get("name") or os.path.basename(path))
        return path, name, []

    if dash.ctx.triggered_id == "online-dataset-load":
        if not online_clicks:
            raise PreventUpdate
        try:
            location = validate_remote_url(custom_url or catalog_url)
            catalog_item = popular_dataset_by_url(location)
            name = (
                catalog_item["name"] if catalog_item
                else source_name_from_location(location)
            )
            extension = os.path.splitext(name)[1].lower()
            if extension not in {".csv", ".txt", ".tsv", ".zip"}:
                raise ValueError(
                    "Ссылка должна вести прямо на .csv, .txt, .tsv или .zip."
                )
            return location, name, []
        except Exception as error:
            return no_update, no_update, _make_error_notif(str(error))

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
    except Exception as error:
        return no_update, no_update, _make_error_notif(
            f"Не удалось открыть диалог выбора файла: {error}"
        )

    return path, os.path.basename(path), []


@app.callback(
    Output("online-dataset-description", "children"),
    Input("online-dataset-catalog", "value"),
)
def describe_online_dataset(value):
    item = popular_dataset_by_url(value)
    return item["description"] if item else "Выберите dataset из каталога."


@app.callback(
    Output('status-message', 'children'),
    Output('sheet-menu-wrapper', 'children'),
    Output('stored-sheet-names', 'data'),
    Output('selected-sheet', 'data'),
    Output('stored-data', 'data'),
    Output('filtered-data', 'data', allow_duplicate=True),
    Output('meta-columns', 'data', allow_duplicate=True),
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),
    Input('source-file-path', 'data'),
    State('source-file-name', 'data'),
    prevent_initial_call=True
)
def on_excel_upload(local_path, local_name):
    """Загрузить локальный или удалённый источник в единый data pipeline.

    Сам файл здесь НЕ читаем: только узнаём список листов и выставляем
    selected-sheet. Чтение делает единственный callback load_selected_sheet —
    иначе файл грузился бы дважды и график перерисовывался лишние разы.
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

                # Для одно-листовой книги используем уже открытую книгу. Раньше
                # после получения sheet_names она закрывалась и тут же
                # открывалась повторно в load_selected_sheet.
                if len(sheets) == 1:
                    sheet_name = sheets[0]
                    df = xl.parse(sheet_name)

            # Один лист прочитан одним проходом; callback выбора листа на этот
            # случай больше не подписан и повторного чтения не будет.
            if len(sheets) == 1:
                meta = meta_from_df(df)
                js = df.to_json(date_format='iso', orient='split')
                # filtered-data has a single owner: callbacks.pipeline.  It will
                # receive active-dataset-data after the source registry is
                # initialized.  Publishing the same payload here used to start
                # the expensive UI cascade twice for wide workbooks.
                return "", dash.no_update, sheets, sheet_name, js, dash.no_update, meta, []

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
            return "", modal, sheets, dash.no_update, dash.no_update, dash.no_update, dash.no_update, []
        elif ext == '.pkl':
            # Для .pkl выбора листа нет — читаем сразу здесь.
            df = pd.read_pickle(local_path)
            js = df.to_json(date_format='iso', orient='split')
            meta = meta_from_df(df)
            return "", dash.no_update, None, None, js, js, meta, []
        elif ext in {'.csv', '.txt', '.tsv'}:
            df, import_info = read_delimited_dataset(local_path, filename)
            js = df.to_json(date_format='iso', orient='split')
            meta = meta_from_df(df)
            meta["import"] = import_info.as_meta()
            return "", dash.no_update, None, None, js, dash.no_update, meta, []
        elif ext == '.zip':
            tables, df, import_info = inspect_archive(local_path, filename)
            if len(tables) > 1:
                return (
                    "", _archive_modal(tables),
                    [item["name"] for item in tables], None,
                    dash.no_update, dash.no_update, dash.no_update, [],
                )
            member = tables[0]["name"]
            js = df.to_json(date_format='iso', orient='split')
            meta = meta_from_df(df)
            meta["import"] = import_info.as_meta()
            return "", dash.no_update, [member], None, js, dash.no_update, meta, []

    except Exception as e:
        notif = _make_error_notif(f"Ошибка загрузки файла: {str(e)}")
        return html.Div(f"Ошибка: {e}", style={'color': 'red'}), dash.no_update, None, dash.no_update, dash.no_update, dash.no_update, dash.no_update, notif

    notif = _make_error_notif(
        "Неподдерживаемый формат: выберите .xlsx, .csv, .txt, .tsv, .zip или .pkl"
    )
    return html.Div("Неподдерживаемый формат", style={'color': 'red'}), dash.no_update, None, dash.no_update, dash.no_update, dash.no_update, dash.no_update, notif


@app.callback(
    Output('sheet-modal', 'opened'),
    Input({'type': 'sheet-select', 'index': ALL}, 'n_clicks'),
    Input({'type': 'archive-select', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def close_modal(sheet_clicks, archive_clicks):
    if not any(sheet_clicks or []) and not any(archive_clicks or []):
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
    selected = _clicked_sheet(n_clicks, ids, dash.ctx.triggered_id)
    if selected is None:
        raise PreventUpdate
    return selected


# ============ Информация о загруженном файле (внизу, серым) ============
@app.callback(
    Output('file-info-bar', 'children'),
    Input('stored-data', 'data'),
    Input('meta-columns', 'data'),
    Input('selected-sheet', 'data'),
    State('source-file-name', 'data'),
    prevent_initial_call=True
)
def update_file_info(stored_json, meta, sheet_name, source_name):
    if not stored_json:
        raise PreventUpdate

    n_rows = meta.get("row_count", "?") if isinstance(meta, dict) else "?"
    n_cols = meta.get("column_count", "?") if isinstance(meta, dict) else "?"
    if n_rows == "?" or n_cols == "?":
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
    import_info = meta.get("import") if isinstance(meta, dict) else None
    if import_info:
        delimiter = {
            "\t": "TAB", r"\s+": "пробел", "\0": "один столбец",
        }.get(import_info.get("delimiter"), import_info.get("delimiter"))
        source_kind = "Интернет" if import_info.get("remote") else "Локальный"
        details = [f"{source_kind} {import_info.get('format', 'TEXT')}"]
        if import_info.get("archive_member"):
            details.append(f"файл {import_info['archive_member']}")
        if import_info.get("encoding"):
            details.append(str(import_info["encoding"]))
        if delimiter:
            details.append(f"разделитель {delimiter}")
        parts.append(" · ".join(details))
    parts.append(f"Строк: {n_rows}")
    parts.append(f"Столбцов: {n_cols}")

    return "  |  ".join(parts)


@app.callback(
    Output('stored-data', 'data', allow_duplicate=True),
    Output('status-message', 'children', allow_duplicate=True),
    Output('sheet-modal', 'opened', allow_duplicate=True),
    Output('filtered-data', 'data', allow_duplicate=True),
    Output('meta-columns', 'data', allow_duplicate=True),
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),
    Input({'type': 'sheet-select', 'index': ALL}, 'n_clicks'),
    State({'type': 'sheet-select', 'index': ALL}, 'id'),
    State('source-file-path', 'data'),
    prevent_initial_call=True
)
def load_selected_sheet(n_clicks, ids, local_path):
    if not any(n_clicks or []):
        raise PreventUpdate
    sheet_name = _clicked_sheet(n_clicks, ids, dash.ctx.triggered_id)
    if sheet_name is None:
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

        # Let callbacks.pipeline publish filtered-data once the active dataset
        # has been initialized.
        return js, "", False, dash.no_update, meta, []
    except Exception as e:
        notif = _make_error_notif(f"Ошибка загрузки листа: {str(e)}")
        return dash.no_update, html.Div(f"Ошибка: {e}", style={'color': 'red'}), False, dash.no_update, dash.no_update, notif


@app.callback(
    Output('stored-data', 'data', allow_duplicate=True),
    Output('status-message', 'children', allow_duplicate=True),
    Output('sheet-modal', 'opened', allow_duplicate=True),
    Output('filtered-data', 'data', allow_duplicate=True),
    Output('meta-columns', 'data', allow_duplicate=True),
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),
    Input({'type': 'archive-select', 'index': ALL}, 'n_clicks'),
    State({'type': 'archive-select', 'index': ALL}, 'id'),
    State('source-file-path', 'data'),
    State('source-file-name', 'data'),
    prevent_initial_call=True,
)
def load_selected_archive_table(n_clicks, ids, source_path, source_name):
    if not any(n_clicks or []):
        raise PreventUpdate
    member_name = _clicked_sheet(n_clicks, ids, dash.ctx.triggered_id)
    if member_name is None:
        raise PreventUpdate
    if not source_path:
        notif = _make_error_notif("ZIP больше недоступен. Выберите архив заново.")
        return no_update, no_update, False, no_update, no_update, notif
    try:
        frame, import_info = read_archive_table(
            source_path, member_name, source_name
        )
        meta = meta_from_df(frame)
        meta["import"] = import_info.as_meta()
        stored = frame.to_json(date_format="iso", orient="split")
        return stored, "", False, no_update, meta, []
    except Exception as error:
        notif = _make_error_notif(f"Ошибка чтения ZIP: {error}")
        return (
            no_update,
            html.Div(f"Ошибка: {error}", style={"color": "red"}),
            False, no_update, no_update, notif,
        )
