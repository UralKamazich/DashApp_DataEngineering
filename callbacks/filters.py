# -*- coding: utf-8 -*-
"""
Callbacks: инициализация фильтров, управление фильтрами, контролы, сохранение состояния.
"""

import pandas as pd
from dash import callback, Output, Input, State, no_update, html, dcc, MATCH, ALL, ctx as _ctx
import dash_mantine_components as dmc
from dash.exceptions import PreventUpdate

from app import app
from utils import classify_simple, read_df_from_store, create_value_control
from components import create_dropdown


# =========================
# Инициализация первого фильтра
# =========================
@app.callback(
    Output("filters-container", "children"),
    Output("filter-count", "data"),
    Output("filters-initialized", "data"),
    Input("stored-data", "data"),
    State("filters-initialized", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=False
)
def init_first_filter(stored_json, inited, meta):
    if not stored_json:
        # первый рендер без данных — пусто и не инициализировано
        return [], 0, False

    # если уже инициализировали для текущего файла — ничего не делаем
    if inited:
        raise PreventUpdate

    # строим первую строку из ПОЛНОГО исходного набора
    try:
        dff0 = read_df_from_store(stored_json, meta)
    except Exception:
        dff0 = pd.DataFrame()

    num_cols, cat_cols, dt_cols = classify_simple(dff0)
    options = [''] + [*cat_cols, *num_cols, *dt_cols]

    row = dmc.Grid(
        id="filter_row_1",
        children=[
            dmc.GridCol([
                dmc.Group([
                    create_dropdown(
                        id={"type": "filter-column", "index": 1},
                        options=options,
                        value="",
                        persistence=False
                    ),
                    dmc.ActionIcon(
                        id={"type": "remove-filter", "index": 1},
                        children="×",
                        color="red",
                        variant="outline",
                        size="xs",
                        disabled=True
                    )
                ], gap="sm")
            ], span=5),
            dmc.GridCol([html.Div(id={"type": "filter-control", "index": 1})], span=6)
        ]
    )
    return [row], 1, True


@app.callback(
    Output("filters-initialized", "data", allow_duplicate=True),
    Input("stored-data", "data"),
    prevent_initial_call=True
)
def reset_filters_flag_on_new_file(_):
    return False


# === Управление фильтрами (через dash.ctx) ===
@app.callback(
    Output("filters-container", "children", allow_duplicate=True),
    Output("filter-count", "data", allow_duplicate=True),
    Output("filters-state", "data", allow_duplicate=True),
    Input("add-filter-btn", "n_clicks"),
    Input({"type": "remove-filter", "index": ALL}, "n_clicks"),
    State("filter-count", "data"),
    State("filters-container", "children"),
    State("filters-state", "data"),
    State("stored-data", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=True
)
def manage_filters(add_clicks, remove_clicks, filter_count, current_filters, filters_state, stored_json, meta):
    trig = _ctx.triggered_id
    if not trig:
        raise PreventUpdate

    # формируем опции колонок из ИСХОДНОГО набора (stored)
    try:
        dff0 = read_df_from_store(stored_json, meta) if stored_json else pd.DataFrame()
    except Exception:
        dff0 = pd.DataFrame()
    num_cols, cat_cols, dt_cols = classify_simple(dff0)
    col_options = [''] + [*cat_cols, *num_cols, *dt_cols]

    def row(filter_id, state):
        current_column = (state or {}).get(str(filter_id), {}).get('column', '')
        return dmc.Grid(
            id=f"filter_row_{filter_id}",
            children=[
                dmc.GridCol([
                    dmc.Group([
                        create_dropdown(
                            id={"type": "filter-column", "index": filter_id},
                            options=col_options,
                            value=current_column,
                            persistence=False
                        ),
                        dmc.ActionIcon(
                            id={"type": "remove-filter", "index": filter_id},
                            children="×", color="red", variant="outline", size="xs",
                            disabled=(filter_id == 1)
                        )
                    ], gap="sm")
                ], span=5),
                dmc.GridCol([html.Div(id={"type": "filter-control", "index": filter_id})], span=6)
            ]
        )

    state = (filters_state or {}).copy()
    cur = list(current_filters or [])
    count = int(filter_count or 0)

    if trig == "add-filter-btn":
        new_id = count + 1
        cur.append(row(new_id, state))
        return cur, new_id, state

    if isinstance(trig, dict) and trig.get("type") == "remove-filter":
        idx = trig.get("index")
        cur = [c for c in cur if c["props"]["id"] != f"filter_row_{idx}"]
        state.pop(str(idx), None)
        return cur, len(cur), state

    # на прочие события контейнер не трогаем
    return no_update, no_update, state


# Обновление контролов фильтра
@app.callback(
    Output({"type": "filter-control", "index": MATCH}, "children"),
    Input({"type": "filter-column", "index": MATCH}, "value"),
    State({"type": "filter-column", "index": MATCH}, "id"),
    State("filters-state", "data"),
    State("stored-data", "data"),          # ← берём исходный датасет
    State("meta-columns", "data"),
    prevent_initial_call=True
)
def update_filter_controls(column, column_id, filters_state, stored_json, meta):
    if not stored_json:
        return html.Div("Загрузите данные")

    try:
        full_dff = read_df_from_store(stored_json, meta)
    except Exception:
        return html.Div("Ошибка чтения данных")

    if full_dff.empty or not column:
        return html.Div("Выберите столбец")

    fid = str(column_id['index'])
    current_value = (filters_state or {}).get(fid, {}).get('value')

    return create_value_control(fid, column, current_value, full_dff)


# Сохранение состояния фильтров + применение к данным
@app.callback(
    Output("filters-state", "data"),
    Input({"type": "filter-column", "index": ALL}, "value"),
    Input({"type": "filter-value", "index": ALL}, "value"),
    State({"type": "filter-column", "index": ALL}, "id"),
    State("filters-state", "data"),
    State("filtered-data", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=True
)
def update_filters_state(columns, values, column_ids, filters_state, filtered_json, meta):
    import logging
    logger = logging.getLogger(__name__)
    # 1) Базовая защита: всегда возвращаем dict
    prev_state = filters_state if isinstance(filters_state, dict) else {}

    if not filtered_json:
        logger.warning("Данные не загружены")
        return prev_state

    try:
        dff = read_df_from_store(filtered_json, meta)
    except Exception:
        logger.warning("Не удалось прочитать filtered-data")
        return prev_state
    if dff.empty:
        logger.warning("Данные пусты")
        return prev_state

    updated = dict(prev_state)
    columns = columns or []
    values = values or []
    column_ids = column_ids or []

    for i, col_id in enumerate(column_ids):
        fid = str(col_id.get('index'))
        col_ok = i < len(columns) and columns[i]
        val_ok = i < len(values) and values[i] not in (None, [], '')
        if not col_ok or not val_ok:
            if fid in updated:
                del updated[fid]
            continue

        updated.setdefault(fid, {})
        updated[fid]['column'] = columns[i]
        updated[fid]['value'] = values[i]

    return updated