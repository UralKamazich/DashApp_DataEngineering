# -*- coding: utf-8 -*-
"""
Callback: отображение плашек колонок исходного и фильтрованного датасетов
в левом сайдбаре.
"""

import pandas as pd
from dash import callback, Output, Input, State, html
import dash_mantine_components as dmc
from dash.exceptions import PreventUpdate

from dash_app import app
from utils import read_df_from_store, classify_simple
from components import make_column_badge


@app.callback(
    Output("columns-original-badges", "children"),
    Output("columns-filtered-badges", "children"),
    Output("columns-filtered-label", "children"),
    Output("columns-sidebar", "style"),
    Input("stored-data", "data"),
    Input("filtered-data", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=False,
)
def update_column_badges(stored_json, filtered_json, meta):
    """Обновляет плашки колонок в левом сайдбаре."""
    sidebar_style = {
        "overflowY": "auto",
        "maxHeight": "calc(100vh - 100px)",
        "padding": "8px",
    }

    # Исходный датасет
    original_badges = []
    try:
        if stored_json:
            df0 = read_df_from_store(stored_json, meta)
            if df0 is not None and not df0.empty:
                num, cat, dt = classify_simple(df0)
                all_types = {}
                for c in num:
                    all_types[c] = "numeric"
                for c in cat:
                    all_types[c] = "categorical"
                for c in dt:
                    all_types[c] = "datetime"
                original_badges = [
                    make_column_badge(c, all_types.get(c, "categorical"))
                    for c in df0.columns
                ]
    except Exception:
        pass

    # Фильтрованный датасет
    filtered_badges = []
    filtered_label = "Фильтрованный датасет"
    try:
        if filtered_json:
            dff = read_df_from_store(filtered_json, meta)
            if dff is not None and not dff.empty:
                num, cat, dt = classify_simple(dff)
                all_types = {}
                for c in num:
                    all_types[c] = "numeric"
                for c in cat:
                    all_types[c] = "categorical"
                for c in dt:
                    all_types[c] = "datetime"
                filtered_badges = [
                    make_column_badge(c, all_types.get(c, "categorical"))
                    for c in dff.columns
                ]
                filtered_label = f"Фильтрованный датасет ({len(dff)} строк)"
    except Exception:
        pass

    if not original_badges and not filtered_badges:
        return [], [], "Нет данных", {**sidebar_style, "display": "none"}

    return original_badges, filtered_badges, filtered_label, sidebar_style