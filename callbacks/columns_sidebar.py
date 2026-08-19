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
    Output("columns-badges", "children"),
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
        "overflowX": "hidden",
        "height": "100%",
        "padding": "8px",
        "fontSize": "10px",
    }

    # Показываем фильтрованный если есть, иначе исходный
    df_to_show = None
    try:
        if filtered_json:
            df_to_show = read_df_from_store(filtered_json, meta)
        elif stored_json:
            df_to_show = read_df_from_store(stored_json, meta)
    except Exception:
        pass

    badges = []
    if df_to_show is not None and not df_to_show.empty:
        num, cat, dt = classify_simple(df_to_show)
        all_types = {}
        for c in num:
            all_types[c] = "numeric"
        for c in cat:
            all_types[c] = "categorical"
        for c in dt:
            all_types[c] = "datetime"
        badges = [
            make_column_badge(c, all_types.get(c, "categorical"))
            for c in df_to_show.columns
        ]

    return badges, sidebar_style