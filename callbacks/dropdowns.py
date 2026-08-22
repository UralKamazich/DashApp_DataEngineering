# -*- coding: utf-8 -*-
"""
Callbacks: обновление опций для всех дропдаунов (update_dropdown_options_all).
"""

from dash_app import app
from dash import callback, Output, Input, State
from utils import read_df_from_store, classify_simple


def _select_options(columns):
    """Return renderer-safe options even when dataframe labels are numeric."""
    return [{"label": str(column), "value": str(column)} for column in columns]


@app.callback(
    [
        Output("dropdown_x", "options"),
        Output("dropdown_y", "options"),
        Output("dropdown_z", "options"),
        Output("dropdown_color", "options"),
        Output("dropdown_size", "options"),
        Output("dropdown_hover_data",  "data"),
        Output("dropdown_corr_columns",  "data"),
        Output("dropdown_facet_row", "options"),
        Output("dropdown_facet_col", "options"),
        Output("dropdown_text", "options"),

        Output("bin-column", "options"),
        Output("agg-keys", "data"),
        Output("agg-cols", "data"),
        Output("txtcopy-cols", "data"),
    ],
    Input("filtered-data", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=False
)
def update_dropdown_options_all(filtered_json, meta):
    empty_axes = [[]] * 10
    empty_bin_options = []
    empty_agg_data = []
    if not filtered_json:
        return (*empty_axes,
                empty_bin_options,
                empty_agg_data, empty_agg_data, empty_agg_data)

    if isinstance(meta, dict) and meta.get("columns") is not None:
        all_cols = [str(column) for column in (meta.get("columns") or [])]
        numeric_cols = [str(column) for column in (meta.get("numeric") or [])]
        categorical_cols = [str(column) for column in (meta.get("categorical") or [])]
    else:
        # Compatibility fallback for old stores without metadata.
        try:
            dff = read_df_from_store(filtered_json, meta)
        except Exception:
            return (*empty_axes,
                    empty_bin_options,
                    empty_agg_data, empty_agg_data, empty_agg_data)
        if dff is None or dff.empty:
            return (*empty_axes,
                    empty_bin_options,
                    empty_agg_data, empty_agg_data, empty_agg_data)
        numeric_cols, categorical_cols, _datetime_cols = classify_simple(dff)
        all_cols = [str(c) for c in dff.columns]

    all_options  = _select_options(all_cols)
    color_options= [{"label": "Нет", "value": "Нет"}] + _select_options(categorical_cols + numeric_cols)
    numeric_opts = _select_options(numeric_cols)
    facet_options= [{"label": "Нет", "value": "Нет"}] + all_options

    bin_options     = _select_options(numeric_cols)
    return [
        all_options,      # X
        all_options,      # Y
        all_options,      # Z
        color_options,    # Color
        numeric_opts,     # Size
        all_options,      # Hover
        numeric_opts,     # Corr
        facet_options,    # Facet row
        facet_options,    # Facet col
        all_options,      # Text
        bin_options,      # bin-column.options
        all_options,      # agg-keys.data
        all_options,      # agg-cols.data
        all_options,      # txtcopy-cols.data
    ]
