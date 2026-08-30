# -*- coding: utf-8 -*-
"""
Callbacks: обновление опций для всех дропдаунов (update_dropdown_options_all).
"""

from dash_app import app
from dash import Output, Input, State
from dash.exceptions import PreventUpdate
from utils import read_df_from_store, classify_simple


def _select_options(columns):
    """Return renderer-safe options even when dataframe labels are numeric."""
    return [{"label": str(column), "value": str(column)} for column in columns]


GRAPH_PATHS = {None, "", "/"}


def _options_for_dataset(filtered_json, meta):
    """Build the shared option groups once from compact column metadata."""
    if not filtered_json:
        return None

    if isinstance(meta, dict) and meta.get("columns") is not None:
        all_cols = [str(column) for column in (meta.get("columns") or [])]
        numeric_cols = [str(column) for column in (meta.get("numeric") or [])]
        categorical_cols = [str(column) for column in (meta.get("categorical") or [])]
    else:
        try:
            dff = read_df_from_store(filtered_json, meta)
        except Exception:
            return None
        if dff is None or dff.empty:
            return None
        numeric_cols, categorical_cols, _datetime_cols = classify_simple(dff)
        all_cols = [str(c) for c in dff.columns]

    all_options = _select_options(all_cols)
    return {
        "all": all_options,
        "numeric": _select_options(numeric_cols),
        "color": [{"label": "Нет", "value": "Нет"}]
        + _select_options(categorical_cols + numeric_cols),
        "facet": [{"label": "Нет", "value": "Нет"}] + all_options,
    }


def update_dropdown_options_all(filtered_json, meta):
    """Compatibility helper returning all option lists in their legacy order."""
    groups = _options_for_dataset(filtered_json, meta)
    if groups is None:
        return tuple([] for _ in range(14))
    return (
        groups["all"], groups["all"], groups["all"], groups["color"],
        groups["numeric"], groups["all"], groups["numeric"],
        groups["facet"], groups["facet"], groups["all"],
        groups["numeric"], groups["all"], groups["all"], groups["all"],
    )


@app.callback(
    [
        Output("dropdown_x", "options"),
        Output("dropdown_y", "options"),
        Output("dropdown_z", "options"),
        Output("dropdown_color", "options"),
        Output("dropdown_size", "options"),
        Output("dropdown_hover_data",  "data"),
        Output("dropdown_facet_row", "options"),
        Output("dropdown_facet_col", "options"),
        Output("dropdown_text", "options"),
        Output("dropdown_hierarchy_levels", "data"),
        Output("dropdown_hierarchy_value", "options"),
    ],
    Input("filtered-data", "data"),
    Input("url", "pathname"),
    State("meta-columns", "data"),
    prevent_initial_call=False
)
def update_graph_dropdown_options(filtered_json, pathname, meta):
    if pathname not in GRAPH_PATHS:
        raise PreventUpdate
    options = update_dropdown_options_all(filtered_json, meta)
    return (*options[:6], *options[7:10], options[0], options[1])


@app.callback(
    Output("dropdown_corr_columns", "data"),
    Input("filtered-data", "data"),
    Input("url", "pathname"),
    State("meta-columns", "data"),
    prevent_initial_call=False,
)
def update_correlation_dropdown_options(filtered_json, pathname, meta):
    if pathname != "/correlation":
        raise PreventUpdate
    return update_dropdown_options_all(filtered_json, meta)[6]


@app.callback(
    Output("bin-column", "options"),
    Output("agg-keys", "data"),
    Output("agg-cols", "data"),
    Output("txtcopy-cols", "data"),
    Output("reshape-wide-index", "data"),
    Output("reshape-wide-names", "data"),
    Output("reshape-wide-values", "data"),
    Output("reshape-long-id", "data"),
    Output("reshape-long-values", "data"),
    Input("filtered-data", "data"),
    Input("url", "pathname"),
    State("meta-columns", "data"),
    prevent_initial_call=False,
)
def update_engineering_dropdown_options(filtered_json, pathname, meta):
    if pathname != "/data-engineering":
        raise PreventUpdate
    options = update_dropdown_options_all(filtered_json, meta)
    all_columns = options[11]
    return (*options[10:14], all_columns, all_columns, all_columns, all_columns, all_columns)
