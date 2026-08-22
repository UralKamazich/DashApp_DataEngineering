# -*- coding: utf-8 -*-
"""Apply per-dataset filters to the active working layer."""

from dash import Input, Output, State
from dash.exceptions import PreventUpdate

from dash_app import app
from callbacks.filters import _clean_filter_state
from utils import apply_filter_conditions, meta_from_df, read_df_from_store


@app.callback(
    Output("filtered-data", "data", allow_duplicate=True),
    Output("meta-columns", "data", allow_duplicate=True),
    Input("filters-applied-state", "data"),
    Input("filter-applied-logic", "data"),
    Input("active-dataset-data", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=True,
)
def apply_filters(filters_state, logic_mode, active_json, meta_state):
    """Apply committed filters to the selected working dataset."""
    if not active_json:
        raise PreventUpdate
    try:
        frame = read_df_from_store(active_json, meta_state)
    except Exception as error:
        raise PreventUpdate from error
    if frame is None or frame.empty:
        raise PreventUpdate

    filters = _clean_filter_state(filters_state)
    frame = apply_filter_conditions(
        frame,
        filters,
        meta_from_df(frame),
        logic_mode or "and",
    )
    return (
        frame.to_json(date_format="iso", orient="split"),
        meta_from_df(frame),
    )
