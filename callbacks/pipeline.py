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
    State("dataset-registry", "data"),
    State("active-dataset-id", "data"),
    prevent_initial_call=True,
)
def apply_filters(
    filters_state,
    logic_mode,
    active_json,
    meta_state,
    registry=None,
    active_id=None,
):
    """Apply committed filters to the selected working dataset."""
    if not active_json:
        raise PreventUpdate
    filters = _clean_filter_state(filters_state)
    if not filters:
        record = (registry or {}).get(str(active_id or ""), {})
        base_meta = record.get("meta") if isinstance(record, dict) else None
        if not (isinstance(base_meta, dict) and base_meta.get("columns") is not None):
            base_meta = meta_state
    else:
        base_meta = None
    if isinstance(base_meta, dict) and base_meta.get("columns") is not None:
        # The unfiltered layer is already serialized and its metadata is ready.
        # Restore metadata from the active dataset record: meta_state may still
        # describe the previously filtered layer after a filter reset.
        return active_json, base_meta
    try:
        frame = read_df_from_store(active_json, meta_state)
    except Exception as error:
        raise PreventUpdate from error
    if frame is None or frame.empty:
        raise PreventUpdate

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
