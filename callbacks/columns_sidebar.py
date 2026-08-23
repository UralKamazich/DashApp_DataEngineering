# -*- coding: utf-8 -*-
"""Render the active dataset channel rail, with optional windowing."""

from __future__ import annotations

import json

from dash import Input, Output, State, html

from dash_app import app
from utils import read_df_from_store, classify_simple
from components import make_column_badge


VIRTUALIZATION_THRESHOLD = 200
DEFAULT_VIRTUAL_WINDOW = 48
VIRTUAL_ROW_HEIGHT = 29


def _column_model(filtered_json, meta):
    """Return column order and types without reading the frame when metadata exists."""
    if filtered_json and isinstance(meta, dict) and meta.get("columns") is not None:
        columns = [str(column) for column in (meta.get("columns") or [])]
        numeric = [str(column) for column in (meta.get("numeric") or [])]
        categorical = [str(column) for column in (meta.get("categorical") or [])]
        datetime = [str(column) for column in (meta.get("datetime") or [])]
        return columns, numeric, categorical, datetime

    try:
        frame = read_df_from_store(filtered_json, meta) if filtered_json else None
    except Exception:
        frame = None
    if frame is None or frame.empty:
        return [], [], [], []
    numeric, categorical, datetime = classify_simple(frame)
    return [str(column) for column in frame.columns], numeric, categorical, datetime


def _window_bounds(total, enabled, window_state):
    if not enabled:
        return 0, total
    state = window_state if isinstance(window_state, dict) else {}
    try:
        start = max(0, int(state.get("start", 0)))
        size = max(16, int(state.get("size", DEFAULT_VIRTUAL_WINDOW)))
    except (TypeError, ValueError):
        start, size = 0, DEFAULT_VIRTUAL_WINDOW
    start = min(start, max(0, total - 1))
    return start, min(total, start + size)


@app.callback(
    Output("columns-badges", "children"),
    Output("dataset-column-catalog", "children"),
    Output("dataset-virtualization-status", "children"),
    Input("filtered-data", "data"),
    Input("dataset-virtualize-columns", "checked"),
    Input("dataset-virtual-window", "data"),
    State("meta-columns", "data"),
    State("dataset-registry", "data"),
    State("active-dataset-id", "data"),
    prevent_initial_call=False,
)
def update_column_badges(
    filtered_json,
    virtualize_columns,
    virtual_window,
    meta,
    registry=None,
    active_id=None,
):
    """Render all channels or only the visible window for wide datasets."""
    columns, numeric, categorical, datetime = _column_model(filtered_json, meta)
    catalog = json.dumps(columns, ensure_ascii=False, separators=(",", ":"))
    if not columns:
        return [], catalog, ""

    all_types = {str(column): "numeric" for column in numeric}
    all_types.update({str(column): "categorical" for column in categorical})
    all_types.update({str(column): "datetime" for column in datetime})

    record = (registry or {}).get(str(active_id or ""), {})
    derived_columns = {
        str(output)
        for step in (record.get("steps") or [])
        for output in (step.get("outputs") or [])
    }

    virtualized = bool(
        virtualize_columns and len(columns) > VIRTUALIZATION_THRESHOLD
    )
    start, end = _window_bounds(len(columns), virtualized, virtual_window)
    children = []
    if virtualized and start:
        children.append(
            html.Div(
                className="column-virtual-spacer",
                style={"height": f"{start * VIRTUAL_ROW_HEIGHT}px"},
                **{"aria-hidden": "true"},
            )
        )

    children.extend(
        make_column_badge(
            column,
            all_types.get(column, "categorical"),
            derived=column in derived_columns,
            dataset_id=active_id,
        )
        for column in columns[start:end]
    )

    if virtualized and end < len(columns):
        children.append(
            html.Div(
                className="column-virtual-spacer",
                style={"height": f"{(len(columns) - end) * VIRTUAL_ROW_HEIGHT}px"},
                **{"aria-hidden": "true"},
            )
        )

    status = f"Виртуализация активна · {len(columns)} каналов" if virtualized else ""
    return children, catalog, status


app.clientside_callback(
    """
    function(activeDatasetId, enabled) {
        var sidebar = document.getElementById("columns-sidebar");
        var isEnabled = Boolean(enabled);
        if (sidebar) {
            sidebar.scrollTop = 0;
            sidebar.setAttribute(
                "data-virtualization-enabled",
                isEnabled ? "true" : "false"
            );
        }
        return {
            start: 0,
            size: 48,
            datasetId: activeDatasetId || null,
            enabled: isEnabled
        };
    }
    """,
    Output("dataset-virtual-window", "data"),
    Input("active-dataset-id", "data"),
    Input("dataset-virtualize-columns", "checked"),
    prevent_initial_call=False,
)
