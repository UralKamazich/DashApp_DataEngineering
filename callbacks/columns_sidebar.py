# -*- coding: utf-8 -*-
"""Render the active dataset channel rail.

The rail used to be built from Dash components: five React components per
column.  With 1000+ columns that put ~5000 extra components into the Dash
tree, and Dash 4 re-evaluates every component on every store update — a wide
workbook froze the window for more than a minute after loading.

Now the server sends a compact column model and a clientside callback draws
plain DOM rows (same classes and data-* attributes as ``make_column_badge``,
so drag-and-drop keeps working).  The full list is always present.
"""

from __future__ import annotations

import json

from dash import Input, Output, State

from dash_app import app
from utils import read_df_from_store, classify_simple
from components import COLUMN_TYPE_LABELS


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


def build_column_rail_model(filtered_json, meta, registry=None, active_id=None):
    """Compact, JSON-friendly description of the channel rail."""
    columns, numeric, categorical, datetime = _column_model(filtered_json, meta)
    all_types = {str(column): "numeric" for column in numeric}
    all_types.update({str(column): "categorical" for column in categorical})
    all_types.update({str(column): "datetime" for column in datetime})

    record = (registry or {}).get(str(active_id or ""), {})
    derived_columns = {
        str(output)
        for step in (record.get("steps") or [])
        for output in (step.get("outputs") or [])
    }
    return {
        "columns": columns,
        "types": [all_types.get(column, "categorical") for column in columns],
        "derived": [index for index, column in enumerate(columns) if column in derived_columns],
        "datasetId": str(active_id or ""),
    }


@app.callback(
    Output("dataset-column-model", "data"),
    Input("filtered-data", "data"),
    State("meta-columns", "data"),
    State("dataset-registry", "data"),
    State("active-dataset-id", "data"),
    prevent_initial_call=False,
)
def update_column_badges(filtered_json, meta, registry=None, active_id=None):
    return build_column_rail_model(filtered_json, meta, registry, active_id)


_TYPE_LABELS_JS = {
    key: {"mark": mark, "name": name} for key, (mark, name) in COLUMN_TYPE_LABELS.items()
}

app.clientside_callback(
    """
    function(model) {
        var container = document.getElementById("columns-badges");
        var catalog = document.getElementById("dataset-column-catalog");
        model = model || {};
        var columns = model.columns || [];
        var types = model.types || [];
        var derived = new Set(model.derived || []);
        var datasetId = model.datasetId || "";
        var labels = __TYPE_LABELS__;
        if (catalog) catalog.textContent = JSON.stringify(columns);
        if (!container) return window.dash_clientside.no_update;

        var fragment = document.createDocumentFragment();
        for (var i = 0; i < columns.length; i++) {
            var name = String(columns[i]);
            var type = types[i] || "categorical";
            var label = labels[type] || {mark: "\\u2026", name: "Другой тип"};
            var isDerived = derived.has(i);

            var badge = document.createElement("div");
            badge.className = "column-badge column-badge--" + type;
            badge.draggable = true;
            badge.title = name + " · " + label.name + (isDerived ? " · производный канал" : "");
            badge.setAttribute("aria-label", "Перетащить канал «" + name + "»");
            badge.setAttribute("data-column-name", name);
            badge.setAttribute("data-column-type", type);
            badge.setAttribute("data-dataset-id", datasetId);
            badge.setAttribute("data-derived", isDerived ? "true" : "false");

            var row = document.createElement("div");
            row.className = "column-channel-row";
            var marker = document.createElement("sup");
            marker.className = "column-type-marker column-type-marker--" + type;
            marker.setAttribute("aria-hidden", "true");
            marker.textContent = label.mark;
            row.appendChild(marker);
            if (isDerived) {
                var fx = document.createElement("sup");
                fx.className = "column-derived-marker";
                fx.title = "Создан в Data Engineering";
                fx.textContent = "fx";
                row.appendChild(fx);
            }
            var text = document.createElement("span");
            text.className = "column-channel-name";
            text.textContent = name;
            row.appendChild(text);
            var handle = document.createElement("span");
            handle.className = "column-drag-handle";
            handle.setAttribute("aria-hidden", "true");
            row.appendChild(handle);

            badge.appendChild(row);
            fragment.appendChild(badge);
        }
        container.replaceChildren(fragment);
        return columns.length;
    }
    """.replace("__TYPE_LABELS__", json.dumps(_TYPE_LABELS_JS, ensure_ascii=False)),
    Output("dataset-column-rendered", "data"),
    Input("dataset-column-model", "data"),
)
