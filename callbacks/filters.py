# -*- coding: utf-8 -*-
"""Callbacks for the global overlay filter panel."""

import json

import pandas as pd
from dash import ALL, MATCH, Input, Output, State, clientside_callback, ctx, html, no_update
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify
import dash_mantine_components as dmc

from dash_app import app
from filter_panel import FILTERS_PANEL
from layout import GRAPH_SETTINGS_PANEL
from utils import classify_simple, create_value_control, read_df_from_store


TYPE_MARKS = {
    "numeric": "123",
    "categorical": "Aa",
    "datetime": "dt",
    "unknown": "…",
}


def _source_frame(stored_json, meta):
    if not stored_json:
        return pd.DataFrame()
    try:
        frame = read_df_from_store(stored_json, meta)
    except Exception:
        return pd.DataFrame()
    frame.columns = [str(column) for column in frame.columns]
    return frame


def _column_type(frame, column):
    if frame is None or frame.empty or not column or column not in frame.columns:
        return "unknown"
    numeric, categorical, datetimes = classify_simple(frame)
    if column in numeric:
        return "numeric"
    if column in datetimes:
        return "datetime"
    if column in categorical:
        return "categorical"
    return "unknown"


def _default_filter_value(frame, column):
    kind = _column_type(frame, column)
    if kind == "numeric":
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.empty and float(values.min()) != float(values.max()):
            return [float(values.min()), float(values.max())]
    if kind == "datetime":
        values = pd.to_datetime(frame[column], errors="coerce").dropna()
        if not values.empty:
            return [values.min().date().isoformat(), values.max().date().isoformat()]
    return []


def _clean_filter_state(filters_state):
    clean = {}
    for filter_id, config in (filters_state or {}).items():
        column = (config or {}).get("column")
        value = (config or {}).get("value")
        if not column or value in (None, "", []):
            continue
        if isinstance(value, list) and any(item is None for item in value):
            continue
        clean[str(filter_id)] = {"column": str(column), "value": value}
    return clean


def _filter_card(filter_id, column, filters_state, frame):
    filter_id = int(filter_id)
    column = str(column) if column else None
    config = (filters_state or {}).get(str(filter_id), {})
    current_value = config.get("value") if config.get("column") == column else None
    kind = _column_type(frame, column)
    options = [{"label": str(item), "value": str(item)} for item in frame.columns]

    return html.Div(
        [
            html.Div(
                [
                    html.Sup(
                        TYPE_MARKS[kind],
                        id={"type": "filter-type-marker", "index": filter_id},
                        className=f"filter-type-marker filter-type-marker--{kind}",
                        **{"aria-hidden": "true"},
                    ),
                    dmc.Select(
                        id={"type": "filter-column", "index": filter_id},
                        data=options,
                        value=column,
                        placeholder="Выберите канал",
                        searchable=True,
                        clearable=True,
                        nothingFoundMessage="Канал не найден",
                        size="xs",
                        className="filter-card-column",
                    ),
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:trash", width=14),
                        id={"type": "remove-filter", "index": filter_id},
                        color="gray",
                        variant="subtle",
                        size="sm",
                        **{"aria-label": "Удалить фильтр"},
                    ),
                ],
                className="filter-card-header",
            ),
            html.Div(
                create_value_control(filter_id, column, current_value, frame),
                id={"type": "filter-control", "index": filter_id},
                className="filter-card-control",
            ),
        ],
        id=f"filter-card-{filter_id}",
        className=f"filter-card filter-card--{kind}",
        **{"data-filter-column": column or ""},
    )


def _serialized_row_count(payload):
    if not payload:
        return 0
    try:
        decoded = json.loads(payload)
        return len(decoded.get("index") or [])
    except Exception:
        return 0


@app.callback(
    Output("filters-drawer", "className"),
    Output("filters-drawer-open-state", "data"),
    Input("filters-side-tab", "n_clicks"),
    Input("filter-drop-store", "data"),
    Input("apply-filters-btn", "n_clicks"),
    State("filters-drawer-open-state", "data"),
    State("filter-close-on-apply", "checked"),
    prevent_initial_call=True,
)
def toggle_filters_drawer(_tab_clicks, _dropped, _apply_clicks, opened, close_on_apply):
    trigger = ctx.triggered_id
    if trigger == "filter-drop-store":
        should_open = True
    elif trigger == "apply-filters-btn":
        if not close_on_apply:
            return no_update, no_update
        should_open = False
    else:
        should_open = not bool(opened)
    panel_class = FILTERS_PANEL.open_class if should_open else FILTERS_PANEL.closed_class
    return panel_class, should_open


@app.callback(
    Output("filters-outside-close-store", "data"),
    Input("filter-close-on-outside", "checked"),
)
def sync_drawer_close_on_outside(close_on_outside):
    return bool(close_on_outside)


clientside_callback(
    """
    function (enabled) {
        if (window.__filtersOutsideAbort) {
            window.__filtersOutsideAbort.abort();
            window.__filtersOutsideAbort = null;
        }
        if (!enabled) {
            return window.dash_clientside.no_update;
        }
        var controller = new AbortController();
        window.__filtersOutsideAbort = controller;
        document.addEventListener("mousedown", function (event) {
            var panel = document.getElementById("filters-drawer");
            if (!panel || !panel.classList.contains("open")) return;
            if (panel.contains(event.target)) return;
            window.dash_clientside.set_props("filters-drawer", {className: "__PANEL_CLOSED_CLASS__"});
            window.dash_clientside.set_props("filters-drawer-open-state", {data: false});
        }, {signal: controller.signal});
        return window.dash_clientside.no_update;
    }
    """.replace("__PANEL_CLOSED_CLASS__", FILTERS_PANEL.closed_class),
    Output("filters-drawer-open-state", "data", allow_duplicate=True),
    Input("filters-outside-close-store", "data"),
    prevent_initial_call=True,
)


clientside_callback(
    """
    function (settingsOpen, filtersOpen) {
        if (settingsOpen) {
            window.dash_clientside.set_props(
                "filters-drawer",
                {className: __FILTERS_SETTINGS_OPEN_CLASS__}
            );
            if (filtersOpen) {
                window.dash_clientside.set_props(
                    "filters-drawer-open-state",
                    {data: false}
                );
            }
            return {panel: "settings", timestamp: Date.now()};
        }
        if (filtersOpen) {
            return window.dash_clientside.no_update;
        }
        window.dash_clientside.set_props(
            "filters-drawer",
            {className: __FILTERS_CLOSED_CLASS__}
        );
        return {panel: null, timestamp: Date.now()};
    }
    """
    .replace(
        "__FILTERS_SETTINGS_OPEN_CLASS__",
        json.dumps(FILTERS_PANEL.closed_class + " filters-drawer--settings-open"),
    )
    .replace("__FILTERS_CLOSED_CLASS__", json.dumps(FILTERS_PANEL.closed_class)),
    Output("right-panels-coordination", "data"),
    Input("drawer-simple-open-state", "data"),
    State("filters-drawer-open-state", "data"),
    prevent_initial_call=True,
)


clientside_callback(
    """
    function (filtersOpen, settingsOpen) {
        if (!filtersOpen || !settingsOpen) {
            return window.dash_clientside.no_update;
        }
        window.dash_clientside.set_props(
            "drawer-simple",
            {className: __SETTINGS_CLOSED_CLASS__}
        );
        window.dash_clientside.set_props(
            "drawer-simple-open-state",
            {data: false}
        );
        return {panel: "filters", timestamp: Date.now()};
    }
    """
    .replace(
        "__SETTINGS_CLOSED_CLASS__",
        json.dumps(GRAPH_SETTINGS_PANEL.slide.closed_class),
    ),
    Output("right-panels-coordination", "data", allow_duplicate=True),
    Input("filters-drawer-open-state", "data"),
    State("drawer-simple-open-state", "data"),
    prevent_initial_call=True,
)


@app.callback(
    Output("filters-container", "children"),
    Output("filter-count", "data"),
    Output("filters-state", "data", allow_duplicate=True),
    Output("filters-applied-state", "data"),
    Input("stored-data", "data"),
    Input("add-filter-btn", "n_clicks"),
    Input({"type": "remove-filter", "index": ALL}, "n_clicks"),
    Input("reset-filters-btn", "n_clicks"),
    Input("apply-filters-btn", "n_clicks"),
    Input("filter-drop-store", "data"),
    State("filter-count", "data"),
    State("filters-container", "children"),
    State("filters-state", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=True,
)
def manage_filters(
    stored_json,
    _add_clicks,
    _remove_clicks,
    _reset_clicks,
    _apply_clicks,
    dropped,
    filter_count,
    current_filters,
    filters_state,
    meta,
):
    trigger = ctx.triggered_id
    current = list(current_filters or [])
    draft = dict(filters_state or {})
    counter = int(filter_count or 0)

    if trigger == "stored-data":
        return [], 0, {}, {}

    if trigger == "apply-filters-btn":
        return no_update, no_update, no_update, _clean_filter_state(draft)

    if trigger == "reset-filters-btn":
        return [], 0, {}, {}

    if isinstance(trigger, dict) and trigger.get("type") == "remove-filter":
        filter_id = str(trigger.get("index"))
        current = [
            component for component in current
            if component.get("props", {}).get("id") != f"filter-card-{filter_id}"
        ]
        draft.pop(filter_id, None)
        return current, counter, draft, no_update

    frame = _source_frame(stored_json, meta)

    column = None
    if trigger == "filter-drop-store":
        column = str((dropped or {}).get("column") or "")
        if not column or column not in frame.columns:
            raise PreventUpdate
    elif trigger != "add-filter-btn":
        raise PreventUpdate

    new_id = counter + 1
    if column:
        draft[str(new_id)] = {
            "column": column,
            "value": _default_filter_value(frame, column),
        }
    current.append(_filter_card(new_id, column, draft, frame))
    return current, new_id, draft, no_update


@app.callback(
    Output({"type": "filter-control", "index": MATCH}, "children"),
    Output({"type": "filter-type-marker", "index": MATCH}, "children"),
    Output({"type": "filter-type-marker", "index": MATCH}, "className"),
    Input({"type": "filter-column", "index": MATCH}, "value"),
    State({"type": "filter-column", "index": MATCH}, "id"),
    State("filters-state", "data"),
    State("stored-data", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=True,
)
def update_filter_control(column, column_id, filters_state, stored_json, meta):
    frame = _source_frame(stored_json, meta)
    filter_id = str(column_id["index"])
    config = (filters_state or {}).get(filter_id, {})
    current_value = config.get("value") if config.get("column") == column else None
    kind = _column_type(frame, column)
    return (
        create_value_control(filter_id, column, current_value, frame),
        TYPE_MARKS[kind],
        f"filter-type-marker filter-type-marker--{kind}",
    )


@app.callback(
    Output("filters-state", "data", allow_duplicate=True),
    Input({"type": "filter-column", "index": ALL}, "value"),
    Input({"type": "filter-value", "index": ALL}, "value"),
    State({"type": "filter-column", "index": ALL}, "id"),
    State({"type": "filter-value", "index": ALL}, "id"),
    State("filters-state", "data"),
    prevent_initial_call=True,
)
def update_filters_state(columns, values, column_ids, value_ids, filters_state):
    updated = dict(filters_state or {})
    values_by_id = {
        str(component_id.get("index")): values[index]
        for index, component_id in enumerate(value_ids or [])
        if index < len(values or [])
    }
    triggered = ctx.triggered_id if isinstance(ctx.triggered_id, dict) else {}
    triggered_type = triggered.get("type")
    triggered_index = str(triggered.get("index")) if triggered else None

    live_ids = set()
    for index, component_id in enumerate(column_ids or []):
        filter_id = str(component_id.get("index"))
        live_ids.add(filter_id)
        column = columns[index] if index < len(columns or []) else None
        if not column:
            updated.pop(filter_id, None)
            continue

        previous = updated.get(filter_id, {})
        column_changed = (
            triggered_type == "filter-column"
            and triggered_index == filter_id
            and previous.get("column") != column
        )
        value = None if column_changed else values_by_id.get(filter_id, previous.get("value"))
        updated[filter_id] = {"column": str(column), "value": value}

    return {
        filter_id: config
        for filter_id, config in updated.items()
        if filter_id in live_ids
    }


@app.callback(
    Output("filter-results-summary", "children"),
    Output("filters-side-tab-count", "children"),
    Output("filters-side-tab", "className"),
    Input("filters-applied-state", "data"),
    Input("stored-data", "data"),
    Input("filtered-data", "data"),
)
def update_filter_summary(applied, stored_json, filtered_json):
    count = len(_clean_filter_state(applied))
    source_rows = _serialized_row_count(stored_json)
    filtered_rows = _serialized_row_count(filtered_json)
    tab_class = FILTERS_PANEL.tab_class
    if count:
        tab_class += " has-active-filters"
    if not source_rows:
        summary = "Загрузите данные"
    elif count:
        summary = f"{filtered_rows:,} из {source_rows:,} строк".replace(",", " ")
    else:
        summary = f"{source_rows:,} строк".replace(",", " ")
    return summary, str(count), tab_class
