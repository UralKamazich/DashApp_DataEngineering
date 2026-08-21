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
from utils import (
    FILTER_CATEGORY_ACTION_ALL,
    FILTER_CATEGORY_ACTION_CLEAR,
    FILTER_CATEGORY_ACTION_INVERT,
    FILTER_CATEGORY_ACTION_VALUES,
    classify_simple,
    create_value_control,
    read_df_from_store,
)


TYPE_MARKS = {
    "numeric": "123",
    "categorical": "Aa",
    "datetime": "dt",
    "unknown": "…",
}

OPERATOR_OPTIONS = {
    "numeric": [
        ("Диапазон", "between"),
        ("> Больше", "gt"),
        ("≥ Не меньше", "gte"),
        ("< Меньше", "lt"),
        ("≤ Не больше", "lte"),
        ("= Равно", "eq"),
        ("≠ Не равно", "ne"),
        ("∅ Пусто", "is_empty"),
        ("≠∅ Не пусто", "not_empty"),
    ],
    "datetime": [
        ("Диапазон", "between"),
        ("После", "after"),
        ("До", "before"),
        ("∅ Пусто", "is_empty"),
        ("≠∅ Не пусто", "not_empty"),
    ],
    "categorical": [
        ("В списке", "in"),
        ("Не в списке", "not_in"),
        ("Содержит", "contains"),
        ("Не содержит", "not_contains"),
        ("Начинается с", "starts_with"),
        ("Заканчивается на", "ends_with"),
        ("∅ Пусто", "is_empty"),
        ("≠∅ Не пусто", "not_empty"),
    ],
}
OPERATOR_OPTIONS["unknown"] = OPERATOR_OPTIONS["categorical"]

DEFAULT_OPERATORS = {
    "numeric": "between",
    "datetime": "between",
    "categorical": "in",
    "unknown": "in",
}
VALUELESS_OPERATORS = {"is_empty", "not_empty"}


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


def _operator_data(kind):
    return [
        {"label": label, "value": value}
        for label, value in OPERATOR_OPTIONS.get(kind, OPERATOR_OPTIONS["unknown"])
    ]


def _normalise_operator(kind, operator):
    allowed = {item["value"] for item in _operator_data(kind)}
    return operator if operator in allowed else DEFAULT_OPERATORS.get(kind, "in")


def _filter_domain(frame, column):
    kind = _column_type(frame, column)
    if kind == "numeric":
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.empty:
            return [float(values.min()), float(values.max())]
    if kind == "datetime":
        values = pd.to_datetime(frame[column], errors="coerce").dropna()
        if not values.empty:
            return [values.min().date().isoformat(), values.max().date().isoformat()]
    return None


def _default_filter_value(frame, column, operator=None):
    kind = _column_type(frame, column)
    operator = _normalise_operator(kind, operator)
    if operator in VALUELESS_OPERATORS:
        return None
    if operator == "between":
        domain = _filter_domain(frame, column)
        if domain and domain[0] != domain[1]:
            return domain
    return []


def _clean_filter_state(filters_state):
    clean = {}
    for filter_id, config in (filters_state or {}).items():
        if (config or {}).get("enabled", True) is False:
            continue
        column = (config or {}).get("column")
        operator = (config or {}).get("operator")
        value = (config or {}).get("value")
        if not column:
            continue
        if operator not in VALUELESS_OPERATORS and value in (None, "", []):
            continue
        if isinstance(value, (list, tuple)) and any(item is None for item in value):
            continue
        if operator == "between" and (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(item is None for item in value)
        ):
            continue
        domain = (config or {}).get("domain")
        if operator == "between" and domain and list(value) == list(domain):
            continue
        cleaned = {"column": str(column), "value": value}
        if operator:
            cleaned["operator"] = operator
        clean[str(filter_id)] = cleaned
    return clean


def _filter_card(filter_id, column, filters_state, frame):
    filter_id = int(filter_id)
    column = str(column) if column else None
    config = (filters_state or {}).get(str(filter_id), {})
    enabled = bool(config.get("enabled", True))
    current_value = config.get("value") if config.get("column") == column else None
    kind = _column_type(frame, column)
    operator = _normalise_operator(kind, config.get("operator"))
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
                        comboboxProps={"withinPortal": True, "zIndex": 10020},
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
            dmc.Select(
                id={"type": "filter-operator", "index": filter_id},
                data=_operator_data(kind),
                value=operator,
                allowDeselect=False,
                size="xs",
                comboboxProps={"withinPortal": True, "zIndex": 10020},
                className="filter-card-operator",
            ),
            html.Div(
                create_value_control(filter_id, column, current_value, frame, operator),
                id={"type": "filter-control", "index": filter_id},
                className="filter-card-control",
            ),
            html.Button(
                type="button",
                id={"type": "filter-enabled", "index": filter_id},
                className=(
                    "filter-enabled-toggle is-enabled"
                    if enabled
                    else "filter-enabled-toggle is-disabled"
                ),
                title=(
                    "Фильтр включён — нажмите, чтобы выключить"
                    if enabled
                    else "Фильтр выключен — нажмите, чтобы включить"
                ),
                **{
                    "aria-label": "Переключить состояние фильтра",
                    "aria-pressed": str(enabled).lower(),
                },
            ),
        ],
        id=f"filter-card-{filter_id}",
        className=f"filter-card filter-card--{kind}",
        **{
            "data-filter-column": column or "",
            "data-filter-operator": operator,
        },
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


@app.callback(
    Output("filters-container", "children"),
    Output("filter-count", "data"),
    Output("filters-state", "data", allow_duplicate=True),
    Output("filters-applied-state", "data"),
    Output("filter-logic-mode", "value"),
    Output("filter-applied-logic", "data"),
    Input("stored-data", "data"),
    Input("add-filter-btn", "n_clicks"),
    Input({"type": "remove-filter", "index": ALL}, "n_clicks"),
    Input("reset-filters-btn", "n_clicks"),
    Input("revert-filters-btn", "n_clicks"),
    Input("apply-filters-btn", "n_clicks"),
    Input("filter-drop-store", "data"),
    State("filter-count", "data"),
    State("filters-container", "children"),
    State("filters-state", "data"),
    State("meta-columns", "data"),
    State("filters-applied-state", "data"),
    State("filter-logic-mode", "value"),
    State("filter-applied-logic", "data"),
    prevent_initial_call=True,
)
def manage_filters(
    stored_json,
    _add_clicks,
    _remove_clicks,
    _reset_clicks,
    _revert_clicks,
    _apply_clicks,
    dropped,
    filter_count,
    current_filters,
    filters_state,
    meta,
    applied_state,
    logic_mode,
    applied_logic,
):
    trigger = ctx.triggered_id
    current = list(current_filters or [])
    draft = dict(filters_state or {})
    counter = int(filter_count or 0)

    if trigger == "stored-data":
        return [], 0, {}, {}, "and", "and"

    if trigger == "apply-filters-btn":
        return (
            no_update,
            no_update,
            no_update,
            draft,
            no_update,
            logic_mode or "and",
        )

    if trigger == "reset-filters-btn":
        return [], 0, {}, {}, "and", "and"

    frame = _source_frame(stored_json, meta)

    if trigger == "revert-filters-btn":
        restored = {}
        cards = []
        for filter_id, config in (applied_state or {}).items():
            column = str((config or {}).get("column") or "")
            if not column or column not in frame.columns:
                continue
            kind = _column_type(frame, column)
            operator = _normalise_operator(kind, (config or {}).get("operator"))
            restored_config = {
                "column": column,
                "operator": operator,
                "value": (config or {}).get("value"),
                "enabled": bool((config or {}).get("enabled", True)),
            }
            domain = _filter_domain(frame, column)
            if domain is not None:
                restored_config["domain"] = domain
            restored[str(filter_id)] = restored_config
            cards.append(_filter_card(filter_id, column, restored, frame))
        counter = max((int(filter_id) for filter_id in restored), default=0)
        return cards, counter, restored, no_update, applied_logic or "and", no_update

    if isinstance(trigger, dict) and trigger.get("type") == "remove-filter":
        filter_id = str(trigger.get("index"))
        current = [
            component for component in current
            if component.get("props", {}).get("id") != f"filter-card-{filter_id}"
        ]
        draft.pop(filter_id, None)
        return current, counter, draft, no_update, no_update, no_update

    column = None
    if trigger == "filter-drop-store":
        column = str((dropped or {}).get("column") or "")
        if not column or column not in frame.columns:
            raise PreventUpdate
    elif trigger != "add-filter-btn":
        raise PreventUpdate

    new_id = counter + 1
    if column:
        kind = _column_type(frame, column)
        operator = DEFAULT_OPERATORS.get(kind, "in")
        draft[str(new_id)] = {
            "column": column,
            "operator": operator,
            "value": _default_filter_value(frame, column, operator),
            "domain": _filter_domain(frame, column),
            "enabled": True,
        }
    current.append(_filter_card(new_id, column, draft, frame))
    return current, new_id, draft, no_update, no_update, no_update


@app.callback(
    Output({"type": "filter-operator", "index": MATCH}, "data"),
    Output({"type": "filter-operator", "index": MATCH}, "value"),
    Output({"type": "filter-type-marker", "index": MATCH}, "children"),
    Output({"type": "filter-type-marker", "index": MATCH}, "className"),
    Input({"type": "filter-column", "index": MATCH}, "value"),
    State({"type": "filter-column", "index": MATCH}, "id"),
    State("filters-state", "data"),
    State("stored-data", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=True,
)
def update_filter_definition(column, column_id, filters_state, stored_json, meta):
    frame = _source_frame(stored_json, meta)
    filter_id = str(column_id["index"])
    config = (filters_state or {}).get(filter_id, {})
    kind = _column_type(frame, column)
    operator = _normalise_operator(
        kind,
        config.get("operator") if config.get("column") == column else None,
    )
    return (
        _operator_data(kind),
        operator,
        TYPE_MARKS[kind],
        f"filter-type-marker filter-type-marker--{kind}",
    )


@app.callback(
    Output({"type": "filter-control", "index": MATCH}, "children"),
    Input({"type": "filter-column", "index": MATCH}, "value"),
    Input({"type": "filter-operator", "index": MATCH}, "value"),
    State({"type": "filter-column", "index": MATCH}, "id"),
    State("filters-state", "data"),
    State("stored-data", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=True,
)
def update_filter_control(column, operator, column_id, filters_state, stored_json, meta):
    frame = _source_frame(stored_json, meta)
    filter_id = str(column_id["index"])
    config = (filters_state or {}).get(filter_id, {})
    same_definition = (
        config.get("column") == column
        and config.get("operator") == operator
    )
    current_value = config.get("value") if same_definition else None
    return create_value_control(filter_id, column, current_value, frame, operator)


@app.callback(
    Output("filters-state", "data", allow_duplicate=True),
    Input({"type": "filter-column", "index": ALL}, "value"),
    Input({"type": "filter-operator", "index": ALL}, "value"),
    Input({"type": "filter-value", "index": ALL}, "value"),
    Input({"type": "filter-category-value", "index": ALL}, "value"),
    Input({"type": "filter-range-value", "index": ALL}, "value"),
    Input({"type": "filter-enabled", "index": ALL}, "n_clicks"),
    State({"type": "filter-column", "index": ALL}, "id"),
    State({"type": "filter-operator", "index": ALL}, "id"),
    State({"type": "filter-value", "index": ALL}, "id"),
    State({"type": "filter-category-value", "index": ALL}, "id"),
    State({"type": "filter-range-value", "index": ALL}, "id"),
    State({"type": "filter-enabled", "index": ALL}, "id"),
    State("filters-state", "data"),
    State("stored-data", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=True,
)
def update_filters_state(
    columns,
    operators,
    values,
    category_values,
    range_values,
    enabled_clicks,
    column_ids,
    operator_ids,
    value_ids,
    category_value_ids,
    range_value_ids,
    enabled_ids,
    filters_state,
    stored_json,
    meta,
):
    updated = dict(filters_state or {})
    frame = _source_frame(stored_json, meta)
    operators_by_id = {
        str(component_id.get("index")): operators[index]
        for index, component_id in enumerate(operator_ids or [])
        if index < len(operators or [])
    }
    values_by_id = {
        str(component_id.get("index")): values[index]
        for index, component_id in enumerate(value_ids or [])
        if index < len(values or [])
    }
    category_values_by_id = {
        str(component_id.get("index")): [
            value
            for value in (category_values[index] or [])
            if value not in FILTER_CATEGORY_ACTION_VALUES
        ]
        for index, component_id in enumerate(category_value_ids or [])
        if index < len(category_values or [])
    }
    ranges_by_id = {
        str(component_id.get("index")): range_values[index]
        for index, component_id in enumerate(range_value_ids or [])
        if index < len(range_values or [])
    }
    enabled_clicks_by_id = {
        str(component_id.get("index")): enabled_clicks[index]
        for index, component_id in enumerate(enabled_ids or [])
        if index < len(enabled_clicks or [])
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
        kind = _column_type(frame, column)
        operator = _normalise_operator(kind, operators_by_id.get(filter_id))
        column_changed = (
            triggered_type == "filter-column"
            and triggered_index == filter_id
            and previous.get("column") != column
        )
        operator_changed = (
            triggered_type == "filter-operator"
            and triggered_index == filter_id
            and previous.get("operator") != operator
        )
        if column_changed or operator_changed:
            value = _default_filter_value(frame, column, operator)
        elif operator in VALUELESS_OPERATORS:
            value = None
        elif kind == "numeric" and operator == "between":
            value = ranges_by_id.get(filter_id, previous.get("value"))
        elif kind == "categorical" and operator in {"in", "not_in"}:
            value = category_values_by_id.get(filter_id, previous.get("value"))
        else:
            value = values_by_id.get(filter_id, previous.get("value"))

        enabled = bool(previous.get("enabled", True))
        if (
            triggered_type == "filter-enabled"
            and triggered_index == filter_id
            and enabled_clicks_by_id.get(filter_id)
        ):
            enabled = not enabled

        config = {
            "column": str(column),
            "operator": operator,
            "value": value,
            "enabled": enabled,
        }
        domain = _filter_domain(frame, column)
        if domain is not None:
            config["domain"] = domain
        updated[filter_id] = config

    return {
        filter_id: config
        for filter_id, config in updated.items()
        if filter_id in live_ids
    }


@app.callback(
    Output({"type": "filter-enabled", "index": MATCH}, "className"),
    Output({"type": "filter-enabled", "index": MATCH}, "title"),
    Output({"type": "filter-enabled", "index": MATCH}, "aria-pressed"),
    Input({"type": "filter-enabled", "index": MATCH}, "n_clicks"),
    State({"type": "filter-enabled", "index": MATCH}, "className"),
    prevent_initial_call=True,
)
def update_filter_enabled_indicator(n_clicks, current_class):
    if not n_clicks:
        raise PreventUpdate
    enabling = "is-disabled" in (current_class or "")
    if enabling:
        return (
            "filter-enabled-toggle is-enabled",
            "Фильтр включён — нажмите, чтобы выключить",
            "true",
        )
    return (
        "filter-enabled-toggle is-disabled",
        "Фильтр выключен — нажмите, чтобы включить",
        "false",
    )


@app.callback(
    Output({"type": "filter-category-value", "index": MATCH}, "value"),
    Input({"type": "filter-category-value", "index": MATCH}, "value"),
    State({"type": "filter-category-value", "index": MATCH}, "data"),
    prevent_initial_call=True,
)
def update_category_selection(selected, options):
    selected = list(selected or [])
    actions = [value for value in selected if value in FILTER_CATEGORY_ACTION_VALUES]
    if not actions:
        raise PreventUpdate

    option_values = []
    for group in options or []:
        items = group.get("items", []) if isinstance(group, dict) else []
        for item in items:
            value = item.get("value") if isinstance(item, dict) else item
            if value not in FILTER_CATEGORY_ACTION_VALUES:
                option_values.append(value)

    action = actions[-1]
    if action == FILTER_CATEGORY_ACTION_ALL:
        return option_values
    if action == FILTER_CATEGORY_ACTION_CLEAR:
        return []
    if action == FILTER_CATEGORY_ACTION_INVERT:
        selected_set = set(selected) - FILTER_CATEGORY_ACTION_VALUES
        return [value for value in option_values if value not in selected_set]
    raise PreventUpdate


@app.callback(
    Output({"type": "filter-number-min", "index": MATCH}, "value"),
    Output({"type": "filter-number-max", "index": MATCH}, "value"),
    Output({"type": "filter-range-value", "index": MATCH}, "value"),
    Input({"type": "filter-number-min", "index": MATCH}, "value"),
    Input({"type": "filter-number-max", "index": MATCH}, "value"),
    Input({"type": "filter-range-value", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def sync_numeric_range(lower, upper, slider_value):
    """Keep exact numeric inputs and the compact range slider in sync."""
    trigger = ctx.triggered_id if isinstance(ctx.triggered_id, dict) else {}
    if trigger.get("type") == "filter-range-value":
        if not isinstance(slider_value, (list, tuple)) or len(slider_value) != 2:
            raise PreventUpdate
        return slider_value[0], slider_value[1], no_update
    if lower is None or upper is None:
        raise PreventUpdate
    lo, hi = sorted((float(lower), float(upper)))
    return no_update, no_update, [lo, hi]


@app.callback(
    Output("filter-draft-status", "children"),
    Output("filter-draft-status", "className"),
    Output("revert-filters-btn", "disabled"),
    Output("apply-filters-btn", "children"),
    Output("apply-filters-btn", "disabled"),
    Input("filters-state", "data"),
    Input("filters-applied-state", "data"),
    Input("filter-logic-mode", "value"),
    Input("filter-applied-logic", "data"),
)
def update_filter_draft_status(draft, applied, draft_logic, applied_logic):
    clean_draft = _clean_filter_state(draft)
    draft_key = json.dumps(draft or {}, sort_keys=True, ensure_ascii=False, default=str)
    applied_key = json.dumps(applied or {}, sort_keys=True, ensure_ascii=False, default=str)
    dirty = draft_key != applied_key or (draft_logic or "and") != (applied_logic or "and")
    count = len(clean_draft)
    label = f"Применить · {count}" if count else "Применить"
    if dirty:
        return (
            "Есть неприменённые изменения",
            "filter-draft-status is-dirty",
            False,
            label,
            False,
        )
    return "Все изменения применены", "filter-draft-status", True, label, True


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
        percent = filtered_rows / source_rows * 100 if source_rows else 0
        rows = f"{filtered_rows:,} из {source_rows:,}".replace(",", " ")
        summary = f"{rows} · {percent:.1f}%".replace(".", ",")
    else:
        summary = f"{source_rows:,} строк".replace(",", " ")
    return summary, str(count), tab_class
