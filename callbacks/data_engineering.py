# -*- coding: utf-8 -*-
"""Draft and atomically execute dataset-aware feature pipelines."""

from __future__ import annotations

from dash import ALL, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from dash_app import app
from dataset_registry import commit_result, get_record, input_payload, suggest_dataset_name
from engineering_ops import execute_pipeline
from utils import meta_from_df, read_df_from_store


def _empty_draft():
    return {"input_id": None, "scope": "base", "steps": []}


def _notification(message, *, color="green", notification_id="de-operation"):
    return [{
        "id": notification_id,
        "title": "Data Engineering",
        "message": message,
        "color": color,
        "action": "show",
        "autoClose": 5500,
    }]


def _queued_step(
    trigger,
    *,
    scope,
    bin_column,
    bin_method,
    bin_groups,
    bin_labels,
    text_columns,
    text_suffix,
    text_strip,
    aggregate_keys,
    aggregate_columns,
    aggregate_metrics,
    exclude_zeros,
    exclude_empty,
):
    if trigger == "btn-grouping":
        if not bin_column:
            raise ValueError("Выберите числовой канал для биннинга.")
        if int(bin_groups or 0) < 2:
            raise ValueError("Количество групп должно быть не меньше двух.")
        return {
            "operation": "binning",
            "label": f"Биннинг · {bin_column}",
            "summary": f"{int(bin_groups)} групп · {bin_method or 'count'}",
            "scope": scope,
            "params": {
                "column": bin_column,
                "method": bin_method or "count",
                "groups": int(bin_groups),
                "label_style": bin_labels or "interval",
            },
        }

    if trigger == "btn-txtcopy":
        columns = list(dict.fromkeys(text_columns or []))
        if not columns:
            raise ValueError("Выберите хотя бы один канал.")
        return {
            "operation": "text_copy",
            "label": f"Текстовая копия · {len(columns)}",
            "summary": ", ".join(columns[:4]) + ("…" if len(columns) > 4 else ""),
            "scope": scope,
            "params": {
                "columns": columns,
                "suffix": text_suffix,
                "strip": bool(text_strip),
            },
        }

    keys = list(dict.fromkeys(aggregate_keys or []))
    columns = list(dict.fromkeys(aggregate_columns or []))
    metrics = list(dict.fromkeys(aggregate_metrics or []))
    if not keys or not columns or not metrics:
        raise ValueError("Выберите ключи, каналы и метрики.")
    return {
        "operation": "group_aggregates",
        "label": f"Агрегаты · {len(metrics)}",
        "summary": f"По: {', '.join(keys[:2])} · {', '.join(metrics[:3])}",
        "scope": scope,
        "params": {
            "keys": keys,
            "columns": columns,
            "metrics": metrics,
            "exclude_zeros": bool(exclude_zeros),
            "exclude_empty": bool(exclude_empty),
        },
    }


@app.callback(
    Output("de-draft-pipeline", "data"),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Output("de-txt-status", "children"),
    Output("de-agg-status", "children", allow_duplicate=True),
    Input("btn-grouping", "n_clicks"),
    Input("btn-txtcopy", "n_clicks"),
    Input("btn-agg", "n_clicks"),
    Input("de-clear-pipeline", "n_clicks"),
    Input("stored-data", "data"),
    Input({"type": "de-remove-step", "index": ALL}, "n_clicks"),
    State({"type": "de-remove-step", "index": ALL}, "id"),
    State("bin-column", "value"),
    State("bin-method", "value"),
    State("bin-k", "value"),
    State("bin-label-style", "value"),
    State("txtcopy-cols", "value"),
    State("txtcopy-suffix", "value"),
    State("txtcopy-strip", "checked"),
    State("agg-keys", "value"),
    State("agg-cols", "value"),
    State("agg-metrics", "value"),
    State("agg-exclude-zeros", "checked"),
    State("agg-exclude-empty", "checked"),
    State("de-draft-pipeline", "data"),
    State("dataset-registry", "data"),
    State("active-dataset-id", "data"),
    State("de-input-dataset", "value"),
    State("de-input-scope", "value"),
    prevent_initial_call=True,
)
def stage_feature_step(
    _bin_clicks,
    _text_clicks,
    _aggregate_clicks,
    _clear_clicks,
    _stored_json,
    remove_clicks,
    remove_ids,
    bin_column,
    bin_method,
    bin_groups,
    bin_labels,
    text_columns,
    text_suffix,
    text_strip,
    aggregate_keys,
    aggregate_columns,
    aggregate_metrics,
    exclude_zeros,
    exclude_empty,
    draft,
    registry,
    active_id,
    input_id,
    input_scope,
):
    trigger = ctx.triggered_id
    current = dict(draft or _empty_draft())
    current["steps"] = list(current.get("steps") or [])

    if trigger == "de-clear-pipeline" or trigger == "stored-data":
        return _empty_draft(), no_update, "", ""

    if isinstance(trigger, dict) and trigger.get("type") == "de-remove-step":
        clicked = {
            int(component_id.get("index"))
            for count, component_id in zip(remove_clicks or [], remove_ids or [])
            if count
        }
        index = int(trigger.get("index"))
        if index not in clicked or index >= len(current["steps"]):
            raise PreventUpdate
        current["steps"].pop(index)
        if not current["steps"]:
            current = _empty_draft()
        return current, no_update, no_update, no_update

    if trigger not in {"btn-grouping", "btn-txtcopy", "btn-agg"}:
        raise PreventUpdate

    input_id = str(input_id or active_id or "")
    scope = input_scope or "base"
    if not get_record(registry, input_id):
        return (
            no_update,
            _notification("Сначала загрузите данные.", color="red", notification_id="de-no-data"),
            "Нет данных." if trigger == "btn-txtcopy" else no_update,
            "Нет данных." if trigger == "btn-agg" else no_update,
        )
    if current["steps"] and (
        str(current.get("input_id")) != input_id or current.get("scope") != scope
    ):
        message = "Очередь уже привязана к другому входу. Очистите её перед сменой источника."
        return (
            no_update,
            _notification(message, color="orange", notification_id="de-queue-context"),
            message if trigger == "btn-txtcopy" else no_update,
            message if trigger == "btn-agg" else no_update,
        )

    try:
        queued = _queued_step(
            trigger,
            scope=scope,
            bin_column=bin_column,
            bin_method=bin_method,
            bin_groups=bin_groups,
            bin_labels=bin_labels,
            text_columns=text_columns,
            text_suffix=text_suffix,
            text_strip=text_strip,
            aggregate_keys=aggregate_keys,
            aggregate_columns=aggregate_columns,
            aggregate_metrics=aggregate_metrics,
            exclude_zeros=exclude_zeros,
            exclude_empty=exclude_empty,
        )
    except (TypeError, ValueError) as error:
        message = str(error)
        return (
            no_update,
            _notification(message, color="orange", notification_id=f"de-{trigger}-invalid"),
            message if trigger == "btn-txtcopy" else no_update,
            message if trigger == "btn-agg" else no_update,
        )

    current["input_id"] = input_id
    current["scope"] = scope
    current["steps"].append(queued)
    status = f"В очереди: {len(current['steps'])}"
    return (
        current,
        no_update,
        status if trigger == "btn-txtcopy" else no_update,
        status if trigger == "btn-agg" else no_update,
    )


@app.callback(
    Output("de-pipeline-list", "children"),
    Output("de-run-pipeline", "children"),
    Output("de-run-pipeline", "disabled"),
    Output("de-clear-pipeline", "disabled"),
    Output("de-queue-context", "children"),
    Output("de-input-dataset", "disabled"),
    Output("de-input-scope", "disabled"),
    Input("de-draft-pipeline", "data"),
    Input("dataset-registry", "data"),
)
def render_draft_pipeline(draft, registry):
    draft = draft or _empty_draft()
    steps = list(draft.get("steps") or [])
    if not steps:
        return (
            html.Div("Добавьте один или несколько шагов", className="de-pipeline-empty"),
            "Выполнить",
            True,
            True,
            "Очередь выполняется одним проходом без промежуточных dataset’ов.",
            False,
            False,
        )

    cards = [
        html.Div(
            [
                html.Div(str(index + 1), className="de-step-index"),
                html.Div(
                    [
                        html.Div(step.get("label") or "Шаг", className="de-step-title"),
                        html.Div(
                            step.get("summary") or "",
                            className="de-step-output",
                            title=step.get("summary") or "",
                        ),
                    ]
                ),
                html.Button(
                    "×",
                    id={"type": "de-remove-step", "index": index},
                    className="de-step-remove",
                    title="Удалить шаг",
                    type="button",
                ),
            ],
            className="de-step-card de-step-card--queued",
        )
        for index, step in enumerate(steps)
    ]
    record = get_record(registry, draft.get("input_id")) or {}
    name = record.get("name") or draft.get("input_id") or "Dataset"
    scope = "после фильтров" if draft.get("scope") == "filtered" else "до фильтров"
    return cards, f"Выполнить · {len(steps)}", False, False, f"{name} · {scope}", True, True


@app.callback(
    Output("de-output-name", "value"),
    Output("de-auto-output-name", "data"),
    Output("de-output-name", "disabled"),
    Input("de-draft-pipeline", "data"),
    Input("de-output-mode", "value"),
    Input("dataset-registry", "data"),
    State("de-output-name", "value"),
    State("de-auto-output-name", "data"),
)
def suggest_output_name(draft, output_mode, registry, current_name, previous_auto_name):
    steps = list((draft or {}).get("steps") or [])
    disabled = output_mode != "new"
    if not steps:
        return "", None, disabled

    candidate = suggest_dataset_name(
        registry,
        steps,
        (draft or {}).get("scope") or "base",
    )
    if not str(current_name or "").strip() or current_name == previous_auto_name:
        return candidate, candidate, disabled
    # A manually edited name remains authoritative while this draft is open.
    return no_update, candidate, disabled


@app.callback(
    Output("dataset-registry", "data", allow_duplicate=True),
    Output("active-dataset-id", "data", allow_duplicate=True),
    Output("active-dataset-data", "data", allow_duplicate=True),
    Output("meta-columns", "data", allow_duplicate=True),
    Output("filters-state", "data", allow_duplicate=True),
    Output("filters-applied-state", "data", allow_duplicate=True),
    Output("filter-logic-mode", "value", allow_duplicate=True),
    Output("filter-applied-logic", "data", allow_duplicate=True),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Output("de-draft-pipeline", "data", allow_duplicate=True),
    Input("de-run-pipeline", "n_clicks"),
    State("de-draft-pipeline", "data"),
    State("dataset-registry", "data"),
    State("active-dataset-id", "data"),
    State("filtered-data", "data"),
    State("de-output-mode", "value"),
    State("de-output-name", "value"),
    prevent_initial_call=True,
)
def run_feature_pipeline(
    _run_clicks,
    draft,
    registry,
    active_id,
    active_filtered,
    output_mode,
    output_name,
):
    draft = draft or _empty_draft()
    queued_steps = list(draft.get("steps") or [])
    if not queued_steps:
        raise PreventUpdate

    input_id = str(draft.get("input_id") or active_id or "")
    input_scope = draft.get("scope") or "base"
    payload, input_meta = input_payload(
        registry,
        input_id,
        input_scope,
        active_id=active_id,
        active_filtered=active_filtered,
    )
    if not payload:
        message = "Исходный dataset больше недоступен."
        return (
            no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update,
            _notification(message, color="red", notification_id="de-pipeline-no-data"),
            no_update,
        )

    try:
        source = read_df_from_store(payload, input_meta)
        result_frame, outputs, committed_steps = execute_pipeline(source, queued_steps)
    except Exception as error:
        message = str(error)
        return (
            no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update,
            _notification(message, color="red", notification_id="de-pipeline-error"),
            no_update,
        )

    result_meta = meta_from_df(result_frame)
    result_payload = result_frame.to_json(date_format="iso", orient="split")
    resolved_output_name = output_name
    if (output_mode or "current") == "new" and not str(output_name or "").strip():
        resolved_output_name = suggest_dataset_name(registry, queued_steps, input_scope)
    updated_registry, result_id = commit_result(
        registry,
        input_id,
        result_payload,
        result_meta,
        committed_steps,
        output_mode=output_mode or "current",
        output_name=resolved_output_name,
    )
    result = get_record(updated_registry, result_id) or {}
    message = f"Выполнено шагов: {len(committed_steps)} · каналов добавлено: {len(outputs)}"

    if (output_mode or "current") == "new":
        return (
            updated_registry,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            _notification(message, notification_id="de-pipeline-ok"),
            _empty_draft(),
        )

    return (
        updated_registry,
        result_id,
        result_payload,
        result_meta,
        result.get("filters_state") or {},
        result.get("filters_applied_state") or {},
        result.get("filter_logic") or "and",
        result.get("filter_applied_logic") or "and",
        _notification(message, notification_id="de-pipeline-ok"),
        _empty_draft(),
    )
