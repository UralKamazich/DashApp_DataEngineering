# -*- coding: utf-8 -*-
"""Dataset registry, switching, lineage rail and active preview."""

from __future__ import annotations

from dash import ALL, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc

from dash_app import app
from dataset_registry import (
    SOURCE_DATASET_ID,
    create_source_registry,
    dataset_options,
    get_record,
    payload_for_record,
    save_runtime_state,
    summarize_transformation_steps,
)
from utils import meta_from_df, read_df_from_store


@app.callback(
    Output("dataset-registry", "data", allow_duplicate=True),
    Output("active-dataset-id", "data", allow_duplicate=True),
    Output("active-dataset-data", "data", allow_duplicate=True),
    Input("stored-data", "data"),
    State("meta-columns", "data"),
    State("source-file-name", "data"),
    prevent_initial_call=True,
)
def initialize_dataset_registry(stored_json, meta, source_name):
    if not stored_json:
        raise PreventUpdate
    try:
        source_meta = meta_from_df(read_df_from_store(stored_json, meta))
    except Exception:
        source_meta = meta or {}
    registry = create_source_registry(stored_json, source_meta, source_name)
    return registry, SOURCE_DATASET_ID, stored_json


@app.callback(
    Output("dataset-tabs-rail", "children"),
    Output("active-dataset-panel-label", "children"),
    Output("graph-dataset-select", "data"),
    Output("de-input-dataset", "data"),
    Input("dataset-registry", "data"),
    Input("active-dataset-id", "data"),
    State("graph-dataset-select", "data"),
    State("de-input-dataset", "data"),
)
def render_dataset_controls(registry, active_id, graph_options, de_options):
    options = dataset_options(registry)
    valid_ids = {option["value"] for option in options}
    if not options:
        return [], "Нет данных", [], []
    active_id = active_id if active_id in valid_ids else options[0]["value"]
    rail = []
    derived_index = 0
    for dataset_id, record in (registry or {}).items():
        if dataset_id == SOURCE_DATASET_ID:
            continue
        derived_index += 1
        short = f"D{derived_index}"
        is_derived = dataset_id != SOURCE_DATASET_ID
        is_active = dataset_id == active_id
        classes = ["dataset-rail-tab"]
        if is_derived:
            classes.append("is-derived")
        if is_active:
            classes.append("is-active")
        name = str(record.get("name") or dataset_id)
        summaries = summarize_transformation_steps(record.get("steps"))
        visible_summaries = summaries[-5:]
        hidden_count = max(0, len(summaries) - len(visible_summaries))
        tooltip_content = html.Div(
            [
                html.Div(name, className="dataset-rail-tooltip-name"),
                html.Div(
                    [
                        *(
                            [
                                html.Div(
                                    f"{len(summaries) - len(visible_summaries) + position}. {summary}",
                                    className="dataset-rail-tooltip-step",
                                )
                                for position, summary in enumerate(visible_summaries, 1)
                            ]
                            if visible_summaries
                            else [html.Div("Без преобразований", className="dataset-rail-tooltip-step")]
                        ),
                        html.Div(
                            f"Ещё преобразований: {hidden_count}",
                            className="dataset-rail-tooltip-more",
                        ) if hidden_count else None,
                    ],
                    className="dataset-rail-tooltip-steps",
                ),
            ],
            className="dataset-rail-tooltip-content",
        )
        rail.append(
            dmc.Tooltip(
                html.Button(
                    short,
                    id={"type": "dataset-rail-tab", "index": dataset_id},
                    className=" ".join(classes),
                    type="button",
                    **{"aria-label": f"Открыть dataset {short}: {name}"},
                ),
                label=tooltip_content,
                position="right",
                openDelay=250,
                withArrow=True,
                multiline=True,
                maw=360,
                withinPortal=True,
                boxWrapperProps={"style": {"width": "32px"}},
            )
        )
    active_record = get_record(registry, active_id) or {}
    active_label = str(active_record.get("name") or active_id)
    graph_data = no_update if graph_options == options else options
    de_data = no_update if de_options == options else options
    return rail, active_label, graph_data, de_data


@app.callback(
    Output("de-input-dataset", "value"),
    Input("dataset-registry", "data"),
    State("de-input-dataset", "value"),
)
def initialize_engineering_input(registry, selected_id):
    """Choose the source once, while preserving an explicit valid selection."""
    options = dataset_options(registry)
    valid_ids = {option["value"] for option in options}
    if selected_id in valid_ids:
        raise PreventUpdate
    if not options:
        return None
    return options[0]["value"]


@app.callback(
    Output("dataset-registry", "data", allow_duplicate=True),
    Output("active-dataset-id", "data", allow_duplicate=True),
    Output("active-dataset-data", "data", allow_duplicate=True),
    Output("meta-columns", "data", allow_duplicate=True),
    Output("filtered-data", "data", allow_duplicate=True),
    Output("filters-state", "data", allow_duplicate=True),
    Output("filters-applied-state", "data", allow_duplicate=True),
    Output("filter-logic-mode", "value", allow_duplicate=True),
    Output("filter-applied-logic", "data", allow_duplicate=True),
    Input("graph-dataset-select", "value"),
    Input("de-input-dataset", "value"),
    Input({"type": "dataset-rail-tab", "index": ALL}, "n_clicks"),
    Input("dataset-side-tab", "n_clicks"),
    State({"type": "dataset-rail-tab", "index": ALL}, "id"),
    State("dataset-registry", "data"),
    State("active-dataset-id", "data"),
    State("filtered-data", "data"),
    State("filters-state", "data"),
    State("filters-applied-state", "data"),
    State("filter-logic-mode", "value"),
    State("filter-applied-logic", "data"),
    prevent_initial_call=True,
)
def activate_dataset(
    graph_selected_id,
    de_selected_id,
    _rail_clicks,
    _dataset_tab_clicks,
    _rail_ids,
    registry,
    active_id,
    filtered_data,
    filters_state,
    filters_applied_state,
    filter_logic,
    filter_applied_logic,
):
    trigger = ctx.triggered_id
    if isinstance(trigger, dict) and trigger.get("type") == "dataset-rail-tab":
        clicked_ids = {
            str(component_id.get("index"))
            for count, component_id in zip(_rail_clicks or [], _rail_ids or [])
            if count
        }
        requested_id = str(trigger.get("index"))
        if requested_id not in clicked_ids:
            # Re-rendering the ALL-pattern button list is not a user click.
            raise PreventUpdate
    elif trigger == "graph-dataset-select":
        requested_id = graph_selected_id
    elif trigger == "de-input-dataset":
        requested_id = de_selected_id
    elif trigger == "dataset-side-tab":
        requested_id = SOURCE_DATASET_ID
    else:
        raise PreventUpdate
    if not requested_id:
        raise PreventUpdate
    if str(requested_id) == str(active_id):
        raise PreventUpdate

    updated = save_runtime_state(
        registry,
        active_id,
        filtered_data=filtered_data,
        filters_state=filters_state,
        filters_applied_state=filters_applied_state,
        filter_logic=filter_logic,
        filter_applied_logic=filter_applied_logic,
    )
    target = get_record(updated, requested_id)
    if not target:
        raise PreventUpdate
    payload = payload_for_record(target)
    filtered = payload_for_record(target, filtered=True) or payload
    return (
        updated,
        str(requested_id),
        payload,
        target.get("meta") or {},
        filtered,
        target.get("filters_state") or {},
        target.get("filters_applied_state") or {},
        target.get("filter_logic") or "and",
        target.get("filter_applied_logic") or "and",
    )


@app.callback(
    Output("dataset-registry", "data", allow_duplicate=True),
    Input("filtered-data", "data"),
    Input("filters-state", "data"),
    Input("filters-applied-state", "data"),
    Input("filter-logic-mode", "value"),
    Input("filter-applied-logic", "data"),
    State("dataset-registry", "data"),
    State("active-dataset-id", "data"),
    prevent_initial_call=True,
)
def sync_active_dataset_runtime(
    filtered_data,
    filters_state,
    filters_applied_state,
    filter_logic,
    filter_applied_logic,
    registry,
    active_id,
):
    current = get_record(registry, active_id)
    if not current:
        raise PreventUpdate
    desired = {
        "filters_state": filters_state or {},
        "filters_applied_state": filters_applied_state or {},
        "filter_logic": filter_logic or "and",
        "filter_applied_logic": filter_applied_logic or "and",
    }
    filtered_changed = ctx.triggered_id == "filtered-data"
    same_filtered = (
        payload_for_record(current, filtered=True) == filtered_data
        if filtered_changed
        else True
    )
    if same_filtered and all(current.get(key) == value for key, value in desired.items()):
        raise PreventUpdate
    return save_runtime_state(
        registry,
        active_id,
        filtered_data=filtered_data if filtered_changed else None,
        **desired,
    )


@app.callback(
    Output("de-dataset-summary", "children"),
    Output("de-active-dataset-badge", "children"),
    Input("dataset-registry", "data"),
    Input("active-dataset-id", "data"),
)
def render_pipeline(registry, active_id):
    record = get_record(registry, active_id)
    if not record:
        return "", "Нет данных"
    meta = record.get("meta") or {}
    rows = meta.get("row_count", 0)
    columns = meta.get("column_count", len(meta.get("columns") or []))
    summary = f"{rows:,} × {columns}".replace(",", " ")
    return summary, str(record.get("name") or active_id)


@app.callback(
    Output("de-preview", "children"),
    Input("active-dataset-data", "data"),
    State("meta-columns", "data"),
)
def render_dataset_preview(payload, meta):
    if not payload:
        return html.Div("Нет данных для предпросмотра", className="de-pipeline-empty")
    try:
        frame = read_df_from_store(payload, meta)
    except Exception:
        return html.Div("Не удалось построить предпросмотр", className="de-pipeline-empty")
    if len(frame.columns) <= 10:
        columns = list(frame.columns)
    else:
        first = list(frame.columns[:3])
        last = list(frame.columns[-7:])
        columns = list(dict.fromkeys([*first, *last]))
    preview = frame.loc[:, columns].head(8)
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(str(column)) for column in columns])),
            html.Tbody(
                [
                    html.Tr([
                        html.Td("" if value is None else str(value), title=str(value))
                        for value in row
                    ])
                    for row in preview.itertuples(index=False, name=None)
                ]
            ),
        ]
    )


@app.callback(
    Output("de-output-mode", "value"),
    Output("de-scope-note", "children"),
    Input("de-input-scope", "value"),
    prevent_initial_call=True,
)
def recommend_dataset_output(scope):
    if scope == "filtered":
        return "new", "Фильтрованная выборка по умолчанию создаёт отдельный dataset."
    return no_update, "Шаг будет рассчитан до фильтров и станет доступен самим фильтрам."


@app.callback(
    Output("dropdown_x", "value", allow_duplicate=True),
    Output("dropdown_y", "value", allow_duplicate=True),
    Output("dropdown_z", "value", allow_duplicate=True),
    Output("dropdown_color", "value", allow_duplicate=True),
    Output("dropdown_size", "value", allow_duplicate=True),
    Output("dropdown_text", "value", allow_duplicate=True),
    Output("dropdown_hover_data", "value", allow_duplicate=True),
    Output("dropdown_facet_row", "value", allow_duplicate=True),
    Output("dropdown_facet_col", "value", allow_duplicate=True),
    Input("active-dataset-id", "data"),
    prevent_initial_call=True,
)
def clear_graph_fields_for_dataset(_active_id):
    if not _active_id:
        raise PreventUpdate
    return None, None, None, None, None, None, [], None, None
