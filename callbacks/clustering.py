# -*- coding: utf-8 -*-
"""Callbacks for the dataset-aware clustering laboratory."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from dash_app import app
from clustering_engine import (
    cache_result,
    cached_result,
    clustering_signature,
    run_clustering,
)
from dataset_registry import (
    commit_result,
    dataset_options,
    get_record,
    input_payload,
    suggest_dataset_name,
)
from utils import meta_from_df, read_df_from_store


def _notification(message, *, color="green", notification_id="cluster"):
    return [{
        "id": notification_id,
        "title": "Кластеризация",
        "message": message,
        "color": color,
        "action": "show",
        "autoClose": 6000,
    }]


def _signature(
    input_id, scope, features, algorithm, k, scaling, missing_policy,
    output_column, include_id, include_distance, include_pca,
):
    return clustering_signature(
        input_id=str(input_id or ""),
        scope=scope or "base",
        features=list(features or []),
        algorithm=algorithm or "kmeans",
        k=int(k or 0),
        scaling=scaling or "standard",
        missing_policy=missing_policy or "drop",
        output_column=str(output_column or "Кластер").strip() or "Кластер",
        include_id=bool(include_id),
        include_distance=bool(include_distance),
        include_pca=bool(include_pca),
    )


def _empty_figure(message, template):
    figure = go.Figure()
    figure.update_layout(
        template=template or "plotly",
        height=340,
        margin=dict(l=35, r=15, t=25, b=35),
        annotations=[dict(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#868e96", size=12),
        )],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return figure


def _projection_figure(analysis, template):
    projection = analysis.get("projection") or {}
    frame = pd.DataFrame({
        "PCA1": projection.get("x") or [],
        "PCA2": projection.get("y") or [],
        "Кластер": projection.get("cluster") or [],
        "Строка": projection.get("row") or [],
    })
    if frame.empty:
        return _empty_figure("Нет данных для проекции", template)
    figure = px.scatter(
        frame,
        x="PCA1",
        y="PCA2",
        color="Кластер",
        hover_data={"Строка": True},
        template=template or "plotly",
        render_mode="webgl",
    )
    figure.update_traces(marker={"size": 6, "opacity": 0.76})
    figure.update_layout(
        height=390,
        margin=dict(l=45, r=20, t=15, b=40),
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    return figure


def _diagnostics_figure(analysis, template):
    diagnostics = analysis.get("diagnostics") or {}
    ks = diagnostics.get("ks") or []
    if not ks:
        return _empty_figure("Недостаточно строк для подбора K", template)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(x=ks, y=diagnostics.get("inertias") or [], name="Inertia", mode="lines+markers"),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(x=ks, y=diagnostics.get("silhouettes") or [], name="Silhouette", mode="lines+markers"),
        secondary_y=True,
    )
    recommended = diagnostics.get("recommended_k")
    if recommended:
        figure.add_vline(x=recommended, line_dash="dot", line_color="#40c057")
    figure.update_xaxes(title_text="K", dtick=1)
    figure.update_yaxes(title_text="Inertia", secondary_y=False)
    figure.update_yaxes(title_text="Silhouette", secondary_y=True)
    figure.update_layout(
        template=template or "plotly",
        height=290,
        margin=dict(l=50, r=55, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    return figure


def _sizes_figure(analysis, template):
    sizes = analysis.get("sizes") or {}
    frame = pd.DataFrame({
        "Кластер": sizes.get("cluster") or [],
        "Строк": sizes.get("count") or [],
    })
    if frame.empty:
        return _empty_figure("Нет распределения кластеров", template)
    figure = px.bar(frame, x="Кластер", y="Строк", color="Кластер", template=template or "plotly")
    figure.update_layout(
        height=290,
        margin=dict(l=45, r=15, t=20, b=55),
        showlegend=False,
    )
    return figure


def _profile_figure(analysis, template):
    profile = analysis.get("profile") or {}
    values = profile.get("values") or []
    if not values:
        return _empty_figure("Нет данных профиля", template)
    figure = go.Figure(go.Heatmap(
        z=values,
        x=profile.get("features") or [],
        y=profile.get("clusters") or [],
        colorscale="RdBu",
        zmid=0,
        colorbar={"title": "z"},
        hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>",
    ))
    figure.update_layout(
        template=template or "plotly",
        height=max(280, 54 * len(profile.get("clusters") or [])),
        margin=dict(l=85, r=25, t=20, b=75),
    )
    return figure


@app.callback(
    Output("cluster-input-dataset", "data"),
    Output("cluster-input-dataset", "value"),
    Output("cluster-dataset-badge", "children"),
    Output("cluster-dataset-badge", "color"),
    Input("dataset-registry", "data"),
    Input("active-dataset-id", "data"),
    State("cluster-input-dataset", "value"),
)
def sync_dataset_selector(registry, active_id, current):
    options = dataset_options(registry)
    available = {item["value"] for item in options}
    selected = current if current in available else active_id
    if selected not in available:
        selected = next(iter(available), None)
    record = get_record(registry, selected)
    if not record:
        return options, selected, "Нет данных", "gray"
    meta = record.get("meta") or {}
    rows = int(meta.get("row_count") or 0)
    columns = int(meta.get("column_count") or len(meta.get("columns") or []))
    label = f"{rows:,} × {columns}".replace(",", " ")
    return options, selected, label, "blue"


@app.callback(
    Output("cluster-cols", "data"),
    Output("cluster-cols", "value"),
    Input("cluster-input-dataset", "value"),
    Input("dataset-registry", "data"),
    State("cluster-cols", "value"),
)
def sync_feature_options(input_id, registry, selected):
    record = get_record(registry, input_id)
    numeric = [str(value) for value in ((record or {}).get("meta") or {}).get("numeric", [])]
    options = [{"label": value, "value": value} for value in numeric]
    kept = [value for value in (selected or []) if value in numeric]
    return options, kept


@app.callback(
    Output("cluster-output-name", "value"),
    Output("cluster-auto-output-name", "data"),
    Output("cluster-output-name", "disabled"),
    Input("cluster-input-scope", "value"),
    Input("cluster-output-mode", "value"),
    Input("dataset-registry", "data"),
    State("cluster-output-name", "value"),
    State("cluster-auto-output-name", "data"),
)
def suggest_cluster_output_name(scope, output_mode, registry, current, previous_auto):
    candidate = suggest_dataset_name(registry, [{"operation": "clustering"}], scope or "base")
    disabled = output_mode != "new"
    if not str(current or "").strip() or current == previous_auto:
        return candidate, candidate, disabled
    return no_update, candidate, disabled


@app.callback(
    Output("cluster-scope-note", "children"),
    Input("cluster-input-scope", "value"),
)
def render_scope_note(scope):
    if scope == "filtered":
        return "Расчёт выполняется только по строкам после применённых фильтров; рекомендуется новый dataset."
    return "Расчёт выполняется по базовому слою выбранного dataset."


@app.callback(
    Output("cluster-analysis", "data"),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("cluster-run", "n_clicks"),
    State("dataset-registry", "data"),
    State("active-dataset-id", "data"),
    State("filtered-data", "data"),
    State("cluster-input-dataset", "value"),
    State("cluster-input-scope", "value"),
    State("cluster-cols", "value"),
    State("cluster-algorithm", "value"),
    State("cluster-k", "value"),
    State("cluster-scaling", "value"),
    State("cluster-missing-policy", "value"),
    State("cluster-output-column", "value"),
    State("cluster-include-id", "checked"),
    State("cluster-include-distance", "checked"),
    State("cluster-include-pca", "checked"),
    prevent_initial_call=True,
)
def calculate_clustering(
    _clicks, registry, active_id, active_filtered, input_id, scope, features,
    algorithm, k, scaling, missing_policy, output_column,
    include_id, include_distance, include_pca,
):
    if not input_id:
        return no_update, _notification("Сначала загрузите dataset.", color="orange")
    payload, meta = input_payload(
        registry,
        input_id,
        scope or "base",
        active_id=active_id,
        active_filtered=active_filtered,
    )
    if not payload:
        return no_update, _notification("Входной dataset недоступен.", color="red")
    signature = _signature(
        input_id, scope, features, algorithm, k, scaling, missing_policy,
        output_column, include_id, include_distance, include_pca,
    )
    try:
        frame = read_df_from_store(payload, meta)
        result = run_clustering(
            frame,
            features=features,
            k=k,
            algorithm=algorithm,
            scaling=scaling,
            missing_policy=missing_policy,
            output_column=output_column,
            include_id=include_id,
            include_distance=include_distance,
            include_pca=include_pca,
            signature=signature,
        )
        result.committed_step["scope"] = scope or "base"
        reference = cache_result(result)
    except Exception as error:
        return no_update, _notification(str(error), color="red", notification_id="cluster-error")

    analysis = result.analysis
    message = (
        f"Рассчитано строк: {analysis['used_rows']} из {analysis['input_rows']} · "
        f"создано каналов: {len(analysis['outputs'])}"
    )
    return {
        "reference": reference,
        "signature": signature,
        "input_id": str(input_id),
        "scope": scope or "base",
        "analysis": analysis,
    }, _notification(message, notification_id="cluster-ready")


@app.callback(
    Output("cluster-projection-graph", "figure"),
    Output("cluster-diagnostics-graph", "figure"),
    Output("cluster-sizes-graph", "figure"),
    Output("cluster-profile-graph", "figure"),
    Output("cluster-analysis-section", "style"),
    Output("cluster-metric-silhouette", "children"),
    Output("cluster-metric-db", "children"),
    Output("cluster-metric-ch", "children"),
    Output("cluster-metric-k", "children"),
    Output("cluster-pca-note", "children"),
    Input("cluster-analysis", "data"),
    Input("dropdown_style", "value"),
)
def render_clustering_analysis(store, template):
    analysis = (store or {}).get("analysis") or {}
    if not analysis:
        empty = _empty_figure("Выберите признаки и нажмите «Рассчитать»", template)
        return empty, empty, empty, empty, {"display": "none"}, "—", "—", "—", "—", ""
    metrics = analysis.get("metrics") or {}
    diagnostics = analysis.get("diagnostics") or {}

    def formatted(value, digits=3):
        return "—" if value is None else f"{float(value):.{digits}f}"

    explained = ((analysis.get("projection") or {}).get("explained_variance") or [0, 0])
    variance = sum(explained[:2]) * 100
    pca_note = f"Показано до 6 000 точек · объяснено {variance:.1f}% дисперсии"
    return (
        _projection_figure(analysis, template),
        _diagnostics_figure(analysis, template),
        _sizes_figure(analysis, template),
        _profile_figure(analysis, template),
        {"display": "grid"},
        formatted(metrics.get("silhouette")),
        formatted(metrics.get("davies_bouldin")),
        formatted(metrics.get("calinski_harabasz"), 1),
        str(diagnostics.get("recommended_k") or "—"),
        pca_note,
    )


@app.callback(
    Output("cluster-run-status", "children"),
    Output("cluster-run-status", "color"),
    Output("cluster-commit", "disabled"),
    Output("cluster-row-status", "children"),
    Input("cluster-analysis", "data"),
    Input("cluster-input-dataset", "value"),
    Input("cluster-input-scope", "value"),
    Input("cluster-cols", "value"),
    Input("cluster-algorithm", "value"),
    Input("cluster-k", "value"),
    Input("cluster-scaling", "value"),
    Input("cluster-missing-policy", "value"),
    Input("cluster-output-column", "value"),
    Input("cluster-include-id", "checked"),
    Input("cluster-include-distance", "checked"),
    Input("cluster-include-pca", "checked"),
)
def validate_current_result(
    store, input_id, scope, features, algorithm, k, scaling, missing_policy,
    output_column, include_id, include_distance, include_pca,
):
    if not store:
        return "Ожидает расчёта", "gray", True, ""
    current = _signature(
        input_id, scope, features, algorithm, k, scaling, missing_policy,
        output_column, include_id, include_distance, include_pca,
    )
    analysis = store.get("analysis") or {}
    rows = f"{analysis.get('used_rows', 0)} / {analysis.get('input_rows', 0)} строк"
    if current != store.get("signature"):
        return "Параметры изменены", "orange", True, f"{rows} · требуется пересчёт"
    if not cached_result(store.get("reference")):
        return "Результат устарел", "red", True, "Повторите расчёт"
    excluded = int(analysis.get("excluded_rows") or 0)
    suffix = f" · исключено {excluded}" if excluded else ""
    return "Результат готов", "green", False, rows + suffix


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
    Input("cluster-commit", "n_clicks"),
    State("cluster-analysis", "data"),
    State("dataset-registry", "data"),
    State("cluster-output-mode", "value"),
    State("cluster-output-name", "value"),
    State("cluster-input-dataset", "value"),
    State("cluster-input-scope", "value"),
    State("cluster-cols", "value"),
    State("cluster-algorithm", "value"),
    State("cluster-k", "value"),
    State("cluster-scaling", "value"),
    State("cluster-missing-policy", "value"),
    State("cluster-output-column", "value"),
    State("cluster-include-id", "checked"),
    State("cluster-include-distance", "checked"),
    State("cluster-include-pca", "checked"),
    prevent_initial_call=True,
)
def commit_clustering_result(
    _clicks, store, registry, output_mode, output_name, input_id, scope,
    features, algorithm, k, scaling, missing_policy, output_column,
    include_id, include_distance, include_pca,
):
    if not store:
        raise PreventUpdate
    current_signature = _signature(
        input_id, scope, features, algorithm, k, scaling, missing_policy,
        output_column, include_id, include_distance, include_pca,
    )
    if current_signature != store.get("signature"):
        return (
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update,
            _notification("Параметры изменились. Сначала повторите расчёт.", color="orange"),
        )
    cached = cached_result(store.get("reference"))
    if not cached:
        return (
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update,
            _notification("Результат больше не доступен. Повторите расчёт.", color="red"),
        )

    result_meta = meta_from_df(cached.frame)
    result_payload = cached.frame.to_json(date_format="iso", orient="split")
    resolved_name = output_name
    if (output_mode or "new") == "new" and not str(output_name or "").strip():
        resolved_name = suggest_dataset_name(
            registry, [{"operation": "clustering"}], scope or "base"
        )
    updated, result_id = commit_result(
        registry,
        input_id,
        result_payload,
        result_meta,
        cached.committed_step,
        output_mode=output_mode or "new",
        output_name=resolved_name,
    )
    record = get_record(updated, result_id) or {}
    outputs = len((store.get("analysis") or {}).get("outputs") or [])
    message = f"Записано каналов: {outputs}"
    if (output_mode or "new") == "new":
        return (
            updated, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update,
            _notification(message + " · создан новый dataset", notification_id="cluster-committed"),
        )
    return (
        updated,
        result_id,
        result_payload,
        result_meta,
        record.get("filters_state") or {},
        record.get("filters_applied_state") or {},
        record.get("filter_logic") or "and",
        record.get("filter_applied_logic") or "and",
        _notification(message, notification_id="cluster-committed"),
    )


app.clientside_callback(
    """
    function(value) {
        var target = document.getElementById('cluster-columns-drop');
        if (!target) return window.dash_clientside.no_update;
        target.setAttribute('data-current-value', JSON.stringify(value || []));
        target.classList.toggle('has-value', Array.isArray(value) && value.length > 0);
        return Date.now();
    }
    """,
    Output("cluster-columns-sync", "data"),
    Input("cluster-cols", "value"),
    prevent_initial_call="initial_duplicate",
)
