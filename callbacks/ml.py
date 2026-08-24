# -*- coding: utf-8 -*-
"""Callbacks for the dataset-aware CatBoost regression workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from dash_app import app
from dataset_export import export_catboost_model, export_frame_to_excel
from dataset_registry import (
    commit_result,
    dataset_options,
    get_record,
    input_payload,
    suggest_dataset_name,
)
from ml_engine import cache_result, cached_result, ml_signature
from ml_models import get_model_adapter
from utils import meta_from_df, read_df_from_store


def _notification(message, *, color="green", notification_id="ml"):
    return [{
        "id": notification_id, "title": "Machine Learning", "message": message,
        "color": color, "action": "show", "autoClose": 7000,
    }]


def _signature(input_id, scope, target, id_column, features, method, test_size,
               folds, group_column, time_column, iterations, depth, learning_rate, l2, loss,
               early_stopping, random_strength, bagging_temperature,
               random_seed, prediction_column, include_residual, compute_shap):
    method = method or "split"
    group_column = group_column if method == "group_cv" else None
    time_column = time_column if method == "time_cv" else None
    return ml_signature(
        input_id=str(input_id or ""), scope=scope or "base", target=str(target or ""),
        id_column=str(id_column or ""), features=list(features or []), method=method,
        test_size=float(test_size or 0), folds=int(folds or 0), iterations=int(iterations or 0),
        group_column=str(group_column or ""), time_column=str(time_column or ""),
        depth=int(depth or 0), learning_rate=float(learning_rate or 0),
        l2=float(l2 or 0), loss=loss or "RMSE", early_stopping=int(early_stopping or 0),
        random_strength=float(random_strength or 0),
        bagging_temperature=float(bagging_temperature or 0), random_seed=int(random_seed or 0),
        prediction_column=str(prediction_column or ""), include_residual=bool(include_residual),
        compute_shap=bool(compute_shap),
    )


def _empty_figure(message, template, height=420):
    figure = go.Figure()
    figure.update_layout(
        template=template or "plotly", height=height,
        margin=dict(l=40, r=20, t=35, b=45),
        annotations=[dict(text=message, x=.5, y=.5, xref="paper", yref="paper",
                          showarrow=False, font=dict(color="#868e96", size=12))],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return figure


def _prediction_figure(analysis, template):
    actual = analysis.get("actual") or []
    predicted = analysis.get("predicted") or []
    if not actual:
        return _empty_figure("Обучите модель", template)
    low = min(min(actual), min(predicted))
    high = max(max(actual), max(predicted))
    figure = go.Figure([
        go.Scattergl(
            x=actual, y=predicted, mode="markers", name="Прогноз",
            marker=dict(size=6, opacity=.62),
            hovertemplate="Факт: %{x:.5g}<br>Прогноз: %{y:.5g}<extra></extra>",
        ),
        go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Идеал",
                   line=dict(dash="dash", width=1.5, color="#868e96")),
    ])
    figure.update_layout(
        template=template or "plotly", height=420,
        title=f"Прогноз vs факт · {analysis.get('evaluation_label', '')}",
        xaxis_title="Факт", yaxis_title="Прогноз", hovermode="closest",
        margin=dict(l=55, r=20, t=55, b=50), legend=dict(orientation="h", y=1.08),
    )
    return figure


def _learning_figure(analysis, template):
    curves = analysis.get("curves") or []
    if not curves:
        return _empty_figure("Нет истории обучения", template)
    figure = go.Figure()
    for curve in curves[:10]:
        figure.add_trace(go.Scatter(
            x=curve.get("iterations"), y=curve.get("validation"), mode="lines",
            name=f"{curve.get('label')} · validation", line=dict(width=1.7),
        ))
        if len(curves) == 1 and curve.get("learn"):
            figure.add_trace(go.Scatter(
                x=curve.get("iterations")[:len(curve["learn"])], y=curve["learn"],
                mode="lines", name="Learn", line=dict(width=1.3, dash="dot"),
            ))
    metric = curves[0].get("metric") or "Ошибка"
    figure.update_layout(
        template=template or "plotly", height=420, title="Кривые обучения",
        xaxis_title="Итерация", yaxis_title=metric,
        margin=dict(l=55, r=20, t=55, b=50), legend=dict(orientation="h", y=1.08),
    )
    return figure


def _importance_figure(items, template, title, x_title):
    top = list(items or [])[:30][::-1]
    if not top:
        return _empty_figure("Нет данных", template)
    figure = go.Figure(go.Bar(
        x=[item["importance"] for item in top],
        y=[item["feature"] for item in top], orientation="h",
        marker_color="#228be6",
        hovertemplate="%{y}<br>%{x:.5g}<extra></extra>",
    ))
    figure.update_layout(
        template=template or "plotly", height=max(420, 24 * len(top) + 90),
        title=title, xaxis_title=x_title,
        margin=dict(l=150, r=20, t=55, b=50),
    )
    return figure


def _shap_figure(analysis, template):
    importance = analysis.get("shap_importance") or []
    values = np.asarray(analysis.get("shap_values") or [], dtype=float)
    sample = pd.DataFrame(analysis.get("shap_sample") or [])
    features = list(analysis.get("features") or [])
    if not importance or values.size == 0 or sample.empty:
        return _empty_figure("SHAP отключён или недоступен", template)
    top_features = [item["feature"] for item in importance[:15]][::-1]
    figure = go.Figure()
    for feature in top_features:
        feature_index = features.index(feature)
        raw = sample[feature] if feature in sample else pd.Series([0] * len(values))
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().sum() >= max(2, len(raw) // 2):
            color = numeric.fillna(numeric.median()).to_numpy(dtype=float)
        else:
            color = pd.factorize(raw.astype(str), sort=True)[0]
        if len(np.unique(color)) > 1:
            color = (color - np.min(color)) / (np.max(color) - np.min(color))
        figure.add_trace(go.Scattergl(
            x=values[:, feature_index], y=[feature] * len(values), mode="markers",
            showlegend=False,
            marker=dict(size=5, opacity=.58, color=color, colorscale="RdBu", showscale=False),
            customdata=np.asarray(raw.astype(str)),
            hovertemplate="%{y}<br>SHAP: %{x:.5g}<br>Значение: %{customdata}<extra></extra>",
        ))
    figure.update_layout(
        template=template or "plotly", height=max(420, 28 * len(top_features) + 80),
        title="SHAP summary · влияние признаков на прогноз",
        xaxis_title="SHAP value", yaxis_title=None,
        margin=dict(l=150, r=20, t=55, b=50),
    )
    return figure


def _diagnostics_figure(analysis, template):
    actual = np.asarray(analysis.get("actual") or [], dtype=float)
    predicted = np.asarray(analysis.get("predicted") or [], dtype=float)
    if actual.size == 0 or predicted.size == 0:
        return _empty_figure("Обучите модель", template)
    residuals = actual - predicted
    figure = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Остаток vs прогноз", "Распределение остатков"),
        horizontal_spacing=.12,
    )
    figure.add_trace(go.Scattergl(
        x=predicted, y=residuals, mode="markers", name="Остаток",
        marker=dict(size=6, opacity=.58),
        hovertemplate="Прогноз: %{x:.5g}<br>Остаток: %{y:.5g}<extra></extra>",
    ), row=1, col=1)
    figure.add_hline(y=0, line_width=1, line_dash="dash", line_color="#868e96", row=1, col=1)
    figure.add_trace(go.Histogram(
        x=residuals, name="Распределение", marker_color="#7950f2", opacity=.78,
        hovertemplate="Остаток: %{x:.5g}<br>Строк: %{y}<extra></extra>",
    ), row=1, col=2)
    figure.update_xaxes(title_text="Прогноз", row=1, col=1)
    figure.update_yaxes(title_text="Факт − прогноз", row=1, col=1)
    figure.update_xaxes(title_text="Остаток", row=1, col=2)
    figure.update_yaxes(title_text="Количество", row=1, col=2)
    figure.update_layout(
        template=template or "plotly", height=420, showlegend=False,
        margin=dict(l=55, r=20, t=55, b=50), bargap=.05,
    )
    return figure


@app.callback(
    Output("ml-input-dataset", "data"), Output("ml-input-dataset", "value"),
    Output("ml-dataset-badge", "children"), Output("ml-dataset-badge", "color"),
    Input("dataset-registry", "data"), Input("active-dataset-id", "data"),
    State("ml-input-dataset", "value"),
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
    label = f"{int(meta.get('row_count') or 0):,} × {int(meta.get('column_count') or 0)}".replace(",", " ")
    return options, selected, label, "blue"


@app.callback(
    Output("ml-target", "data"), Output("ml-target", "value"),
    Output("ml-id-column", "data"), Output("ml-id-column", "value"),
    Output("ml-features", "data"), Output("ml-features", "value"),
    Output("ml-group-column", "data"), Output("ml-group-column", "value"),
    Output("ml-time-column", "data"), Output("ml-time-column", "value"),
    Input("ml-input-dataset", "value"), Input("dataset-registry", "data"),
    State("ml-target", "value"), State("ml-id-column", "value"),
    State("ml-features", "value"), State("ml-group-column", "value"),
    State("ml-time-column", "value"),
)
def sync_column_options(input_id, registry, target, id_column, selected_features,
                        group_column, time_column):
    meta = ((get_record(registry, input_id) or {}).get("meta") or {})
    columns = [str(value) for value in meta.get("columns", [])]
    numeric = [str(value) for value in meta.get("numeric", [])]
    all_options = [{"label": value, "value": value} for value in columns]
    numeric_options = [{"label": value, "value": value} for value in numeric]
    target = target if target in numeric else None
    id_column = id_column if id_column in columns else None
    kept = [value for value in (selected_features or []) if value in columns and value != target]
    group_column = group_column if group_column in columns else None
    time_column = time_column if time_column in columns else None
    return (
        numeric_options, target, all_options, id_column, all_options, kept,
        all_options, group_column, all_options, time_column,
    )


@app.callback(
    Output("ml-features", "value", allow_duplicate=True),
    Input("ml-select-numeric", "n_clicks"),
    State("ml-input-dataset", "value"), State("dataset-registry", "data"),
    State("ml-target", "value"), State("ml-id-column", "value"),
    prevent_initial_call=True,
)
def select_all_numeric(_clicks, input_id, registry, target, id_column):
    meta = ((get_record(registry, input_id) or {}).get("meta") or {})
    return [str(value) for value in meta.get("numeric", [])
            if value not in {target, id_column}]


@app.callback(
    Output("ml-test-size", "disabled"), Output("ml-folds", "disabled"),
    Output("ml-group-column", "disabled"), Output("ml-time-column", "disabled"),
    Output("ml-validation-hint", "children"),
    Input("ml-method", "value"),
)
def toggle_validation_controls(method):
    hints = {
        "split": "Быстрая случайная проверка. Для связанных объектов она может завысить качество.",
        "cv": "KFold устойчивее одной случайной выборки, но не защищает от смешивания связанных объектов.",
        "group_cv": "Группы не смешиваются между train и validation. Для скважин укажите скважину или месторождение.",
        "time_cv": "Строки сортируются по выбранному каналу; модель проверяется только на будущем относительно train.",
    }
    return (
        method != "split", method == "split", method != "group_cv",
        method != "time_cv", hints.get(method, hints["split"]),
    )


@app.callback(
    Output("ml-iterations", "value"), Output("ml-depth", "value"),
    Output("ml-learning-rate", "value"), Output("ml-l2", "value"),
    Output("ml-early-stopping", "value"),
    Input("ml-preset", "value"), prevent_initial_call=True,
)
def apply_preset(preset):
    values = {
        "draft": (250, 5, .12, 4, 35),
        "balanced": (800, 6, .05, 3, 80),
        "quality": (2000, 8, .025, 3, 180),
    }.get(preset)
    if not values:
        return (no_update,) * 5
    return values


@app.callback(
    Output("ml-output-name", "value"), Output("ml-auto-output-name", "data"),
    Input("ml-input-scope", "value"),
    Input("dataset-registry", "data"), State("ml-output-name", "value"),
    State("ml-auto-output-name", "data"),
)
def suggest_output_name(scope, registry, current, previous_auto):
    candidate = suggest_dataset_name(
        registry, [{"operation": "catboost_regression"}], scope or "base"
    )
    if not str(current or "").strip() or current == previous_auto:
        return candidate, candidate
    return no_update, candidate


RUN_STATES = [
    State("dataset-registry", "data"), State("active-dataset-id", "data"),
    State("filtered-data", "data"), State("ml-input-dataset", "value"),
    State("ml-input-scope", "value"), State("ml-target", "value"),
    State("ml-id-column", "value"), State("ml-features", "value"),
    State("ml-method", "value"), State("ml-test-size", "value"),
    State("ml-folds", "value"), State("ml-group-column", "value"),
    State("ml-time-column", "value"), State("ml-iterations", "value"),
    State("ml-depth", "value"), State("ml-learning-rate", "value"),
    State("ml-l2", "value"), State("ml-loss", "value"),
    State("ml-early-stopping", "value"), State("ml-random-strength", "value"),
    State("ml-bagging-temperature", "value"), State("ml-random-seed", "value"),
    State("ml-prediction-column", "value"), State("ml-include-residual", "checked"),
    State("ml-compute-shap", "checked"),
]


@app.callback(
    Output("ml-analysis", "data"),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("ml-run", "n_clicks"), *RUN_STATES,
    running=[(Output("ml-run", "loading"), True, False)],
    prevent_initial_call=True,
)
def train_model(_clicks, registry, active_id, filtered_data, input_id, scope,
                target, id_column, features, method, test_size, folds,
                group_column, time_column, iterations, depth, learning_rate, l2, loss, early_stopping,
                random_strength, bagging_temperature, random_seed,
                prediction_column, include_residual, compute_shap):
    if not input_id:
        return no_update, _notification("Сначала загрузите dataset.", color="orange")
    group_column = group_column if method == "group_cv" else None
    time_column = time_column if method == "time_cv" else None
    payload, meta = input_payload(
        registry, input_id, scope or "base",
        active_id=active_id, active_filtered=filtered_data,
    )
    if not payload:
        return no_update, _notification("Входной dataset недоступен.", color="red")
    signature = _signature(
        input_id, scope, target, id_column, features, method, test_size, folds,
        group_column, time_column,
        iterations, depth, learning_rate, l2, loss, early_stopping,
        random_strength, bagging_temperature, random_seed, prediction_column,
        include_residual, compute_shap,
    )
    try:
        frame = read_df_from_store(payload, meta)
        result = get_model_adapter("catboost").run(
            frame, target=target, features=features, id_column=id_column,
            method=method, test_size=test_size, folds=folds,
            group_column=group_column, time_column=time_column, iterations=iterations,
            depth=depth, learning_rate=learning_rate, l2_leaf_reg=l2,
            loss_function=loss, random_seed=random_seed,
            early_stopping_rounds=early_stopping, random_strength=random_strength,
            bagging_temperature=bagging_temperature,
            prediction_column=prediction_column, include_residual=include_residual,
            compute_shap=compute_shap, signature=signature,
        )
        result.committed_step["scope"] = scope or "base"
        reference = cache_result(result)
    except Exception as error:
        return no_update, _notification(str(error), color="red", notification_id="ml-error")
    analysis = result.analysis
    message = (
        f"Модель готова · {analysis['training_rows']} строк · "
        f"{len(analysis['features'])} признаков · MAE {analysis['metrics']['mae']:.4g}"
    )
    return {
        "reference": reference, "signature": signature, "input_id": str(input_id),
        "scope": scope or "base", "analysis": analysis,
    }, _notification(message, notification_id="ml-ready")


@app.callback(
    Output("ml-prediction-graph", "figure"), Output("ml-learning-graph", "figure"),
    Output("ml-importance-graph", "figure"), Output("ml-shap-graph", "figure"),
    Output("ml-diagnostics-graph", "figure"),
    Output("ml-prediction-table", "data"), Output("ml-prediction-table", "columns"),
    Output("ml-metric-mae", "children"), Output("ml-metric-rmse", "children"),
    Output("ml-metric-mape", "children"), Output("ml-metric-r2", "children"),
    Output("ml-metric-baseline", "children"), Output("ml-evaluation-note", "children"),
    Output("ml-shap-note", "children"), Output("ml-log", "children"),
    Input("ml-analysis", "data"), Input("dropdown_style", "value"),
)
def render_analysis(store, template):
    analysis = (store or {}).get("analysis") or {}
    if not analysis:
        empty = _empty_figure("Выберите цель и признаки, затем обучите модель", template)
        return empty, empty, empty, empty, empty, [], [], "—", "—", "—", "—", "—", "", "", ""
    metrics = analysis.get("metrics") or {}
    baseline = analysis.get("baseline") or {}
    def number(value, digits=4, suffix=""):
        return "—" if value is None else f"{float(value):.{digits}g}{suffix}"
    rows = analysis.get("preview") or []
    columns = [{"name": name, "id": name} for name in (rows[0].keys() if rows else [])]
    params = analysis.get("params") or {}
    log = "\n".join([
        f"Метод: {analysis.get('evaluation_label')}",
        f"Цель: {analysis.get('target')}",
        f"Признаки ({len(analysis.get('features') or [])}): {', '.join(analysis.get('features') or [])}",
        f"Категориальные: {', '.join(analysis.get('categorical_features') or []) or 'нет'}",
        f"Строки: {analysis.get('training_rows')} обучающих из {analysis.get('input_rows')}",
        f"Исключено из обучения из-за пустой цели: {analysis.get('excluded_target_rows')}",
        f"Параметры: {params}",
        f"Финальное количество деревьев: {analysis.get('final_iterations')}",
        "Оценка качества рассчитана только на test/OOF; финальная модель переобучена на всех строках с известной целью.",
    ])
    shown = min(6000, int(analysis.get("evaluation_rows") or 0))
    note = (
        f"{analysis.get('evaluation_label')} · оценено {analysis.get('evaluation_rows')} строк"
        + (f" · на графике выборка {shown}" if shown < analysis.get("evaluation_rows", 0) else "")
    )
    return (
        _prediction_figure(analysis, template), _learning_figure(analysis, template),
        _importance_figure(analysis.get("feature_importance"), template,
                           "Важность признаков CatBoost", "Feature importance"),
        _shap_figure(analysis, template), _diagnostics_figure(analysis, template),
        rows, columns,
        number(metrics.get("mae")), number(metrics.get("rmse")),
        number(metrics.get("mape"), 4, "%"), number(metrics.get("r2")),
        number(baseline.get("mae")), note, analysis.get("shap_note") or "", log,
    )


SIGNATURE_INPUTS = [
    Input("ml-input-dataset", "value"), Input("ml-input-scope", "value"),
    Input("ml-target", "value"), Input("ml-id-column", "value"),
    Input("ml-features", "value"), Input("ml-method", "value"),
    Input("ml-test-size", "value"), Input("ml-folds", "value"),
    Input("ml-group-column", "value"), Input("ml-time-column", "value"),
    Input("ml-iterations", "value"), Input("ml-depth", "value"),
    Input("ml-learning-rate", "value"), Input("ml-l2", "value"),
    Input("ml-loss", "value"), Input("ml-early-stopping", "value"),
    Input("ml-random-strength", "value"), Input("ml-bagging-temperature", "value"),
    Input("ml-random-seed", "value"), Input("ml-prediction-column", "value"),
    Input("ml-include-residual", "checked"), Input("ml-compute-shap", "checked"),
]


@app.callback(
    Output("ml-run-status", "children"), Output("ml-run-status", "color"),
    Output("ml-commit", "disabled"), Output("ml-export-excel", "disabled"),
    Output("ml-save-model", "disabled"),
    Output("ml-row-status", "children"),
    Input("ml-analysis", "data"), *SIGNATURE_INPUTS,
)
def validate_current_result(store, *values):
    if not store:
        return "Ожидает запуска", "gray", True, True, True, ""
    current = _signature(*values)
    analysis = store.get("analysis") or {}
    rows = f"{analysis.get('training_rows', 0)} / {analysis.get('input_rows', 0)} строк"
    if current != store.get("signature"):
        return "Параметры изменены", "orange", True, True, True, rows + " · требуется пересчёт"
    if not cached_result(store.get("reference")):
        return "Результат устарел", "red", True, True, True, "Повторите обучение"
    return "Модель готова", "green", False, False, False, rows + " · доступны dataset, Excel и CBM"


@app.callback(
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("ml-export-excel", "n_clicks"), State("ml-analysis", "data"),
    State("ml-output-name", "value"), State("source-file-path", "data"),
    State("source-file-name", "data"), prevent_initial_call=True,
)
def export_ml_result(_clicks, store, output_name, source_path, source_name):
    cached = cached_result((store or {}).get("reference"))
    if not cached:
        return _notification("Сначала обучите модель.", color="orange", notification_id="ml-export")
    analysis = (store or {}).get("analysis") or {}
    try:
        path = export_frame_to_excel(
            cached.frame,
            source_path=source_path,
            source_name=source_name,
            dataset_name=str(output_name or "CatBoost"),
            created_at=analysis.get("created_at"),
        )
    except Exception as error:
        return _notification(str(error), color="red", notification_id="ml-export-error")
    return _notification(
        f"Excel сохранён рядом с исходником: {path}", notification_id="ml-export-ready"
    )


@app.callback(
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("ml-save-model", "n_clicks"), State("ml-analysis", "data"),
    State("ml-output-name", "value"), State("source-file-path", "data"),
    State("source-file-name", "data"), prevent_initial_call=True,
)
def save_ml_model(_clicks, store, output_name, source_path, source_name):
    cached = cached_result((store or {}).get("reference"))
    if not cached:
        return _notification(
            "Сначала обучите модель.", color="orange", notification_id="ml-model-export"
        )
    analysis = (store or {}).get("analysis") or {}
    try:
        path = export_catboost_model(
            cached.model,
            source_path=source_path,
            source_name=source_name,
            experiment_name=str(output_name or "CatBoost"),
            created_at=analysis.get("created_at"),
        )
    except Exception as error:
        return _notification(
            str(error), color="red", notification_id="ml-model-export-error"
        )
    return _notification(
        f"Модель CatBoost сохранена: {path}", notification_id="ml-model-export-ready"
    )


@app.callback(
    Output("ml-experiment-history", "data"),
    Input("ml-analysis", "data"), State("ml-experiment-history", "data"),
    State("dataset-registry", "data"),
    prevent_initial_call=True,
)
def remember_experiment(store, history, registry):
    analysis = (store or {}).get("analysis") or {}
    reference = str((store or {}).get("reference") or "")
    if not analysis or not reference:
        raise PreventUpdate
    records = list(history or [])
    if any(str(item.get("reference") or "") == reference for item in records):
        raise PreventUpdate
    input_id = str((store or {}).get("input_id") or "")
    input_record = get_record(registry, input_id) or {}
    records.append({
        "reference": reference,
        "signature": str((store or {}).get("signature") or ""),
        "model": "CatBoost",
        "task": "Регрессия",
        "dataset": str(input_record.get("name") or input_id),
        "dataset_id": input_id,
        "scope": str((store or {}).get("scope") or "base"),
        "target": str(analysis.get("target") or ""),
        "features": list(analysis.get("features") or []),
        "validation": str(analysis.get("evaluation_label") or ""),
        "training_rows": int(analysis.get("training_rows") or 0),
        "evaluation_rows": int(analysis.get("evaluation_rows") or 0),
        "feature_count": len(analysis.get("features") or []),
        "mae": (analysis.get("metrics") or {}).get("mae"),
        "rmse": (analysis.get("metrics") or {}).get("rmse"),
        "r2": (analysis.get("metrics") or {}).get("r2"),
        "params": dict(analysis.get("params") or {}),
        "created_at": str(analysis.get("created_at") or ""),
    })
    return records[-200:]


@app.callback(
    Output("ml-experiments-table", "data"), Output("ml-experiments-table", "columns"),
    Output("ml-experiments-graph", "figure"), Output("ml-history-count", "children"),
    Output("ml-history-count", "color"), Output("ml-history-best", "children"),
    Input("ml-experiment-history", "data"), Input("dropdown_style", "value"),
)
def render_experiment_history(history, template):
    records = list(history or [])
    if not records:
        return (
            [], [], _empty_figure("Запусков пока нет", template, height=330),
            "0 запусков", "gray", "Лучший MAE: —",
        )

    rows = []
    for index, item in enumerate(reversed(records), 1):
        created = str(item.get("created_at") or "").replace("T", " ")[:19]
        rows.append({
            "№": len(records) - index + 1,
            "Время": created,
            "Модель": item.get("model"),
            "Dataset": item.get("dataset"),
            "Слой": "после фильтров" if item.get("scope") == "filtered" else "до фильтров",
            "Цель": item.get("target"),
            "Проверка": item.get("validation"),
            "Строк train": item.get("training_rows"),
            "Строк eval": item.get("evaluation_rows"),
            "Признаков": item.get("feature_count"),
            "MAE": item.get("mae"),
            "RMSE": item.get("rmse"),
            "R²": item.get("r2"),
        })
    columns = [{"name": name, "id": name} for name in rows[0]]

    chart_records = records[-30:]
    labels, values, hover = [], [], []
    first_run = max(1, len(records) - len(chart_records) + 1)
    for run_number, item in enumerate(chart_records, first_run):
        if item.get("mae") is None:
            continue
        labels.append(f"#{run_number}")
        values.append(float(item["mae"]))
        hover.append(f"{item.get('model')} · {item.get('target')}<br>{item.get('validation')}")
    if values:
        best_value = min(values)
        colors = ["#40c057" if value == best_value else "#7950f2" for value in values]
        figure = go.Figure(go.Bar(
            x=labels, y=values, marker_color=colors, customdata=hover,
            hovertemplate="%{customdata}<br>MAE: %{y:.5g}<extra></extra>",
        ))
        figure.update_layout(
            template=template or "plotly", height=330,
            margin=dict(l=48, r=18, t=20, b=42),
            xaxis_title="Запуск", yaxis_title="MAE",
        )
        best_label = f"Лучший MAE: {best_value:.5g}"
    else:
        figure = _empty_figure("Нет MAE для сравнения", template, height=330)
        best_label = "Лучший MAE: —"
    return rows, columns, figure, f"{len(records)} запусков", "violet", best_label


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
    Input("ml-commit", "n_clicks"), State("ml-analysis", "data"),
    State("dataset-registry", "data"), State("ml-output-name", "value"),
    *[State(item.component_id, item.component_property)
                                      for item in SIGNATURE_INPUTS],
    prevent_initial_call=True,
)
def commit_prediction(_clicks, store, registry, output_name, *values):
    if not store:
        raise PreventUpdate
    if _signature(*values) != store.get("signature"):
        result = (no_update,) * 8
        return *result, _notification("Параметры изменились. Повторите обучение.", color="orange")
    cached = cached_result(store.get("reference"))
    if not cached:
        result = (no_update,) * 8
        return *result, _notification("Результат недоступен. Повторите обучение.", color="red")
    input_id, scope = values[0], values[1]
    payload = cached.frame.to_json(date_format="iso", orient="split")
    meta = meta_from_df(cached.frame)
    resolved_name = output_name
    if not str(output_name or "").strip():
        resolved_name = suggest_dataset_name(
            registry, [{"operation": "catboost_regression"}], scope or "base"
        )
    updated, result_id = commit_result(
        registry, input_id, payload, meta, cached.committed_step,
        output_mode="new", output_name=resolved_name,
    )
    outputs = len((store.get("analysis") or {}).get("outputs") or [])
    return (
        updated, no_update, no_update, no_update, no_update, no_update, no_update, no_update,
        _notification(f"Записано каналов: {outputs} · создан новый dataset",
                      notification_id="ml-committed"),
    )


app.clientside_callback(
    """
    function(value) {
        var target = document.getElementById('ml-features-drop');
        if (!target) return window.dash_clientside.no_update;
        target.setAttribute('data-current-value', JSON.stringify(value || []));
        target.classList.toggle('has-value', Array.isArray(value) && value.length > 0);
        return Date.now();
    }
    """,
    Output("ml-columns-sync", "data"), Input("ml-features", "value"),
)
