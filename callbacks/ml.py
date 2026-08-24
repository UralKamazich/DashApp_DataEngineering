# -*- coding: utf-8 -*-
"""Callbacks for the dataset-aware CatBoost regression/classification workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, html, no_update
import dash_mantine_components as dmc
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
from ml_jobs import cancel_ml_job, ml_job_snapshot, submit_ml_job, take_ml_job_result
from ml_data_profile import missingness_figure, profile_dataset, target_figure
from ml_models import get_model_adapter
from ml_tuning import tune_catboost_parameters
from utils import meta_from_df, read_df_from_store


def _notification(message, *, color="green", notification_id="ml"):
    return [{
        "id": notification_id, "title": "Machine Learning", "message": message,
        "color": color, "action": "show", "autoClose": 7000,
    }]


def _format_bytes(value):
    size = float(value or 0)
    for suffix in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if size < 1024 or suffix == "ТБ":
            return f"{size:.0f} {suffix}" if suffix in {"Б", "КБ"} else f"{size:.1f} {suffix}"
        size /= 1024


def _profile_list(items, *, empty="Сигналов нет"):
    if not items:
        return dmc.Text(empty, size="9px", c="dimmed")
    icons = {"error": "×", "warning": "!", "info": "i"}
    colors = {"error": "red", "warning": "yellow", "info": "blue"}
    result = []
    for item in items:
        severity = str(item.get("severity") or "info")
        result.append(html.Div([
            dmc.Badge(
                icons.get(severity, "i"), color=colors.get(severity, "blue"),
                variant="filled", circle=True, size="xs",
            ),
            html.Div([
                html.Div(str(item.get("title") or ""), className="ml-profile-list-title"),
                html.Div(str(item.get("detail") or ""), className="ml-profile-list-detail"),
            ], className="ml-profile-list-copy"),
        ], className=f"ml-profile-list-item ml-profile-list-item--{severity}"))
    return result


def _profile_target_summary(profile):
    target = (profile or {}).get("target") or {}
    if not target.get("selected"):
        return dmc.Text("Выберите цель для проверки постановки задачи.", size="9px", c="dimmed")
    task = target.get("task")
    task_label = {
        "classification": "Классификация", "regression": "Регрессия",
    }.get(task, "Не определено")
    items = [
        {"severity": "info", "title": target.get("name"),
         "detail": f"{task_label} · уникальных значений: {target.get('unique', 0)}"},
    ]
    if task == "classification":
        items.append({
            "severity": "warning" if float(target.get("minority_share") or 100) < 10 else "info",
            "title": f"Классов: {target.get('unique', 0)}",
            "detail": (
                f"Минимальный класс: {float(target.get('minority_share') or 0):.1f}% · "
                f"отношение max/min: {float(target.get('imbalance_ratio') or 0):.2f}"
            ),
        })
    elif task == "regression" and target.get("mean") is not None:
        items.append({
            "severity": "info", "title": "Диапазон",
            "detail": (
                f"{target.get('min'):.5g} … {target.get('max'):.5g} · "
                f"среднее {target.get('mean'):.5g} · σ {target.get('std'):.5g}"
            ),
        })
    if float(target.get("missing_percent") or 0) > 0:
        items.append({
            "severity": "warning", "title": "Пропуски в цели",
            "detail": f"{target.get('missing_percent'):.1f}% строк будут исключены из обучения.",
        })
    return _profile_list(items)


def _signature(input_id, scope, task, run_mode, tuning_trials, target, id_column,
               features, method, test_size,
               folds, group_column, time_column, iterations, depth, learning_rate, l2, loss,
               class_weights, early_stopping, random_strength, bagging_temperature,
               random_seed, prediction_column, include_residual, include_confidence,
               compute_shap):
    method = method or "split"
    group_column = group_column if method == "group_cv" else None
    time_column = time_column if method == "time_cv" else None
    return ml_signature(
        input_id=str(input_id or ""), scope=scope or "base",
        task=task or "regression", run_mode=run_mode or "single",
        tuning_trials=int(tuning_trials or 0), target=str(target or ""),
        id_column=str(id_column or ""), features=list(features or []), method=method,
        test_size=float(test_size or 0), folds=int(folds or 0), iterations=int(iterations or 0),
        group_column=str(group_column or ""), time_column=str(time_column or ""),
        depth=int(depth or 0), learning_rate=float(learning_rate or 0),
        l2=float(l2 or 0), loss=loss or "RMSE",
        class_weights=class_weights or "none",
        early_stopping=int(early_stopping or 0),
        random_strength=float(random_strength or 0),
        bagging_temperature=float(bagging_temperature or 0), random_seed=int(random_seed or 0),
        prediction_column=str(prediction_column or ""), include_residual=bool(include_residual),
        include_confidence=bool(include_confidence),
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
    if analysis.get("task") == "classification":
        labels = list(analysis.get("class_labels") or [])
        matrix = analysis.get("confusion_matrix") or []
        if not labels or not matrix:
            return _empty_figure("Обучите модель", template)
        figure = go.Figure(go.Heatmap(
            z=matrix, x=labels, y=labels, colorscale="Blues", showscale=True,
            text=matrix, texttemplate="%{text}",
            hovertemplate="Факт: %{y}<br>Прогноз: %{x}<br>Строк: %{z}<extra></extra>",
        ))
        figure.update_layout(
            template=template or "plotly", height=420,
            title=f"Матрица ошибок · {analysis.get('evaluation_label', '')}",
            xaxis_title="Прогноз", yaxis_title="Факт", yaxis_autorange="reversed",
            margin=dict(l=70, r=20, t=55, b=60),
        )
        return figure
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
    if importance and (values.size == 0 or sample.empty):
        return _importance_figure(
            importance, template, "Среднее |SHAP| по классам", "mean |SHAP|"
        )
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
    if analysis.get("task") == "classification":
        actual = np.asarray(analysis.get("actual_labels") or [], dtype=str)
        predicted = np.asarray(analysis.get("predicted_labels") or [], dtype=str)
        confidence = np.asarray(analysis.get("confidence") or [], dtype=float)
        if actual.size == 0 or confidence.size == 0:
            return _empty_figure("Обучите модель", template)
        correct = actual == predicted
        figure = go.Figure()
        figure.add_trace(go.Histogram(
            x=confidence[correct], name="Верно", marker_color="#40c057", opacity=.72,
            nbinsx=20,
        ))
        figure.add_trace(go.Histogram(
            x=confidence[~correct], name="Ошибка", marker_color="#fa5252", opacity=.72,
            nbinsx=20,
        ))
        figure.update_layout(
            template=template or "plotly", height=420, barmode="overlay",
            title="Уверенность классификации",
            xaxis_title="Максимальная вероятность класса", yaxis_title="Количество",
            margin=dict(l=55, r=20, t=55, b=50),
            legend=dict(orientation="h", y=1.08),
        )
        return figure
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


def _tuning_figure(analysis, template):
    tuning = analysis.get("tuning") or {}
    trials = list(tuning.get("trials") or [])
    if not trials:
        return _empty_figure("Автоподбор параметров не запускался", template)
    metric_name = str(tuning.get("metric_name") or "Метрика")
    best_trial = int(tuning.get("best_trial") or 0)
    custom = []
    for item in trials:
        custom.append(
            f"depth={item.get('depth')}<br>learning rate={item.get('learning_rate'):.4g}"
            f"<br>L2={item.get('l2_leaf_reg'):.4g}"
            f"<br>random strength={item.get('random_strength'):.4g}"
            f"<br>bagging temperature={item.get('bagging_temperature'):.4g}"
            f"<br>деревьев={item.get('best_iteration')}"
        )
    colors = [
        "#40c057" if int(item.get("trial") or 0) == best_trial else "#7950f2"
        for item in trials
    ]
    figure = go.Figure(go.Scatter(
        x=[item.get("trial") for item in trials],
        y=[item.get("score") for item in trials],
        mode="lines+markers", customdata=custom,
        marker=dict(size=9, color=colors), line=dict(width=1.2, color="#7950f2"),
        hovertemplate=(
            f"Попытка %{{x}}<br>{metric_name}: %{{y:.5g}}<br>"
            "%{customdata}<extra></extra>"
        ),
    ))
    direction = "максимум" if tuning.get("higher_is_better") else "минимум"
    figure.update_layout(
        template=template or "plotly", height=420,
        title=(
            f"Автоподбор параметров · {metric_name} ({direction}) · "
            f"лучшая попытка #{best_trial}"
        ),
        xaxis_title="Попытка", yaxis_title=metric_name,
        margin=dict(l=55, r=20, t=55, b=50),
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
    Output("ml-profile-dataset", "data"), Output("ml-profile-dataset", "value"),
    Input("dataset-registry", "data"), Input("active-dataset-id", "data"),
    State("ml-profile-dataset", "value"),
)
def sync_profile_dataset_selector(registry, active_id, current):
    options = dataset_options(registry)
    available = {item["value"] for item in options}
    selected = current if current in available else active_id
    if selected not in available:
        selected = next(iter(available), None)
    return options, selected


@app.callback(
    Output("ml-profile-target", "data"), Output("ml-profile-target", "value"),
    Input("ml-profile-dataset", "value"), Input("dataset-registry", "data"),
    State("ml-profile-target", "value"),
)
def sync_profile_target_options(input_id, registry, current):
    meta = ((get_record(registry, input_id) or {}).get("meta") or {})
    columns = [str(value) for value in meta.get("columns", [])]
    options = [{"label": value, "value": value} for value in columns]
    return options, current if current in columns else None


@app.callback(
    Output("ml-profile-status", "children"), Output("ml-profile-status", "color"),
    Output("ml-profile-rows", "children"), Output("ml-profile-columns", "children"),
    Output("ml-profile-numeric", "children"), Output("ml-profile-categorical", "children"),
    Output("ml-profile-missing", "children"), Output("ml-profile-memory", "children"),
    Output("ml-profile-issues", "children"),
    Output("ml-profile-target-summary", "children"),
    Output("ml-profile-recommendations", "children"),
    Output("ml-profile-missing-graph", "figure"),
    Output("ml-profile-target-graph", "figure"),
    Output("ml-profile-table", "data"), Output("ml-profile-table", "columns"),
    Output("ml-profile-sample-note", "children"),
    Output("ml-profile-analysis", "data"),
    Input("ml-profile-dataset", "value"), Input("ml-profile-scope", "value"),
    Input("ml-profile-target", "value"), Input("ml-profile-task", "value"),
    Input("dataset-registry", "data"), Input("active-dataset-id", "data"),
    Input("filtered-data", "data"), Input("dropdown_style", "value"),
)
def render_data_profile(input_id, scope, target, task_mode, registry, active_id,
                        active_filtered, template):
    payload, meta = input_payload(
        registry, input_id, scope or "base",
        active_id=active_id, active_filtered=active_filtered,
    )
    frame = read_df_from_store(payload, meta) if payload else pd.DataFrame()
    profile = profile_dataset(frame, target=target, task_mode=task_mode or "auto")
    summary = profile["summary"]
    if not len(frame):
        status_label, status_color = "Нет данных", "gray"
    else:
        status_label, status_color = {
            "critical": ("Есть критические сигналы", "red"),
            "warning": ("Требует проверки", "yellow"),
            "ready": ("Готово к моделированию", "green"),
        }.get(profile.get("status"), ("Готово", "green"))

    recommendation_items = [
        {"severity": "info", **item}
        for item in profile.get("recommendations", [])
    ]
    table_rows = [
        {key: value for key, value in row.items() if key != "_risk"}
        for row in profile.get("columns", [])
    ]
    numeric_table_columns = {"Пропуски, %", "Уникальных", "Уникальность, %", "Макс. доля, %"}
    table_columns = [
        {
            "name": name, "id": name,
            **({"type": "numeric"} if name in numeric_table_columns else {}),
        }
        for name in (list(table_rows[0]) if table_rows else [])
    ]
    sample_note = (
        f"Тяжёлые проверки: выборка {summary['sample_rows']:,} из {summary['rows']:,} строк"
        if summary.get("sampled") else f"Проверены все {summary['rows']:,} строк"
    ).replace(",", " ")
    return (
        status_label, status_color,
        f"{summary['rows']:,}".replace(",", " "),
        f"{summary['columns']:,}".replace(",", " "),
        str(summary["numeric"]), str(summary["categorical"]),
        f"{summary['missing_percent']:.1f}%", _format_bytes(summary["memory_bytes"]),
        _profile_list(profile.get("issues", []), empty="Критических сигналов нет"),
        _profile_target_summary(profile),
        _profile_list(recommendation_items, empty="Рекомендации появятся после анализа"),
        missingness_figure(profile, template), target_figure(frame, profile, template),
        table_rows, table_columns, sample_note, profile,
    )


@app.callback(
    Output("ml-target", "data"), Output("ml-target", "value"),
    Output("ml-id-column", "data"), Output("ml-id-column", "value"),
    Output("ml-features", "data"), Output("ml-features", "value"),
    Output("ml-group-column", "data"), Output("ml-group-column", "value"),
    Output("ml-time-column", "data"), Output("ml-time-column", "value"),
    Input("ml-input-dataset", "value"), Input("dataset-registry", "data"),
    Input("ml-task", "value"),
    State("ml-target", "value"), State("ml-id-column", "value"),
    State("ml-features", "value"), State("ml-group-column", "value"),
    State("ml-time-column", "value"),
)
def sync_column_options(input_id, registry, task, target, id_column, selected_features,
                        group_column, time_column):
    meta = ((get_record(registry, input_id) or {}).get("meta") or {})
    columns = [str(value) for value in meta.get("columns", [])]
    numeric = [str(value) for value in meta.get("numeric", [])]
    all_options = [{"label": value, "value": value} for value in columns]
    numeric_options = [{"label": value, "value": value} for value in numeric]
    target_columns = numeric if task != "classification" else columns
    target_options = numeric_options if task != "classification" else all_options
    target = target if target in target_columns else None
    id_column = id_column if id_column in columns else None
    kept = [value for value in (selected_features or []) if value in columns and value != target]
    group_column = group_column if group_column in columns else None
    time_column = time_column if time_column in columns else None
    return (
        target_options, target, all_options, id_column, all_options, kept,
        all_options, group_column, all_options, time_column,
    )


@app.callback(
    Output("ml-workspace-title", "children"), Output("ml-shell-task-badge", "children"),
    Output("ml-target-label", "children"), Output("ml-loss", "data"),
    Output("ml-loss", "value"), Output("ml-class-weights-wrap", "style"),
    Output("ml-residual-wrap", "style"), Output("ml-confidence-wrap", "style"),
    Output("ml-prediction-column", "value"),
    Output("ml-metric-mae-label", "children"), Output("ml-metric-mae-note", "children"),
    Output("ml-metric-rmse-label", "children"), Output("ml-metric-rmse-note", "children"),
    Output("ml-metric-mape-label", "children"), Output("ml-metric-mape-note", "children"),
    Output("ml-metric-r2-label", "children"), Output("ml-metric-r2-note", "children"),
    Output("ml-metric-baseline-label", "children"),
    Output("ml-metric-baseline-note", "children"),
    Input("ml-task", "value"), State("ml-prediction-column", "value"),
)
def configure_task(task, prediction_column):
    classification = task == "classification"
    if classification:
        prediction = (
            "Класс CatBoost"
            if prediction_column in {None, "", "Прогноз CatBoost"}
            else prediction_column
        )
        return (
            "CatBoost · классификация", "Классификация", "Целевой канал · класс",
            ["Auto", "Logloss", "MultiClass"], "Auto", {}, {"display": "none"}, {},
            prediction,
            "Accuracy", "выше — лучше",
            "Balanced accuracy", "учитывает дисбаланс",
            "F1 weighted", "баланс precision / recall",
            "ROC AUC", "binary / multiclass OVR",
            "Baseline accuracy", "мажоритарный класс",
        )
    prediction = (
        "Прогноз CatBoost"
        if prediction_column in {None, "", "Класс CatBoost"}
        else prediction_column
    )
    return (
        "CatBoost · регрессия", "Регрессия", "Целевой канал · число",
        ["RMSE", "MAE", "MAPE", "Quantile"], "RMSE", {"display": "none"}, {},
        {"display": "none"}, prediction,
        "MAE", "ниже — лучше", "RMSE", "ниже — лучше",
        "MAPE", "% · нули исключены", "R²", "выше — лучше",
        "Baseline MAE", "прогноз средним",
    )


@app.callback(
    Output("ml-tuning-trials-wrap", "style"), Output("ml-tuning-hint", "style"),
    Output("ml-run", "children"),
    Input("ml-run-mode", "value"),
)
def configure_run_mode(run_mode):
    if run_mode == "tune":
        return {}, {}, "Автоподбор и обучение"
    return {"display": "none"}, {"display": "none"}, "Обучить модель"


@app.callback(Output("ml-method", "data"), Input("ml-task", "value"))
def validation_method_options(task):
    random_label = (
        "Stratified KFold · классы распределены"
        if task == "classification" else "KFold · случайные фолды"
    )
    split_label = (
        "Train / test · стратифицированное"
        if task == "classification" else "Train / test · случайное"
    )
    return [
        {"label": split_label, "value": "split"},
        {"label": random_label, "value": "cv"},
        {"label": "GroupKFold · группы не смешиваются", "value": "group_cv"},
        {"label": "TimeSeriesSplit · прошлое → будущее", "value": "time_cv"},
    ]


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
    Input("ml-method", "value"), Input("ml-task", "value"),
)
def toggle_validation_controls(method, task):
    classification = task == "classification"
    hints = {
        "split": (
            "Стратифицированная проверка сохраняет доли классов. Для связанных объектов она может завысить качество."
            if classification else
            "Быстрая случайная проверка. Для связанных объектов она может завысить качество."
        ),
        "cv": (
            "Stratified KFold сохраняет доли классов, но не защищает от смешивания связанных объектов."
            if classification else
            "KFold устойчивее одной случайной выборки, но не защищает от смешивания связанных объектов."
        ),
        "group_cv": "Группы не смешиваются между train и validation. Для скважин укажите скважину или месторождение.",
        "time_cv": "Строки сортируются по выбранному каналу; модель проверяется только на будущем относительно train.",
    }
    return (
        method != "split", method == "split", method != "group_cv",
        method != "time_cv", hints.get(method, hints["split"]),
    )


@app.callback(
    Output("ml-preset", "data"), Output("ml-preset", "value"),
    Input("ml-task", "value"), Input("ml-tuning-presets", "data"),
    State("ml-preset", "value"),
)
def preset_options(task, presets, current):
    options = [
        {"label": "Быстрый черновик", "value": "draft"},
        {"label": "Баланс", "value": "balanced"},
        {"label": "Высокое качество", "value": "quality"},
        {"label": "Вручную", "value": "custom"},
    ]
    available = bool((presets or {}).get(task or "regression"))
    if available:
        options.append({
            "label": "Результат автоподбора", "value": "tuning_result",
        })
    valid = {item["value"] for item in options}
    return options, current if current in valid else "balanced"


@app.callback(
    Output("ml-iterations", "value"), Output("ml-depth", "value"),
    Output("ml-learning-rate", "value"), Output("ml-l2", "value"),
    Output("ml-early-stopping", "value"),
    Output("ml-random-strength", "value"),
    Output("ml-bagging-temperature", "value"),
    Input("ml-preset", "value"), State("ml-tuning-presets", "data"),
    State("ml-task", "value"), prevent_initial_call=True,
)
def apply_preset(preset, presets, task):
    values = {
        "draft": (250, 5, .12, 4, 35, 1, 1),
        "balanced": (800, 6, .05, 3, 80, 1, 1),
        "quality": (2000, 8, .025, 3, 180, 1, 1),
    }.get(preset)
    if preset == "tuning_result":
        params = ((presets or {}).get(task or "regression") or {}).get("params") or {}
        if params:
            return (
                params.get("iterations", no_update),
                params.get("depth", no_update),
                params.get("learning_rate", no_update),
                params.get("l2_leaf_reg", no_update),
                no_update,
                params.get("random_strength", no_update),
                params.get("bagging_temperature", no_update),
            )
    if not values:
        return (no_update,) * 7
    return values


@app.callback(
    Output("ml-tuning-presets", "data"),
    Input("ml-analysis", "data"), State("ml-tuning-presets", "data"),
    prevent_initial_call=True,
)
def remember_tuning_preset(store, presets):
    analysis = (store or {}).get("analysis") or {}
    tuning = analysis.get("tuning") or {}
    params = tuning.get("best_params") or {}
    if not params:
        raise PreventUpdate
    task = str(analysis.get("task") or "regression")
    updated = dict(presets or {})
    updated[task] = {
        "params": dict(params),
        "metric_name": tuning.get("metric_name"),
        "metric_value": tuning.get("best_value"),
        "target": analysis.get("target"),
        "created_at": analysis.get("created_at"),
    }
    return updated


@app.callback(
    Output("ml-output-name", "value"), Output("ml-auto-output-name", "data"),
    Input("ml-input-scope", "value"),
    Input("dataset-registry", "data"), Input("ml-task", "value"),
    State("ml-output-name", "value"),
    State("ml-auto-output-name", "data"),
)
def suggest_output_name(scope, registry, task, current, previous_auto):
    operation = (
        "catboost_classification" if task == "classification"
        else "catboost_regression"
    )
    candidate = suggest_dataset_name(
        registry, [{"operation": operation}], scope or "base"
    )
    if not str(current or "").strip() or current == previous_auto:
        return candidate, candidate
    return no_update, candidate


RUN_STATES = [
    State("dataset-registry", "data"), State("active-dataset-id", "data"),
    State("filtered-data", "data"), State("ml-input-dataset", "value"),
    State("ml-input-scope", "value"), State("ml-task", "value"),
    State("ml-run-mode", "value"), State("ml-tuning-trials", "value"),
    State("ml-target", "value"),
    State("ml-id-column", "value"), State("ml-features", "value"),
    State("ml-method", "value"), State("ml-test-size", "value"),
    State("ml-folds", "value"), State("ml-group-column", "value"),
    State("ml-time-column", "value"), State("ml-iterations", "value"),
    State("ml-depth", "value"), State("ml-learning-rate", "value"),
    State("ml-l2", "value"), State("ml-loss", "value"),
    State("ml-class-weights", "value"),
    State("ml-early-stopping", "value"), State("ml-random-strength", "value"),
    State("ml-bagging-temperature", "value"), State("ml-random-seed", "value"),
    State("ml-prediction-column", "value"), State("ml-include-residual", "checked"),
    State("ml-include-confidence", "checked"),
    State("ml-compute-shap", "checked"),
]


@app.callback(
    Output("ml-job-state", "data"),
    Output("ml-job-poll", "disabled"),
    Output("ml-analysis", "data"),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("ml-run", "n_clicks"), *RUN_STATES,
    prevent_initial_call=True,
)
def train_model(_clicks, registry, active_id, filtered_data, input_id, scope, task,
                run_mode, tuning_trials,
                target, id_column, features, method, test_size, folds,
                group_column, time_column, iterations, depth, learning_rate, l2, loss,
                class_weights, early_stopping, random_strength, bagging_temperature, random_seed,
                prediction_column, include_residual, include_confidence, compute_shap):
    if not input_id:
        return no_update, True, no_update, _notification(
            "Сначала загрузите dataset.", color="orange"
        )
    group_column = group_column if method == "group_cv" else None
    time_column = time_column if method == "time_cv" else None
    payload, meta = input_payload(
        registry, input_id, scope or "base",
        active_id=active_id, active_filtered=filtered_data,
    )
    if not payload:
        return no_update, True, no_update, _notification(
            "Входной dataset недоступен.", color="red"
        )
    signature = _signature(
        input_id, scope, task, run_mode, tuning_trials, target, id_column,
        features, method, test_size, folds,
        group_column, time_column,
        iterations, depth, learning_rate, l2, loss, class_weights, early_stopping,
        random_strength, bagging_temperature, random_seed, prediction_column,
        include_residual, include_confidence, compute_shap,
    )
    parameters = {
        "target": target, "features": list(features or []), "id_column": id_column,
        "method": method, "test_size": test_size, "folds": folds,
        "group_column": group_column, "time_column": time_column,
        "iterations": iterations, "depth": depth, "learning_rate": learning_rate,
        "l2_leaf_reg": l2, "loss_function": loss, "random_seed": random_seed,
        "early_stopping_rounds": early_stopping,
        "random_strength": random_strength, "bagging_temperature": bagging_temperature,
        "prediction_column": prediction_column,
        "compute_shap": compute_shap, "signature": signature,
    }
    if task == "classification":
        parameters.update({
            "auto_class_weights": class_weights or "none",
            "include_confidence": include_confidence,
        })
    else:
        parameters["include_residual"] = include_residual

    def execute(report_progress, cancel_event):
        report_progress(1, "Чтение входного dataset")
        frame = read_df_from_store(payload, meta)
        tuning = None
        if run_mode == "tune":
            tuning = tune_catboost_parameters(
                frame, task=task or "regression",
                target=target, features=list(features or []), method=method,
                test_size=test_size, folds=folds, group_column=group_column,
                time_column=time_column, iterations=iterations, depth=depth,
                learning_rate=learning_rate, l2_leaf_reg=l2,
                loss_function=loss, random_seed=random_seed,
                early_stopping_rounds=early_stopping,
                random_strength=random_strength,
                bagging_temperature=bagging_temperature,
                auto_class_weights=class_weights or "none",
                trials=tuning_trials, progress_callback=report_progress,
                cancel_event=cancel_event,
            )
            parameters.update(tuning["best_params"])
        result = get_model_adapter("catboost").run(
            frame, task=task or "regression", **parameters, progress_callback=report_progress,
            cancel_event=cancel_event,
        )
        result.analysis["run_mode"] = run_mode or "single"
        result.analysis["tuning"] = tuning
        result.analysis["resolved_signature"] = signature
        if tuning:
            result.analysis["resolved_signature"] = _signature(
                input_id, scope, task, run_mode, tuning_trials, target, id_column,
                features, method, test_size, folds, group_column, time_column,
                parameters["iterations"], parameters["depth"],
                parameters["learning_rate"], parameters["l2_leaf_reg"], loss,
                class_weights, early_stopping, parameters["random_strength"],
                parameters["bagging_temperature"], random_seed, prediction_column,
                include_residual, include_confidence, compute_shap,
            )
            result.committed_step.setdefault("params", {})["tuning"] = {
                "trials": tuning["trials_count"],
                "best_trial": tuning["best_trial"],
                "metric": tuning["metric_name"],
                "best_value": tuning["best_value"],
            }
        result.committed_step["scope"] = scope or "base"
        return result

    job_id = submit_ml_job(execute)
    snapshot = ml_job_snapshot(job_id) or {
        "job_id": job_id, "status": "queued", "progress": 0, "message": "В очереди",
    }
    snapshot.update({
        "signature": signature, "input_id": str(input_id), "scope": scope or "base",
        "task": task or "regression",
        "run_mode": run_mode or "single",
    })
    message = (
        f"CatBoost: автоподбор {int(tuning_trials or 0)} конфигураций поставлен в очередь."
        if run_mode == "tune" else "CatBoost поставлен в фоновую очередь."
    )
    return snapshot, False, None, _notification(message, notification_id="ml-started")


@app.callback(
    Output("ml-job-state", "data", allow_duplicate=True),
    Output("ml-job-poll", "disabled", allow_duplicate=True),
    Output("ml-analysis", "data", allow_duplicate=True),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("ml-job-poll", "n_intervals"), State("ml-job-state", "data"),
    prevent_initial_call=True,
)
def poll_ml_job(_ticks, job_store):
    job_id = str((job_store or {}).get("job_id") or "")
    if not job_id:
        raise PreventUpdate
    snapshot = ml_job_snapshot(job_id)
    if not snapshot:
        missing = {**(job_store or {}), "status": "failed", "message": "Задание недоступно"}
        return missing, True, no_update, _notification(
            "Фоновое задание потеряно. Запустите обучение повторно.",
            color="red", notification_id="ml-job-missing",
        )
    snapshot.update({
        "signature": (job_store or {}).get("signature"),
        "input_id": (job_store or {}).get("input_id"),
        "scope": (job_store or {}).get("scope") or "base",
        "task": (job_store or {}).get("task") or "regression",
        "run_mode": (job_store or {}).get("run_mode") or "single",
    })
    status = snapshot.get("status")
    if status in {"queued", "running", "cancelling"}:
        return snapshot, False, no_update, no_update
    if status == "cancelled":
        return snapshot, True, no_update, _notification(
            "Обучение отменено.", color="orange", notification_id="ml-cancelled"
        )
    if status == "failed":
        return snapshot, True, no_update, _notification(
            snapshot.get("error") or "Ошибка обучения.",
            color="red", notification_id="ml-error",
        )

    result = take_ml_job_result(job_id)
    if result is None:
        return snapshot, True, no_update, _notification(
            "Результат фонового задания недоступен.",
            color="red", notification_id="ml-result-missing",
        )
    reference = cache_result(result)
    analysis = result.analysis
    store = {
        "reference": reference,
        "signature": snapshot.get("signature"),
        "input_id": snapshot.get("input_id"),
        "scope": snapshot.get("scope") or "base",
        "resolved_signature": analysis.get("resolved_signature"),
        "analysis": analysis,
    }
    metric_name = str(analysis.get("primary_metric_name") or "Метрика")
    metric_value = analysis.get("primary_metric_value")
    metric_text = "—" if metric_value is None else f"{float(metric_value):.4g}"
    message = (
        f"Модель готова · {analysis['training_rows']} строк · "
        f"{len(analysis['features'])} признаков · {metric_name} {metric_text}"
    )
    return snapshot, True, store, _notification(message, notification_id="ml-ready")


@app.callback(
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("ml-cancel", "n_clicks"), State("ml-job-state", "data"),
    prevent_initial_call=True,
)
def cancel_training(_clicks, job_store):
    if not cancel_ml_job((job_store or {}).get("job_id")):
        return _notification(
            "Активного обучения нет.", color="gray", notification_id="ml-cancel-empty"
        )
    return _notification(
        "Запрошена остановка CatBoost.", color="orange", notification_id="ml-cancel-requested"
    )


@app.callback(
    Output("ml-prediction-graph", "figure"), Output("ml-learning-graph", "figure"),
    Output("ml-importance-graph", "figure"), Output("ml-shap-graph", "figure"),
    Output("ml-diagnostics-graph", "figure"),
    Output("ml-tuning-graph", "figure"),
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
        return (
            empty, empty, empty, empty, empty, empty, [], [],
            "—", "—", "—", "—", "—", "", "", "",
        )
    metrics = analysis.get("metrics") or {}
    baseline = analysis.get("baseline") or {}
    def number(value, digits=4, suffix=""):
        return "—" if value is None else f"{float(value):.{digits}g}{suffix}"
    rows = analysis.get("preview") or []
    columns = [{"name": name, "id": name} for name in (rows[0].keys() if rows else [])]
    params = analysis.get("params") or {}
    task = analysis.get("task") or "regression"
    task_label = "Классификация" if task == "classification" else "Регрессия"
    log = "\n".join([
        f"Задача: {task_label}",
        f"Метод: {analysis.get('evaluation_label')}",
        f"Цель: {analysis.get('target')}",
        f"Признаки ({len(analysis.get('features') or [])}): {', '.join(analysis.get('features') or [])}",
        f"Категориальные: {', '.join(analysis.get('categorical_features') or []) or 'нет'}",
        f"Строки: {analysis.get('training_rows')} обучающих из {analysis.get('input_rows')}",
        f"Исключено из обучения из-за пустой цели: {analysis.get('excluded_target_rows')}",
        f"Параметры: {params}",
        f"Финальное количество деревьев: {analysis.get('final_iterations')}",
        *([f"Автоподбор: {analysis['tuning'].get('trials_count')} попыток · "
            f"лучшая #{analysis['tuning'].get('best_trial')} · "
            f"{analysis['tuning'].get('metric_name')} "
            f"{analysis['tuning'].get('best_value'):.5g}"]
          if analysis.get("tuning") else []),
        *([f"Классы ({analysis.get('class_count')}): {', '.join(analysis.get('class_labels') or [])}"]
          if task == "classification" else []),
        "Оценка качества рассчитана только на test/OOF; финальная модель переобучена на всех строках с известной целью.",
    ])
    shown = min(6000, int(analysis.get("evaluation_rows") or 0))
    note = (
        f"{analysis.get('evaluation_label')} · оценено {analysis.get('evaluation_rows')} строк"
        + (f" · на графике выборка {shown}" if shown < analysis.get("evaluation_rows", 0) else "")
    )
    if task == "classification":
        metric_values = (
            number(metrics.get("accuracy")),
            number(metrics.get("balanced_accuracy")),
            number(metrics.get("f1")),
            number(metrics.get("roc_auc")),
            number(baseline.get("accuracy")),
        )
    else:
        metric_values = (
            number(metrics.get("mae")), number(metrics.get("rmse")),
            number(metrics.get("mape"), 4, "%"), number(metrics.get("r2")),
            number(baseline.get("mae")),
        )
    return (
        _prediction_figure(analysis, template), _learning_figure(analysis, template),
        _importance_figure(analysis.get("feature_importance"), template,
                           "Важность признаков CatBoost", "Feature importance"),
        _shap_figure(analysis, template), _diagnostics_figure(analysis, template),
        _tuning_figure(analysis, template),
        rows, columns, *metric_values,
        note, analysis.get("shap_note") or "", log,
    )


SIGNATURE_INPUTS = [
    Input("ml-input-dataset", "value"), Input("ml-input-scope", "value"),
    Input("ml-task", "value"), Input("ml-run-mode", "value"),
    Input("ml-tuning-trials", "value"), Input("ml-target", "value"),
    Input("ml-id-column", "value"),
    Input("ml-features", "value"), Input("ml-method", "value"),
    Input("ml-test-size", "value"), Input("ml-folds", "value"),
    Input("ml-group-column", "value"), Input("ml-time-column", "value"),
    Input("ml-iterations", "value"), Input("ml-depth", "value"),
    Input("ml-learning-rate", "value"), Input("ml-l2", "value"),
    Input("ml-loss", "value"), Input("ml-class-weights", "value"),
    Input("ml-early-stopping", "value"),
    Input("ml-random-strength", "value"), Input("ml-bagging-temperature", "value"),
    Input("ml-random-seed", "value"), Input("ml-prediction-column", "value"),
    Input("ml-include-residual", "checked"),
    Input("ml-include-confidence", "checked"), Input("ml-compute-shap", "checked"),
]


@app.callback(
    Output("ml-run-status", "children"), Output("ml-run-status", "color"),
    Output("ml-commit", "disabled"), Output("ml-export-excel", "disabled"),
    Output("ml-save-model", "disabled"),
    Output("ml-run", "disabled"), Output("ml-cancel", "disabled"),
    Output("ml-row-status", "children"),
    Input("ml-analysis", "data"), Input("ml-job-state", "data"), *SIGNATURE_INPUTS,
)
def validate_current_result(store, job_store, *values):
    job_status = str((job_store or {}).get("status") or "idle")
    if job_status in {"queued", "running", "cancelling"}:
        progress = float((job_store or {}).get("progress") or 0)
        message = str((job_store or {}).get("message") or "Обучение")
        return (
            f"Обучение {progress:.0f}%", "violet", True, True, True,
            True, job_status == "cancelling", message,
        )
    if not store:
        return "Ожидает запуска", "gray", True, True, True, False, True, ""
    current = _signature(*values)
    analysis = store.get("analysis") or {}
    rows = f"{analysis.get('training_rows', 0)} / {analysis.get('input_rows', 0)} строк"
    valid_signatures = {
        str(store.get("signature") or ""),
        str(store.get("resolved_signature") or ""),
    }
    if current not in valid_signatures:
        return (
            "Параметры изменены", "orange", True, True, True,
            False, True, rows + " · требуется пересчёт",
        )
    if not cached_result(store.get("reference")):
        return (
            "Результат устарел", "red", True, True, True,
            False, True, "Повторите обучение",
        )
    return (
        "Модель готова", "green", False, False, False,
        False, True, rows + " · доступны dataset, Excel и CBM",
    )


@app.callback(
    Output("ml-job-progress", "value"), Output("ml-job-message", "children"),
    Input("ml-job-state", "data"),
)
def render_job_progress(job_store):
    status = str((job_store or {}).get("status") or "idle")
    progress = float((job_store or {}).get("progress") or 0)
    message = str((job_store or {}).get("message") or "")
    if status == "idle":
        return 0, ""
    if status == "completed":
        return 100, "Модель готова"
    return progress, message


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
        "task": str(analysis.get("task") or "regression"),
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
        "accuracy": (analysis.get("metrics") or {}).get("accuracy"),
        "balanced_accuracy": (analysis.get("metrics") or {}).get("balanced_accuracy"),
        "f1": (analysis.get("metrics") or {}).get("f1"),
        "roc_auc": (analysis.get("metrics") or {}).get("roc_auc"),
        "primary_metric_name": str(analysis.get("primary_metric_name") or "MAE"),
        "primary_metric_value": analysis.get("primary_metric_value"),
        "higher_is_better": bool(analysis.get("higher_is_better")),
        "run_mode": str(analysis.get("run_mode") or "single"),
        "tuning_trials": int(((analysis.get("tuning") or {}).get("trials_count") or 0)),
        "tuning_best_trial": (analysis.get("tuning") or {}).get("best_trial"),
        "params": dict(analysis.get("params") or {}),
        "created_at": str(analysis.get("created_at") or ""),
    })
    return records[-200:]


@app.callback(
    Output("ml-experiments-table", "data"), Output("ml-experiments-table", "columns"),
    Output("ml-experiments-graph", "figure"), Output("ml-history-count", "children"),
    Output("ml-history-count", "color"), Output("ml-history-best", "children"),
    Output("ml-history-chart-title", "children"),
    Input("ml-experiment-history", "data"), Input("dropdown_style", "value"),
)
def render_experiment_history(history, template):
    records = list(history or [])
    if not records:
        return (
            [], [], _empty_figure("Запусков пока нет", template, height=330),
            "0 запусков", "gray", "Лучшая метрика: —", "Сравнение запусков",
        )

    rows = []
    for index, item in enumerate(reversed(records), 1):
        created = str(item.get("created_at") or "").replace("T", " ")[:19]
        rows.append({
            "№": len(records) - index + 1,
            "Время": created,
            "Модель": item.get("model"),
            "Задача": (
                "Классификация"
                if item.get("task") == "classification" else "Регрессия"
            ),
            "Режим": (
                f"Автоподбор · {item.get('tuning_trials')}"
                if item.get("run_mode") == "tune" else "Один запуск"
            ),
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
            "Accuracy": item.get("accuracy"),
            "Balanced accuracy": item.get("balanced_accuracy"),
            "F1": item.get("f1"),
            "ROC AUC": item.get("roc_auc"),
        })
    columns = [{"name": name, "id": name} for name in rows[0]]

    latest_task = records[-1].get("task") or "regression"
    chart_records = [
        (run_number, item)
        for run_number, item in enumerate(records, 1)
        if (item.get("task") or "regression") == latest_task
    ][-30:]
    metric_name = "Accuracy" if latest_task == "classification" else "MAE"
    metric_key = "accuracy" if latest_task == "classification" else "mae"
    higher_is_better = latest_task == "classification"
    labels, values, hover = [], [], []
    for run_number, item in chart_records:
        if item.get(metric_key) is None:
            continue
        labels.append(f"#{run_number}")
        values.append(float(item[metric_key]))
        hover.append(f"{item.get('model')} · {item.get('target')}<br>{item.get('validation')}")
    if values:
        best_value = max(values) if higher_is_better else min(values)
        colors = ["#40c057" if value == best_value else "#7950f2" for value in values]
        figure = go.Figure(go.Bar(
            x=labels, y=values, marker_color=colors, customdata=hover,
            hovertemplate=f"%{{customdata}}<br>{metric_name}: %{{y:.5g}}<extra></extra>",
        ))
        figure.update_layout(
            template=template or "plotly", height=330,
            margin=dict(l=48, r=18, t=20, b=42),
            xaxis_title="Запуск", yaxis_title=metric_name,
        )
        best_label = f"Лучший {metric_name}: {best_value:.5g}"
    else:
        figure = _empty_figure(f"Нет {metric_name} для сравнения", template, height=330)
        best_label = f"Лучший {metric_name}: —"
    chart_title = f"Сравнение {metric_name} · " + (
        "классификация" if latest_task == "classification" else "регрессия"
    )
    return (
        rows, columns, figure, f"{len(records)} запусков", "violet", best_label,
        chart_title,
    )


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
    current_signature = _signature(*values)
    valid_signatures = {
        str(store.get("signature") or ""),
        str(store.get("resolved_signature") or ""),
    }
    if current_signature not in valid_signatures:
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
        operation = (
            "catboost_classification" if values[2] == "classification"
            else "catboost_regression"
        )
        resolved_name = suggest_dataset_name(
            registry, [{"operation": operation}], scope or "base"
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
