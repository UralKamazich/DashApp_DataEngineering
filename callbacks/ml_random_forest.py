# -*- coding: utf-8 -*-
"""Callbacks for the Random Forest subpage of ML Studio."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate

from dash_app import app
from callbacks.ml import (
    _diagnostics_figure,
    _empty_figure,
    _importance_figure,
    _notification,
    _prediction_figure,
)
from dataset_export import export_frame_to_excel, export_sklearn_model
from dataset_registry import (
    commit_result,
    dataset_options,
    get_record,
    input_payload,
    suggest_dataset_name,
)
from ml_engine import cache_result, cached_result, ml_signature
from ml_jobs import cancel_ml_job, ml_job_snapshot, submit_ml_job, take_ml_job_result
from ml_models import get_model_adapter
from utils import meta_from_df, read_df_from_store


def _rf_signature(input_id, scope, task, target, id_column, features, method,
                  test_size, folds, group_column, time_column, n_estimators,
                  max_depth, min_samples_leaf, min_samples_split, max_features,
                  bootstrap, max_samples, oob_score, criterion, class_weight,
                  random_seed, prediction_column, include_residual,
                  include_confidence):
    method = method or "split"
    return ml_signature(
        model="random_forest", input_id=str(input_id or ""), scope=scope or "base",
        task=task or "regression", target=str(target or ""),
        id_column=str(id_column or ""), features=list(features or []), method=method,
        test_size=float(test_size or 0), folds=int(folds or 0),
        group_column=str(group_column or "") if method == "group_cv" else "",
        time_column=str(time_column or "") if method == "time_cv" else "",
        n_estimators=int(n_estimators or 0), max_depth=int(max_depth or 0),
        min_samples_leaf=int(min_samples_leaf or 0),
        min_samples_split=int(min_samples_split or 0),
        max_features=str(max_features or "sqrt"), bootstrap=bool(bootstrap),
        max_samples=float(max_samples or 0), oob_score=bool(oob_score),
        criterion=str(criterion or ""), class_weight=str(class_weight or "none"),
        random_seed=int(random_seed or 0),
        prediction_column=str(prediction_column or ""),
        include_residual=bool(include_residual),
        include_confidence=bool(include_confidence),
    )


@app.callback(
    Output("rf-input-dataset", "data"), Output("rf-input-dataset", "value"),
    Output("rf-dataset-badge", "children"), Output("rf-dataset-badge", "color"),
    Input("dataset-registry", "data"), Input("active-dataset-id", "data"),
    State("rf-input-dataset", "value"),
)
def sync_rf_dataset_selector(registry, active_id, current):
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
    return options, selected, label, "teal"


@app.callback(
    Output("rf-target", "data"), Output("rf-target", "value"),
    Output("rf-id-column", "data"), Output("rf-id-column", "value"),
    Output("rf-features", "data"), Output("rf-features", "value"),
    Output("rf-group-column", "data"), Output("rf-group-column", "value"),
    Output("rf-time-column", "data"), Output("rf-time-column", "value"),
    Input("rf-input-dataset", "value"), Input("dataset-registry", "data"),
    Input("rf-task", "value"),
    State("rf-target", "value"), State("rf-id-column", "value"),
    State("rf-features", "value"), State("rf-group-column", "value"),
    State("rf-time-column", "value"),
)
def sync_rf_column_options(input_id, registry, task, target, id_column, features,
                           group_column, time_column):
    meta = ((get_record(registry, input_id) or {}).get("meta") or {})
    columns = [str(value) for value in meta.get("columns", [])]
    numeric = [str(value) for value in meta.get("numeric", [])]
    all_options = [{"label": value, "value": value} for value in columns]
    target_columns = columns if task == "classification" else numeric
    target_options = [{"label": value, "value": value} for value in target_columns]
    target = target if target in target_columns else None
    kept = [value for value in (features or []) if value in columns and value != target]
    return (
        target_options, target,
        all_options, id_column if id_column in columns else None,
        all_options, kept,
        all_options, group_column if group_column in columns else None,
        all_options, time_column if time_column in columns else None,
    )


@app.callback(
    Output("rf-workspace-title", "children"),
    Output("ml-shell-task-badge", "children", allow_duplicate=True),
    Output("rf-target-label", "children"),
    Output("rf-criterion", "data"), Output("rf-criterion", "value"),
    Output("rf-class-weight-wrap", "style"),
    Output("rf-residual-wrap", "style"), Output("rf-confidence-wrap", "style"),
    Output("rf-prediction-column", "value"),
    Output("rf-metric-mae-label", "children"), Output("rf-metric-mae-note", "children"),
    Output("rf-metric-rmse-label", "children"), Output("rf-metric-rmse-note", "children"),
    Output("rf-metric-r2-label", "children"), Output("rf-metric-r2-note", "children"),
    Output("rf-metric-baseline-label", "children"),
    Output("rf-metric-baseline-note", "children"),
    Output("rf-metric-oob-label", "children"), Output("rf-metric-oob-note", "children"),
    Input("rf-task", "value"), State("rf-prediction-column", "value"),
    prevent_initial_call=True,
)
def configure_rf_task(task, prediction_column):
    if task == "classification":
        prediction = (
            "Класс Random Forest"
            if prediction_column in {None, "", "Прогноз Random Forest"}
            else prediction_column
        )
        return (
            "Random Forest · классификация", "Классификация", "Целевой канал · класс",
            [
                {"label": "Gini", "value": "gini"},
                {"label": "Entropy", "value": "entropy"},
                {"label": "Log loss", "value": "log_loss"},
            ], "gini", {}, {"display": "none"}, {}, prediction,
            "Accuracy", "выше — лучше",
            "Balanced accuracy", "учитывает дисбаланс",
            "F1 weighted", "баланс precision / recall",
            "Baseline accuracy", "мажоритарный класс",
            "OOB accuracy", "вне bootstrap-выборки",
        )
    prediction = (
        "Прогноз Random Forest"
        if prediction_column in {None, "", "Класс Random Forest"}
        else prediction_column
    )
    return (
        "Random Forest · регрессия", "Регрессия", "Целевой канал · число",
        [
            {"label": "Squared error", "value": "squared_error"},
            {"label": "Absolute error", "value": "absolute_error"},
            {"label": "Poisson", "value": "poisson"},
        ], "squared_error", {"display": "none"}, {}, {"display": "none"}, prediction,
        "MAE", "ниже — лучше", "RMSE", "ниже — лучше",
        "R²", "выше — лучше", "Baseline MAE", "прогноз средним",
        "OOB R²", "вне bootstrap-выборки",
    )


@app.callback(Output("rf-method", "data"), Input("rf-task", "value"))
def rf_validation_options(task):
    return [
        {"label": (
            "Train / test · стратифицированное"
            if task == "classification" else "Train / test · случайное"
        ), "value": "split"},
        {"label": (
            "Stratified KFold · классы распределены"
            if task == "classification" else "KFold · случайные фолды"
        ), "value": "cv"},
        {"label": "GroupKFold · группы не смешиваются", "value": "group_cv"},
        {"label": "TimeSeriesSplit · прошлое → будущее", "value": "time_cv"},
    ]


@app.callback(
    Output("rf-features", "value", allow_duplicate=True),
    Input("rf-select-numeric", "n_clicks"),
    State("rf-input-dataset", "value"), State("dataset-registry", "data"),
    State("rf-target", "value"), State("rf-id-column", "value"),
    prevent_initial_call=True,
)
def select_rf_numeric(_clicks, input_id, registry, target, id_column):
    meta = ((get_record(registry, input_id) or {}).get("meta") or {})
    return [str(value) for value in meta.get("numeric", [])
            if value not in {target, id_column}]


@app.callback(
    Output("rf-test-size", "disabled"), Output("rf-folds", "disabled"),
    Output("rf-group-column", "disabled"), Output("rf-time-column", "disabled"),
    Output("rf-validation-hint", "children"),
    Input("rf-method", "value"), Input("rf-task", "value"),
)
def toggle_rf_validation(method, task):
    hints = {
        "split": "Быстрая проверка на отложенной части данных.",
        "cv": "Несколько фолдов дают более устойчивую оценку качества.",
        "group_cv": "Одна группа не попадёт одновременно в train и validation.",
        "time_cv": "Модель проверяется только на более поздних строках.",
    }
    if task == "classification" and method == "split":
        hints["split"] = "Стратификация сохраняет доли классов в train и test."
    return (
        method != "split", method == "split", method != "group_cv",
        method != "time_cv", hints.get(method, hints["split"]),
    )


@app.callback(
    Output("rf-n-estimators", "value"), Output("rf-max-depth", "value"),
    Output("rf-min-samples-leaf", "value"), Output("rf-min-samples-split", "value"),
    Output("rf-max-features", "value"), Output("rf-max-samples", "value"),
    Input("rf-preset", "value"), prevent_initial_call=True,
)
def apply_rf_preset(preset):
    values = {
        "draft": (150, 10, 2, 4, "sqrt", .8),
        "balanced": (600, 0, 2, 2, "sqrt", .85),
        "quality": (1200, 0, 1, 2, "0.8", .9),
    }.get(preset)
    return values if values else (no_update,) * 6


@app.callback(
    Output("rf-max-samples", "disabled"), Output("rf-oob-score", "disabled"),
    Input("rf-bootstrap", "checked"),
)
def toggle_rf_bootstrap(bootstrap):
    return not bootstrap, not bootstrap


@app.callback(
    Output("rf-output-name", "value"), Output("rf-auto-output-name", "data"),
    Input("rf-input-scope", "value"), Input("dataset-registry", "data"),
    Input("rf-task", "value"), State("rf-output-name", "value"),
    State("rf-auto-output-name", "data"),
)
def suggest_rf_output_name(scope, registry, task, current, previous_auto):
    operation = (
        "random_forest_classification"
        if task == "classification" else "random_forest_regression"
    )
    candidate = suggest_dataset_name(registry, [{"operation": operation}], scope or "base")
    if not str(current or "").strip() or current == previous_auto:
        return candidate, candidate
    return no_update, candidate


RF_SIGNATURE_INPUTS = [
    Input("rf-input-dataset", "value"), Input("rf-input-scope", "value"),
    Input("rf-task", "value"), Input("rf-target", "value"),
    Input("rf-id-column", "value"), Input("rf-features", "value"),
    Input("rf-method", "value"), Input("rf-test-size", "value"),
    Input("rf-folds", "value"), Input("rf-group-column", "value"),
    Input("rf-time-column", "value"), Input("rf-n-estimators", "value"),
    Input("rf-max-depth", "value"), Input("rf-min-samples-leaf", "value"),
    Input("rf-min-samples-split", "value"), Input("rf-max-features", "value"),
    Input("rf-bootstrap", "checked"), Input("rf-max-samples", "value"),
    Input("rf-oob-score", "checked"), Input("rf-criterion", "value"),
    Input("rf-class-weight", "value"), Input("rf-random-seed", "value"),
    Input("rf-prediction-column", "value"),
    Input("rf-include-residual", "checked"),
    Input("rf-include-confidence", "checked"),
]

RF_RUN_STATES = [
    State("dataset-registry", "data"), State("active-dataset-id", "data"),
    State("filtered-data", "data"),
    *[State(item.component_id, item.component_property) for item in RF_SIGNATURE_INPUTS],
]


@app.callback(
    Output("rf-job-state", "data"), Output("rf-job-poll", "disabled"),
    Output("rf-analysis", "data"),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("rf-run", "n_clicks"), *RF_RUN_STATES, prevent_initial_call=True,
)
def train_random_forest(_clicks, registry, active_id, filtered_data, *values):
    (
        input_id, scope, task, target, id_column, features, method, test_size,
        folds, group_column, time_column, n_estimators, max_depth,
        min_samples_leaf, min_samples_split, max_features, bootstrap,
        max_samples, oob_score, criterion, class_weight, random_seed,
        prediction_column, include_residual, include_confidence,
    ) = values
    if not input_id:
        return no_update, True, no_update, _notification(
            "Сначала загрузите dataset.", color="orange", notification_id="rf-no-data"
        )
    payload, meta = input_payload(
        registry, input_id, scope or "base",
        active_id=active_id, active_filtered=filtered_data,
    )
    if not payload:
        return no_update, True, no_update, _notification(
            "Входной dataset недоступен.", color="red", notification_id="rf-input-error"
        )
    signature = _rf_signature(*values)
    parameters = {
        "target": target, "features": list(features or []), "id_column": id_column,
        "method": method, "test_size": test_size, "folds": folds,
        "group_column": group_column if method == "group_cv" else None,
        "time_column": time_column if method == "time_cv" else None,
        "n_estimators": n_estimators, "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf, "min_samples_split": min_samples_split,
        "max_features": max_features, "bootstrap": bootstrap,
        "max_samples": max_samples, "oob_score": oob_score,
        "criterion": criterion, "random_seed": random_seed,
        "prediction_column": prediction_column, "signature": signature,
    }
    if task == "classification":
        parameters.update({
            "class_weight": class_weight or "none",
            "include_confidence": include_confidence,
        })
    else:
        parameters["include_residual"] = include_residual

    def execute(report_progress, cancel_event):
        report_progress(1, "Чтение входного dataset")
        frame = read_df_from_store(payload, meta)
        result = get_model_adapter("random-forest").run(
            frame, task=task or "regression", **parameters,
            progress_callback=report_progress, cancel_event=cancel_event,
        )
        result.analysis["resolved_signature"] = signature
        result.analysis["run_mode"] = "single"
        result.committed_step["scope"] = scope or "base"
        return result

    job_id = submit_ml_job(execute)
    snapshot = ml_job_snapshot(job_id) or {
        "job_id": job_id, "status": "queued", "progress": 0, "message": "В очереди",
    }
    snapshot.update({
        "signature": signature, "input_id": str(input_id), "scope": scope or "base",
        "task": task or "regression", "model": "Random Forest",
    })
    return snapshot, False, None, _notification(
        "Random Forest поставлен в фоновую очередь.", notification_id="rf-started"
    )


@app.callback(
    Output("rf-job-state", "data", allow_duplicate=True),
    Output("rf-job-poll", "disabled", allow_duplicate=True),
    Output("rf-analysis", "data", allow_duplicate=True),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("rf-job-poll", "n_intervals"), State("rf-job-state", "data"),
    prevent_initial_call=True,
)
def poll_random_forest(_ticks, job_store):
    job_id = str((job_store or {}).get("job_id") or "")
    if not job_id:
        raise PreventUpdate
    snapshot = ml_job_snapshot(job_id)
    if not snapshot:
        return {**(job_store or {}), "status": "failed"}, True, no_update, _notification(
            "Фоновое задание недоступно.", color="red", notification_id="rf-job-missing"
        )
    for key in ("signature", "input_id", "scope", "task", "model"):
        snapshot[key] = (job_store or {}).get(key)
    status = snapshot.get("status")
    if status in {"queued", "running", "cancelling"}:
        return snapshot, False, no_update, no_update
    if status == "cancelled":
        return snapshot, True, no_update, _notification(
            "Обучение отменено.", color="orange", notification_id="rf-cancelled"
        )
    if status == "failed":
        return snapshot, True, no_update, _notification(
            snapshot.get("error") or "Ошибка обучения.",
            color="red", notification_id="rf-error",
        )
    result = take_ml_job_result(job_id)
    if result is None:
        return snapshot, True, no_update, _notification(
            "Результат задания недоступен.", color="red", notification_id="rf-result-missing"
        )
    reference = cache_result(result)
    analysis = result.analysis
    store = {
        "reference": reference, "signature": snapshot.get("signature"),
        "input_id": snapshot.get("input_id"), "scope": snapshot.get("scope") or "base",
        "resolved_signature": analysis.get("resolved_signature"), "analysis": analysis,
    }
    metric = analysis.get("primary_metric_value")
    metric_text = "—" if metric is None else f"{float(metric):.4g}"
    return snapshot, True, store, _notification(
        f"Random Forest готов · {analysis['training_rows']} строк · "
        f"{len(analysis['features'])} признаков · "
        f"{analysis.get('primary_metric_name')} {metric_text}",
        notification_id="rf-ready",
    )


@app.callback(
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("rf-cancel", "n_clicks"), State("rf-job-state", "data"),
    prevent_initial_call=True,
)
def cancel_random_forest(_clicks, job_store):
    if not cancel_ml_job((job_store or {}).get("job_id")):
        return _notification("Активного обучения нет.", color="gray", notification_id="rf-cancel-empty")
    return _notification(
        "Запрошена остановка Random Forest.", color="orange",
        notification_id="rf-cancel-requested",
    )


def _validation_figure(analysis, template):
    folds = list(analysis.get("fold_metrics") or [])
    if not folds:
        return _empty_figure("Нет результатов валидации", template)
    classification = analysis.get("task") == "classification"
    metric_key = "balanced_accuracy" if classification else "mae"
    metric_label = "Balanced accuracy" if classification else "MAE"
    values = [item.get(metric_key) for item in folds]
    figure = go.Figure(go.Bar(
        x=[f"Fold {item.get('fold')}" for item in folds], y=values,
        marker_color="#20c997",
        hovertemplate=f"%{{x}}<br>{metric_label}: %{{y:.5g}}<extra></extra>",
    ))
    overall = (analysis.get("metrics") or {}).get(metric_key)
    if overall is not None:
        figure.add_hline(
            y=float(overall), line_dash="dot", line_color="#7950f2",
            annotation_text=f"Общий {metric_label}: {float(overall):.4g}",
        )
    figure.update_layout(
        template=template or "plotly", height=420,
        title=f"Стабильность проверки · {analysis.get('evaluation_label', '')}",
        xaxis_title="Проверка", yaxis_title=metric_label,
        margin=dict(l=55, r=20, t=55, b=50),
    )
    return figure


@app.callback(
    Output("rf-prediction-graph", "figure"), Output("rf-validation-graph", "figure"),
    Output("rf-importance-graph", "figure"), Output("rf-diagnostics-graph", "figure"),
    Output("rf-prediction-table", "data"), Output("rf-prediction-table", "columns"),
    Output("rf-metric-mae", "children"), Output("rf-metric-rmse", "children"),
    Output("rf-metric-r2", "children"), Output("rf-metric-baseline", "children"),
    Output("rf-metric-oob", "children"), Output("rf-metric-gap", "children"),
    Output("rf-overfit-status", "children"), Output("rf-overfit-status", "color"),
    Output("rf-evaluation-note", "children"), Output("rf-log", "children"),
    Input("rf-analysis", "data"), Input("dropdown_style", "value"),
)
def render_rf_analysis(store, template):
    analysis = (store or {}).get("analysis") or {}
    if not analysis:
        empty = _empty_figure("Выберите цель и признаки, затем обучите модель", template)
        return (
            empty, empty, empty, empty, [], [],
            "—", "—", "—", "—", "—", "—", "Нет оценки", "gray", "", "",
        )
    metrics = analysis.get("metrics") or {}
    baseline = analysis.get("baseline") or {}
    overfitting = analysis.get("overfitting") or {}
    classification = analysis.get("task") == "classification"
    number = lambda value: "—" if value is None else f"{float(value):.4g}"
    if classification:
        metric_values = (
            number(metrics.get("accuracy")), number(metrics.get("balanced_accuracy")),
            number(metrics.get("f1")), number(baseline.get("accuracy")),
        )
        gap_suffix = " п.п."
    else:
        metric_values = (
            number(metrics.get("mae")), number(metrics.get("rmse")),
            number(metrics.get("r2")), number(baseline.get("mae")),
        )
        gap_suffix = "%"
    gap = overfitting.get("gap_percent")
    rows = analysis.get("preview") or []
    columns = [{"name": name, "id": name} for name in (rows[0].keys() if rows else [])]
    oob = analysis.get("oob_score")
    log = "\n".join([
        f"Модель: Random Forest",
        f"Задача: {'Классификация' if classification else 'Регрессия'}",
        f"Метод: {analysis.get('evaluation_label')}",
        f"Цель: {analysis.get('target')}",
        f"Признаки ({len(analysis.get('features') or [])}): {', '.join(analysis.get('features') or [])}",
        f"Категориальные: {', '.join(analysis.get('categorical_features') or []) or 'нет'}",
        f"Строки: {analysis.get('training_rows')} из {analysis.get('input_rows')}",
        f"Вычислитель: CPU · все доступные ядра",
        f"OOB: {number(oob)}",
        f"Контроль переобучения: {overfitting.get('detail') or 'нет оценки'}",
        f"Параметры: {analysis.get('params')}",
        "Оценка качества рассчитана на test/OOF; финальная модель обучена на всех строках с известной целью.",
    ])
    note = (
        f"{analysis.get('evaluation_label')} · оценено {analysis.get('evaluation_rows')} строк"
        " · CPU"
    )
    return (
        _prediction_figure(analysis, template), _validation_figure(analysis, template),
        _importance_figure(
            analysis.get("feature_importance"), template,
            "Важность признаков Random Forest", "Суммарная важность",
        ),
        _diagnostics_figure(analysis, template), rows, columns,
        *metric_values, number(oob),
        "—" if gap is None else f"{float(gap):.1f}{gap_suffix}",
        str(overfitting.get("label") or "Нет оценки"),
        str(overfitting.get("color") or "gray"), note, log,
    )


@app.callback(
    Output("rf-run-status", "children"), Output("rf-run-status", "color"),
    Output("rf-commit", "disabled"), Output("rf-export-excel", "disabled"),
    Output("rf-save-model", "disabled"), Output("rf-run", "disabled"),
    Output("rf-cancel", "disabled"), Output("rf-row-status", "children"),
    Input("rf-analysis", "data"), Input("rf-job-state", "data"),
    *RF_SIGNATURE_INPUTS,
)
def validate_rf_result(store, job_store, *values):
    status = str((job_store or {}).get("status") or "idle")
    if status in {"queued", "running", "cancelling"}:
        progress = float((job_store or {}).get("progress") or 0)
        return (
            f"Обучение {progress:.0f}%", "teal", True, True, True,
            True, status == "cancelling", str((job_store or {}).get("message") or ""),
        )
    if not store:
        return "Ожидает запуска", "gray", True, True, True, False, True, ""
    current = _rf_signature(*values)
    analysis = store.get("analysis") or {}
    rows = f"{analysis.get('training_rows', 0)} / {analysis.get('input_rows', 0)} строк"
    valid = {str(store.get("signature") or ""), str(store.get("resolved_signature") or "")}
    if current not in valid:
        return "Параметры изменены", "orange", True, True, True, False, True, rows + " · требуется пересчёт"
    if not cached_result(store.get("reference")):
        return "Результат устарел", "red", True, True, True, False, True, "Повторите обучение"
    return "Модель готова", "green", False, False, False, False, True, rows + " · доступны dataset, Excel и joblib"


@app.callback(
    Output("rf-job-progress", "value"), Output("rf-job-message", "children"),
    Input("rf-job-state", "data"),
)
def render_rf_progress(job_store):
    status = str((job_store or {}).get("status") or "idle")
    if status == "idle":
        return 0, ""
    if status == "completed":
        return 100, "Модель готова"
    return float((job_store or {}).get("progress") or 0), str((job_store or {}).get("message") or "")


@app.callback(
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("rf-export-excel", "n_clicks"), State("rf-analysis", "data"),
    State("rf-output-name", "value"), State("source-file-path", "data"),
    State("source-file-name", "data"), prevent_initial_call=True,
)
def export_rf_result(_clicks, store, output_name, source_path, source_name):
    cached = cached_result((store or {}).get("reference"))
    if not cached:
        return _notification("Сначала обучите модель.", color="orange", notification_id="rf-export")
    analysis = (store or {}).get("analysis") or {}
    try:
        path = export_frame_to_excel(
            cached.frame, source_path=source_path, source_name=source_name,
            dataset_name=str(output_name or "Random Forest"),
            created_at=analysis.get("created_at"),
        )
    except Exception as error:
        return _notification(str(error), color="red", notification_id="rf-export-error")
    return _notification(f"Excel сохранён рядом с исходником: {path}", notification_id="rf-export-ready")


@app.callback(
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("rf-save-model", "n_clicks"), State("rf-analysis", "data"),
    State("rf-output-name", "value"), State("source-file-path", "data"),
    State("source-file-name", "data"), prevent_initial_call=True,
)
def save_rf_model(_clicks, store, output_name, source_path, source_name):
    cached = cached_result((store or {}).get("reference"))
    if not cached:
        return _notification("Сначала обучите модель.", color="orange", notification_id="rf-model-export")
    analysis = (store or {}).get("analysis") or {}
    try:
        path = export_sklearn_model(
            cached.model, source_path=source_path, source_name=source_name,
            experiment_name=str(output_name or "Random Forest"),
            created_at=analysis.get("created_at"),
        )
    except Exception as error:
        return _notification(str(error), color="red", notification_id="rf-model-error")
    return _notification(f"Модель Random Forest сохранена: {path}", notification_id="rf-model-ready")


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
    Input("rf-commit", "n_clicks"), State("rf-analysis", "data"),
    State("dataset-registry", "data"), State("rf-output-name", "value"),
    *[State(item.component_id, item.component_property) for item in RF_SIGNATURE_INPUTS],
    prevent_initial_call=True,
)
def commit_rf_prediction(_clicks, store, registry, output_name, *values):
    if not store:
        raise PreventUpdate
    if _rf_signature(*values) not in {
        str(store.get("signature") or ""), str(store.get("resolved_signature") or ""),
    }:
        return *((no_update,) * 8), _notification(
            "Параметры изменились. Повторите обучение.", color="orange"
        )
    cached = cached_result(store.get("reference"))
    if not cached:
        return *((no_update,) * 8), _notification(
            "Результат недоступен. Повторите обучение.", color="red"
        )
    input_id, scope, task = values[0], values[1], values[2]
    resolved_name = str(output_name or "").strip()
    if not resolved_name:
        operation = (
            "random_forest_classification"
            if task == "classification" else "random_forest_regression"
        )
        resolved_name = suggest_dataset_name(registry, [{"operation": operation}], scope or "base")
    updated, _result_id = commit_result(
        registry, input_id,
        cached.frame.to_json(date_format="iso", orient="split"),
        meta_from_df(cached.frame), cached.committed_step,
        output_mode="new", output_name=resolved_name,
    )
    outputs = len(((store.get("analysis") or {}).get("outputs") or []))
    return (
        updated, no_update, no_update, no_update, no_update, no_update, no_update, no_update,
        _notification(f"Записано каналов: {outputs} · создан новый dataset", notification_id="rf-committed"),
    )


@app.callback(
    Output("ml-experiment-history", "data", allow_duplicate=True),
    Input("rf-analysis", "data"), State("ml-experiment-history", "data"),
    State("dataset-registry", "data"), prevent_initial_call=True,
)
def remember_rf_experiment(store, history, registry):
    analysis = (store or {}).get("analysis") or {}
    reference = str((store or {}).get("reference") or "")
    if not analysis or not reference:
        raise PreventUpdate
    records = list(history or [])
    if any(str(item.get("reference") or "") == reference for item in records):
        raise PreventUpdate
    input_id = str((store or {}).get("input_id") or "")
    record = get_record(registry, input_id) or {}
    metrics = analysis.get("metrics") or {}
    records.append({
        "reference": reference, "signature": str((store or {}).get("signature") or ""),
        "model": "Random Forest", "task": str(analysis.get("task") or "regression"),
        "dataset": str(record.get("name") or input_id), "dataset_id": input_id,
        "scope": str((store or {}).get("scope") or "base"),
        "target": str(analysis.get("target") or ""),
        "features": list(analysis.get("features") or []),
        "validation": str(analysis.get("evaluation_label") or ""),
        "training_rows": int(analysis.get("training_rows") or 0),
        "evaluation_rows": int(analysis.get("evaluation_rows") or 0),
        "feature_count": len(analysis.get("features") or []),
        "mae": metrics.get("mae"), "rmse": metrics.get("rmse"),
        "r2": metrics.get("r2"), "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "f1": metrics.get("f1"), "roc_auc": metrics.get("roc_auc"),
        "primary_metric_name": str(analysis.get("primary_metric_name") or "MAE"),
        "primary_metric_value": analysis.get("primary_metric_value"),
        "higher_is_better": bool(analysis.get("higher_is_better")),
        "run_mode": "Один запуск", "tuning_trials": 0,
        "params": dict(analysis.get("params") or {}), "compute_device": "CPU",
        "overfitting": dict(analysis.get("overfitting") or {}),
        "created_at": str(analysis.get("created_at") or ""),
    })
    return records[-200:]


app.clientside_callback(
    """
    function(value) {
        var target = document.getElementById('rf-features-drop');
        if (!target) return window.dash_clientside.no_update;
        target.setAttribute('data-current-value', JSON.stringify(value || []));
        target.classList.toggle('has-value', Array.isArray(value) && value.length > 0);
        return Date.now();
    }
    """,
    Output("rf-columns-sync", "data"), Input("rf-features", "value"),
)
