# -*- coding: utf-8 -*-
"""Callbacks for the tabular Neural Network subpage of ML Studio."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, State, no_update
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
from torch_tabular_engine import torch_runtime
from utils import meta_from_df, read_df_from_store


def _nn_signature(input_id, scope, task, target, id_column, features, method,
                  test_size, folds, group_column, time_column, engine,
                  compute_device, hidden_layers,
                  activation, solver, max_iter, learning_rate, alpha,
                  batch_size, early_stopping, validation_fraction, patience,
                  tolerance, min_category_frequency, permutation_repeats,
                  class_balance, random_seed, prediction_column,
                  include_residual, include_confidence):
    method = method or "split"
    return ml_signature(
        model="neural_network", input_id=str(input_id or ""),
        scope=scope or "base", task=task or "regression",
        target=str(target or ""), id_column=str(id_column or ""),
        features=list(features or []), method=method,
        test_size=float(test_size or 0), folds=int(folds or 0),
        group_column=str(group_column or "") if method == "group_cv" else "",
        time_column=str(time_column or "") if method == "time_cv" else "",
        engine=str(engine or "pytorch"), compute_device=str(compute_device or "auto"),
        hidden_layers=str(hidden_layers or ""), activation=str(activation or "relu"),
        solver=str(solver or "adam"), max_iter=int(max_iter or 0),
        learning_rate=float(learning_rate or 0), alpha=float(alpha or 0),
        batch_size=int(batch_size or 0), early_stopping=bool(early_stopping),
        validation_fraction=float(validation_fraction or 0),
        patience=int(patience or 0), tolerance=float(tolerance or 0),
        min_category_frequency=int(min_category_frequency or 1),
        permutation_repeats=int(permutation_repeats or 0),
        class_balance=str(class_balance or "none"),
        random_seed=int(random_seed or 0),
        prediction_column=str(prediction_column or ""),
        include_residual=bool(include_residual),
        include_confidence=bool(include_confidence),
    )


@app.callback(
    Output("nn-input-dataset", "data"), Output("nn-input-dataset", "value"),
    Output("nn-dataset-badge", "children"), Output("nn-dataset-badge", "color"),
    Input("dataset-registry", "data"), Input("active-dataset-id", "data"),
    State("nn-input-dataset", "value"),
)
def sync_nn_dataset_selector(registry, active_id, current):
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
    Output("nn-target", "data"), Output("nn-target", "value"),
    Output("nn-id-column", "data"), Output("nn-id-column", "value"),
    Output("nn-features", "data"), Output("nn-features", "value"),
    Output("nn-group-column", "data"), Output("nn-group-column", "value"),
    Output("nn-time-column", "data"), Output("nn-time-column", "value"),
    Input("nn-input-dataset", "value"), Input("dataset-registry", "data"),
    Input("nn-task", "value"),
    State("nn-target", "value"), State("nn-id-column", "value"),
    State("nn-features", "value"), State("nn-group-column", "value"),
    State("nn-time-column", "value"),
)
def sync_nn_column_options(input_id, registry, task, target, id_column, features,
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
    Output("nn-workspace-title", "children"),
    Output("ml-shell-task-badge", "children", allow_duplicate=True),
    Output("nn-target-label", "children"),
    Output("nn-class-balance-wrap", "style"),
    Output("nn-residual-wrap", "style"), Output("nn-confidence-wrap", "style"),
    Output("nn-prediction-column", "value"),
    Output("nn-metric-mae-label", "children"),
    Output("nn-metric-mae-note", "children"),
    Output("nn-metric-rmse-label", "children"),
    Output("nn-metric-rmse-note", "children"),
    Output("nn-metric-r2-label", "children"),
    Output("nn-metric-r2-note", "children"),
    Output("nn-metric-baseline-label", "children"),
    Output("nn-metric-baseline-note", "children"),
    Input("nn-task", "value"), State("nn-prediction-column", "value"),
    prevent_initial_call=True,
)
def configure_nn_task(task, prediction_column):
    if task == "classification":
        prediction = (
            "Класс Neural Network"
            if prediction_column in {None, "", "Прогноз Neural Network"}
            else prediction_column
        )
        return (
            "Neural Network · классификация", "Классификация",
            "Целевой канал · класс", {}, {"display": "none"}, {}, prediction,
            "Accuracy", "выше — лучше",
            "Balanced accuracy", "учитывает дисбаланс",
            "F1 weighted", "баланс precision / recall",
            "Baseline accuracy", "мажоритарный класс",
        )
    prediction = (
        "Прогноз Neural Network"
        if prediction_column in {None, "", "Класс Neural Network"}
        else prediction_column
    )
    return (
        "Neural Network · регрессия", "Регрессия", "Целевой канал · число",
        {"display": "none"}, {}, {"display": "none"}, prediction,
        "MAE", "ниже — лучше", "RMSE", "ниже — лучше",
        "R²", "выше — лучше", "Baseline MAE", "прогноз средним",
    )


@app.callback(Output("nn-method", "data"), Input("nn-task", "value"))
def nn_validation_options(task):
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
    Output("nn-features", "value", allow_duplicate=True),
    Input("nn-select-numeric", "n_clicks"),
    State("nn-input-dataset", "value"), State("dataset-registry", "data"),
    State("nn-target", "value"), State("nn-id-column", "value"),
    prevent_initial_call=True,
)
def select_nn_numeric(_clicks, input_id, registry, target, id_column):
    meta = ((get_record(registry, input_id) or {}).get("meta") or {})
    return [
        str(value) for value in meta.get("numeric", [])
        if value not in {target, id_column}
    ]


@app.callback(
    Output("nn-test-size", "disabled"), Output("nn-folds", "disabled"),
    Output("nn-group-column", "disabled"), Output("nn-time-column", "disabled"),
    Output("nn-validation-hint", "children"),
    Input("nn-method", "value"), Input("nn-task", "value"),
)
def toggle_nn_validation(method, task):
    hints = {
        "split": "Быстрая проверка на отложенной части данных.",
        "cv": "Несколько обучений дают более устойчивую оценку, но требуют больше времени.",
        "group_cv": "Одна группа не попадёт одновременно в train и validation.",
        "time_cv": "Сеть проверяется только на более поздних строках.",
    }
    if task == "classification" and method == "split":
        hints["split"] = "Стратификация сохраняет доли классов в train и test."
    return (
        method != "split", method == "split", method != "group_cv",
        method != "time_cv", hints.get(method, hints["split"]),
    )


@app.callback(
    Output("nn-hidden-layers", "value"), Output("nn-max-iter", "value"),
    Output("nn-learning-rate", "value"), Output("nn-alpha", "value"),
    Output("nn-batch-size", "value"), Output("nn-patience", "value"),
    Output("nn-validation-fraction", "value"),
    Input("nn-preset", "value"), prevent_initial_call=True,
)
def apply_nn_preset(preset):
    values = {
        "draft": ("32", 200, .003, .001, 64, 15, .15),
        "balanced": ("64, 32", 500, .001, .0001, 64, 30, .15),
        "deep": ("128, 64, 32", 1000, .0005, .001, 128, 50, .2),
    }.get(preset)
    return values if values else (no_update,) * 7


@app.callback(
    Output("nn-compute-device", "value"),
    Output("nn-compute-device", "disabled"),
    Output("nn-compute-hint", "children"), Output("nn-compute-hint", "c"),
    Output("nn-solver", "data"), Output("nn-solver", "value"),
    Input("nn-engine", "value"), Input("nn-input-dataset", "value"),
    Input("dataset-registry", "data"),
    State("nn-compute-device", "value"), State("nn-solver", "value"),
)
def configure_nn_compute(engine, input_id, registry, current_device, current_solver):
    if engine == "sklearn":
        options = [
            {"label": "Adam", "value": "adam"},
            {"label": "SGD", "value": "sgd"},
            {"label": "L-BFGS · малые данные", "value": "lbfgs"},
        ]
        solver = current_solver if current_solver in {"adam", "sgd", "lbfgs"} else "adam"
        return "cpu", True, "sklearn MLP работает только на CPU.", "dimmed", options, solver
    runtime = torch_runtime()
    options = [
        {"label": "Adam", "value": "adam"},
        {"label": "SGD", "value": "sgd"},
    ]
    solver = current_solver if current_solver in {"adam", "sgd"} else "adam"
    device = current_device if current_device in {"auto", "cpu", "mps"} else "auto"
    rows = int((((get_record(registry, input_id) or {}).get("meta") or {}).get("row_count") or 0))
    if runtime["mps_available"]:
        if rows and rows < 1000:
            hint = (
                f"Metal/MPS доступен · {rows:,} строк: CPU обычно быстрее; "
                "Auto всё равно выберет GPU."
            ).replace(",", " ")
            color = "orange"
        else:
            hint = "Metal/MPS доступен · Auto выберет GPU видеоядра Mac."
            color = "teal"
        return device, False, hint, color, options, solver
    if device == "mps":
        device = "auto"
    return device, False, "Metal/MPS не обнаружен · Auto использует CPU.", "orange", options, solver


@app.callback(
    Output("nn-early-stopping", "disabled"),
    Output("nn-patience", "disabled"),
    Output("nn-validation-fraction", "disabled"),
    Input("nn-solver", "value"), Input("nn-early-stopping", "checked"),
)
def toggle_nn_early_stopping(solver, enabled):
    available = solver != "lbfgs"
    active = available and bool(enabled)
    return not available, not active, not active


@app.callback(
    Output("nn-output-name", "value"), Output("nn-auto-output-name", "data"),
    Input("nn-input-scope", "value"), Input("dataset-registry", "data"),
    Input("nn-task", "value"), State("nn-output-name", "value"),
    State("nn-auto-output-name", "data"),
)
def suggest_nn_output_name(scope, registry, task, current, previous_auto):
    operation = (
        "neural_network_classification"
        if task == "classification" else "neural_network_regression"
    )
    candidate = suggest_dataset_name(registry, [{"operation": operation}], scope or "base")
    if not str(current or "").strip() or current == previous_auto:
        return candidate, candidate
    return no_update, candidate


NN_SIGNATURE_INPUTS = [
    Input("nn-input-dataset", "value"), Input("nn-input-scope", "value"),
    Input("nn-task", "value"), Input("nn-target", "value"),
    Input("nn-id-column", "value"), Input("nn-features", "value"),
    Input("nn-method", "value"), Input("nn-test-size", "value"),
    Input("nn-folds", "value"), Input("nn-group-column", "value"),
    Input("nn-time-column", "value"), Input("nn-engine", "value"),
    Input("nn-compute-device", "value"), Input("nn-hidden-layers", "value"),
    Input("nn-activation", "value"), Input("nn-solver", "value"),
    Input("nn-max-iter", "value"), Input("nn-learning-rate", "value"),
    Input("nn-alpha", "value"), Input("nn-batch-size", "value"),
    Input("nn-early-stopping", "checked"),
    Input("nn-validation-fraction", "value"), Input("nn-patience", "value"),
    Input("nn-tolerance", "value"), Input("nn-min-category-frequency", "value"),
    Input("nn-permutation-repeats", "value"), Input("nn-class-balance", "value"),
    Input("nn-random-seed", "value"), Input("nn-prediction-column", "value"),
    Input("nn-include-residual", "checked"),
    Input("nn-include-confidence", "checked"),
]

NN_RUN_STATES = [
    State("dataset-registry", "data"), State("active-dataset-id", "data"),
    State("filtered-data", "data"),
    *[State(item.component_id, item.component_property) for item in NN_SIGNATURE_INPUTS],
]


@app.callback(
    Output("nn-job-state", "data"), Output("nn-job-poll", "disabled"),
    Output("nn-analysis", "data"),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("nn-run", "n_clicks"), *NN_RUN_STATES, prevent_initial_call=True,
)
def train_neural_network(_clicks, registry, active_id, filtered_data, *values):
    (
        input_id, scope, task, target, id_column, features, method, test_size,
        folds, group_column, time_column, engine, compute_device,
        hidden_layers, activation, solver,
        max_iter, learning_rate, alpha, batch_size, early_stopping,
        validation_fraction, patience, tolerance, min_category_frequency,
        permutation_repeats, class_balance, random_seed, prediction_column,
        include_residual, include_confidence,
    ) = values
    if not input_id:
        return no_update, True, no_update, _notification(
            "Сначала загрузите dataset.", color="orange", notification_id="nn-no-data"
        )
    payload, meta = input_payload(
        registry, input_id, scope or "base",
        active_id=active_id, active_filtered=filtered_data,
    )
    if not payload:
        return no_update, True, no_update, _notification(
            "Входной dataset недоступен.", color="red", notification_id="nn-input-error"
        )
    signature = _nn_signature(*values)
    parameters = {
        "target": target, "features": list(features or []), "id_column": id_column,
        "method": method, "test_size": test_size, "folds": folds,
        "group_column": group_column if method == "group_cv" else None,
        "time_column": time_column if method == "time_cv" else None,
        "engine": engine, "compute_device": compute_device,
        "hidden_layers": hidden_layers, "activation": activation, "solver": solver,
        "max_iter": max_iter, "learning_rate_init": learning_rate,
        "alpha": alpha, "batch_size": batch_size,
        "early_stopping": early_stopping,
        "validation_fraction": validation_fraction,
        "n_iter_no_change": patience, "tolerance": tolerance,
        "min_category_frequency": min_category_frequency,
        "permutation_repeats": permutation_repeats,
        "random_seed": random_seed, "prediction_column": prediction_column,
        "signature": signature,
    }
    if task == "classification":
        parameters.update({
            "class_balance": class_balance or "none",
            "include_confidence": include_confidence,
        })
    else:
        parameters["include_residual"] = include_residual

    def execute(report_progress, cancel_event):
        report_progress(1, "Чтение входного dataset")
        frame = read_df_from_store(payload, meta)
        result = get_model_adapter("neural-networks").run(
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
        "signature": signature, "input_id": str(input_id),
        "scope": scope or "base", "task": task or "regression",
        "model": "Neural Network",
    })
    return snapshot, False, None, _notification(
        "Neural Network поставлена в фоновую очередь.", notification_id="nn-started"
    )


@app.callback(
    Output("nn-job-state", "data", allow_duplicate=True),
    Output("nn-job-poll", "disabled", allow_duplicate=True),
    Output("nn-analysis", "data", allow_duplicate=True),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("nn-job-poll", "n_intervals"), State("nn-job-state", "data"),
    prevent_initial_call=True,
)
def poll_neural_network(_ticks, job_store):
    job_id = str((job_store or {}).get("job_id") or "")
    if not job_id:
        raise PreventUpdate
    snapshot = ml_job_snapshot(job_id)
    if not snapshot:
        return {**(job_store or {}), "status": "failed"}, True, no_update, _notification(
            "Фоновое задание недоступно.", color="red", notification_id="nn-job-missing"
        )
    for key in ("signature", "input_id", "scope", "task", "model"):
        snapshot[key] = (job_store or {}).get(key)
    status = snapshot.get("status")
    if status in {"queued", "running", "cancelling"}:
        return snapshot, False, no_update, no_update
    if status == "cancelled":
        return snapshot, True, no_update, _notification(
            "Обучение отменено.", color="orange", notification_id="nn-cancelled"
        )
    if status == "failed":
        return snapshot, True, no_update, _notification(
            snapshot.get("error") or "Ошибка обучения.",
            color="red", notification_id="nn-error",
        )
    result = take_ml_job_result(job_id)
    if result is None:
        return snapshot, True, no_update, _notification(
            "Результат задания недоступен.", color="red",
            notification_id="nn-result-missing",
        )
    reference = cache_result(result)
    analysis = result.analysis
    store = {
        "reference": reference, "signature": snapshot.get("signature"),
        "input_id": snapshot.get("input_id"),
        "scope": snapshot.get("scope") or "base",
        "resolved_signature": analysis.get("resolved_signature"),
        "analysis": analysis,
    }
    metric = analysis.get("primary_metric_value")
    metric_text = "—" if metric is None else f"{float(metric):.4g}"
    return snapshot, True, store, _notification(
        f"Neural Network готова · {analysis['training_rows']} строк · "
        f"{len(analysis['features'])} признаков · "
        f"{analysis.get('primary_metric_name')} {metric_text}",
        notification_id="nn-ready",
    )


@app.callback(
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("nn-cancel", "n_clicks"), State("nn-job-state", "data"),
    prevent_initial_call=True,
)
def cancel_neural_network(_clicks, job_store):
    if not cancel_ml_job((job_store or {}).get("job_id")):
        return _notification(
            "Активного обучения нет.", color="gray", notification_id="nn-cancel-empty"
        )
    return _notification(
        "Запрошена остановка Neural Network.", color="orange",
        notification_id="nn-cancel-requested",
    )


def _learning_figure(analysis, template):
    curve = analysis.get("final_learning_curve") or {}
    epochs = curve.get("epochs") or []
    loss = curve.get("loss") or []
    if not epochs or not loss:
        return _empty_figure("История обучения недоступна", template)
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=epochs, y=loss, mode="lines", name="Train loss",
        line={"color": "#228be6", "width": 2},
        hovertemplate="Эпоха %{x}<br>Loss: %{y:.5g}<extra></extra>",
    ))
    validation = curve.get("validation") or []
    if validation:
        validation_epochs = list(range(1, len(validation) + 1))
        figure.add_trace(go.Scatter(
            x=validation_epochs, y=validation, mode="lines", name="Validation score",
            yaxis="y2", line={"color": "#20c997", "width": 2},
            hovertemplate="Эпоха %{x}<br>Score: %{y:.5g}<extra></extra>",
        ))
    figure.update_layout(
        template=template or "plotly", height=420,
        title=f"Обучение финальной сети · {curve.get('n_iter', len(epochs))} эпох",
        xaxis_title="Эпоха", yaxis_title="Loss",
        yaxis2={
            "title": "Validation score", "overlaying": "y", "side": "right",
            "showgrid": False,
        } if validation else None,
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 55, "r": 55, "t": 60, "b": 50},
    )
    return figure


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
        marker_color="#7950f2",
        hovertemplate=f"%{{x}}<br>{metric_label}: %{{y:.5g}}<extra></extra>",
    ))
    overall = (analysis.get("metrics") or {}).get(metric_key)
    if overall is not None:
        figure.add_hline(
            y=float(overall), line_dash="dot", line_color="#20c997",
            annotation_text=f"Общий {metric_label}: {float(overall):.4g}",
        )
    figure.update_layout(
        template=template or "plotly", height=420,
        title=f"Стабильность проверки · {analysis.get('evaluation_label', '')}",
        xaxis_title="Проверка", yaxis_title=metric_label,
        margin={"l": 55, "r": 20, "t": 55, "b": 50},
    )
    return figure


@app.callback(
    Output("nn-prediction-graph", "figure"), Output("nn-learning-graph", "figure"),
    Output("nn-validation-graph", "figure"), Output("nn-importance-graph", "figure"),
    Output("nn-diagnostics-graph", "figure"),
    Output("nn-prediction-table", "data"), Output("nn-prediction-table", "columns"),
    Output("nn-metric-mae", "children"), Output("nn-metric-rmse", "children"),
    Output("nn-metric-r2", "children"), Output("nn-metric-baseline", "children"),
    Output("nn-metric-epochs", "children"), Output("nn-metric-gap", "children"),
    Output("nn-overfit-status", "children"), Output("nn-overfit-status", "color"),
    Output("nn-evaluation-note", "children"), Output("nn-log", "children"),
    Input("nn-analysis", "data"), Input("dropdown_style", "value"),
)
def render_nn_analysis(store, template):
    analysis = (store or {}).get("analysis") or {}
    if not analysis:
        empty = _empty_figure("Выберите цель и признаки, затем обучите модель", template)
        return (
            empty, empty, empty, empty, empty, [], [],
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
    params = analysis.get("params") or {}
    compute = analysis.get("compute") or {}
    layers = " × ".join(str(value) for value in params.get("hidden_layer_sizes") or [])
    curve = analysis.get("final_learning_curve") or {}
    log = "\n".join([
        f"Модель: Neural Network · {analysis.get('engine') or 'MLP'}",
        f"Задача: {'Классификация' if classification else 'Регрессия'}",
        f"Метод: {analysis.get('evaluation_label')}",
        f"Цель: {analysis.get('target')}",
        f"Архитектура: {layers or '—'}",
        f"Признаки ({len(analysis.get('features') or [])}): {', '.join(analysis.get('features') or [])}",
        f"После кодирования: около {analysis.get('encoded_feature_estimate')} входов",
        f"Категориальные: {', '.join(analysis.get('categorical_features') or []) or 'нет'}",
        f"Строки: {analysis.get('training_rows')} из {analysis.get('input_rows')}",
        f"Эпохи финальной модели: {analysis.get('epochs_run')}",
        f"Early stopping фактически: {'да' if curve.get('early_stopping') else 'нет'}",
        f"Вычислитель: {compute.get('resolved') or 'CPU'}",
        f"PyTorch: {compute.get('torch_version') or 'не используется'}",
        f"Контроль переобучения: {overfitting.get('detail') or 'нет оценки'}",
        f"Параметры: {params}",
        "Качество рассчитано на test/OOF; финальная сеть обучена на всех строках с известной целью.",
    ])
    note = (
        f"{analysis.get('evaluation_label')} · оценено {analysis.get('evaluation_rows')} строк"
        f" · {compute.get('resolved') or 'CPU'}"
    )
    return (
        _prediction_figure(analysis, template), _learning_figure(analysis, template),
        _validation_figure(analysis, template),
        _importance_figure(
            analysis.get("feature_importance"), template,
            "Permutation importance Neural Network", "Снижение качества",
        ),
        _diagnostics_figure(analysis, template), rows, columns,
        *metric_values, str(analysis.get("epochs_run") or "—"),
        "—" if gap is None else f"{float(gap):.1f}{gap_suffix}",
        str(overfitting.get("label") or "Нет оценки"),
        str(overfitting.get("color") or "gray"), note, log,
    )


@app.callback(
    Output("nn-run-status", "children"), Output("nn-run-status", "color"),
    Output("nn-commit", "disabled"), Output("nn-export-excel", "disabled"),
    Output("nn-save-model", "disabled"), Output("nn-run", "disabled"),
    Output("nn-cancel", "disabled"), Output("nn-row-status", "children"),
    Input("nn-analysis", "data"), Input("nn-job-state", "data"),
    *NN_SIGNATURE_INPUTS,
)
def validate_nn_result(store, job_store, *values):
    status = str((job_store or {}).get("status") or "idle")
    if status in {"queued", "running", "cancelling"}:
        progress = float((job_store or {}).get("progress") or 0)
        return (
            f"Обучение {progress:.0f}%", "teal", True, True, True,
            True, status == "cancelling", str((job_store or {}).get("message") or ""),
        )
    if not store:
        return "Ожидает запуска", "gray", True, True, True, False, True, ""
    current = _nn_signature(*values)
    analysis = store.get("analysis") or {}
    rows = f"{analysis.get('training_rows', 0)} / {analysis.get('input_rows', 0)} строк"
    valid = {str(store.get("signature") or ""), str(store.get("resolved_signature") or "")}
    if current not in valid:
        return (
            "Параметры изменены", "orange", True, True, True, False, True,
            rows + " · требуется пересчёт",
        )
    if not cached_result(store.get("reference")):
        return (
            "Результат устарел", "red", True, True, True, False, True,
            "Повторите обучение",
        )
    return (
        "Модель готова", "green", False, False, False, False, True,
        rows + " · доступны dataset, Excel и joblib",
    )


@app.callback(
    Output("nn-job-progress", "value"), Output("nn-job-message", "children"),
    Input("nn-job-state", "data"),
)
def render_nn_progress(job_store):
    status = str((job_store or {}).get("status") or "idle")
    if status == "idle":
        return 0, ""
    if status == "completed":
        return 100, "Модель готова"
    return (
        float((job_store or {}).get("progress") or 0),
        str((job_store or {}).get("message") or ""),
    )


@app.callback(
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("nn-export-excel", "n_clicks"), State("nn-analysis", "data"),
    State("nn-output-name", "value"), State("source-file-path", "data"),
    State("source-file-name", "data"), prevent_initial_call=True,
)
def export_nn_result(_clicks, store, output_name, source_path, source_name):
    cached = cached_result((store or {}).get("reference"))
    if not cached:
        return _notification(
            "Сначала обучите модель.", color="orange", notification_id="nn-export"
        )
    analysis = (store or {}).get("analysis") or {}
    try:
        path = export_frame_to_excel(
            cached.frame, source_path=source_path, source_name=source_name,
            dataset_name=str(output_name or "Neural Network"),
            created_at=analysis.get("created_at"),
        )
    except Exception as error:
        return _notification(str(error), color="red", notification_id="nn-export-error")
    return _notification(
        f"Excel сохранён рядом с исходником: {path}", notification_id="nn-export-ready"
    )


@app.callback(
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("nn-save-model", "n_clicks"), State("nn-analysis", "data"),
    State("nn-output-name", "value"), State("source-file-path", "data"),
    State("source-file-name", "data"), prevent_initial_call=True,
)
def save_nn_model(_clicks, store, output_name, source_path, source_name):
    cached = cached_result((store or {}).get("reference"))
    if not cached:
        return _notification(
            "Сначала обучите модель.", color="orange", notification_id="nn-model-export"
        )
    analysis = (store or {}).get("analysis") or {}
    try:
        path = export_sklearn_model(
            cached.model, source_path=source_path, source_name=source_name,
            experiment_name=str(output_name or "Neural Network"),
            created_at=analysis.get("created_at"),
        )
    except Exception as error:
        return _notification(str(error), color="red", notification_id="nn-model-error")
    return _notification(
        f"Модель Neural Network сохранена: {path}", notification_id="nn-model-ready"
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
    Input("nn-commit", "n_clicks"), State("nn-analysis", "data"),
    State("dataset-registry", "data"), State("nn-output-name", "value"),
    *[State(item.component_id, item.component_property) for item in NN_SIGNATURE_INPUTS],
    prevent_initial_call=True,
)
def commit_nn_prediction(_clicks, store, registry, output_name, *values):
    if not store:
        raise PreventUpdate
    if _nn_signature(*values) not in {
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
            "neural_network_classification"
            if task == "classification" else "neural_network_regression"
        )
        resolved_name = suggest_dataset_name(
            registry, [{"operation": operation}], scope or "base"
        )
    updated, _result_id = commit_result(
        registry, input_id,
        cached.frame.to_json(date_format="iso", orient="split"),
        meta_from_df(cached.frame), cached.committed_step,
        output_mode="new", output_name=resolved_name,
    )
    outputs = len(((store.get("analysis") or {}).get("outputs") or []))
    return (
        updated, no_update, no_update, no_update, no_update, no_update, no_update, no_update,
        _notification(
            f"Записано каналов: {outputs} · создан новый dataset",
            notification_id="nn-committed",
        ),
    )


@app.callback(
    Output("ml-experiment-history", "data", allow_duplicate=True),
    Input("nn-analysis", "data"), State("ml-experiment-history", "data"),
    State("dataset-registry", "data"), prevent_initial_call=True,
)
def remember_nn_experiment(store, history, registry):
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
        "model": "Neural Network", "task": str(analysis.get("task") or "regression"),
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
        "params": dict(analysis.get("params") or {}),
        "compute_device": str((analysis.get("compute") or {}).get("resolved") or "CPU"),
        "overfitting": dict(analysis.get("overfitting") or {}),
        "created_at": str(analysis.get("created_at") or ""),
    })
    return records[-200:]


app.clientside_callback(
    """
    function(value) {
        var target = document.getElementById('nn-features-drop');
        if (!target) return window.dash_clientside.no_update;
        target.setAttribute('data-current-value', JSON.stringify(value || []));
        target.classList.toggle('has-value', Array.isArray(value) && value.length > 0);
        return Date.now();
    }
    """,
    Output("nn-columns-sync", "data"), Input("nn-features", "value"),
)
