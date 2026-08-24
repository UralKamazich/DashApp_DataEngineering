# -*- coding: utf-8 -*-
"""CatBoost regression/classification and a short-lived server result cache."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from uuid import uuid4

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)


MAX_SHAP_ROWS = 300
MAX_CACHED_RESULTS = 12


@lru_cache(maxsize=1)
def available_gpu_count():
    """Return the CUDA devices visible to the installed CatBoost runtime."""
    try:
        return max(0, int(get_gpu_device_count()))
    except Exception:
        return 0


def resolve_compute_device(requested="auto"):
    requested = str(requested or "auto").strip().lower()
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("Вычислитель должен быть Auto, CPU или GPU.")
    gpu_count = available_gpu_count()
    if requested == "gpu" and gpu_count < 1:
        raise ValueError(
            "CatBoost не обнаружил CUDA‑GPU. Выберите Auto или CPU. "
            "На macOS обучение CatBoost выполняется на CPU."
        )
    resolved = "GPU" if requested == "gpu" or (requested == "auto" and gpu_count > 0) else "CPU"
    return {
        "requested": requested,
        "resolved": resolved,
        "gpu_count": gpu_count,
    }


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Обучение отменено.")


class _CatBoostProgressCallback:
    def __init__(self, report, cancel_event, *, start, span, expected_iterations, label):
        self.report = report
        self.cancel_event = cancel_event
        self.start = float(start)
        self.span = float(span)
        self.expected_iterations = max(1, int(expected_iterations or 1))
        self.label = label

    def after_iteration(self, info):
        iteration = max(1, int(getattr(info, "iteration", 1)))
        fraction = min(1.0, iteration / self.expected_iterations)
        if self.report:
            self.report(self.start + self.span * fraction, self.label)
        return not (self.cancel_event is not None and self.cancel_event.is_set())


def _fit_model(model, train_pool, *, eval_set=None, fit_kwargs=None,
               progress_callback=None, cancel_event=None, progress_start=0,
               progress_span=1, expected_iterations=1, label="Обучение"):
    _check_cancel(cancel_event)
    kwargs = dict(fit_kwargs or {})
    if eval_set is not None:
        kwargs["eval_set"] = eval_set
    task_type = str((model.get_params() or {}).get("task_type") or "CPU").upper()
    if task_type != "GPU" and (progress_callback or cancel_event is not None):
        kwargs["callbacks"] = [_CatBoostProgressCallback(
            progress_callback, cancel_event,
            start=progress_start, span=progress_span,
            expected_iterations=expected_iterations, label=label,
        )]
    elif task_type == "GPU" and progress_callback:
        progress_callback(progress_start, f"{label} · GPU")
    model.fit(train_pool, **kwargs)
    _check_cancel(cancel_event)
    if progress_callback:
        progress_callback(progress_start + progress_span, label)


@dataclass
class CachedMLResult:
    frame: pd.DataFrame
    analysis: dict
    committed_step: dict
    signature: str
    model: object


_RESULT_CACHE: dict[str, CachedMLResult] = {}


def ml_signature(**parameters) -> str:
    normalized = json.dumps(parameters, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def cache_result(result: CachedMLResult) -> str:
    reference = uuid4().hex
    _RESULT_CACHE[reference] = result
    while len(_RESULT_CACHE) > MAX_CACHED_RESULTS:
        oldest = next(iter(_RESULT_CACHE))
        _RESULT_CACHE.pop(oldest, None)
    return reference


def cached_result(reference) -> CachedMLResult | None:
    return _RESULT_CACHE.get(str(reference or ""))


def _unique_name(columns, base):
    name = str(base or "Прогноз").strip() or "Прогноз"
    if name not in columns:
        return name
    index = 2
    while f"{name}_{index}" in columns:
        index += 1
    return f"{name}_{index}"


def _mape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual != 0)
    if not valid.any():
        return None
    return float(np.mean(np.abs((actual[valid] - predicted[valid]) / actual[valid])) * 100)


def _metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mse = float(mean_squared_error(actual, predicted))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mape": _mape(actual, predicted),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def _classification_metrics(actual, predicted, probabilities=None, labels=None):
    actual = np.asarray(actual, dtype=str)
    predicted = np.asarray(predicted, dtype=str)
    result = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
        "logloss": None,
        "roc_auc": None,
    }
    if probabilities is None or labels is None:
        return result
    probabilities = np.asarray(probabilities, dtype=float)
    labels = [str(value) for value in labels]
    try:
        result["logloss"] = float(log_loss(actual, probabilities, labels=labels))
    except ValueError:
        pass
    try:
        if len(labels) == 2:
            binary_actual = (actual == labels[1]).astype(int)
            if len(np.unique(binary_actual)) == 2:
                result["roc_auc"] = float(roc_auc_score(binary_actual, probabilities[:, 1]))
        elif len(labels) > 2 and len(np.unique(actual)) == len(labels):
            result["roc_auc"] = float(roc_auc_score(
                actual, probabilities, labels=labels,
                multi_class="ovr", average="weighted",
            ))
    except ValueError:
        pass
    return result


def _prepare_features(frame, features):
    prepared = frame[list(features)].copy()
    categorical = []
    for index, column in enumerate(features):
        series = prepared[column]
        if pd.api.types.is_numeric_dtype(series):
            prepared[column] = pd.to_numeric(series, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        else:
            categorical.append(index)
            prepared[column] = series.astype("string").fillna("<NA>").astype(str)
    return prepared, categorical


def _pool(x, y, categorical):
    return Pool(x, y, cat_features=categorical or None)


def _model_params(*, iterations, depth, learning_rate, l2_leaf_reg,
                  loss_function, random_seed, random_strength,
                  bagging_temperature, compute_device="auto"):
    if int(iterations) < 1 or int(iterations) > 20_000:
        raise ValueError("Количество деревьев должно быть от 1 до 20 000.")
    if int(depth) < 1 or int(depth) > 16:
        raise ValueError("Глубина дерева должна быть от 1 до 16.")
    if not 0 < float(learning_rate) <= 1:
        raise ValueError("Скорость обучения должна быть в диапазоне (0; 1].")
    if float(l2_leaf_reg) < 0:
        raise ValueError("L2-регуляризация не может быть отрицательной.")
    if float(random_strength or 0) < 0 or float(bagging_temperature or 0) < 0:
        raise ValueError("Random strength и bagging temperature не могут быть отрицательными.")
    if loss_function not in {"RMSE", "MAE", "MAPE", "Quantile"}:
        raise ValueError("Выбрана неподдерживаемая функция потерь.")
    compute = resolve_compute_device(compute_device)
    return {
        "iterations": int(iterations),
        "depth": int(depth),
        "learning_rate": float(learning_rate),
        "l2_leaf_reg": float(l2_leaf_reg),
        "loss_function": loss_function,
        "random_seed": int(random_seed),
        "random_strength": float(random_strength or 0),
        "bagging_temperature": float(bagging_temperature or 0),
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": -1,
        "task_type": compute["resolved"],
    }


def _classifier_params(*, iterations, depth, learning_rate, l2_leaf_reg,
                       loss_function, random_seed, random_strength,
                       bagging_temperature, auto_class_weights, class_count,
                       compute_device="auto"):
    resolved_loss = str(loss_function or "Auto")
    if resolved_loss == "Auto":
        resolved_loss = "Logloss" if int(class_count) == 2 else "MultiClass"
    if resolved_loss not in {"Logloss", "MultiClass"}:
        raise ValueError("Выбрана неподдерживаемая функция потерь классификации.")
    if int(class_count) > 2 and resolved_loss == "Logloss":
        raise ValueError("Для многоклассовой цели используйте Auto или MultiClass.")
    if int(class_count) == 2 and resolved_loss == "MultiClass":
        raise ValueError("Для бинарной цели используйте Auto или Logloss.")
    base = _model_params(
        iterations=iterations, depth=depth, learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg, loss_function="RMSE",
        random_seed=random_seed, random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        compute_device=compute_device,
    )
    base["loss_function"] = resolved_loss
    if auto_class_weights == "balanced":
        base["auto_class_weights"] = "Balanced"
    return base


def _prediction_labels(values):
    return np.asarray(values, dtype=object).reshape(-1).astype(str)


def _curve(model, label):
    result = model.get_evals_result() or {}
    validation = result.get("validation") or result.get("validation_0") or {}
    learn = result.get("learn") or {}
    metric_name = next(iter(validation), None)
    if not metric_name:
        return None
    validation_values = validation.get(metric_name) or []
    learn_values = learn.get(metric_name) or []
    return {
        "label": label,
        "metric": metric_name,
        "iterations": list(range(1, len(validation_values) + 1)),
        "validation": [float(value) for value in validation_values],
        "learn": [float(value) for value in learn_values],
    }


def _overfitting_summary(task, training_metrics, validation_metrics):
    """Build a conservative train/validation gap diagnostic."""
    if task == "classification":
        metric = "Balanced accuracy"
        train_value = training_metrics.get("balanced_accuracy")
        validation_value = validation_metrics.get("balanced_accuracy")
        if train_value is None or validation_value is None:
            return {"status": "unknown", "label": "Нет оценки", "color": "gray"}
        gap = max(0.0, float(train_value) - float(validation_value))
        gap_percent = gap * 100
        if gap_percent <= 5:
            status, label, color = "low", "Низкий риск", "green"
        elif gap_percent <= 15:
            status, label, color = "moderate", "Умеренный риск", "yellow"
        else:
            status, label, color = "high", "Высокий риск", "red"
        detail = (
            f"{metric}: train {float(train_value):.4g} → validation "
            f"{float(validation_value):.4g} · разрыв {gap_percent:.1f} п.п."
        )
    else:
        metric = "MAE"
        train_value = training_metrics.get("mae")
        validation_value = validation_metrics.get("mae")
        if train_value is None or validation_value is None:
            return {"status": "unknown", "label": "Нет оценки", "color": "gray"}
        denominator = max(abs(float(validation_value)), 1e-12)
        gap_percent = max(0.0, (float(validation_value) - float(train_value)) / denominator * 100)
        if gap_percent <= 15:
            status, label, color = "low", "Низкий риск", "green"
        elif gap_percent <= 35:
            status, label, color = "moderate", "Умеренный риск", "yellow"
        else:
            status, label, color = "high", "Высокий риск", "red"
        detail = (
            f"{metric}: train {float(train_value):.4g} → validation "
            f"{float(validation_value):.4g} · разрыв {gap_percent:.1f}%"
        )
    return {
        "status": status, "label": label, "color": color,
        "metric": metric, "train": float(train_value),
        "validation": float(validation_value), "gap_percent": float(gap_percent),
        "detail": detail,
    }


def run_catboost_regression(
    frame: pd.DataFrame,
    *,
    target: str,
    features,
    id_column=None,
    method="split",
    test_size=0.2,
    folds=5,
    group_column=None,
    time_column=None,
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    loss_function="RMSE",
    random_seed=42,
    early_stopping_rounds=80,
    use_best_iteration=True,
    compute_device="auto",
    random_strength=1.0,
    bagging_temperature=1.0,
    prediction_column="Прогноз CatBoost",
    include_residual=True,
    compute_shap=True,
    signature="",
    progress_callback=None,
    cancel_event=None,
):
    """Train/evaluate a regressor and fit a final model on all labelled rows."""
    if frame is None or frame.empty:
        raise ValueError("Входной dataset пуст.")
    if progress_callback:
        progress_callback(2, "Подготовка данных")
    _check_cancel(cancel_event)
    target = str(target or "")
    if target not in frame.columns:
        raise ValueError("Выберите числовой целевой канал.")
    selected = list(dict.fromkeys(str(value) for value in (features or [])))
    selected = [value for value in selected if value in frame.columns]
    control_columns = {
        target,
        str(id_column or ""),
    }
    if method == "group_cv":
        control_columns.add(str(group_column or ""))
    if method == "time_cv":
        control_columns.add(str(time_column or ""))
    selected = [value for value in selected if value not in control_columns]
    if not selected:
        raise ValueError("Выберите хотя бы один признак модели.")

    y_all = pd.to_numeric(frame[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    train_mask = y_all.notna()
    if int(train_mask.sum()) < 5:
        raise ValueError("После исключения пустой цели осталось меньше пяти строк.")
    x_all, categorical = _prepare_features(frame, selected)
    x_trainable = x_all.loc[train_mask]
    y_trainable = y_all.loc[train_mask].astype(float)

    compute = resolve_compute_device(compute_device)
    params = _model_params(
        iterations=iterations, depth=depth, learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg, loss_function=loss_function,
        random_seed=random_seed, random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        compute_device=compute["resolved"],
    )
    fit_kwargs = {"use_best_model": bool(use_best_iteration)}
    if int(early_stopping_rounds or 0) > 0:
        fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)

    evaluation_actual = []
    evaluation_predicted = []
    evaluation_indices = []
    curves = []
    fold_metrics = []

    validation_fits = int(folds or 0) if method in {"cv", "group_cv", "time_cv"} else 1
    fit_count = validation_fits + 1
    fit_span = 84.0 / fit_count
    completed_fits = 0

    if method in {"cv", "group_cv", "time_cv"}:
        folds = int(folds or 0)
        if folds < 2 or folds >= len(x_trainable):
            raise ValueError("Число фолдов должно быть от 2 до количества обучающих строк.")
        split_source = x_trainable
        split_groups = None
        if method == "group_cv":
            group_column = str(group_column or "")
            if group_column not in frame.columns:
                raise ValueError("Для групповой проверки выберите канал группы.")
            split_groups = (
                frame.loc[train_mask, group_column]
                .astype("string").fillna("<NA>").astype(str)
            )
            unique_groups = int(split_groups.nunique(dropna=False))
            if unique_groups < folds:
                raise ValueError(
                    f"Для {folds} фолдов нужно не меньше {folds} групп; найдено {unique_groups}."
                )
            splitter = GroupKFold(n_splits=folds)
            splits = splitter.split(split_source, y_trainable, groups=split_groups)
            evaluation_label = f"Group OOF, {folds} folds · {group_column}"
        elif method == "time_cv":
            time_column = str(time_column or "")
            if time_column not in frame.columns:
                raise ValueError("Для временной проверки выберите канал времени или порядка.")
            time_values = frame.loc[train_mask, time_column]
            if pd.api.types.is_numeric_dtype(time_values):
                order_values = pd.to_numeric(time_values, errors="coerce")
            else:
                order_values = pd.to_datetime(time_values, errors="coerce", utc=True)
            if order_values.isna().any():
                missing = int(order_values.isna().sum())
                raise ValueError(
                    f"В канале порядка {missing} пустых или нераспознанных значений."
                )
            order = np.argsort(order_values.to_numpy(), kind="stable")
            split_source = x_trainable.iloc[order]
            splitter = TimeSeriesSplit(n_splits=folds)
            splits = splitter.split(split_source)
            evaluation_label = f"Time series OOF, {folds} folds · {time_column}"
        else:
            splitter = KFold(n_splits=folds, shuffle=True, random_state=int(random_seed))
            splits = splitter.split(split_source)
            evaluation_label = f"OOF, {folds} folds"

        for fold_number, (train_idx, valid_idx) in enumerate(splits, 1):
            x_fit = split_source.iloc[train_idx]
            x_valid = split_source.iloc[valid_idx]
            y_fit = y_trainable.loc[x_fit.index]
            y_valid = y_trainable.loc[x_valid.index]
            model = CatBoostRegressor(**params)
            _fit_model(
                model, _pool(x_fit, y_fit, categorical),
                eval_set=_pool(x_valid, y_valid, categorical), fit_kwargs=fit_kwargs,
                progress_callback=progress_callback, cancel_event=cancel_event,
                progress_start=4 + completed_fits * fit_span, progress_span=fit_span,
                expected_iterations=params["iterations"], label=f"Обучение · fold {fold_number}/{folds}",
            )
            completed_fits += 1
            predicted = np.asarray(model.predict(x_valid), dtype=float)
            evaluation_actual.extend(y_valid.tolist())
            evaluation_predicted.extend(predicted.tolist())
            evaluation_indices.extend(x_valid.index.tolist())
            fold_metrics.append({"fold": fold_number, **_metrics(y_valid, predicted)})
            curve = _curve(model, f"Fold {fold_number}")
            if curve:
                curves.append(curve)
    else:
        test_size = float(test_size or 0)
        if not 0 < test_size < 1:
            raise ValueError("Доля test должна быть в диапазоне (0; 1).")
        indices = np.arange(len(x_trainable))
        train_idx, valid_idx = train_test_split(
            indices, test_size=test_size, random_state=int(random_seed)
        )
        if len(train_idx) < 2 or len(valid_idx) < 2:
            raise ValueError("Недостаточно строк для выбранной доли test.")
        x_fit, y_fit = x_trainable.iloc[train_idx], y_trainable.iloc[train_idx]
        x_valid, y_valid = x_trainable.iloc[valid_idx], y_trainable.iloc[valid_idx]
        model = CatBoostRegressor(**params)
        _fit_model(
            model, _pool(x_fit, y_fit, categorical),
            eval_set=_pool(x_valid, y_valid, categorical), fit_kwargs=fit_kwargs,
            progress_callback=progress_callback, cancel_event=cancel_event,
            progress_start=4, progress_span=fit_span,
            expected_iterations=params["iterations"], label="Обучение · train/test",
        )
        completed_fits = 1
        predicted = np.asarray(model.predict(x_valid), dtype=float)
        evaluation_actual = y_valid.tolist()
        evaluation_predicted = predicted.tolist()
        evaluation_indices = x_valid.index.tolist()
        fold_metrics.append({"fold": 1, **_metrics(y_valid, predicted)})
        curve = _curve(model, "Train / test")
        if curve:
            curves.append(curve)
        evaluation_label = f"Test, {test_size:.0%}"

    overall_metrics = _metrics(evaluation_actual, evaluation_predicted)
    baseline_value = float(y_trainable.mean())
    baseline_metrics = _metrics(
        evaluation_actual, np.full(len(evaluation_actual), baseline_value)
    )

    best_iterations = [
        int(np.argmin(curve["validation"])) + 1
        for curve in curves if curve.get("validation")
    ]
    final_iterations = int(params["iterations"])
    if best_iterations and bool(use_best_iteration):
        final_iterations = max(1, int(round(float(np.mean(best_iterations)))))
    final_params = {**params, "iterations": final_iterations}
    final_model = CatBoostRegressor(**final_params)
    _fit_model(
        final_model, _pool(x_trainable, y_trainable, categorical),
        progress_callback=progress_callback, cancel_event=cancel_event,
        progress_start=4 + completed_fits * fit_span, progress_span=fit_span,
        expected_iterations=final_iterations, label="Финальная модель · все строки",
    )
    all_predictions = np.asarray(final_model.predict(x_all), dtype=float)
    training_predictions = np.asarray(final_model.predict(x_trainable), dtype=float)
    training_metrics = _metrics(y_trainable, training_predictions)
    overfitting = _overfitting_summary("regression", training_metrics, overall_metrics)
    _check_cancel(cancel_event)
    if progress_callback:
        progress_callback(91, "Важность признаков")

    prediction_name = _unique_name(frame.columns, prediction_column)
    result_frame = frame.copy()
    result_frame[prediction_name] = all_predictions
    output_columns = [prediction_name]
    residual_name = None
    if include_residual:
        residual_name = _unique_name(result_frame.columns, f"Остаток {target}")
        result_frame[residual_name] = y_all - all_predictions
        output_columns.append(residual_name)

    importances = final_model.get_feature_importance(
        _pool(x_trainable, y_trainable, categorical)
    )
    feature_importance = sorted(
        [{"feature": feature, "importance": float(value)}
         for feature, value in zip(selected, importances)],
        key=lambda item: item["importance"], reverse=True,
    )

    shap_importance, shap_sample, shap_values = [], [], []
    shap_note = "SHAP отключён"
    if compute_shap:
        _check_cancel(cancel_event)
        if progress_callback:
            progress_callback(95, "Расчёт SHAP")
        sample_size = min(MAX_SHAP_ROWS, len(x_trainable))
        sample = x_trainable.sample(sample_size, random_state=int(random_seed))
        sample_y = y_trainable.loc[sample.index]
        matrix = np.asarray(final_model.get_feature_importance(
            _pool(sample, sample_y, categorical), type="ShapValues"
        ))[:, :-1]
        means = np.mean(np.abs(matrix), axis=0)
        shap_importance = sorted(
            [{"feature": feature, "importance": float(value)}
             for feature, value in zip(selected, means)],
            key=lambda item: item["importance"], reverse=True,
        )
        shap_sample = sample.astype(object).where(sample.notna(), None).to_dict("records")
        shap_values = matrix.tolist()
        shap_note = f"SHAP рассчитан на {sample_size} строках"

    _check_cancel(cancel_event)
    if progress_callback:
        progress_callback(99, "Подготовка результатов")

    preview = []
    for actual, predicted, row_index in zip(
        evaluation_actual, evaluation_predicted, evaluation_indices
    ):
        row = {
            "Строка": str(row_index), "Факт": float(actual),
            "Прогноз": float(predicted), "Ошибка": float(actual - predicted),
        }
        if id_column in frame.columns:
            row["ID"] = str(frame.loc[row_index, id_column])
        preview.append(row)

    evaluation_rows = len(evaluation_actual)
    if evaluation_rows > 6000:
        generator = np.random.default_rng(int(random_seed))
        chart_positions = np.sort(generator.choice(evaluation_rows, 6000, replace=False))
        chart_actual = [evaluation_actual[position] for position in chart_positions]
        chart_predicted = [evaluation_predicted[position] for position in chart_positions]
    else:
        chart_actual = evaluation_actual
        chart_predicted = evaluation_predicted

    analysis = {
        "task": "regression", "method": method,
        "evaluation_label": evaluation_label, "target": target,
        "group_column": str(group_column or ""),
        "time_column": str(time_column or ""),
        "features": selected,
        "categorical_features": [selected[index] for index in categorical],
        "input_rows": int(len(frame)), "training_rows": int(train_mask.sum()),
        "excluded_target_rows": int((~train_mask).sum()),
        "metrics": overall_metrics,
        "training_metrics": training_metrics,
        "overfitting": overfitting,
        "baseline": {"value": baseline_value, **baseline_metrics},
        "fold_metrics": fold_metrics, "evaluation_rows": evaluation_rows,
        "actual": chart_actual, "predicted": chart_predicted,
        "preview": preview[:200],
        "feature_importance": feature_importance, "curves": curves,
        "shap_importance": shap_importance, "shap_sample": shap_sample,
        "shap_values": shap_values, "shap_note": shap_note,
        "outputs": output_columns, "prediction_column": prediction_name,
        "residual_column": residual_name, "params": final_params,
        "best_iterations": best_iterations, "final_iterations": final_iterations,
        "use_best_iteration": bool(use_best_iteration),
        "compute": compute,
        "primary_metric_name": "MAE",
        "primary_metric_value": overall_metrics.get("mae"),
        "higher_is_better": False,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    step = {
        "type": "catboost_regression", "operation": "catboost_regression",
        "label": "CatBoost regression", "inputs": [target, *selected],
        "outputs": output_columns,
        "params": {
            "method": method, "target": target, "features": selected,
            "group_column": str(group_column or ""),
            "time_column": str(time_column or ""),
            "iterations": final_iterations, "depth": int(depth),
            "learning_rate": float(learning_rate), "metrics": overall_metrics,
            "compute_device": compute["resolved"],
            "use_best_iteration": bool(use_best_iteration),
        },
    }
    return CachedMLResult(result_frame, analysis, step, signature, final_model)


def run_catboost_classification(
    frame: pd.DataFrame,
    *,
    target: str,
    features,
    id_column=None,
    method="split",
    test_size=0.2,
    folds=5,
    group_column=None,
    time_column=None,
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    loss_function="Auto",
    random_seed=42,
    early_stopping_rounds=80,
    use_best_iteration=True,
    compute_device="auto",
    random_strength=1.0,
    bagging_temperature=1.0,
    auto_class_weights="none",
    prediction_column="Класс CatBoost",
    include_confidence=True,
    compute_shap=True,
    signature="",
    progress_callback=None,
    cancel_event=None,
):
    """Train/evaluate a classifier and append aligned prediction channels."""
    if frame is None or frame.empty:
        raise ValueError("Входной dataset пуст.")
    if progress_callback:
        progress_callback(2, "Подготовка данных")
    _check_cancel(cancel_event)

    target = str(target or "")
    if target not in frame.columns:
        raise ValueError("Выберите целевой канал-класс.")
    raw_target = frame[target].astype("string")
    train_mask = raw_target.notna() & raw_target.str.strip().ne("")
    if int(train_mask.sum()) < 8:
        raise ValueError("После исключения пустой цели осталось меньше восьми строк.")
    y_trainable = raw_target.loc[train_mask].astype(str)
    class_labels = sorted(y_trainable.unique().tolist())
    class_count = len(class_labels)
    if class_count < 2:
        raise ValueError("Для классификации нужно минимум два класса.")
    if class_count > 100 or class_count > max(20, int(len(y_trainable) * .25)):
        raise ValueError(
            f"В цели найдено {class_count} классов — канал похож на непрерывный. "
            "Сначала преобразуйте его в категории."
        )

    selected = list(dict.fromkeys(str(value) for value in (features or [])))
    selected = [value for value in selected if value in frame.columns]
    control_columns = {target, str(id_column or "")}
    if method == "group_cv":
        control_columns.add(str(group_column or ""))
    if method == "time_cv":
        control_columns.add(str(time_column or ""))
    selected = [value for value in selected if value not in control_columns]
    if not selected:
        raise ValueError("Выберите хотя бы один признак модели.")

    x_all, categorical = _prepare_features(frame, selected)
    x_trainable = x_all.loc[train_mask]
    compute = resolve_compute_device(compute_device)
    params = _classifier_params(
        iterations=iterations, depth=depth, learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg, loss_function=loss_function,
        random_seed=random_seed, random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        auto_class_weights=auto_class_weights, class_count=class_count,
        compute_device=compute["resolved"],
    )
    fit_kwargs = {"use_best_model": bool(use_best_iteration)}
    if int(early_stopping_rounds or 0) > 0:
        fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)

    def aligned_probabilities(model, values):
        raw = np.asarray(model.predict_proba(values), dtype=float)
        model_labels = [str(value) for value in np.asarray(model.classes_).reshape(-1)]
        aligned = np.zeros((len(values), class_count), dtype=float)
        for source_index, label in enumerate(model_labels):
            if label in class_labels:
                aligned[:, class_labels.index(label)] = raw[:, source_index]
        row_sums = aligned.sum(axis=1, keepdims=True)
        return np.divide(
            aligned, row_sums, out=np.full_like(aligned, 1 / class_count),
            where=row_sums > 0,
        )

    evaluation_actual, evaluation_predicted = [], []
    evaluation_probabilities, evaluation_indices = [], []
    curves, fold_metrics = [], []
    validation_fits = int(folds or 0) if method in {"cv", "group_cv", "time_cv"} else 1
    fit_count = validation_fits + 1
    fit_span = 84.0 / fit_count
    completed_fits = 0

    if method in {"cv", "group_cv", "time_cv"}:
        folds = int(folds or 0)
        if folds < 2 or folds >= len(x_trainable):
            raise ValueError("Число фолдов должно быть от 2 до количества обучающих строк.")
        split_source = x_trainable
        if method == "group_cv":
            group_column = str(group_column or "")
            if group_column not in frame.columns:
                raise ValueError("Для групповой проверки выберите канал группы.")
            groups = frame.loc[train_mask, group_column].astype("string").fillna("<NA>").astype(str)
            unique_groups = int(groups.nunique(dropna=False))
            if unique_groups < folds:
                raise ValueError(
                    f"Для {folds} фолдов нужно не меньше {folds} групп; найдено {unique_groups}."
                )
            splits = GroupKFold(n_splits=folds).split(split_source, y_trainable, groups=groups)
            evaluation_label = f"Group OOF, {folds} folds · {group_column}"
        elif method == "time_cv":
            time_column = str(time_column or "")
            if time_column not in frame.columns:
                raise ValueError("Для временной проверки выберите канал времени или порядка.")
            time_values = frame.loc[train_mask, time_column]
            order_values = (
                pd.to_numeric(time_values, errors="coerce")
                if pd.api.types.is_numeric_dtype(time_values)
                else pd.to_datetime(time_values, errors="coerce", utc=True)
            )
            if order_values.isna().any():
                raise ValueError(
                    f"В канале порядка {int(order_values.isna().sum())} пустых или нераспознанных значений."
                )
            order = np.argsort(order_values.to_numpy(), kind="stable")
            split_source = x_trainable.iloc[order]
            splits = TimeSeriesSplit(n_splits=folds).split(split_source)
            evaluation_label = f"Time series OOF, {folds} folds · {time_column}"
        else:
            minimum_class = int(y_trainable.value_counts().min())
            if minimum_class < folds:
                raise ValueError(
                    f"Для {folds} стратифицированных фолдов в каждом классе нужно минимум {folds} строк."
                )
            splits = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=int(random_seed)
            ).split(split_source, y_trainable)
            evaluation_label = f"Stratified OOF, {folds} folds"

        for fold_number, (train_idx, valid_idx) in enumerate(splits, 1):
            x_fit, x_valid = split_source.iloc[train_idx], split_source.iloc[valid_idx]
            y_fit, y_valid = y_trainable.loc[x_fit.index], y_trainable.loc[x_valid.index]
            if y_fit.nunique() < 2:
                raise ValueError(
                    f"В train fold {fold_number} остался один класс. Измените порядок, группы или число фолдов."
                )
            model = CatBoostClassifier(**params)
            _fit_model(
                model, _pool(x_fit, y_fit, categorical),
                eval_set=_pool(x_valid, y_valid, categorical), fit_kwargs=fit_kwargs,
                progress_callback=progress_callback, cancel_event=cancel_event,
                progress_start=4 + completed_fits * fit_span, progress_span=fit_span,
                expected_iterations=params["iterations"],
                label=f"Обучение · fold {fold_number}/{folds}",
            )
            completed_fits += 1
            probabilities = aligned_probabilities(model, x_valid)
            predicted = _prediction_labels(model.predict(x_valid))
            evaluation_actual.extend(y_valid.astype(str).tolist())
            evaluation_predicted.extend(predicted.tolist())
            evaluation_probabilities.extend(probabilities.tolist())
            evaluation_indices.extend(x_valid.index.tolist())
            fold_metrics.append({
                "fold": fold_number,
                **_classification_metrics(y_valid, predicted, probabilities, class_labels),
            })
            curve = _curve(model, f"Fold {fold_number}")
            if curve:
                curves.append(curve)
    else:
        test_size = float(test_size or 0)
        if not 0 < test_size < 1:
            raise ValueError("Доля test должна быть в диапазоне (0; 1).")
        counts = y_trainable.value_counts()
        if int(counts.min()) < 2:
            raise ValueError("Для train/test в каждом классе нужно минимум две строки.")
        test_rows = int(np.ceil(len(x_trainable) * test_size))
        train_rows = len(x_trainable) - test_rows
        if test_rows < class_count or train_rows < class_count:
            raise ValueError(
                "Выбранная доля test не оставляет хотя бы по одной строке каждого класса "
                "в train и test. Измените долю или объедините редкие классы."
            )
        indices = np.arange(len(x_trainable))
        train_idx, valid_idx = train_test_split(
            indices, test_size=test_size, random_state=int(random_seed),
            stratify=y_trainable.to_numpy(),
        )
        x_fit, x_valid = x_trainable.iloc[train_idx], x_trainable.iloc[valid_idx]
        y_fit, y_valid = y_trainable.iloc[train_idx], y_trainable.iloc[valid_idx]
        model = CatBoostClassifier(**params)
        _fit_model(
            model, _pool(x_fit, y_fit, categorical),
            eval_set=_pool(x_valid, y_valid, categorical), fit_kwargs=fit_kwargs,
            progress_callback=progress_callback, cancel_event=cancel_event,
            progress_start=4, progress_span=fit_span,
            expected_iterations=params["iterations"], label="Обучение · stratified train/test",
        )
        completed_fits = 1
        probabilities = aligned_probabilities(model, x_valid)
        predicted = _prediction_labels(model.predict(x_valid))
        evaluation_actual = y_valid.astype(str).tolist()
        evaluation_predicted = predicted.tolist()
        evaluation_probabilities = probabilities.tolist()
        evaluation_indices = x_valid.index.tolist()
        fold_metrics.append({
            "fold": 1,
            **_classification_metrics(y_valid, predicted, probabilities, class_labels),
        })
        curve = _curve(model, "Train / test")
        if curve:
            curves.append(curve)
        evaluation_label = f"Stratified test, {test_size:.0%}"

    probability_matrix = np.asarray(evaluation_probabilities, dtype=float)
    overall_metrics = _classification_metrics(
        evaluation_actual, evaluation_predicted, probability_matrix, class_labels
    )
    majority_label = str(y_trainable.value_counts().idxmax())
    class_priors = y_trainable.value_counts(normalize=True).reindex(class_labels, fill_value=0).to_numpy()
    baseline_probabilities = np.tile(class_priors, (len(evaluation_actual), 1))
    baseline_metrics = _classification_metrics(
        evaluation_actual,
        np.full(len(evaluation_actual), majority_label, dtype=object),
        baseline_probabilities,
        class_labels,
    )

    best_iterations = [
        int(np.argmin(curve["validation"])) + 1
        for curve in curves if curve.get("validation")
    ]
    final_iterations = int(params["iterations"])
    if best_iterations and bool(use_best_iteration):
        final_iterations = max(1, int(round(float(np.mean(best_iterations)))))
    final_params = {**params, "iterations": final_iterations}
    final_model = CatBoostClassifier(**final_params)
    _fit_model(
        final_model, _pool(x_trainable, y_trainable, categorical),
        progress_callback=progress_callback, cancel_event=cancel_event,
        progress_start=4 + completed_fits * fit_span, progress_span=fit_span,
        expected_iterations=final_iterations, label="Финальная модель · все строки",
    )
    all_predictions = _prediction_labels(final_model.predict(x_all))
    all_probabilities = aligned_probabilities(final_model, x_all)
    all_confidence = np.max(all_probabilities, axis=1)
    training_predictions = _prediction_labels(final_model.predict(x_trainable))
    training_probabilities = aligned_probabilities(final_model, x_trainable)
    training_metrics = _classification_metrics(
        y_trainable, training_predictions, training_probabilities, class_labels
    )
    overfitting = _overfitting_summary("classification", training_metrics, overall_metrics)
    _check_cancel(cancel_event)
    if progress_callback:
        progress_callback(91, "Важность признаков")

    prediction_name = _unique_name(frame.columns, prediction_column or "Класс CatBoost")
    result_frame = frame.copy()
    result_frame[prediction_name] = all_predictions
    output_columns = [prediction_name]
    confidence_name = None
    if include_confidence:
        confidence_name = _unique_name(result_frame.columns, "Уверенность CatBoost")
        result_frame[confidence_name] = all_confidence
        output_columns.append(confidence_name)

    importances = final_model.get_feature_importance(
        _pool(x_trainable, y_trainable, categorical)
    )
    feature_importance = sorted(
        [{"feature": feature, "importance": float(value)}
         for feature, value in zip(selected, importances)],
        key=lambda item: item["importance"], reverse=True,
    )

    shap_importance, shap_sample, shap_values = [], [], []
    shap_note = "SHAP отключён"
    if compute_shap:
        _check_cancel(cancel_event)
        if progress_callback:
            progress_callback(95, "Расчёт SHAP")
        sample_size = min(MAX_SHAP_ROWS, len(x_trainable))
        sample = x_trainable.sample(sample_size, random_state=int(random_seed))
        sample_y = y_trainable.loc[sample.index]
        raw_shap = np.asarray(final_model.get_feature_importance(
            _pool(sample, sample_y, categorical), type="ShapValues"
        ))
        if raw_shap.ndim == 2:
            matrix = raw_shap[:, :-1]
            means = np.mean(np.abs(matrix), axis=0)
            shap_values = matrix.tolist()
        else:
            matrix = raw_shap[:, :, :-1]
            means = np.mean(np.abs(matrix), axis=(0, 1))
        shap_importance = sorted(
            [{"feature": feature, "importance": float(value)}
             for feature, value in zip(selected, means)],
            key=lambda item: item["importance"], reverse=True,
        )
        shap_sample = sample.astype(object).where(sample.notna(), None).to_dict("records")
        shap_note = (
            f"SHAP рассчитан на {sample_size} строках"
            + (" · многоклассовое среднее |SHAP|" if raw_shap.ndim == 3 else "")
        )

    confidence = probability_matrix.max(axis=1)
    preview = []
    for actual, predicted, score, row_index in zip(
        evaluation_actual, evaluation_predicted, confidence, evaluation_indices
    ):
        row = {
            "Строка": str(row_index), "Факт": str(actual),
            "Прогноз": str(predicted), "Уверенность": float(score),
            "Верно": "Да" if str(actual) == str(predicted) else "Нет",
        }
        if id_column in frame.columns:
            row["ID"] = str(frame.loc[row_index, id_column])
        preview.append(row)

    evaluation_rows = len(evaluation_actual)
    if evaluation_rows > 6000:
        generator = np.random.default_rng(int(random_seed))
        positions = np.sort(generator.choice(evaluation_rows, 6000, replace=False))
    else:
        positions = np.arange(evaluation_rows)
    shown_actual = [evaluation_actual[position] for position in positions]
    shown_predicted = [evaluation_predicted[position] for position in positions]
    shown_confidence = [float(confidence[position]) for position in positions]
    matrix = confusion_matrix(evaluation_actual, evaluation_predicted, labels=class_labels)

    _check_cancel(cancel_event)
    if progress_callback:
        progress_callback(99, "Подготовка результатов")
    analysis = {
        "task": "classification", "method": method,
        "evaluation_label": evaluation_label, "target": target,
        "group_column": str(group_column or ""), "time_column": str(time_column or ""),
        "features": selected,
        "categorical_features": [selected[index] for index in categorical],
        "class_labels": class_labels, "class_count": class_count,
        "class_distribution": {
            str(label): int(count)
            for label, count in y_trainable.value_counts().items()
        },
        "input_rows": int(len(frame)), "training_rows": int(train_mask.sum()),
        "excluded_target_rows": int((~train_mask).sum()),
        "metrics": overall_metrics,
        "training_metrics": training_metrics,
        "overfitting": overfitting,
        "baseline": {"class": majority_label, **baseline_metrics},
        "fold_metrics": fold_metrics, "evaluation_rows": evaluation_rows,
        "actual_labels": shown_actual, "predicted_labels": shown_predicted,
        "confidence": shown_confidence, "confusion_matrix": matrix.tolist(),
        "preview": preview[:200], "feature_importance": feature_importance,
        "curves": curves, "shap_importance": shap_importance,
        "shap_sample": shap_sample, "shap_values": shap_values,
        "shap_note": shap_note, "outputs": output_columns,
        "prediction_column": prediction_name, "confidence_column": confidence_name,
        "params": final_params, "best_iterations": best_iterations,
        "final_iterations": final_iterations,
        "use_best_iteration": bool(use_best_iteration),
        "compute": compute,
        "primary_metric_name": "Accuracy",
        "primary_metric_value": overall_metrics.get("accuracy"),
        "higher_is_better": True,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    step = {
        "type": "catboost_classification", "operation": "catboost_classification",
        "label": "CatBoost classification", "inputs": [target, *selected],
        "outputs": output_columns,
        "params": {
            "method": method, "target": target, "features": selected,
            "classes": class_labels, "iterations": final_iterations,
            "depth": int(depth), "learning_rate": float(learning_rate),
            "metrics": overall_metrics,
            "compute_device": compute["resolved"],
            "use_best_iteration": bool(use_best_iteration),
        },
    }
    return CachedMLResult(result_frame, analysis, step, signature, final_model)
