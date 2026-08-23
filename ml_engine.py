# -*- coding: utf-8 -*-
"""CatBoost regression helpers and a short-lived server-side result cache."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from uuid import uuid4

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


MAX_SHAP_ROWS = 300


@dataclass
class CachedMLResult:
    frame: pd.DataFrame
    analysis: dict
    committed_step: dict
    signature: str
    model: CatBoostRegressor


_RESULT_CACHE: dict[str, CachedMLResult] = {}


def ml_signature(**parameters) -> str:
    normalized = json.dumps(parameters, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def cache_result(result: CachedMLResult) -> str:
    reference = uuid4().hex
    _RESULT_CACHE[reference] = result
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
                  bagging_temperature):
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
    }


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


def run_catboost_regression(
    frame: pd.DataFrame,
    *,
    target: str,
    features,
    id_column=None,
    method="split",
    test_size=0.2,
    folds=5,
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    loss_function="RMSE",
    random_seed=42,
    early_stopping_rounds=80,
    random_strength=1.0,
    bagging_temperature=1.0,
    prediction_column="Прогноз CatBoost",
    include_residual=True,
    compute_shap=True,
    signature="",
):
    """Train/evaluate a regressor and fit a final model on all labelled rows."""
    if frame is None or frame.empty:
        raise ValueError("Входной dataset пуст.")
    target = str(target or "")
    if target not in frame.columns:
        raise ValueError("Выберите числовой целевой канал.")
    selected = list(dict.fromkeys(str(value) for value in (features or [])))
    selected = [value for value in selected if value in frame.columns]
    selected = [value for value in selected if value not in {target, str(id_column or "")}]
    if not selected:
        raise ValueError("Выберите хотя бы один признак модели.")

    y_all = pd.to_numeric(frame[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    train_mask = y_all.notna()
    if int(train_mask.sum()) < 5:
        raise ValueError("После исключения пустой цели осталось меньше пяти строк.")
    x_all, categorical = _prepare_features(frame, selected)
    x_trainable = x_all.loc[train_mask]
    y_trainable = y_all.loc[train_mask].astype(float)

    params = _model_params(
        iterations=iterations, depth=depth, learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg, loss_function=loss_function,
        random_seed=random_seed, random_strength=random_strength,
        bagging_temperature=bagging_temperature,
    )
    fit_kwargs = {}
    if int(early_stopping_rounds or 0) > 0:
        fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)

    evaluation_actual = []
    evaluation_predicted = []
    evaluation_indices = []
    curves = []
    fold_metrics = []

    if method == "cv":
        folds = int(folds or 0)
        if folds < 2 or folds > len(x_trainable):
            raise ValueError("Число фолдов должно быть от 2 до количества обучающих строк.")
        splitter = KFold(n_splits=folds, shuffle=True, random_state=int(random_seed))
        for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x_trainable), 1):
            x_fit, y_fit = x_trainable.iloc[train_idx], y_trainable.iloc[train_idx]
            x_valid, y_valid = x_trainable.iloc[valid_idx], y_trainable.iloc[valid_idx]
            model = CatBoostRegressor(**params)
            model.fit(_pool(x_fit, y_fit, categorical),
                      eval_set=_pool(x_valid, y_valid, categorical), **fit_kwargs)
            predicted = np.asarray(model.predict(x_valid), dtype=float)
            evaluation_actual.extend(y_valid.tolist())
            evaluation_predicted.extend(predicted.tolist())
            evaluation_indices.extend(x_valid.index.tolist())
            fold_metrics.append({"fold": fold_number, **_metrics(y_valid, predicted)})
            curve = _curve(model, f"Fold {fold_number}")
            if curve:
                curves.append(curve)
        evaluation_label = f"OOF, {folds} folds"
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
        model.fit(_pool(x_fit, y_fit, categorical),
                  eval_set=_pool(x_valid, y_valid, categorical), **fit_kwargs)
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
    if best_iterations and int(early_stopping_rounds or 0) > 0:
        final_iterations = max(1, int(round(float(np.mean(best_iterations)))))
    final_params = {**params, "iterations": final_iterations}
    final_model = CatBoostRegressor(**final_params)
    final_model.fit(_pool(x_trainable, y_trainable, categorical))
    all_predictions = np.asarray(final_model.predict(x_all), dtype=float)

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
        "method": method, "evaluation_label": evaluation_label, "target": target,
        "features": selected,
        "categorical_features": [selected[index] for index in categorical],
        "input_rows": int(len(frame)), "training_rows": int(train_mask.sum()),
        "excluded_target_rows": int((~train_mask).sum()),
        "metrics": overall_metrics,
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
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    step = {
        "type": "catboost_regression", "operation": "catboost_regression",
        "label": "CatBoost regression", "inputs": [target, *selected],
        "outputs": output_columns,
        "params": {
            "method": method, "target": target, "features": selected,
            "iterations": final_iterations, "depth": int(depth),
            "learning_rate": float(learning_rate), "metrics": overall_metrics,
        },
    }
    return CachedMLResult(result_frame, analysis, step, signature, final_model)
