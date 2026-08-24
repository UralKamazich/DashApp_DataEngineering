# -*- coding: utf-8 -*-
"""Lightweight, cancellable CatBoost parameter search."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from ml_engine import (
    _check_cancel,
    _classification_metrics,
    _classifier_params,
    _fit_model,
    _metrics,
    _model_params,
    _pool,
    _prediction_labels,
    _prepare_features,
)


def _log_uniform(generator, low, high):
    return float(math.exp(generator.uniform(math.log(low), math.log(high))))


def _candidate_parameters(generator, baseline, trial_number):
    if trial_number == 1:
        return dict(baseline)
    return {
        "depth": int(generator.integers(3, 11)),
        "learning_rate": _log_uniform(generator, .015, .2),
        "l2_leaf_reg": _log_uniform(generator, .5, 12),
        "random_strength": _log_uniform(generator, .1, 3),
        "bagging_temperature": float(generator.uniform(0, 2)),
    }


def _holdout_indices(frame, train_mask, x, y, *, task, method, test_size,
                     folds, group_column, time_column, random_seed):
    positions = np.arange(len(x))
    if method == "group_cv":
        group_column = str(group_column or "")
        if group_column not in frame.columns:
            raise ValueError("Для автоподбора с GroupKFold выберите канал группы.")
        groups = frame.loc[train_mask, group_column].astype("string").fillna("<NA>").astype(str)
        unique_groups = int(groups.nunique(dropna=False))
        if unique_groups < 2:
            raise ValueError("Для Group holdout нужно минимум две группы.")
        split_count = min(max(2, int(folds or 2)), unique_groups)
        return (*next(GroupKFold(n_splits=split_count).split(x, y, groups=groups)),
                "Group holdout")
    if method == "time_cv":
        time_column = str(time_column or "")
        if time_column not in frame.columns:
            raise ValueError("Для временного автоподбора выберите канал времени или порядка.")
        raw = frame.loc[train_mask, time_column]
        order_values = (
            pd.to_numeric(raw, errors="coerce")
            if pd.api.types.is_numeric_dtype(raw)
            else pd.to_datetime(raw, errors="coerce", utc=True)
        )
        if order_values.isna().any():
            raise ValueError("Канал порядка содержит пустые или нераспознанные значения.")
        order = np.argsort(order_values.to_numpy(), kind="stable")
        split_count = min(max(2, int(folds or 2)), len(x) - 1)
        train_idx, valid_idx = list(TimeSeriesSplit(n_splits=split_count).split(order))[-1]
        return order[train_idx], order[valid_idx], "Последний временной holdout"
    if method == "cv":
        split_count = min(max(2, int(folds or 2)), len(x) - 1)
        if task == "classification":
            minimum_class = int(pd.Series(y).value_counts().min())
            split_count = min(split_count, minimum_class)
            if split_count < 2:
                raise ValueError("Для автоподбора в каждом классе нужно минимум две строки.")
            splitter = StratifiedKFold(
                n_splits=split_count, shuffle=True, random_state=int(random_seed)
            )
            return (*next(splitter.split(x, y)), "Первый stratified fold")
        splitter = KFold(n_splits=split_count, shuffle=True, random_state=int(random_seed))
        return (*next(splitter.split(x)), "Первый fold")

    test_size = float(test_size or .2)
    if not 0 < test_size < 1:
        raise ValueError("Доля test должна быть в диапазоне (0; 1).")
    stratify = y if task == "classification" else None
    train_idx, valid_idx = train_test_split(
        positions, test_size=test_size, random_state=int(random_seed), stratify=stratify,
    )
    return train_idx, valid_idx, "Train / test holdout"


def tune_catboost_parameters(
    frame: pd.DataFrame,
    *,
    task="regression",
    target,
    features,
    method="split",
    test_size=.2,
    folds=5,
    group_column=None,
    time_column=None,
    iterations=800,
    depth=6,
    learning_rate=.05,
    l2_leaf_reg=3,
    loss_function="RMSE",
    random_seed=42,
    early_stopping_rounds=80,
    use_best_iteration=True,
    compute_device="auto",
    random_strength=1,
    bagging_temperature=1,
    auto_class_weights="none",
    trials=12,
    progress_callback=None,
    cancel_event=None,
):
    """Search five tree parameters on one validation holdout."""
    task = "classification" if task == "classification" else "regression"
    trials = int(trials or 0)
    if trials < 3 or trials > 60:
        raise ValueError("Количество попыток автоподбора должно быть от 3 до 60.")
    target = str(target or "")
    if target not in frame.columns:
        raise ValueError("Выберите целевой канал.")

    selected = list(dict.fromkeys(str(value) for value in (features or [])))
    excluded = {target, str(group_column or ""), str(time_column or "")}
    selected = [value for value in selected if value in frame.columns and value not in excluded]
    if not selected:
        raise ValueError("Выберите хотя бы один признак модели.")

    if task == "classification":
        raw_target = frame[target].astype("string")
        train_mask = raw_target.notna() & raw_target.str.strip().ne("")
        y = raw_target.loc[train_mask].astype(str)
        class_count = int(y.nunique())
        if class_count < 2:
            raise ValueError("Для классификации нужно минимум два класса.")
        if class_count > 100 or class_count > max(20, int(len(y) * .25)):
            raise ValueError(
                f"В цели найдено {class_count} классов — канал похож на непрерывный."
            )
        if int(y.value_counts().min()) < 2:
            raise ValueError("Для автоподбора в каждом классе нужно минимум две строки.")
    else:
        raw_target = pd.to_numeric(frame[target], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        train_mask = raw_target.notna()
        y = raw_target.loc[train_mask].astype(float)
        class_count = 0
    if len(y) < 8:
        raise ValueError("Для автоподбора осталось меньше восьми обучающих строк.")

    x_all, categorical = _prepare_features(frame, selected)
    x = x_all.loc[train_mask]
    train_idx, valid_idx, validation_label = _holdout_indices(
        frame, train_mask, x, y, task=task, method=method,
        test_size=test_size, folds=folds, group_column=group_column,
        time_column=time_column, random_seed=random_seed,
    )
    x_fit, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
    y_fit, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    if task == "classification" and y_fit.nunique() < 2:
        raise ValueError("В обучающей части автоподбора остался только один класс.")

    baseline = {
        "depth": int(depth), "learning_rate": float(learning_rate),
        "l2_leaf_reg": float(l2_leaf_reg),
        "random_strength": float(random_strength or 0),
        "bagging_temperature": float(bagging_temperature or 0),
    }
    generator = np.random.default_rng(int(random_seed))
    results = []
    best = None
    span = 36.0 / trials
    for trial_number in range(1, trials + 1):
        _check_cancel(cancel_event)
        candidate = _candidate_parameters(generator, baseline, trial_number)
        common = {
            "iterations": int(iterations), "random_seed": int(random_seed),
            **candidate,
        }
        if task == "classification":
            params = _classifier_params(
                **common, loss_function=loss_function,
                auto_class_weights=auto_class_weights, class_count=class_count,
                compute_device=compute_device,
            )
            model = CatBoostClassifier(**params)
        else:
            params = _model_params(
                **common, loss_function=loss_function,
                compute_device=compute_device,
            )
            model = CatBoostRegressor(**params)
        fit_kwargs = {"use_best_model": bool(use_best_iteration)}
        if int(early_stopping_rounds or 0) > 0:
            fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
        _fit_model(
            model, _pool(x_fit, y_fit, categorical),
            eval_set=_pool(x_valid, y_valid, categorical), fit_kwargs=fit_kwargs,
            progress_callback=progress_callback, cancel_event=cancel_event,
            progress_start=2 + (trial_number - 1) * span, progress_span=span,
            expected_iterations=int(iterations),
            label=f"Автоподбор параметров · {trial_number}/{trials}",
        )
        if task == "classification":
            predicted = _prediction_labels(model.predict(x_valid))
            score = _classification_metrics(y_valid, predicted)["accuracy"]
            metric_name, higher_is_better = "Accuracy", True
        else:
            predicted = np.asarray(model.predict(x_valid), dtype=float)
            score = _metrics(y_valid, predicted)["mae"]
            metric_name, higher_is_better = "MAE", False
        best_iteration = int(model.get_best_iteration()) + 1
        if best_iteration < 1:
            best_iteration = int(iterations)
        item = {
            "trial": trial_number, "score": float(score),
            "best_iteration": best_iteration, **candidate,
        }
        results.append(item)
        if best is None or (
            score > best["score"] if higher_is_better else score < best["score"]
        ):
            best = item

    best_params = {
        key: best[key]
        for key in (
            "depth", "learning_rate", "l2_leaf_reg",
            "random_strength", "bagging_temperature",
        )
    }
    best_params["iterations"] = int(best["best_iteration"])
    return {
        "enabled": True, "trials_count": trials, "trials": results,
        "metric_name": metric_name, "higher_is_better": higher_is_better,
        "best_value": float(best["score"]), "best_trial": int(best["trial"]),
        "best_params": best_params, "validation_label": validation_label,
    }
