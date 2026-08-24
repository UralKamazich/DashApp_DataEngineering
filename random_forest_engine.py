# -*- coding: utf-8 -*-
"""Random Forest regression/classification for the dataset-aware ML workspace."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

from ml_engine import (
    CachedMLResult,
    _check_cancel,
    _classification_metrics,
    _metrics,
    _overfitting_summary,
    _unique_name,
)


def _prepare_features(frame, features):
    prepared = frame[list(features)].copy()
    numeric, categorical = [], []
    for column in features:
        if pd.api.types.is_numeric_dtype(prepared[column]):
            numeric.append(column)
            prepared[column] = pd.to_numeric(
                prepared[column], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)
        else:
            categorical.append(column)
            values = prepared[column].astype("string")
            prepared[column] = pd.Series(
                values.to_numpy(dtype=object, na_value=np.nan),
                index=prepared.index,
            )
    return prepared, numeric, categorical


def _resolved_max_features(value):
    text = str(value or "sqrt").strip().lower()
    if text in {"sqrt", "log2"}:
        return text
    if text in {"all", "none", "1.0"}:
        return 1.0
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError("Доля признаков должна быть sqrt, log2, all или числом от 0 до 1.") from error
    if not 0 < number <= 1:
        raise ValueError("Доля признаков должна быть больше 0 и не больше 1.")
    return number


def _model_parameters(*, n_estimators, max_depth, min_samples_leaf,
                      min_samples_split, max_features, bootstrap, max_samples,
                      random_seed, oob_score, class_weight=None, criterion=None):
    trees = int(n_estimators or 0)
    if not 10 <= trees <= 5000:
        raise ValueError("Количество деревьев должно быть от 10 до 5 000.")
    depth = int(max_depth or 0)
    if depth < 0 or depth > 200:
        raise ValueError("Максимальная глубина должна быть от 0 до 200; 0 — без ограничения.")
    leaf = int(min_samples_leaf or 0)
    split = int(min_samples_split or 0)
    if leaf < 1:
        raise ValueError("Минимум строк в листе должен быть не меньше 1.")
    if split < 2:
        raise ValueError("Минимум строк для разбиения должен быть не меньше 2.")
    use_bootstrap = bool(bootstrap)
    sample_fraction = float(max_samples or 0)
    if sample_fraction and not 0 < sample_fraction <= 1:
        raise ValueError("Доля строк bootstrap должна быть в диапазоне (0; 1].")
    params = {
        "n_estimators": trees,
        "max_depth": depth or None,
        "min_samples_leaf": leaf,
        "min_samples_split": split,
        "max_features": _resolved_max_features(max_features),
        "bootstrap": use_bootstrap,
        "max_samples": (sample_fraction or None) if use_bootstrap else None,
        "oob_score": bool(oob_score) and use_bootstrap,
        "random_state": int(random_seed or 0),
        "n_jobs": -1,
    }
    if criterion:
        params["criterion"] = criterion
    if class_weight and class_weight != "none":
        params["class_weight"] = class_weight
    return params


def _pipeline(numeric, categorical, estimator):
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent", keep_empty_features=True,
                )),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]),
            categorical,
        ))
    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers, remainder="drop")),
        ("model", estimator),
    ])


def _fit(model, x, y, *, progress_callback, cancel_event, progress, label):
    _check_cancel(cancel_event)
    if progress_callback:
        progress_callback(progress, label)
    model.fit(x, y)
    _check_cancel(cancel_event)


def _splits(frame, train_mask, x, y, *, task, method, test_size, folds,
            group_column, time_column, random_seed):
    if method == "split":
        fraction = float(test_size or 0)
        if not 0 < fraction < 1:
            raise ValueError("Доля test должна быть в диапазоне (0; 1).")
        positions = np.arange(len(x))
        stratify = None
        label = f"Test, {fraction:.0%}"
        if task == "classification":
            counts = y.value_counts()
            if int(counts.min()) < 2:
                raise ValueError("Для train/test в каждом классе нужно минимум две строки.")
            class_count = int(y.nunique())
            test_rows = int(np.ceil(len(x) * fraction))
            if test_rows < class_count or len(x) - test_rows < class_count:
                raise ValueError("Доля test не оставляет по одной строке каждого класса в train и test.")
            stratify = y.to_numpy()
            label = f"Stratified test, {fraction:.0%}"
        train_idx, valid_idx = train_test_split(
            positions, test_size=fraction, random_state=int(random_seed),
            stratify=stratify,
        )
        return [(train_idx, valid_idx)], x, label

    count = int(folds or 0)
    if count < 2 or count >= len(x):
        raise ValueError("Число фолдов должно быть от 2 до количества обучающих строк.")
    split_source = x
    if method == "group_cv":
        group_column = str(group_column or "")
        if group_column not in frame.columns:
            raise ValueError("Для групповой проверки выберите канал группы.")
        groups = frame.loc[train_mask, group_column].astype("string").fillna("<NA>").astype(str)
        if int(groups.nunique(dropna=False)) < count:
            raise ValueError(f"Для {count} фолдов нужно не меньше {count} групп.")
        return list(GroupKFold(n_splits=count).split(x, y, groups=groups)), x, (
            f"Group OOF, {count} folds · {group_column}"
        )
    if method == "time_cv":
        time_column = str(time_column or "")
        if time_column not in frame.columns:
            raise ValueError("Для временной проверки выберите канал времени или порядка.")
        values = frame.loc[train_mask, time_column]
        order_values = (
            pd.to_numeric(values, errors="coerce")
            if pd.api.types.is_numeric_dtype(values)
            else pd.to_datetime(values, errors="coerce", utc=True)
        )
        if order_values.isna().any():
            raise ValueError(
                f"В канале порядка {int(order_values.isna().sum())} пустых или нераспознанных значений."
            )
        order = np.argsort(order_values.to_numpy(), kind="stable")
        split_source = x.iloc[order]
        return list(TimeSeriesSplit(n_splits=count).split(split_source)), split_source, (
            f"Time series OOF, {count} folds · {time_column}"
        )
    if task == "classification":
        if int(y.value_counts().min()) < count:
            raise ValueError(
                f"Для {count} фолдов в каждом классе нужно минимум {count} строк."
            )
        splitter = StratifiedKFold(n_splits=count, shuffle=True, random_state=int(random_seed))
        return list(splitter.split(x, y)), x, f"Stratified OOF, {count} folds"
    splitter = KFold(n_splits=count, shuffle=True, random_state=int(random_seed))
    return list(splitter.split(x)), x, f"OOF, {count} folds"


def _feature_importance(pipeline, numeric, categorical):
    raw = np.asarray(pipeline.named_steps["model"].feature_importances_, dtype=float)
    preprocessor = pipeline.named_steps["preprocessor"]
    result = []
    if numeric and "num" in preprocessor.output_indices_:
        values = raw[preprocessor.output_indices_["num"]]
        result.extend(
            {"feature": feature, "importance": float(value)}
            for feature, value in zip(numeric, values)
        )
    if categorical and "cat" in preprocessor.output_indices_:
        values = raw[preprocessor.output_indices_["cat"]]
        encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        offset = 0
        for feature, categories in zip(categorical, encoder.categories_):
            width = len(categories)
            result.append({
                "feature": feature,
                "importance": float(values[offset:offset + width].sum()),
            })
            offset += width
    return sorted(result, key=lambda item: item["importance"], reverse=True)


def _common_input(frame, *, target, features, id_column, method,
                  group_column, time_column, task):
    if frame is None or frame.empty:
        raise ValueError("Входной dataset пуст.")
    target = str(target or "")
    if target not in frame.columns:
        raise ValueError(
            "Выберите целевой канал-класс."
            if task == "classification" else "Выберите числовой целевой канал."
        )
    selected = list(dict.fromkeys(str(value) for value in (features or [])))
    control = {target, str(id_column or "")}
    if method == "group_cv":
        control.add(str(group_column or ""))
    if method == "time_cv":
        control.add(str(time_column or ""))
    selected = [value for value in selected if value in frame.columns and value not in control]
    if not selected:
        raise ValueError("Выберите хотя бы один признак модели.")
    return target, selected


def run_random_forest_regression(
    frame: pd.DataFrame, *, target: str, features, id_column=None,
    method="split", test_size=0.2, folds=5, group_column=None, time_column=None,
    n_estimators=500, max_depth=0, min_samples_leaf=1, min_samples_split=2,
    max_features="sqrt", bootstrap=True, max_samples=0.8, oob_score=True,
    criterion="squared_error", random_seed=42,
    prediction_column="Прогноз Random Forest", include_residual=True,
    signature="", progress_callback=None, cancel_event=None,
):
    if progress_callback:
        progress_callback(2, "Подготовка данных")
    target, selected = _common_input(
        frame, target=target, features=features, id_column=id_column, method=method,
        group_column=group_column, time_column=time_column, task="regression",
    )
    y_all = pd.to_numeric(frame[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    train_mask = y_all.notna()
    if int(train_mask.sum()) < 5:
        raise ValueError("После исключения пустой цели осталось меньше пяти строк.")
    x_all, numeric, categorical = _prepare_features(frame, selected)
    x_trainable, y_trainable = x_all.loc[train_mask], y_all.loc[train_mask].astype(float)
    params = _model_parameters(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
        max_features=max_features, bootstrap=bootstrap, max_samples=max_samples,
        random_seed=random_seed, oob_score=oob_score, criterion=criterion,
    )
    split_items, split_source, evaluation_label = _splits(
        frame, train_mask, x_trainable, y_trainable, task="regression",
        method=method, test_size=test_size, folds=folds,
        group_column=group_column, time_column=time_column, random_seed=random_seed,
    )
    evaluation_actual, evaluation_predicted, evaluation_indices = [], [], []
    fold_metrics = []
    span = 72 / (len(split_items) + 1)
    for fold_number, (train_idx, valid_idx) in enumerate(split_items, 1):
        x_fit, x_valid = split_source.iloc[train_idx], split_source.iloc[valid_idx]
        y_fit, y_valid = y_trainable.loc[x_fit.index], y_trainable.loc[x_valid.index]
        model = _pipeline(numeric, categorical, RandomForestRegressor(**params))
        _fit(
            model, x_fit, y_fit, progress_callback=progress_callback,
            cancel_event=cancel_event, progress=5 + (fold_number - 1) * span,
            label=f"Random Forest · проверка {fold_number}/{len(split_items)}",
        )
        predicted = np.asarray(model.predict(x_valid), dtype=float)
        evaluation_actual.extend(y_valid.tolist())
        evaluation_predicted.extend(predicted.tolist())
        evaluation_indices.extend(x_valid.index.tolist())
        fold_metrics.append({"fold": fold_number, **_metrics(y_valid, predicted)})

    overall_metrics = _metrics(evaluation_actual, evaluation_predicted)
    baseline_value = float(y_trainable.mean())
    baseline_metrics = _metrics(
        evaluation_actual, np.full(len(evaluation_actual), baseline_value)
    )
    final_model = _pipeline(numeric, categorical, RandomForestRegressor(**params))
    _fit(
        final_model, x_trainable, y_trainable, progress_callback=progress_callback,
        cancel_event=cancel_event, progress=5 + len(split_items) * span,
        label="Random Forest · финальная модель",
    )
    all_predictions = np.asarray(final_model.predict(x_all), dtype=float)
    training_metrics = _metrics(y_trainable, final_model.predict(x_trainable))
    overfitting = _overfitting_summary("regression", training_metrics, overall_metrics)
    prediction_name = _unique_name(frame.columns, prediction_column)
    result_frame = frame.copy()
    result_frame[prediction_name] = all_predictions
    outputs = [prediction_name]
    residual_name = None
    if include_residual:
        residual_name = _unique_name(result_frame.columns, f"Остаток {target}")
        result_frame[residual_name] = y_all - all_predictions
        outputs.append(residual_name)
    importance = _feature_importance(final_model, numeric, categorical)
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
    analysis = {
        "model": "Random Forest", "task": "regression", "method": method,
        "evaluation_label": evaluation_label, "target": target,
        "group_column": str(group_column or ""), "time_column": str(time_column or ""),
        "features": selected, "categorical_features": categorical,
        "input_rows": int(len(frame)), "training_rows": int(train_mask.sum()),
        "excluded_target_rows": int((~train_mask).sum()),
        "metrics": overall_metrics, "training_metrics": training_metrics,
        "overfitting": overfitting,
        "baseline": {"value": baseline_value, **baseline_metrics},
        "fold_metrics": fold_metrics, "evaluation_rows": len(evaluation_actual),
        "actual": evaluation_actual[:6000], "predicted": evaluation_predicted[:6000],
        "preview": preview[:200], "feature_importance": importance,
        "outputs": outputs, "prediction_column": prediction_name,
        "residual_column": residual_name, "params": params,
        "oob_score": getattr(final_model.named_steps["model"], "oob_score_", None),
        "compute": {"requested": "cpu", "resolved": "CPU", "parallel_jobs": -1},
        "primary_metric_name": "MAE", "primary_metric_value": overall_metrics.get("mae"),
        "higher_is_better": False,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    step = {
        "type": "random_forest_regression", "operation": "random_forest_regression",
        "label": "Random Forest regression", "inputs": [target, *selected],
        "outputs": outputs,
        "params": {
            "method": method, "target": target, "features": selected,
            "n_estimators": params["n_estimators"], "max_depth": params["max_depth"],
            "metrics": overall_metrics, "compute_device": "CPU",
        },
    }
    if progress_callback:
        progress_callback(99, "Подготовка результатов")
    return CachedMLResult(result_frame, analysis, step, signature, final_model)


def run_random_forest_classification(
    frame: pd.DataFrame, *, target: str, features, id_column=None,
    method="split", test_size=0.2, folds=5, group_column=None, time_column=None,
    n_estimators=500, max_depth=0, min_samples_leaf=1, min_samples_split=2,
    max_features="sqrt", bootstrap=True, max_samples=0.8, oob_score=True,
    criterion="gini", class_weight="balanced", random_seed=42,
    prediction_column="Класс Random Forest", include_confidence=True,
    signature="", progress_callback=None, cancel_event=None,
):
    if progress_callback:
        progress_callback(2, "Подготовка данных")
    target, selected = _common_input(
        frame, target=target, features=features, id_column=id_column, method=method,
        group_column=group_column, time_column=time_column, task="classification",
    )
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
    x_all, numeric, categorical = _prepare_features(frame, selected)
    x_trainable = x_all.loc[train_mask]
    params = _model_parameters(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
        max_features=max_features, bootstrap=bootstrap, max_samples=max_samples,
        random_seed=random_seed, oob_score=oob_score,
        class_weight=class_weight, criterion=criterion,
    )
    split_items, split_source, evaluation_label = _splits(
        frame, train_mask, x_trainable, y_trainable, task="classification",
        method=method, test_size=test_size, folds=folds,
        group_column=group_column, time_column=time_column, random_seed=random_seed,
    )

    def aligned_probabilities(model, values):
        raw = np.asarray(model.predict_proba(values), dtype=float)
        labels = [str(value) for value in model.named_steps["model"].classes_]
        aligned = np.zeros((len(values), class_count), dtype=float)
        for source_index, label in enumerate(labels):
            if label in class_labels:
                aligned[:, class_labels.index(label)] = raw[:, source_index]
        sums = aligned.sum(axis=1, keepdims=True)
        return np.divide(
            aligned, sums, out=np.full_like(aligned, 1 / class_count), where=sums > 0,
        )

    evaluation_actual, evaluation_predicted = [], []
    evaluation_probabilities, evaluation_indices, fold_metrics = [], [], []
    span = 72 / (len(split_items) + 1)
    for fold_number, (train_idx, valid_idx) in enumerate(split_items, 1):
        x_fit, x_valid = split_source.iloc[train_idx], split_source.iloc[valid_idx]
        y_fit, y_valid = y_trainable.loc[x_fit.index], y_trainable.loc[x_valid.index]
        if y_fit.nunique() < 2:
            raise ValueError(f"В train fold {fold_number} остался один класс.")
        model = _pipeline(numeric, categorical, RandomForestClassifier(**params))
        _fit(
            model, x_fit, y_fit, progress_callback=progress_callback,
            cancel_event=cancel_event, progress=5 + (fold_number - 1) * span,
            label=f"Random Forest · проверка {fold_number}/{len(split_items)}",
        )
        predicted = np.asarray(model.predict(x_valid), dtype=object).astype(str)
        probabilities = aligned_probabilities(model, x_valid)
        evaluation_actual.extend(y_valid.astype(str).tolist())
        evaluation_predicted.extend(predicted.tolist())
        evaluation_probabilities.extend(probabilities.tolist())
        evaluation_indices.extend(x_valid.index.tolist())
        fold_metrics.append({
            "fold": fold_number,
            **_classification_metrics(y_valid, predicted, probabilities, class_labels),
        })
    probability_matrix = np.asarray(evaluation_probabilities, dtype=float)
    overall_metrics = _classification_metrics(
        evaluation_actual, evaluation_predicted, probability_matrix, class_labels
    )
    majority = str(y_trainable.value_counts().idxmax())
    priors = y_trainable.value_counts(normalize=True).reindex(class_labels, fill_value=0).to_numpy()
    baseline_metrics = _classification_metrics(
        evaluation_actual,
        np.full(len(evaluation_actual), majority, dtype=object),
        np.tile(priors, (len(evaluation_actual), 1)), class_labels,
    )
    final_model = _pipeline(numeric, categorical, RandomForestClassifier(**params))
    _fit(
        final_model, x_trainable, y_trainable, progress_callback=progress_callback,
        cancel_event=cancel_event, progress=5 + len(split_items) * span,
        label="Random Forest · финальная модель",
    )
    all_predictions = np.asarray(final_model.predict(x_all), dtype=object).astype(str)
    all_probabilities = aligned_probabilities(final_model, x_all)
    training_predictions = np.asarray(final_model.predict(x_trainable), dtype=object).astype(str)
    training_probabilities = aligned_probabilities(final_model, x_trainable)
    training_metrics = _classification_metrics(
        y_trainable, training_predictions, training_probabilities, class_labels
    )
    overfitting = _overfitting_summary("classification", training_metrics, overall_metrics)
    prediction_name = _unique_name(frame.columns, prediction_column or "Класс Random Forest")
    result_frame = frame.copy()
    result_frame[prediction_name] = all_predictions
    outputs = [prediction_name]
    confidence_name = None
    if include_confidence:
        confidence_name = _unique_name(result_frame.columns, "Уверенность Random Forest")
        result_frame[confidence_name] = all_probabilities.max(axis=1)
        outputs.append(confidence_name)
    importance = _feature_importance(final_model, numeric, categorical)
    confidence = probability_matrix.max(axis=1)
    preview = []
    for actual, predicted, score, row_index in zip(
        evaluation_actual, evaluation_predicted, confidence, evaluation_indices
    ):
        row = {
            "Строка": str(row_index), "Факт": str(actual), "Прогноз": str(predicted),
            "Уверенность": float(score),
            "Верно": "Да" if str(actual) == str(predicted) else "Нет",
        }
        if id_column in frame.columns:
            row["ID"] = str(frame.loc[row_index, id_column])
        preview.append(row)
    matrix = confusion_matrix(evaluation_actual, evaluation_predicted, labels=class_labels)
    analysis = {
        "model": "Random Forest", "task": "classification", "method": method,
        "evaluation_label": evaluation_label, "target": target,
        "group_column": str(group_column or ""), "time_column": str(time_column or ""),
        "features": selected, "categorical_features": categorical,
        "class_labels": class_labels, "class_count": class_count,
        "input_rows": int(len(frame)), "training_rows": int(train_mask.sum()),
        "excluded_target_rows": int((~train_mask).sum()),
        "metrics": overall_metrics, "training_metrics": training_metrics,
        "overfitting": overfitting,
        "baseline": {"class": majority, **baseline_metrics},
        "fold_metrics": fold_metrics, "evaluation_rows": len(evaluation_actual),
        "actual_labels": evaluation_actual[:6000],
        "predicted_labels": evaluation_predicted[:6000],
        "confidence": confidence[:6000].tolist(), "confusion_matrix": matrix.tolist(),
        "preview": preview[:200], "feature_importance": importance,
        "outputs": outputs, "prediction_column": prediction_name,
        "confidence_column": confidence_name, "params": params,
        "oob_score": getattr(final_model.named_steps["model"], "oob_score_", None),
        "compute": {"requested": "cpu", "resolved": "CPU", "parallel_jobs": -1},
        "primary_metric_name": "Accuracy",
        "primary_metric_value": overall_metrics.get("accuracy"),
        "higher_is_better": True,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    step = {
        "type": "random_forest_classification",
        "operation": "random_forest_classification",
        "label": "Random Forest classification", "inputs": [target, *selected],
        "outputs": outputs,
        "params": {
            "method": method, "target": target, "features": selected,
            "classes": class_labels, "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"], "metrics": overall_metrics,
            "compute_device": "CPU",
        },
    }
    if progress_callback:
        progress_callback(99, "Подготовка результатов")
    return CachedMLResult(result_frame, analysis, step, signature, final_model)
