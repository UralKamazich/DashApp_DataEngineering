# -*- coding: utf-8 -*-
"""Tabular neural-network regression/classification for the ML workspace."""

from __future__ import annotations

from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from ml_engine import (
    CachedMLResult,
    _check_cancel,
    _classification_metrics,
    _metrics,
    _overfitting_summary,
    _unique_name,
)
from random_forest_engine import _common_input, _prepare_features, _splits
from torch_tabular_engine import fit_torch_tabular, torch_runtime


def _hidden_layers(value):
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not parts:
        raise ValueError("Укажите архитектуру скрытых слоёв, например 64, 32.")
    try:
        layers = tuple(int(item) for item in parts)
    except (TypeError, ValueError) as error:
        raise ValueError("Слои задаются целыми числами через запятую: 64, 32.") from error
    if len(layers) > 6:
        raise ValueError("Допускается не больше шести скрытых слоёв.")
    if any(value < 1 or value > 2048 for value in layers):
        raise ValueError("В каждом скрытом слое должно быть от 1 до 2 048 нейронов.")
    if sum(layers) > 8192:
        raise ValueError("Суммарно допускается не больше 8 192 нейронов.")
    return layers


def _model_parameters(*, hidden_layers, activation, solver, max_iter,
                      learning_rate_init, alpha, batch_size, early_stopping,
                      validation_fraction, n_iter_no_change, tolerance,
                      random_seed, min_category_frequency):
    layers = _hidden_layers(hidden_layers)
    activation = str(activation or "relu")
    if activation not in {"relu", "tanh", "logistic", "identity"}:
        raise ValueError("Выбрана неподдерживаемая функция активации.")
    solver = str(solver or "adam")
    if solver not in {"adam", "sgd", "lbfgs"}:
        raise ValueError("Выбран неподдерживаемый оптимизатор.")
    epochs = int(max_iter or 0)
    if not 20 <= epochs <= 5000:
        raise ValueError("Количество эпох должно быть от 20 до 5 000.")
    rate = float(learning_rate_init or 0)
    if not 0 < rate <= 1:
        raise ValueError("Learning rate должен быть в диапазоне (0; 1].")
    regularization = float(alpha or 0)
    if regularization < 0:
        raise ValueError("L2-регуляризация не может быть отрицательной.")
    batch = int(batch_size or 0)
    if batch < 1 or batch > 65_536:
        raise ValueError("Размер batch должен быть от 1 до 65 536.")
    fraction = float(validation_fraction or 0)
    if not .05 <= fraction <= .4:
        raise ValueError("Внутренняя validation-доля должна быть от 0.05 до 0.4.")
    patience = int(n_iter_no_change or 0)
    if patience < 2 or patience > 500:
        raise ValueError("Patience должен быть от 2 до 500 эпох.")
    tolerance = float(tolerance or 0)
    if tolerance <= 0:
        raise ValueError("Tolerance должен быть больше нуля.")
    category_frequency = int(min_category_frequency or 1)
    if category_frequency < 1 or category_frequency > 10_000:
        raise ValueError("Минимальная частота категории должна быть от 1 до 10 000.")
    return {
        "hidden_layer_sizes": layers,
        "activation": activation,
        "solver": solver,
        "max_iter": epochs,
        "learning_rate_init": rate,
        "alpha": regularization,
        "batch_size": batch,
        "early_stopping": bool(early_stopping) and solver != "lbfgs",
        "validation_fraction": fraction,
        "n_iter_no_change": patience,
        "tol": tolerance,
        "random_state": int(random_seed or 0),
        "min_category_frequency": category_frequency,
    }


def _resolved_engine(engine, compute_device):
    engine = str(engine or "pytorch").strip().lower()
    if engine not in {"sklearn", "pytorch"}:
        raise ValueError("Движок должен быть sklearn или PyTorch.")
    compute_device = str(compute_device or "auto").strip().lower()
    if compute_device not in {"auto", "cpu", "mps"}:
        raise ValueError("Вычислитель должен быть Auto, CPU или GPU · MPS.")
    if engine == "sklearn":
        return engine, "cpu"
    return engine, compute_device


def _encoded_feature_estimate(frame, numeric, categorical, min_frequency):
    estimate = len(numeric)
    for column in categorical:
        counts = frame[column].value_counts(dropna=False)
        if min_frequency > 1:
            frequent = int((counts >= min_frequency).sum())
            rare = int((counts < min_frequency).any())
            estimate += frequent + rare
        else:
            estimate += max(1, int(len(counts)))
    return estimate


def _preprocessor(numeric, categorical, min_category_frequency):
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        onehot_kwargs = {
            "handle_unknown": "infrequent_if_exist",
            "sparse_output": False,
        }
        if int(min_category_frequency or 1) > 1:
            onehot_kwargs["min_frequency"] = int(min_category_frequency)
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent", keep_empty_features=True,
                )),
                ("onehot", OneHotEncoder(**onehot_kwargs)),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def _effective_early_stopping(task, y, params):
    if not params["early_stopping"]:
        return False
    validation_rows = int(np.ceil(len(y) * params["validation_fraction"]))
    if validation_rows < 2 or len(y) - validation_rows < 2:
        return False
    if task == "classification":
        counts = pd.Series(y).value_counts()
        return (
            int(counts.min()) >= 2
            and validation_rows >= len(counts)
            and len(y) - validation_rows >= len(counts)
        )
    return True


def _pipeline(task, numeric, categorical, params, y):
    model_params = {
        key: value for key, value in params.items()
        if key != "min_category_frequency"
    }
    model_params["early_stopping"] = _effective_early_stopping(task, y, params)
    if model_params["solver"] != "lbfgs":
        effective_rows = len(y)
        if model_params["early_stopping"]:
            effective_rows -= int(np.ceil(len(y) * model_params["validation_fraction"]))
        model_params["batch_size"] = min(
            int(model_params["batch_size"]), max(1, effective_rows)
        )
    preprocessor = _preprocessor(
        numeric, categorical, params["min_category_frequency"]
    )
    if task == "classification":
        estimator = MLPClassifier(**model_params)
    else:
        regressor = MLPRegressor(**model_params)
        estimator = TransformedTargetRegressor(
            regressor=regressor, transformer=StandardScaler(),
        )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def _fit(model, x, y, *, task, class_balance, progress_callback,
         cancel_event, progress, label):
    _check_cancel(cancel_event)
    if progress_callback:
        progress_callback(progress, label)
    fit_kwargs = {}
    if task == "classification" and class_balance == "balanced":
        fit_kwargs["model__sample_weight"] = compute_sample_weight(
            class_weight="balanced", y=y,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y, **fit_kwargs)
    _check_cancel(cancel_event)


def _fitted_mlp(model, task):
    fitted = model.named_steps["model"]
    return fitted if task == "classification" else fitted.regressor_


def _history(model, task, label):
    fitted = _fitted_mlp(model, task)
    loss = [float(value) for value in (getattr(fitted, "loss_curve_", None) or [])]
    validation = [
        float(value) for value in (getattr(fitted, "validation_scores_", None) or [])
    ]
    return {
        "label": label,
        "epochs": list(range(1, len(loss) + 1)),
        "loss": loss,
        "validation": validation,
        "n_iter": int(getattr(fitted, "n_iter_", len(loss)) or len(loss)),
        "early_stopping": bool(getattr(fitted, "early_stopping", False)),
        "best_validation_score": getattr(fitted, "best_validation_score_", None),
    }


def _train_model(task, x, y, numeric, categorical, params, *, engine,
                 compute_device, class_balance, class_labels,
                 progress_callback, cancel_event, progress_start, progress_span,
                 label):
    if engine == "pytorch":
        model, history, compute = fit_torch_tabular(
            x, y,
            preprocessor=_preprocessor(
                numeric, categorical, params["min_category_frequency"]
            ),
            task=task, classes=class_labels,
            hidden_layers=params["hidden_layer_sizes"],
            activation=params["activation"], solver=params["solver"],
            max_iter=params["max_iter"],
            learning_rate=params["learning_rate_init"], alpha=params["alpha"],
            batch_size=params["batch_size"],
            early_stopping=_effective_early_stopping(task, y, params),
            validation_fraction=params["validation_fraction"],
            patience=params["n_iter_no_change"], tolerance=params["tol"],
            class_balance=class_balance, random_seed=params["random_state"],
            compute_device=compute_device, progress_callback=progress_callback,
            cancel_event=cancel_event, progress_start=progress_start,
            progress_span=progress_span,
        )
        history["label"] = label
        return model, history, compute
    model = _pipeline(task, numeric, categorical, params, y)
    _fit(
        model, x, y, task=task, class_balance=class_balance,
        progress_callback=progress_callback, cancel_event=cancel_event,
        progress=progress_start, label=label,
    )
    return model, _history(model, task, label), {
        "requested": "cpu", "resolved": "CPU", "device": "cpu",
        "engine": "sklearn MLP", "mps_available": torch_runtime()["mps_available"],
    }


def _model_classes(model, task):
    if hasattr(model, "classes_"):
        return [str(value) for value in model.classes_]
    return [str(value) for value in _fitted_mlp(model, task).classes_]


def _permutation_feature_importance(model, x, y, features, *, task, repeats,
                                    random_seed, cancel_event):
    _check_cancel(cancel_event)
    count = int(repeats or 0)
    if count < 1:
        return []
    actual = np.asarray(y)
    baseline_prediction = np.asarray(model.predict(x))
    if task == "classification":
        baseline_score = float(balanced_accuracy_score(actual.astype(str), baseline_prediction.astype(str)))
    else:
        baseline_score = -float(mean_absolute_error(actual.astype(float), baseline_prediction.astype(float)))
    rng = np.random.default_rng(int(random_seed or 0))
    importance = []
    for feature in features:
        changes = []
        for _repeat in range(count):
            _check_cancel(cancel_event)
            permuted = x.copy()
            values = permuted[feature].to_numpy(copy=True)
            rng.shuffle(values)
            permuted[feature] = values
            predicted = np.asarray(model.predict(permuted))
            if task == "classification":
                score = float(balanced_accuracy_score(
                    actual.astype(str), predicted.astype(str)
                ))
            else:
                score = -float(mean_absolute_error(
                    actual.astype(float), predicted.astype(float)
                ))
            changes.append(baseline_score - score)
        importance.append({
            "feature": str(feature),
            "importance": float(np.mean(changes)),
            "std": float(np.std(changes)),
        })
    return sorted(importance, key=lambda item: item["importance"], reverse=True)


def _common_setup(frame, *, target, features, id_column, method,
                  group_column, time_column, task, params):
    target, selected = _common_input(
        frame, target=target, features=features, id_column=id_column,
        method=method, group_column=group_column, time_column=time_column,
        task=task,
    )
    x_all, numeric, categorical = _prepare_features(frame, selected)
    estimate = _encoded_feature_estimate(
        x_all, numeric, categorical, params["min_category_frequency"]
    )
    estimated_bytes = max(1, len(frame)) * max(1, estimate) * 8
    if estimate > 50_000 or estimated_bytes > 1_500_000_000:
        raise ValueError(
            f"После кодирования получится около {estimate:,} признаков. "
            "Увеличьте минимальную частоту категории или уберите ID-подобные каналы."
        )
    return target, selected, x_all, numeric, categorical, estimate


def run_neural_network_regression(
    frame: pd.DataFrame, *, target: str, features, id_column=None,
    method="split", test_size=.2, folds=5, group_column=None, time_column=None,
    hidden_layers="64, 32", activation="relu", solver="adam", max_iter=500,
    learning_rate_init=.001, alpha=.0001, batch_size=64,
    early_stopping=True, validation_fraction=.15, n_iter_no_change=30,
    tolerance=.0001, min_category_frequency=2, permutation_repeats=3,
    engine="pytorch", compute_device="auto",
    random_seed=42, prediction_column="Прогноз Neural Network",
    include_residual=True, signature="", progress_callback=None,
    cancel_event=None,
):
    if progress_callback:
        progress_callback(2, "Подготовка данных")
    engine, compute_device = _resolved_engine(engine, compute_device)
    params = _model_parameters(
        hidden_layers=hidden_layers, activation=activation, solver=solver,
        max_iter=max_iter, learning_rate_init=learning_rate_init, alpha=alpha,
        batch_size=batch_size, early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change, tolerance=tolerance,
        random_seed=random_seed, min_category_frequency=min_category_frequency,
    )
    target, selected, x_all, numeric, categorical, encoded_estimate = _common_setup(
        frame, target=target, features=features, id_column=id_column, method=method,
        group_column=group_column, time_column=time_column,
        task="regression", params=params,
    )
    y_all = pd.to_numeric(frame[target], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    train_mask = y_all.notna()
    if int(train_mask.sum()) < 10:
        raise ValueError("После исключения пустой цели осталось меньше десяти строк.")
    x_trainable = x_all.loc[train_mask]
    y_trainable = y_all.loc[train_mask].astype(float)
    split_items, split_source, evaluation_label = _splits(
        frame, train_mask, x_trainable, y_trainable, task="regression",
        method=method, test_size=test_size, folds=folds,
        group_column=group_column, time_column=time_column,
        random_seed=random_seed,
    )
    actual_values, predicted_values, evaluation_indices = [], [], []
    fold_metrics, curves = [], []
    importance_source = None
    span = 70 / (len(split_items) + 1)
    for fold_number, (train_idx, valid_idx) in enumerate(split_items, 1):
        x_fit, x_valid = split_source.iloc[train_idx], split_source.iloc[valid_idx]
        y_fit, y_valid = y_trainable.loc[x_fit.index], y_trainable.loc[x_valid.index]
        model, curve, _fold_compute = _train_model(
            "regression", x_fit, y_fit, numeric, categorical, params,
            engine=engine, compute_device=compute_device, class_balance="none",
            class_labels=None, progress_callback=progress_callback,
            cancel_event=cancel_event,
            progress_start=5 + (fold_number - 1) * span,
            progress_span=span * .9,
            label=f"Fold {fold_number}",
        )
        predicted = np.asarray(model.predict(x_valid), dtype=float)
        actual_values.extend(y_valid.tolist())
        predicted_values.extend(predicted.tolist())
        evaluation_indices.extend(x_valid.index.tolist())
        fold_metrics.append({"fold": fold_number, **_metrics(y_valid, predicted)})
        curves.append(curve)
        if importance_source is None:
            importance_source = (model, x_valid, y_valid)

    metrics = _metrics(actual_values, predicted_values)
    baseline_value = float(y_trainable.mean())
    baseline = _metrics(actual_values, np.full(len(actual_values), baseline_value))
    final_model, final_history, final_compute = _train_model(
        "regression", x_trainable, y_trainable, numeric, categorical, params,
        engine=engine, compute_device=compute_device, class_balance="none",
        class_labels=None, progress_callback=progress_callback,
        cancel_event=cancel_event,
        progress_start=5 + len(split_items) * span, progress_span=span * .9,
        label="Финальная модель",
    )
    all_predictions = np.asarray(final_model.predict(x_all), dtype=float)
    training_metrics = _metrics(y_trainable, final_model.predict(x_trainable))
    overfitting = _overfitting_summary("regression", training_metrics, metrics)
    importance = []
    if int(permutation_repeats or 0) > 0 and importance_source:
        if progress_callback:
            progress_callback(88, "Permutation importance")
        importance = _permutation_feature_importance(
            *importance_source, selected, task="regression",
            repeats=permutation_repeats, random_seed=random_seed,
            cancel_event=cancel_event,
        )
    prediction_name = _unique_name(frame.columns, prediction_column)
    result_frame = frame.copy()
    result_frame[prediction_name] = all_predictions
    outputs = [prediction_name]
    residual_name = None
    if include_residual:
        residual_name = _unique_name(result_frame.columns, f"Остаток {target}")
        result_frame[residual_name] = y_all - all_predictions
        outputs.append(residual_name)
    preview = []
    for actual, predicted, row_index in zip(
        actual_values, predicted_values, evaluation_indices
    ):
        row = {
            "Строка": str(row_index), "Факт": float(actual),
            "Прогноз": float(predicted), "Ошибка": float(actual - predicted),
        }
        if id_column in frame.columns:
            row["ID"] = str(frame.loc[row_index, id_column])
        preview.append(row)
    analysis_params = dict(params)
    analysis_params["hidden_layer_sizes"] = list(params["hidden_layer_sizes"])
    analysis_params["engine"] = engine
    analysis_params["compute_device"] = compute_device
    analysis = {
        "model": "Neural Network",
        "engine": "PyTorch" if engine == "pytorch" else "sklearn MLP",
        "task": "regression", "method": method,
        "evaluation_label": evaluation_label, "target": target,
        "group_column": str(group_column or ""), "time_column": str(time_column or ""),
        "features": selected, "numeric_features": numeric,
        "categorical_features": categorical, "encoded_feature_estimate": encoded_estimate,
        "input_rows": int(len(frame)), "training_rows": int(train_mask.sum()),
        "excluded_target_rows": int((~train_mask).sum()),
        "metrics": metrics, "training_metrics": training_metrics,
        "overfitting": overfitting,
        "baseline": {"value": baseline_value, **baseline},
        "fold_metrics": fold_metrics, "evaluation_rows": len(actual_values),
        "actual": actual_values[:6000], "predicted": predicted_values[:6000],
        "preview": preview[:200], "feature_importance": importance,
        "learning_curves": curves, "final_learning_curve": final_history,
        "epochs_run": final_history["n_iter"],
        "outputs": outputs, "prediction_column": prediction_name,
        "residual_column": residual_name, "params": analysis_params,
        "compute": final_compute,
        "primary_metric_name": "MAE", "primary_metric_value": metrics.get("mae"),
        "higher_is_better": False,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    step = {
        "type": "neural_network_regression",
        "operation": "neural_network_regression",
        "label": "Neural Network regression", "inputs": [target, *selected],
        "outputs": outputs,
        "params": {
            "method": method, "target": target, "features": selected,
            "hidden_layers": list(params["hidden_layer_sizes"]),
            "metrics": metrics,
            "engine": "PyTorch" if engine == "pytorch" else "sklearn MLP",
            "compute_device": final_compute.get("resolved", "CPU"),
        },
    }
    if progress_callback:
        progress_callback(99, "Подготовка результатов")
    return CachedMLResult(result_frame, analysis, step, signature, final_model)


def run_neural_network_classification(
    frame: pd.DataFrame, *, target: str, features, id_column=None,
    method="split", test_size=.2, folds=5, group_column=None, time_column=None,
    hidden_layers="64, 32", activation="relu", solver="adam", max_iter=500,
    learning_rate_init=.001, alpha=.0001, batch_size=64,
    early_stopping=True, validation_fraction=.15, n_iter_no_change=30,
    tolerance=.0001, min_category_frequency=2, permutation_repeats=3,
    class_balance="balanced", engine="pytorch", compute_device="auto",
    random_seed=42,
    prediction_column="Класс Neural Network", include_confidence=True,
    signature="", progress_callback=None, cancel_event=None,
):
    if progress_callback:
        progress_callback(2, "Подготовка данных")
    engine, compute_device = _resolved_engine(engine, compute_device)
    params = _model_parameters(
        hidden_layers=hidden_layers, activation=activation, solver=solver,
        max_iter=max_iter, learning_rate_init=learning_rate_init, alpha=alpha,
        batch_size=batch_size, early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change, tolerance=tolerance,
        random_seed=random_seed, min_category_frequency=min_category_frequency,
    )
    target, selected, x_all, numeric, categorical, encoded_estimate = _common_setup(
        frame, target=target, features=features, id_column=id_column, method=method,
        group_column=group_column, time_column=time_column,
        task="classification", params=params,
    )
    raw_target = frame[target].astype("string")
    train_mask = raw_target.notna() & raw_target.str.strip().ne("")
    if int(train_mask.sum()) < 12:
        raise ValueError("После исключения пустой цели осталось меньше двенадцати строк.")
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
    x_trainable = x_all.loc[train_mask]
    split_items, split_source, evaluation_label = _splits(
        frame, train_mask, x_trainable, y_trainable, task="classification",
        method=method, test_size=test_size, folds=folds,
        group_column=group_column, time_column=time_column,
        random_seed=random_seed,
    )

    def aligned_probabilities(model, values):
        raw = np.asarray(model.predict_proba(values), dtype=float)
        labels = _model_classes(model, "classification")
        aligned = np.zeros((len(values), class_count), dtype=float)
        for source_index, label in enumerate(labels):
            if label in class_labels:
                aligned[:, class_labels.index(label)] = raw[:, source_index]
        sums = aligned.sum(axis=1, keepdims=True)
        return np.divide(
            aligned, sums, out=np.full_like(aligned, 1 / class_count), where=sums > 0,
        )

    actual_values, predicted_values, probabilities_values = [], [], []
    evaluation_indices, fold_metrics, curves = [], [], []
    importance_source = None
    span = 70 / (len(split_items) + 1)
    for fold_number, (train_idx, valid_idx) in enumerate(split_items, 1):
        x_fit, x_valid = split_source.iloc[train_idx], split_source.iloc[valid_idx]
        y_fit, y_valid = y_trainable.loc[x_fit.index], y_trainable.loc[x_valid.index]
        if y_fit.nunique() < 2:
            raise ValueError(f"В train fold {fold_number} остался один класс.")
        model, curve, _fold_compute = _train_model(
            "classification", x_fit, y_fit, numeric, categorical, params,
            engine=engine, compute_device=compute_device,
            class_balance=class_balance, class_labels=class_labels,
            progress_callback=progress_callback, cancel_event=cancel_event,
            progress_start=5 + (fold_number - 1) * span,
            progress_span=span * .9,
            label=f"Fold {fold_number}",
        )
        predicted = np.asarray(model.predict(x_valid), dtype=object).astype(str)
        probabilities = aligned_probabilities(model, x_valid)
        actual_values.extend(y_valid.astype(str).tolist())
        predicted_values.extend(predicted.tolist())
        probabilities_values.extend(probabilities.tolist())
        evaluation_indices.extend(x_valid.index.tolist())
        fold_metrics.append({
            "fold": fold_number,
            **_classification_metrics(y_valid, predicted, probabilities, class_labels),
        })
        curves.append(curve)
        if importance_source is None:
            importance_source = (model, x_valid, y_valid)

    probability_matrix = np.asarray(probabilities_values, dtype=float)
    metrics = _classification_metrics(
        actual_values, predicted_values, probability_matrix, class_labels
    )
    majority = str(y_trainable.value_counts().idxmax())
    priors = y_trainable.value_counts(normalize=True).reindex(
        class_labels, fill_value=0
    ).to_numpy()
    baseline = _classification_metrics(
        actual_values, np.full(len(actual_values), majority, dtype=object),
        np.tile(priors, (len(actual_values), 1)), class_labels,
    )
    final_model, final_history, final_compute = _train_model(
        "classification", x_trainable, y_trainable,
        numeric, categorical, params, engine=engine,
        compute_device=compute_device, class_balance=class_balance,
        class_labels=class_labels, progress_callback=progress_callback,
        cancel_event=cancel_event,
        progress_start=5 + len(split_items) * span, progress_span=span * .9,
        label="Финальная модель",
    )
    all_predictions = np.asarray(final_model.predict(x_all), dtype=object).astype(str)
    all_probabilities = aligned_probabilities(final_model, x_all)
    train_predictions = np.asarray(
        final_model.predict(x_trainable), dtype=object
    ).astype(str)
    train_probabilities = aligned_probabilities(final_model, x_trainable)
    training_metrics = _classification_metrics(
        y_trainable, train_predictions, train_probabilities, class_labels
    )
    overfitting = _overfitting_summary("classification", training_metrics, metrics)
    importance = []
    if int(permutation_repeats or 0) > 0 and importance_source:
        if progress_callback:
            progress_callback(88, "Permutation importance")
        importance = _permutation_feature_importance(
            *importance_source, selected, task="classification",
            repeats=permutation_repeats, random_seed=random_seed,
            cancel_event=cancel_event,
        )
    prediction_name = _unique_name(
        frame.columns, prediction_column or "Класс Neural Network"
    )
    result_frame = frame.copy()
    result_frame[prediction_name] = all_predictions
    outputs = [prediction_name]
    confidence_name = None
    if include_confidence:
        confidence_name = _unique_name(result_frame.columns, "Уверенность Neural Network")
        result_frame[confidence_name] = all_probabilities.max(axis=1)
        outputs.append(confidence_name)
    confidence = probability_matrix.max(axis=1)
    preview = []
    for actual, predicted, score, row_index in zip(
        actual_values, predicted_values, confidence, evaluation_indices
    ):
        row = {
            "Строка": str(row_index), "Факт": str(actual),
            "Прогноз": str(predicted), "Уверенность": float(score),
            "Верно": "Да" if str(actual) == str(predicted) else "Нет",
        }
        if id_column in frame.columns:
            row["ID"] = str(frame.loc[row_index, id_column])
        preview.append(row)
    matrix = confusion_matrix(actual_values, predicted_values, labels=class_labels)
    analysis_params = dict(params)
    analysis_params["hidden_layer_sizes"] = list(params["hidden_layer_sizes"])
    analysis_params["class_balance"] = str(class_balance or "none")
    analysis_params["engine"] = engine
    analysis_params["compute_device"] = compute_device
    analysis = {
        "model": "Neural Network",
        "engine": "PyTorch" if engine == "pytorch" else "sklearn MLP",
        "task": "classification", "method": method,
        "evaluation_label": evaluation_label, "target": target,
        "group_column": str(group_column or ""), "time_column": str(time_column or ""),
        "features": selected, "numeric_features": numeric,
        "categorical_features": categorical, "encoded_feature_estimate": encoded_estimate,
        "class_labels": class_labels, "class_count": class_count,
        "input_rows": int(len(frame)), "training_rows": int(train_mask.sum()),
        "excluded_target_rows": int((~train_mask).sum()),
        "metrics": metrics, "training_metrics": training_metrics,
        "overfitting": overfitting, "baseline": {"class": majority, **baseline},
        "fold_metrics": fold_metrics, "evaluation_rows": len(actual_values),
        "actual_labels": actual_values[:6000],
        "predicted_labels": predicted_values[:6000],
        "confidence": confidence[:6000].tolist(),
        "confusion_matrix": matrix.tolist(),
        "preview": preview[:200], "feature_importance": importance,
        "learning_curves": curves, "final_learning_curve": final_history,
        "epochs_run": final_history["n_iter"],
        "outputs": outputs, "prediction_column": prediction_name,
        "confidence_column": confidence_name, "params": analysis_params,
        "compute": final_compute,
        "primary_metric_name": "Balanced accuracy",
        "primary_metric_value": metrics.get("balanced_accuracy"),
        "higher_is_better": True,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    step = {
        "type": "neural_network_classification",
        "operation": "neural_network_classification",
        "label": "Neural Network classification", "inputs": [target, *selected],
        "outputs": outputs,
        "params": {
            "method": method, "target": target, "features": selected,
            "hidden_layers": list(params["hidden_layer_sizes"]),
            "metrics": metrics,
            "engine": "PyTorch" if engine == "pytorch" else "sklearn MLP",
            "compute_device": final_compute.get("resolved", "CPU"),
        },
    }
    if progress_callback:
        progress_callback(99, "Подготовка результатов")
    return CachedMLResult(result_frame, analysis, step, signature, final_model)
