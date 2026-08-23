# -*- coding: utf-8 -*-
"""Data Engineering operations shared by the UI pipeline."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _unique_name(columns, base):
    if base not in columns:
        return base
    index = 2
    while f"{base}_{index}" in columns:
        index += 1
    return f"{base}_{index}"


def apply_binning(df, column, method, groups, label_style):
    if not column or column not in df.columns:
        raise ValueError("Выберите числовой канал для биннинга.")
    groups = int(groups or 0)
    if groups < 2:
        raise ValueError("Количество групп должно быть не меньше двух.")
    values = pd.to_numeric(df[column], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        raise ValueError("В выбранном канале нет числовых значений.")
    if method == "width":
        bins = pd.cut(valid, bins=groups, include_lowest=True, duplicates="drop")
    else:
        bins = pd.qcut(valid, q=groups, duplicates="drop")
    output = _unique_name(df.columns, f"Группа({column})")
    if label_style == "index":
        labels = pd.Series(
            [f"Группа {code + 1}" if code >= 0 else pd.NA for code in bins.cat.codes],
            index=valid.index,
            dtype="string",
        )
    else:
        labels = bins.astype("string")
    df[output] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[labels.index, output] = labels
    return df, [output], {
        "type": "binning",
        "label": f"Биннинг · {column}",
        "inputs": [column],
        "outputs": [output],
        "params": {"method": method, "groups": groups, "labels": label_style},
    }


def apply_text_copy(df, columns, suffix, strip_values):
    selected = [column for column in (columns or []) if column in df.columns]
    if not selected:
        raise ValueError("Выберите хотя бы один канал.")
    suffix = str(suffix or "_txt").strip() or "_txt"
    if not suffix.startswith("_"):
        suffix = "_" + suffix
    outputs = []
    for column in selected:
        output = _unique_name(df.columns, f"{column}{suffix}")
        values = df[column].astype("string")
        if strip_values:
            values = values.str.strip()
        df[output] = values
        outputs.append(output)
    return df, outputs, {
        "type": "text_copy",
        "label": f"Текстовая копия · {len(outputs)}",
        "inputs": selected,
        "outputs": outputs,
        "params": {"suffix": suffix, "strip": bool(strip_values)},
    }


def apply_group_aggregates(
    df,
    keys,
    columns,
    metrics,
    exclude_zeros=False,
    exclude_empty=True,
):
    keys = [column for column in (keys or []) if column in df.columns]
    columns = [column for column in (columns or []) if column in df.columns]
    metrics = list(dict.fromkeys(metrics or []))
    if not keys or not columns or not metrics:
        raise ValueError("Выберите ключи, каналы и метрики.")

    groupers = [df[column] for column in keys]
    outputs = []
    skipped = []

    def transform(series, function):
        try:
            return series.groupby(groupers, dropna=False, sort=False).transform(function)
        except TypeError:
            return series.groupby(groupers, sort=False).transform(function)

    def apply(series, function):
        try:
            return series.groupby(groupers, dropna=False, sort=False).transform(function)
        except TypeError:
            return series.groupby(groupers, sort=False).transform(function)

    for column in columns:
        numeric = pd.api.types.is_numeric_dtype(df[column])
        numeric_values = pd.to_numeric(df[column], errors="coerce") if numeric else None
        if numeric:
            prepared = numeric_values.copy()
            if not exclude_empty:
                prepared = prepared.fillna(0)
            if exclude_zeros:
                prepared = prepared.mask(prepared == 0, np.nan)

        for metric in metrics:
            result = None
            if metric in {"mean", "median", "sum", "min", "max", "std"}:
                if numeric:
                    result = transform(prepared, metric)
            elif metric == "count":
                if exclude_empty:
                    source = prepared if numeric else df[column]
                    result = transform(source, "count")
                else:
                    result = apply(df[column], lambda values: len(values))
            elif metric == "nunique":
                source = df[column]
                if not exclude_empty:
                    source = source.astype("object").where(pd.notna(source), "<EMPTY>")
                if exclude_zeros and numeric:
                    converted = pd.to_numeric(source, errors="coerce")
                    source = converted.mask(converted == 0, np.nan)
                result = apply(source, lambda values: values.nunique(dropna=exclude_empty))
            elif metric == "mode":
                source = prepared if numeric else df[column]
                if not numeric and not exclude_empty:
                    source = source.astype("object").where(pd.notna(source), "<EMPTY>")

                def first_mode(values):
                    modes = values.mode(dropna=exclude_empty)
                    return modes.iloc[0] if len(modes) else np.nan

                result = apply(source, first_mode)
            elif metric == "cumsum" and numeric:
                result = prepared.groupby(groupers, sort=False).cumsum()

            if result is None:
                skipped.append(f"{column}:{metric}")
                continue
            output = _unique_name(df.columns, f"{column}_{metric}")
            df[output] = result
            outputs.append(output)

    if not outputs:
        raise ValueError("Нет совместимых сочетаний каналов и метрик.")
    return df, outputs, {
        "type": "group_aggregates",
        "label": f"Агрегаты по группам · {len(outputs)}",
        "inputs": list(dict.fromkeys([*keys, *columns])),
        "outputs": outputs,
        "params": {
            "keys": keys,
            "metrics": metrics,
            "exclude_zeros": bool(exclude_zeros),
            "exclude_empty": bool(exclude_empty),
            "skipped": skipped,
        },
    }


def execute_pipeline(df, queued_steps):
    """Execute queued feature operations on one copy and return one result."""
    result = df.copy()
    all_outputs = []
    committed_steps = []

    for position, queued in enumerate(queued_steps or [], 1):
        operation = (queued or {}).get("operation")
        params = dict((queued or {}).get("params") or {})
        try:
            if operation == "binning":
                result, outputs, committed = apply_binning(
                    result,
                    params.get("column"),
                    params.get("method") or "count",
                    params.get("groups"),
                    params.get("label_style") or "interval",
                )
            elif operation == "text_copy":
                result, outputs, committed = apply_text_copy(
                    result,
                    params.get("columns"),
                    params.get("suffix"),
                    bool(params.get("strip", True)),
                )
            elif operation == "group_aggregates":
                result, outputs, committed = apply_group_aggregates(
                    result,
                    params.get("keys"),
                    params.get("columns"),
                    params.get("metrics"),
                    bool(params.get("exclude_zeros")),
                    bool(params.get("exclude_empty", True)),
                )
            else:
                raise ValueError("Неизвестный тип операции.")
        except Exception as error:
            label = (queued or {}).get("label") or operation or "Операция"
            raise ValueError(f"Шаг {position} · {label}: {error}") from error

        committed["scope"] = (queued or {}).get("scope") or "base"
        all_outputs.extend(outputs)
        committed_steps.append(committed)

    if not committed_steps:
        raise ValueError("Конвейер пуст.")
    return result, all_outputs, committed_steps
