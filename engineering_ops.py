# -*- coding: utf-8 -*-
"""Data Engineering operations shared by the UI pipeline."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


MAX_RESHAPE_COLUMNS = 5_000


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


def _validated_columns(df, columns, label, *, required=True):
    selected = list(dict.fromkeys(str(column) for column in (columns or [])))
    missing = [column for column in selected if column not in df.columns]
    if missing:
        raise ValueError(f"Не найдены каналы {label}: {', '.join(missing[:4])}.")
    if required and not selected:
        raise ValueError(f"Выберите {label}.")
    return selected


def _flatten_pivot_columns(columns, separator, reserved):
    """Flatten pandas pivot columns without silently creating duplicates."""
    separator = str(separator or "__")
    used = set(str(column) for column in reserved)
    flattened = []
    for column in columns:
        parts = column if isinstance(column, tuple) else (column,)
        base = separator.join(
            str(part) for part in parts if part is not None and str(part) != ""
        ) or "Значение"
        name = _unique_name(used, base)
        used.add(name)
        flattened.append(name)
    return flattened


def apply_long_to_wide(
    df,
    index_columns,
    names_from,
    value_columns,
    aggregation="error",
    separator="__",
):
    """Pivot a long table into a wide table with explicit duplicate handling."""
    if not df.columns.is_unique:
        raise ValueError("Long → Wide требует уникальные названия каналов.")
    index_columns = _validated_columns(df, index_columns, "идентификаторы строк")
    value_columns = _validated_columns(df, value_columns, "каналы значений")
    if not names_from or names_from not in df.columns:
        raise ValueError("Выберите канал, из значений которого будут созданы заголовки.")

    overlap = set(index_columns) & ({names_from} | set(value_columns))
    if overlap or names_from in value_columns:
        raise ValueError("Идентификаторы, заголовки и значения должны быть разными каналами.")

    aggregation = str(aggregation or "error")
    allowed = {"error", "first", "last", "sum", "mean", "min", "max", "count"}
    if aggregation not in allowed:
        raise ValueError("Неизвестная политика обработки повторяющихся строк.")

    unique_headers = int(df[names_from].nunique(dropna=False))
    projected_columns = unique_headers * len(value_columns)
    if projected_columns > MAX_RESHAPE_COLUMNS:
        raise ValueError(
            f"Long → Wide создаст около {projected_columns} каналов. "
            f"Допустимо не более {MAX_RESHAPE_COLUMNS}; отфильтруйте значения заголовка."
        )

    selected = [*index_columns, names_from, *value_columns]
    working = df[selected].copy()
    header_empty = "__DASHAPP_EMPTY_HEADER__"
    working[names_from] = working[names_from].where(
        pd.notna(working[names_from]), header_empty
    )

    pivot_keys = [*index_columns, names_from]
    if aggregation == "error":
        duplicate_count = int(working.duplicated(pivot_keys, keep=False).sum())
        if duplicate_count:
            raise ValueError(
                f"Найдено {duplicate_count} строк с повторяющейся комбинацией "
                "идентификаторов и заголовка. Выберите способ агрегации."
            )
        pivoted = working.pivot(
            index=index_columns,
            columns=names_from,
            values=value_columns[0] if len(value_columns) == 1 else value_columns,
        )
    else:
        pivoted = working.pivot_table(
            index=index_columns,
            columns=names_from,
            values=value_columns[0] if len(value_columns) == 1 else value_columns,
            aggfunc=aggregation,
            sort=False,
            observed=True,
        )

    header_order = list(pd.unique(working[names_from]))
    if len(value_columns) == 1:
        desired_columns = [
            header for header in header_order if header in pivoted.columns
        ]
    else:
        desired_columns = [
            (value_column, header)
            for value_column in value_columns
            for header in header_order
            if (value_column, header) in pivoted.columns
        ]
    pivoted = pivoted.reindex(columns=desired_columns)

    pivoted.columns = _flatten_pivot_columns(
        pivoted.columns, separator, reserved=index_columns
    )
    pivoted.columns.name = None
    result = pivoted.reset_index()
    result = result.rename(columns={header_empty: "(пусто)"})
    # The empty-header marker is normally embedded in a flattened string.
    result.columns = [
        str(column).replace(header_empty, "(пусто)") for column in result.columns
    ]
    outputs = [column for column in result.columns if column not in index_columns]
    return result, outputs, {
        "type": "long_to_wide",
        "label": "Long → Wide",
        "inputs": selected,
        "outputs": outputs,
        "params": {
            "index": index_columns,
            "names_from": names_from,
            "values_from": value_columns,
            "aggregation": aggregation,
            "separator": str(separator or "__"),
            "rows_before": len(df),
            "rows_after": len(result),
        },
    }


def apply_wide_to_long(
    df,
    id_columns,
    value_columns=None,
    variable_name="Переменная",
    value_name="Значение",
    drop_empty=False,
):
    """Unpivot selected wide columns into variable/value rows."""
    if not df.columns.is_unique:
        raise ValueError("Wide → Long требует уникальные названия каналов.")
    id_columns = _validated_columns(
        df, id_columns, "идентификаторы строк", required=False
    )
    value_columns = _validated_columns(
        df, value_columns, "разворачиваемые каналы", required=False
    )
    if not value_columns:
        value_columns = [column for column in df.columns if column not in id_columns]
    if not value_columns:
        raise ValueError("Нет каналов, которые можно развернуть в Long.")
    if set(id_columns) & set(value_columns):
        raise ValueError("Идентификаторы не должны входить в разворачиваемые каналы.")

    variable_name = str(variable_name or "Переменная").strip() or "Переменная"
    value_name = str(value_name or "Значение").strip() or "Значение"
    if variable_name == value_name:
        raise ValueError("Имена каналов переменной и значения должны различаться.")
    collisions = ({variable_name, value_name} & set(id_columns))
    if collisions:
        raise ValueError(
            "Новые имена каналов совпадают с идентификаторами: "
            + ", ".join(sorted(collisions))
        )

    result = df.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name=variable_name,
        value_name=value_name,
    )
    if drop_empty:
        result = result.dropna(subset=[value_name]).reset_index(drop=True)
    outputs = [variable_name, value_name]
    return result, outputs, {
        "type": "wide_to_long",
        "label": "Wide → Long",
        "inputs": [*id_columns, *value_columns],
        "outputs": outputs,
        "params": {
            "id_columns": id_columns,
            "value_columns": value_columns,
            "variable_name": variable_name,
            "value_name": value_name,
            "drop_empty": bool(drop_empty),
            "rows_before": len(df),
            "rows_after": len(result),
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
            elif operation == "long_to_wide":
                result, outputs, committed = apply_long_to_wide(
                    result,
                    params.get("index_columns"),
                    params.get("names_from"),
                    params.get("value_columns"),
                    params.get("aggregation") or "error",
                    params.get("separator") or "__",
                )
            elif operation == "wide_to_long":
                result, outputs, committed = apply_wide_to_long(
                    result,
                    params.get("id_columns"),
                    params.get("value_columns"),
                    params.get("variable_name") or "Переменная",
                    params.get("value_name") or "Значение",
                    bool(params.get("drop_empty")),
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
