# -*- coding: utf-8 -*-
"""Model-independent dataset diagnostics for the ML workspace."""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype


PROFILE_SAMPLE_ROWS = 50_000
HIGH_MISSING_PERCENT = 30.0
NEAR_CONSTANT_SHARE = 99.5
HIGH_UNIQUE_SHARE = 98.0


def _empty_figure(message, template="plotly", height=270):
    figure = go.Figure()
    figure.update_layout(
        template=template or "plotly", height=height,
        margin=dict(l=35, r=15, t=25, b=35),
        annotations=[dict(
            text=message, x=.5, y=.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color="#868e96", size=11),
        )],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return figure


def _column_type(series):
    if is_datetime64_any_dtype(series):
        return "Дата/время"
    if is_bool_dtype(series):
        return "Логический"
    if is_numeric_dtype(series):
        return "Числовой"
    return "Категориальный"


def _compact_names(values, limit=8):
    names = [str(value) for value in values]
    if len(names) <= limit:
        return ", ".join(names)
    return f"{', '.join(names[:limit])} +{len(names) - limit}"


def _infer_task(series):
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"
    if not is_numeric_dtype(non_null):
        return "classification"
    unique = int(non_null.nunique(dropna=True))
    threshold = max(12, min(50, int(math.sqrt(len(non_null))) + 2))
    return "classification" if unique <= threshold else "regression"


def _name_matches(name, patterns):
    normalized = re.sub(r"[^a-zа-я0-9]+", "_", str(name).lower())
    return any(re.search(pattern, normalized) for pattern in patterns)


def _severity_rank(value):
    return {"error": 3, "warning": 2, "info": 1}.get(value, 0)


def profile_dataset(frame, *, target=None, task_mode="auto", sample_rows=PROFILE_SAMPLE_ROWS):
    """Return a JSON-friendly passport without modifying the input frame."""
    if frame is None or not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame()

    row_count, column_count = map(int, frame.shape)
    target = str(target) if target is not None and str(target) in frame.columns else None
    sampled = row_count > int(sample_rows)
    sample = (
        frame.sample(n=int(sample_rows), random_state=42)
        if sampled else frame
    )
    sample_count = int(len(sample))

    if column_count:
        missing_counts = frame.isna().sum()
        missing_total = int(missing_counts.sum())
    else:
        missing_counts = pd.Series(dtype="int64")
        missing_total = 0
    cell_count = row_count * column_count
    missing_percent = 100.0 * missing_total / cell_count if cell_count else 0.0

    sample_memory = int(sample.memory_usage(index=True, deep=True).sum())
    memory_bytes = int(
        sample_memory * (row_count / max(1, sample_count))
    ) if sampled else sample_memory
    duplicate_count_sample = int(sample.duplicated().sum()) if sample_count else 0
    duplicate_percent = (
        100.0 * duplicate_count_sample / sample_count if sample_count else 0.0
    )

    numeric_columns = [str(c) for c in frame.columns if is_numeric_dtype(frame[c])]
    datetime_columns = [str(c) for c in frame.columns if is_datetime64_any_dtype(frame[c])]
    categorical_columns = [
        str(c) for c in frame.columns
        if str(c) not in set(numeric_columns) | set(datetime_columns)
    ]

    leak_candidates = set()
    leak_details = {}
    if target and target in sample and sample_count:
        target_sample = sample[target]
        for column in sample.columns:
            column = str(column)
            if column == target:
                continue
            feature = sample[column]
            valid = feature.notna() & target_sample.notna()
            if int(valid.sum()) < 5:
                continue
            left = feature.loc[valid]
            right = target_sample.loc[valid]
            if left.astype(str).reset_index(drop=True).equals(
                right.astype(str).reset_index(drop=True)
            ):
                leak_candidates.add(column)
                leak_details[column] = "совпадает с целью"
                continue
            if is_numeric_dtype(left) and is_numeric_dtype(right):
                left_numeric = pd.to_numeric(left, errors="coerce")
                right_numeric = pd.to_numeric(right, errors="coerce")
                if left_numeric.nunique() > 1 and right_numeric.nunique() > 1:
                    correlation = left_numeric.corr(right_numeric)
                    if pd.notna(correlation) and abs(float(correlation)) >= .995:
                        leak_candidates.add(column)
                        leak_details[column] = f"|corr|={abs(float(correlation)):.3f}"

    column_rows = []
    constant_columns = []
    near_constant_columns = []
    high_missing_columns = []
    high_cardinality_columns = []
    id_candidates = []
    for raw_column in frame.columns:
        column = str(raw_column)
        full_series = frame[raw_column]
        sampled_series = sample[raw_column] if raw_column in sample else full_series
        missing = int(missing_counts.get(raw_column, 0))
        missing_pct = 100.0 * missing / row_count if row_count else 0.0
        unique = int(sampled_series.nunique(dropna=True))
        unique_share = 100.0 * unique / max(1, int(sampled_series.notna().sum()))
        non_null = sampled_series.dropna()
        top_share = (
            100.0 * int(non_null.value_counts(dropna=False).iloc[0]) / len(non_null)
            if len(non_null) else 0.0
        )
        kind = _column_type(full_series)
        flags = []
        if missing_pct >= 100:
            flags.append("Пустой")
        elif unique <= 1:
            flags.append("Константа")
            constant_columns.append(column)
        elif top_share >= NEAR_CONSTANT_SHARE:
            flags.append("Почти константа")
            near_constant_columns.append(column)
        if missing_pct >= HIGH_MISSING_PERCENT and missing_pct < 100:
            flags.append("Много пропусков")
            high_missing_columns.append(column)
        if kind in {"Категориальный", "Логический"} and unique >= 50 and unique_share >= 50:
            flags.append("Высокая кардинальность")
            high_cardinality_columns.append(column)
        if unique_share >= HIGH_UNIQUE_SHARE and row_count >= 10:
            if kind != "Числовой" or _name_matches(
                column, [r"(^|_)id($|_)", r"uuid", r"код", r"номер", r"key"]
            ):
                flags.append("Возможный ID")
                id_candidates.append(column)
        if column in leak_candidates:
            flags.append("Возможная утечка")
        risk_score = (
            (3 if column in leak_candidates else 0)
            + (2 if missing_pct >= HIGH_MISSING_PERCENT else 0)
            + (2 if unique <= 1 else 0)
            + (1 if top_share >= NEAR_CONSTANT_SHARE else 0)
            + (1 if column in high_cardinality_columns else 0)
            + (1 if column in id_candidates else 0)
        )
        column_rows.append({
            "Канал": column,
            "Тип": kind,
            "Пропуски, %": round(missing_pct, 2),
            "Уникальных": unique,
            "Уникальность, %": round(unique_share, 2),
            "Макс. доля, %": round(top_share, 2),
            "Сигналы": " · ".join(flags) if flags else "—",
            "_risk": risk_score,
        })
    column_rows.sort(key=lambda item: (-item["_risk"], -item["Пропуски, %"], item["Канал"]))

    task = "unknown"
    target_profile = {"selected": False, "task": "unknown"}
    if target:
        inferred = _infer_task(frame[target])
        task = inferred if task_mode in {None, "", "auto"} else str(task_mode)
        target_series = frame[target]
        target_non_null = target_series.dropna()
        target_missing = int(target_series.isna().sum())
        target_profile = {
            "selected": True,
            "name": target,
            "task": task,
            "inferred_task": inferred,
            "missing": target_missing,
            "missing_percent": round(100.0 * target_missing / max(1, row_count), 2),
            "unique": int(target_non_null.nunique(dropna=True)),
        }
        if task == "classification":
            counts = target_non_null.astype(str).value_counts(dropna=False)
            target_profile["classes"] = [
                {"label": str(label), "count": int(value)}
                for label, value in counts.items()
            ]
            if len(counts):
                target_profile["minority_share"] = round(
                    100.0 * int(counts.iloc[-1]) / max(1, int(counts.sum())), 2
                )
                target_profile["imbalance_ratio"] = round(
                    int(counts.iloc[0]) / max(1, int(counts.iloc[-1])), 2
                )
        elif task == "regression":
            numeric_target = pd.to_numeric(target_non_null, errors="coerce").dropna()
            if len(numeric_target):
                target_profile.update({
                    "min": float(numeric_target.min()),
                    "max": float(numeric_target.max()),
                    "mean": float(numeric_target.mean()),
                    "std": float(numeric_target.std(ddof=0)),
                })

    issues = []

    def add_issue(severity, title, detail):
        issues.append({"severity": severity, "title": title, "detail": detail})

    if row_count == 0 or column_count == 0:
        add_issue("error", "Нет данных", "Выберите непустой dataset.")
    if 0 < row_count < 30:
        add_issue("warning", "Мало строк", f"Всего {row_count}: оценка качества будет нестабильной.")
    feature_count = max(0, column_count - (1 if target else 0))
    rows_per_feature = row_count / max(1, feature_count)
    if feature_count and rows_per_feature < 3:
        add_issue(
            "warning", "Признаков больше, чем позволяет выборка",
            f"{rows_per_feature:.2f} строки на признак. Нужен отбор признаков и строгая проверка.",
        )
    elif feature_count and rows_per_feature < 10:
        add_issue(
            "info", "Низкое отношение строк к признакам",
            f"{rows_per_feature:.1f} строки на признак — контролируйте переобучение.",
        )
    if constant_columns:
        add_issue(
            "warning", "Константные каналы",
            f"{len(constant_columns)}: {_compact_names(constant_columns)}. Их лучше исключить.",
        )
    if near_constant_columns:
        add_issue(
            "info", "Почти константные каналы",
            f"{len(near_constant_columns)}: {_compact_names(near_constant_columns)}.",
        )
    if high_missing_columns:
        add_issue(
            "warning", "Много пропусков",
            f"≥{HIGH_MISSING_PERCENT:.0f}% в {len(high_missing_columns)} каналах: "
            f"{_compact_names(high_missing_columns)}.",
        )
    if duplicate_percent >= 5:
        add_issue(
            "warning", "Повторяющиеся строки",
            f"Около {duplicate_percent:.1f}% в "
            f"{'выборке' if sampled else 'dataset'}.",
        )
    if leak_candidates:
        details = [f"{name} ({leak_details[name]})" for name in sorted(leak_candidates)]
        add_issue(
            "error", "Возможная утечка цели",
            _compact_names(details, limit=6) + ". Проверьте происхождение каналов.",
        )
    if target:
        if target_profile.get("missing_percent", 0) > 0:
            add_issue(
                "warning", "Пропуски в цели",
                f"{target_profile['missing_percent']:.1f}% строк не участвуют в обучении.",
            )
        if target_profile.get("unique", 0) <= 1:
            add_issue("error", "Цель постоянна", "Модель не сможет обучиться на одной величине.")
        if task == "classification":
            minority = float(target_profile.get("minority_share") or 0)
            if 0 < minority < 10:
                add_issue(
                    "warning", "Сильный дисбаланс классов",
                    f"Минимальный класс занимает {minority:.1f}% известных значений.",
                )
            if target_profile.get("unique", 0) > 50:
                add_issue(
                    "warning", "Слишком много классов",
                    f"В цели {target_profile['unique']} классов — проверьте постановку задачи.",
                )
        if task == "regression" and not is_numeric_dtype(frame[target]):
            add_issue("error", "Нечисловая цель", "Для регрессии выберите числовой канал.")
    else:
        add_issue("info", "Цель не выбрана", "Общая диагностика готова; выберите цель для ML-проверок.")

    recommendations = []

    def recommend(title, detail):
        recommendations.append({"title": title, "detail": detail})

    time_candidates = [
        column for column in map(str, frame.columns)
        if column in datetime_columns or _name_matches(
            column, [r"дата", r"время", r"date", r"time", r"timestamp", r"year", r"год"]
        )
    ]
    group_candidates = []
    for raw_column in frame.columns:
        column = str(raw_column)
        unique = int(sample[raw_column].nunique(dropna=True)) if sample_count else 0
        if 2 <= unique <= max(2, int(sample_count * .4)) and _name_matches(
            column, [r"well", r"скваж", r"месторожд", r"групп", r"объект", r"field"]
        ):
            group_candidates.append(column)
    if time_candidates:
        recommend(
            "Проверка по времени",
            f"Если модель прогнозирует будущее, используйте TimeSeriesSplit: "
            f"{_compact_names(time_candidates, 5)}.",
        )
    if group_candidates:
        recommend(
            "Разделение по группам",
            f"Чтобы связанные объекты не смешивались, рассмотрите GroupKFold: "
            f"{_compact_names(group_candidates, 5)}.",
        )
    if not time_candidates and not group_candidates:
        recommend(
            "Схема проверки",
            "Используйте KFold для небольшой выборки; для большого dataset достаточно фиксированного test.",
        )
    if high_cardinality_columns:
        recommend(
            "Категории высокой мощности",
            f"CatBoost подходит для этих каналов без one-hot: "
            f"{_compact_names(high_cardinality_columns, 5)}.",
        )
    if id_candidates:
        recommend(
            "Идентификаторы",
            f"Не подавайте уникальные ID как признаки без причины: {_compact_names(id_candidates, 5)}.",
        )
    if target and task == "classification" and float(target_profile.get("minority_share") or 100) < 20:
        recommend(
            "Баланс классов",
            "Используйте стратификацию и проверьте автоматические веса классов.",
        )
    if rows_per_feature < 10 and feature_count:
        recommend(
            "Отбор признаков",
            "Сначала удалите константы, возможные утечки и ID; затем сравните сокращённый набор по CV.",
        )

    issues.sort(key=lambda item: -_severity_rank(item["severity"]))
    summary = {
        "rows": row_count,
        "columns": column_count,
        "numeric": len(numeric_columns),
        "categorical": len(categorical_columns),
        "datetime": len(datetime_columns),
        "missing_percent": round(missing_percent, 2),
        "memory_bytes": memory_bytes,
        "duplicate_percent": round(duplicate_percent, 2),
        "sampled": sampled,
        "sample_rows": sample_count,
        "rows_per_feature": round(rows_per_feature, 3),
    }
    status = "critical" if any(item["severity"] == "error" for item in issues) else (
        "warning" if any(item["severity"] == "warning" for item in issues) else "ready"
    )
    return {
        "summary": summary,
        "target": target_profile,
        "task": task,
        "issues": issues,
        "recommendations": recommendations,
        "columns": column_rows,
        "status": status,
        "leak_candidates": sorted(leak_candidates),
    }


def missingness_figure(profile, template="plotly"):
    rows = [
        item for item in (profile or {}).get("columns", [])
        if float(item.get("Пропуски, %") or 0) > 0
    ]
    rows = sorted(rows, key=lambda item: float(item["Пропуски, %"]), reverse=True)[:20][::-1]
    if not rows:
        return _empty_figure("Пропусков не обнаружено", template)
    figure = go.Figure(go.Bar(
        x=[item["Пропуски, %"] for item in rows],
        y=[item["Канал"] for item in rows], orientation="h",
        marker_color="#228be6",
        hovertemplate="%{y}<br>Пропуски: %{x:.2f}%<extra></extra>",
    ))
    figure.update_layout(
        template=template or "plotly", height=max(270, 19 * len(rows) + 85),
        margin=dict(l=145, r=15, t=25, b=40), xaxis_title="Пропуски, %",
    )
    return figure


def target_figure(frame, profile, template="plotly"):
    target = (profile or {}).get("target") or {}
    if not target.get("selected"):
        return _empty_figure("Выберите целевой канал", template)
    name = target.get("name")
    if name not in frame:
        return _empty_figure("Целевой канал недоступен", template)
    series = frame[name].dropna()
    if target.get("task") == "classification":
        counts = series.astype(str).value_counts().head(20)
        figure = go.Figure(go.Bar(
            x=[str(value) for value in counts.index], y=counts.astype(int).tolist(),
            marker_color="#7950f2",
            hovertemplate="Класс: %{x}<br>Строк: %{y}<extra></extra>",
        ))
        figure.update_layout(
            template=template or "plotly", height=270,
            margin=dict(l=45, r=15, t=25, b=70),
            xaxis_title="Класс", yaxis_title="Строки",
        )
        return figure
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return _empty_figure("Для регрессии нужна числовая цель", template)
    figure = go.Figure(go.Histogram(x=numeric, nbinsx=30, marker_color="#7950f2"))
    figure.update_layout(
        template=template or "plotly", height=270,
        margin=dict(l=45, r=15, t=25, b=45),
        xaxis_title=str(name), yaxis_title="Строки",
    )
    return figure
