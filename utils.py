# -*- coding: utf-8 -*-
"""
Вспомогательные функции: классификация, мета-данные, фильтрация, уведомления, сортировка легенды.
"""

import re
import logging
import os

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import html, dcc
from pandas.api.types import is_numeric_dtype
from io import StringIO
from flask import request

logger = logging.getLogger(__name__)


def _shutdown_server():
    func = request.environ.get("werkzeug.server.shutdown")
    if func:
        func()
    else:
        os._exit(0)


def classify_simple(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """
    Простая классификация без ухищрений:
    - datetime: dtype начинается с 'datetime' (в т.ч. timezone-aware)
    - numeric: is_numeric_dtype (кроме datetime)
    - categorical: всё остальное (в т.ч. string, object)
    """
    if df is None or df.empty:
        return [], [], []

    # 1) datetime по dtype
    datetime_cols = [c for c in df.columns if str(df[c].dtype).startswith('datetime')]

    # 2) numeric (исключая datetime)
    datetime_set = set(datetime_cols)
    numeric_cols = [c for c in df.columns if c not in datetime_set and is_numeric_dtype(df[c])]

    # 3) categorical = остальные
    used = datetime_set | set(numeric_cols)
    categorical_cols = [c for c in df.columns if c not in used]

    return numeric_cols, categorical_cols, datetime_cols


def meta_from_df(df: pd.DataFrame) -> dict:
    num, cat, dt = classify_simple(df)
    return {"numeric": num, "categorical": cat, "datetime": dt, "columns": list(map(str, df.columns))}


def read_df_from_store(json_str: str | None, meta: dict | None = None, *, dayfirst: bool = True) -> pd.DataFrame:
    """Читает DataFrame из dcc.Store (orient='split') и восстанавливает datetime-колонки по meta.

    Round-trip df.to_json(date_format='iso') -> pd.read_json(...) нередко возвращает datetime как строки.
    Поэтому после чтения принудительно парсим колонки из meta['datetime'] обратно в datetime64.
    """
    if not json_str:
        return pd.DataFrame()

    df = pd.read_json(StringIO(json_str), orient="split")

    dt_cols = []
    if isinstance(meta, dict):
        dt_cols = meta.get("datetime") or []

    for c in dt_cols:
        if c in df.columns:
            # iso-строки корректно распарсятся; dayfirst полезен, если в данных встречаются локальные форматы дат
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=dayfirst)

    return df


def safe_get_columns(df: pd.DataFrame, cols):
    """
    Вернёт пересечение имен из cols с df.columns.
    НЕ завязана на df.empty (пустые таблицы с правильными именами допустимы).
    """
    all_cols = set(map(str, getattr(df, "columns", []) or []))
    want = [str(c) for c in (cols or [])]
    return [c for c in want if c in all_cols]


def _is_numeric_col(df, col, meta) -> bool:
    if not col or col not in df.columns:
        return False
    if meta and col in (meta.get("numeric") or []):
        return True
    try:
        return is_numeric_dtype(df[col])
    except Exception:
        return False


def _ensure_box_ready(df, x_col: str, y_col: str, meta, req_id: str):
    """
    Строгая логика:
      - Если задан Y: он ОБЯЗАН быть числовым и непустым.
      - Если Y не задан: допускаем 1-аргументный box ТОЛЬКО если X числовой и непустой.
      - Если X категориальный, обязателен числовой Y.
    """
    # 1) Задан Y
    if y_col:
        if _is_numeric_col(df, y_col, meta) and df[y_col].notna().any():
            return True, None
        return False, f"Box: столбец Y ('{y_col}') должен быть ЧИСЛОВЫМ и содержать данные. [id={req_id}]"

    # 2) Y не задан -> разрешаем только если X числовой
    if x_col and _is_numeric_col(df, x_col, meta) and df[x_col].notna().any():
        return True, None

    # 3) Иначе — некорректно (категориальный X без Y и т.п.)
    return False, (
        f"Box: при категориальном X ('{x_col}') обязателен числовой Y. "
        f"Либо выберите числовой X без Y. [id={req_id}]"
    )


def _count_points_in_fig(fig) -> int:
    total = 0
    try:
        for tr in fig.data:
            xs = getattr(tr, "x", None)
            ys = getattr(tr, "y", None)
            if xs is not None:
                try: total += len(xs)
                except Exception: pass
            if ys is not None:
                try: total += len(ys)
                except Exception: pass
    except Exception:
        pass
    return int(total)


def _make_error_notif(msg: str):
    try:
        # Возвращаем список dict — ровно то, что ждёт sendNotifications
        return [{
            "id": "notifications-show",
            "title": "Ошибка!",
            "message": msg,
            "color": "red",
            "loading": False,
            "action": "show",        # ← ПОКАЗАТЬ (а не update)
            "autoClose": 6000,
            "style": {"fontSize": 20},
            # icon — можно добавить позже, когда всё заработает стабильно
        }]
    except Exception:
        return [{
                "action": "show",
                "id": "my-id",
                "message": "Ошибка!",
                # other props like title, color, icon, etc.
            }]


def apply_custom_colors_safely(fig, custom_colors):
    """Безопасно применяет пользовательские цвета к фигуре, избегая ошибок с типами трасс."""
    try:
        for i, trace in enumerate(fig.data):
            idx = str(i)
            if idx in custom_colors:
                # Для Box и Violin используем line.color
                if isinstance(trace, (go.Box, go.Violin)):
                    trace.line = getattr(trace, 'line', go.box.Line(color=custom_colors[idx]))
                    trace.line.color = custom_colors[idx]
                # Для Scatter и других с marker
                elif hasattr(trace, 'marker'):
                    trace.marker = getattr(trace, 'marker', go.scatter.Marker(color=custom_colors[idx]))
                    trace.marker.color = custom_colors[idx]
                else:
                    logger.warning(f"Не удалось установить цвет для трассы типа {type(trace).__name__}")
        return fig
    except Exception as e:
        logger.error(f"Ошибка при применении пользовательских цветов: {e}")
        return fig  # Возвращаем фигуру без изменений


def _apply_filters_once(frame: pd.DataFrame, filters_state: dict, meta: dict) -> pd.DataFrame:
    """Векторизованное применение фильтров одной маской за один проход."""
    if frame.empty or not filters_state:
        return frame
    mask = pd.Series(True, index=frame.index)
    numeric_cols = set(meta.get("numeric", []))
    for fdata in filters_state.values():
        col, val = fdata.get('column'), fdata.get('value')
        if not col or col not in frame.columns or val in (None, [], ''):
            continue
        if col in numeric_cols and isinstance(val, list) and len(val) == 2:
            lo, hi = val
            mask &= frame[col].between(lo, hi, inclusive="both")
        else:
            mask &= frame[col].isin(val) if isinstance(val, list) else (frame[col] == val)
    return frame[mask]


def _empty_fig():
    return go.Figure()


def needs_text_axis(col: str, meta: dict, force_text: list = None) -> bool:
    """Определяем, должна ли ось быть категориальной."""
    if not col:
        return False
    if col in (force_text or []):
        return True
    if col in (meta.get("datetime") or []):
        return False
    return col not in (meta.get("numeric") or [])


# ====== Сортировка легенды (перестановка трэйсов) ======
def _sort_legend_traces(fig: go.Figure, mode: str, custom_order_str: str | None = None):
    if not fig or not fig.data:
        return
    if mode == "original":
        return
    named, unnamed = [], []
    for tr in fig.data:
        name = getattr(tr, "name", None)
        (unnamed if name is None else named).append(tr if name is None else (tr, str(name)))
    if mode == "alphabetical":
        named_sorted = sorted(named, key=lambda t: t[1].lower())
        fig.data = tuple([t[0] for t in named_sorted] + unnamed); return
    if mode == "custom":
        custom = [x.strip() for x in (custom_order_str or "").split(",") if x.strip()]
        order_map = {val: i for i, val in enumerate(custom)}
        max_pos = len(custom)
        named_sorted = sorted(named, key=lambda t: (order_map.get(t[1], max_pos), t[1].lower()))
        fig.data = tuple([t[0] for t in named_sorted] + unnamed); return


def hide_xlabels_on_upper_facets(fig: go.Figure) -> go.Figure:
    """
    Убирает подписи X-оси (и тики, и title) у всех фасеток, кроме нижнего ряда.
    Работает для px-фасеток: определяем ряд по domain привязанной y-оси (anchor).
    """
    # Собираем домены всех y-осей
    y_domains = {}
    for k in fig.layout:
        if str(k).startswith("yaxis"):
            yaxis = getattr(fig.layout, k)
            dom = getattr(yaxis, "domain", None)
            if isinstance(dom, (list, tuple)) and len(dom) == 2:
                y_domains[k] = dom
    if not y_domains:
        return fig

    # Нижний ряд = минимальный y0
    bottom_y0 = min(dom[0] for dom in y_domains.values())

    # Проходим по всем x-осям и гасим подписи там, где ряд не нижний
    for k in fig.layout:
        if str(k).startswith("xaxis"):
            xaxis = getattr(fig.layout, k)
            anchor = getattr(xaxis, "anchor", None) or "y"   # 'y', 'y2', ...
            yname = f"yaxis{'' if anchor == 'y' else anchor[1:]}"
            dom = y_domains.get(yname)

            if dom and dom[0] > bottom_y0 + 1e-9:
                # Глушим подписи и тики
                xaxis.showticklabels = False
                xaxis.ticks = ""          # уберёт «чёрточки»
                xaxis.ticktext = None     # на всякий случай
                xaxis.tickvals = None
                xaxis.mirror = False      # чтобы не всплыли сверху из-за mirror
                # Уберём и заголовок оси X в верхних рядах
                if getattr(xaxis, "title", None):
                    xaxis.title.text = None

    return fig


# Обновление контролов фильтра
def create_value_control(filter_id, column, current_value=None, dff: pd.DataFrame | None = None):
    dff = dff if dff is not None else pd.DataFrame()
    if not column or (dff.empty or column not in dff.columns):
        return html.Div("Выберите столбец")

    numeric_cols, categorical_cols, datetime_cols = classify_simple(dff)

    if column in numeric_cols:
        min_val, max_val = float(dff[column].min()), float(dff[column].max())
        return dcc.RangeSlider(
            id={"type": "filter-value", "index": filter_id},
            min=min_val,
            max=max_val,
            value=current_value if current_value else [min_val, max_val],
            marks={min_val: str(min_val), max_val: str(max_val)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    else:
        unique_values = dff[column].dropna().unique().tolist()
        return dcc.Dropdown(
            id={"type": "filter-value", "index": filter_id},
            options=[{'label': str(val), 'value': val} for val in unique_values],
            value=current_value if current_value else [],
            multi=True,
            placeholder="Выберите значения...",
            style={'font-size': '12px', 'width': '100%', 'min-width': '200px', 'max-width': '300px'}
        )