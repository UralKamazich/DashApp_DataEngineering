# -*- coding: utf-8 -*-
import re
import dash
from dash import Dash, _dash_renderer, dash_table, dcc, callback, Output, Input, html, State, ALL, MATCH, no_update
_dash_renderer._set_react_version("18.2.0")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import plotly.express as px
import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from functools import lru_cache
from pathlib import Path
import logging
import base64
from io import StringIO
import json
import os
import io
import re
from dash_iconify import DashIconify
from pandas.api.types import is_numeric_dtype
import plotly.io as pio
from dash import ctx  # новый API контекста
import logging, uuid, traceback
from logging.handlers import RotatingFileHandler
from flask import request

logger = logging.getLogger("dash-app")
NOISY_DEBUG = False
if not logger.handlers:
    logger.setLevel(logging.WARNING)  # Уменьшаем уровень логирования до WARNING, чтобы убрать info-сообщения
    _handler = RotatingFileHandler("dash_app.log", maxBytes=5_000_000, backupCount=3, encoding='utf-8')
    _fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    _handler.setFormatter(_fmt)
    logger.addHandler(_handler)
# =========================
# Константы и настройки
# =========================
COLOR_THRESHOLD = 10
MAX_FILTERS = 10
PORT = 8090
DEFAULT_X = "Дата"
DEFAULT_Y = "Добыча"
DEFAULT_Z = "Забойное давление"
DEFAULT_BUBBLE_SIZE = "Дебит"
DEFAULT_HOVER_COLS = ["Скважина", "Пластовое давление"]
STYLE = {"margin": 10}

legend_config = {
    "top-left-inside": dict(x=0.01, y=0.99, xanchor="left", yanchor="top", orientation="v"),
    "top-center-outside": dict(x=0.5, y=1.01, xanchor="center", yanchor="bottom", orientation="h"),
    "top-right-inside": dict(x=0.99, y=0.99, xanchor="right", yanchor="top", orientation="v"),
    "top-right-outside": dict(x=1.02, y=1, xanchor="left", yanchor="top", orientation="v"),
    "bottom-outside": dict(x=0.5, y=-0.12, xanchor="center", yanchor="top", orientation="h"),
}

# Пользовательский шаблон
seaborn_custom = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Arial, sans-serif", size=12, color="#000000"),
        title_font=dict(family="Arial, sans-serif", size=16, color="#000000"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(
            gridcolor="#D3D3D3",
            gridwidth=1,
            zerolinecolor="#D3D3D3",
            zerolinewidth=1,
            showline=True,
            linecolor="#000000",
            linewidth=2,
            mirror=True
        ),
        yaxis=dict(
            gridcolor="#D3D3D3",
            gridwidth=1,
            zerolinecolor="#D3D3D3",
            zerolinewidth=1,
            showline=True,
            linecolor="#000000",
            linewidth=2,
            mirror=True
        ),
        colorway=[
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
    )
)
pio.templates["seaborn_custom"] = seaborn_custom

PLOTLY_STYLES = [
    "plotly", "ggplot2", "seaborn", "seaborn_custom", "simple_white", "plotly_white", "plotly_dark"
]

# =========================
# Логирование
# =========================
logging.basicConfig(level=logging.WARNING)  # Уменьшаем глобальное логирование
logger = logging.getLogger(__name__)

# =========================
# Вспомогательные функции
# =========================

def _shutdown_server():
    func = request.environ.get("werkzeug.server.shutdown")
    if func:
        func()
    else:
        os._exit(0)
# === Простая классификация и метаданные ===
    if not patterns:
        return df
    low_patterns = [p.lower() for p in patterns]
    cols = list(map(str, df.columns))
    for c in cols:
        lc = c.lower()
        if any(p in lc for p in low_patterns):
            df[c] = df[c].astype("string")  # принудительно категориальный/текстовый
    return df

from pandas.api.types import is_numeric_dtype

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

# ==== LOG UTILS ==============================================================
# Убрали _col_debug, так как это ненужное логирование
# ============================================================================

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

    
NOTIF_POSITION = "bottom-right"

# def _make_error_notif(msg: str):
#     # Возвращаем список dict — ровно то, что ждёт sendNotifications
#     return [{
#         "id": "notif1",
#         "title": "Ошибка!",
#         "message": msg,
#         "color": "red",
#         "loading": False,
#         "action": "show",        # ← ПОКАЗАТЬ (а не update)
#         "autoClose": 6000,
#         # icon — можно добавить позже, когда всё заработает стабильно
#     }]
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

def needs_text_axis(col: str, meta: dict, force_text: list) -> bool:
    """Определяем, должна ли ось быть категориальной."""
    if not col:
        return False
    if col in (force_text or []):
        return True
    if col in (meta.get("datetime") or []):
        return False
    return col not in (meta.get("numeric") or [])


# =========================
# UI элементы (Dropdowns, etc.)
# =========================
dropdown_style = dmc.Select(
    id="dropdown_style",
    label="Стиль графика",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    data=[{"label": style.capitalize(), "value": style} for style in PLOTLY_STYLES],
    value="plotly",
    clearable=False,
    persistence=True
)

def create_dropdown(id, options, value=None, clearable=False, multi=False, persistence=True):
    return dcc.Dropdown(
        id=id,
        value=value,  # оставляем None как есть
        options=[{'label': str(col), 'value': col} for col in options],
        clearable=clearable,
        multi=multi,
        maxHeight=500,
        persistence=persistence,
        className="dd",
        style={
            "width": "95%",
            "maxWidth": "95%",
            "minWidth": 0,
            "fontSize": "12px"
        }
    )
def create_multiselect(id, options, value=None, searchable=True, clearable=True, persistence=False):
    """
    Универсальный MultiSelect c галочками.
    options — такой же список [{'label':..., 'value':...}], как и у dcc.Dropdown,
    мы преобразуем его в data для Mantine.
    """
    data = [{"label": str(o["label"]), "value": str(o["value"])} for o in (options or [])]
    return dmc.MultiSelect(
        id=id,
        data=data,
        value=(value or []),
        searchable=searchable,
        clearable=clearable,
        nothingFoundMessage="Ничего не найдено",
        maxDropdownHeight=500,
        comboboxProps={"shadow": "md"},
        persistence=persistence,
        style={
            "width": "95%",
            "maxWidth": "95%",
            "minWidth": 0,
            "fontSize": "12px",
            "hidePickedOptions": True
        }
        # по желанию позже можно включить: hidePickedOptions=True
    )

dropdown_x = create_dropdown("dropdown_x", [], None, clearable=True)
dropdown_y = create_dropdown("dropdown_y", [], None, clearable=True)
dropdown_z = create_dropdown("dropdown_z", [], None, clearable=True)
dropdown_color = create_dropdown("dropdown_color", [], None, clearable=True)
dropdown_size = create_dropdown("dropdown_size", [], None, clearable=True)
#dropdown_hover_data = create_dropdown("dropdown_hover_data", [], [], multi=True, clearable=True)
dropdown_text = create_dropdown("dropdown_text",[], None, clearable=True)
#dropdown_corr_columns = create_dropdown("dropdown_corr_columns", [], [], multi=True, clearable=True)
dropdown_facet_row = create_dropdown("dropdown_facet_row", [], None, clearable=True)
dropdown_facet_col = create_dropdown("dropdown_facet_col", [], None, clearable=True)

dropdown_hover_data = create_multiselect("dropdown_hover_data", [], value=[], clearable=True)
dropdown_corr_columns = create_multiselect("dropdown_corr_columns", [], value=[], clearable=True)
dropdown_cluster_cols=dmc.MultiSelect(
    id="cluster-cols",
    data=[],  # наполним из meta.numeric
    value=[],
    searchable=True,
    clearable=True,
    nothingFoundMessage="Ничего не найдено",
    maxDropdownHeight=500,
    comboboxProps={"shadow": "md"},
    style={"width": "100%", "fontSize": "12px"}
)

# === Data Engineering: агрегаты по группам (Group Aggregations) ===
agg_keys_select = dmc.MultiSelect(
    id="agg-keys",
    data=[],          # наполним из meta (все столбцы)
    value=[],
    searchable=True,
    clearable=True,
    nothingFoundMessage="Ничего не найдено",
    maxDropdownHeight=500,
    comboboxProps={"shadow": "md"},
    persistence=True,
    style={"width": "100%", "fontSize": "12px"}
)

agg_cols_select = dmc.MultiSelect(
    id="agg-cols",
    data=[],          # наполним из meta (все столбцы)
    value=[],
    searchable=True,
    clearable=True,
    nothingFoundMessage="Ничего не найдено",
    maxDropdownHeight=500,
    comboboxProps={"shadow": "md"},
    persistence=True,
    style={"width": "100%", "fontSize": "12px"}
)

agg_metrics_select = dmc.CheckboxGroup(
    id="agg-metrics",
    label="Параметры расчёта",
    value=["mean", "median", "mode"],
    children=[
        dmc.Checkbox(label="Среднее", value="mean"),
        dmc.Checkbox(label="Медиана", value="median"),
        dmc.Checkbox(label="Мода", value="mode"),
        dmc.Checkbox(label="Сумма", value="sum"),
        dmc.Checkbox(label="Кумулятивная сумма", value="cumsum"),
        dmc.Checkbox(label="Мин", value="min"),
        dmc.Checkbox(label="Макс", value="max"),
        dmc.Checkbox(label="Std", value="std"),
        dmc.Checkbox(label="Count", value="count"),
        dmc.Checkbox(label="Nunique", value="nunique"),
    ],
)



agg_exclude_zeros_switch = dmc.Switch(
    id="agg-exclude-zeros",
    label="Исключать нули из расчётов",
    checked=False,
    onLabel="Да",
    offLabel="Нет",
    size="md"
)

agg_exclude_empty_switch = dmc.Switch(
    id="agg-exclude-empty",
    label="Исключать пустые (NaN) из расчётов",
    checked=True,
    onLabel="Да",
    offLabel="Нет",
    size="md"
)


# ========= Data Engineering: text copy module =========
txtcopy_cols_select = dmc.MultiSelect(
    id="txtcopy-cols",
    data=[],
    value=[],
    searchable=True,
    clearable=True,
    nothingFoundMessage="Ничего не найдено",
    maxDropdownHeight=500,
    comboboxProps={"shadow": "md"},
    persistence=True,
    style={"width": "100%", "fontSize": "12px"}
)

txtcopy_suffix_input = dmc.TextInput(
    id="txtcopy-suffix",
    label="Суффикс для новой колонки",
    value="_txt",
    size="xs",
    style={"width": "220px"}
)

txtcopy_strip_switch = dmc.Switch(
    id="txtcopy-strip",
    label="Обрезать пробелы по краям (strip)",
    checked=True,
    onLabel="Да",
    offLabel="Нет",
    size="md"
)



add_filter_button = dmc.Button("Добавить фильтр", id="add-filter-btn", size="xs", variant="outline", leftSection=html.Div("+"))
segmentedcontrol = dmc.SegmentedControl(
    id="segmented", value="Scatter",
    data=[
        {"value": "Scatter", "label": "Scatter"},
        {"value": "3D_Scatter", "label": "3D Scatter"},
        {"value": "Bar", "label": "Bar"},
        {"value": "Box", "label": "Box"},
        {"value": "Line", "label": "Line"},
        {"value": "Polar", "label": "Треугольный график"},
        {"value": "Hist", "label": "Гистограмма"},
        {"value": "Pie", "label": "Круговая диаграмма"},
        {"value": "Correlation", "label": "Коррелограмма"},
        {"value": "Violin", "label": "Violin"},
        {"value": "Ridge", "label": "Ridge Plot"},
        {"value": "ScatterMatrix",   "label": "Scatter Matrix"},
        {"value": "Parcoords",       "label": "Parallel Coordinates"},
        {"value": "Sunburst",        "label": "Sunburst"},
        {"value": "Treemap",         "label": "Treemap"},
        {"value": "DensityHeat",     "label": "Density Heatmap"},
        {"value": "DensityContour",  "label": "Density Contour"},

    ], size="sm"
)

SwitchBubble = dmc.Switch(id="SwitchBubble", size="xs", radius="sm", label="Bubble", checked=False)
update_graf = dmc.Tooltip(
    label="Обновить / Отобразить график",
    withArrow=True,
    children=dmc.ActionIcon(
        id="update-graf",                 # тот же ID — логика не меняется
        variant="light",
        size="xl",
        radius="xl",
        children=DashIconify(icon="lucide:refresh-ccw", width=18)
        # альтернативы: DashIconify(icon="tabler:refresh", width=18) или "mdi:refresh"
    )
)


download_button = dmc.Tooltip(
    label="Сохранить в HTML",
    withArrow=True,
    children=dmc.ActionIcon(
        id="download-button",            # тот же ID
        variant="light",
        size="xl",
        radius="xl",
        children=DashIconify(icon="tabler:file-type-html", width=18)
        # альтернатива: DashIconify(icon="lucide:download", width=18)
    )
)

DownloadFile = dcc.Download(id="download-file")

excel_download_button = dmc.Tooltip(
    label="Сохранить датасет в Excel",
    withArrow=True,
    children=dmc.ActionIcon(
        id="download-excel-button",
        variant="light",
        size="xl",
        radius="xl",
        children=DashIconify(icon="vscode-icons:file-type-excel2", width=18)
    )
)

DownloadExcel = dcc.Download(id="download-excel")

copy_button = dmc.Tooltip(
    label="Копировать как PNG",
    withArrow=True,
    children=dmc.ActionIcon(
        id="copy-png-button",            # тот же ID
        variant="light",
        size="xl",
        radius="xl",
        children=DashIconify(icon="lucide:image-down", width=18)
        # альтернатива: DashIconify(icon="tabler:copy", width=18)
    )
)


copy_trigger = dcc.Clipboard(id="clipboard", style={"display": "none"})

dropdown_text_pozition = dmc.Select(
    id="dropdown_text_pozition",
    label="Положение подписи",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    value="middle center",
    data=['top left', 'top center', 'top right', 'middle left',
          'middle center', 'middle right', 'bottom left', 'bottom center', 'bottom right'],
    clearable=False, persistence=True
)
dropdown_category_ascending = dmc.Select(
    id="dropdown_category_ascending",
    label="Сортировка категорий (оси)",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    value="total ascending",
    data=[
        "trace", "category ascending", "category descending", "total ascending", "total descending",
        "min ascending", "min descending", "max ascending", "max descending", "sum ascending", "sum descending",
        "mean ascending", "mean descending", "geometric mean ascending", "geometric mean descending",
        "median ascending", "median descending"
    ],
    clearable=False, persistence=True
)
dropdown_axes_category = dmc.Select(
    id="dropdown_axes_category",
    label="Ось сортировки (катег.)",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    value="auto",               # было "y"
    data=["auto", "x", "y"],    # добавили auto
    clearable=False, persistence=True
)
dropdown_overlay = dmc.Select(
    id="dropdown_overlay",
    label="Наложение в гистограмме",
    value="overlay",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    data=["overlay", "stack", "relative", "group"],
    clearable=False, persistence=True
)
dropdown_legend = dmc.Select(
    id="dropdown_legend",
    label="Расположение легенды",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    data=[
        {"label": "Слева сверху (внутри)", "value": "top-left-inside"},
        {"label": "Сверху по центру (вне)", "value": "top-center-outside"},
        {"label": "Справа сверху (внутри)", "value": "top-right-inside"},
        {"label": "Справа сверху (вне)", "value": "top-right-outside"},
        {"label": "Снизу (вне)", "value": "bottom-outside"}
    ],
    value="top-right-outside",
)

# Порядок легенды + пользовательский список
dropdown_legend_order = dmc.Select(
    id="dropdown_legend_order",
    label="Сортировка легенды",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    value="alphabetical",
    data=[
        {"label": "По алфавиту", "value": "alphabetical"},
        {"label": "Как в данных", "value": "original"},
        {"label": "Пользовательский", "value": "custom"},
    ],
    clearable=False,
    persistence=True,
)
input_legend_custom_order = dmc.TextInput(
    id="input_legend_custom_order",
    label="Порядок легенды (через запятую)",
    placeholder="Пример: A, C, B",
    persistence=True,
)


# === Новый блок: ГРУППИРОВКА ЧИСЛОВОГО СТОЛБЦА ===
bin_column_select = create_dropdown("bin-column", [], None, clearable=True, persistence=False)
bin_method = dmc.SegmentedControl(
    id="bin-method",
    data=[{"label": "Равные интервалы", "value": "width"},
          {"label": "Равное число значений", "value": "count"}],
    value="count", size="xs",fullWidth=True
)
bin_k = dmc.NumberInput(
    id="bin-k", value=5, min=2, max=20, step=1, debounce=True
)
bin_label_style = dmc.SegmentedControl(
    id="bin-label-style",
    data=[{"label": "Диапазоны", "value": "interval"},
          {"label": "Индексы", "value": "index"}],
    value="interval", size="xs", fullWidth=True
)

# =========================
# Приложение
# =========================
app = Dash(__name__, suppress_callback_exceptions=True,title="DataAnalize ver.1.0.21 by Muslimov Ural")
server = app.server

STYLE_CARD = {
    'maxWidth': '100%',
    'padding': '20px',
    'boxSizing': 'border-box',
    "margin": "10px"
}
PAPER_BASE = {"height": "auto", "overflow": "visible"}
initial_fig = go.Figure()
app.layout = dmc.MantineProvider(
    children=[
        # --- Stores ---
        dcc.Store(id='stored-data', data=False, storage_type="memory"),
        dcc.Store(id='filtered-data'),
        #dcc.Store(id='enhanced-data'),
        dcc.Store(id='bin-applied-name'),
        dcc.Store(id='filter-count', data=1),
        dcc.Store(id='filters-state', data={}),
        dcc.Store(id='stored-sheet-names'),
        dcc.Store(id='selected-sheet'),
        dcc.Store(id='sheet-modal-toggle', data=True),
        dcc.Store(id='custom-colors', data={}),
        #dcc.Store(id='processed-data'),     # новый: обработанные данные — источник правды
        dcc.Store(id='meta-columns'),       # новый: метаданные о колонках
        dcc.Store(id='cluster-metrics'),  
                dcc.Store(id='source-file-path'),
        dcc.Store(id='source-file-name'),
# --- Color modal ---
        dmc.Modal(
            id="color-modal",
            title="Выберите цвета для классов",
            children=[
                dmc.Group([
                    dmc.Text("Режим выбора цвета:"),
                    dmc.Switch(
                        id="color-mode-toggle",
                        onLabel="Ручной",
                        offLabel="Авто",
                        checked=False,
                        size="md"
                    ),
                ]),
                dmc.Stack(id="color-inputs"),
                dmc.Button("Применить", id="apply-colors")
            ],
            opened=False, size="auto"
        ),

        dmc.NotificationContainer(
            id="notifications-container"),
        #html.Div(id="notifications-container"),
        copy_trigger,

        dmc.Grid([
            dmc.GridCol([
                dmc.Paper([
                    dmc.Group([
                        dmc.Button("Выбрать файл (.xlsx, .pkl)", id="pick-file-btn", size="xs"),
                        dmc.Text(id="file-path-message", size="xs", c="dimmed", style={"wordBreak": "break-all"})
                    ], gap="sm", align="center"),

                                        dmc.Modal(
                        id="de-modal",
                        title="Data Engineering",
                        opened=False,
                        size="xl",
                        withCloseButton=True,
                        closeOnClickOutside=True,
                        closeOnEscape=True,
                        overlayProps={"opacity": 0.15},
                        children=[
                            dmc.Tabs(
                                value="de",
                                children=[
                                    dmc.TabsList([
                                        dmc.TabsTab("Data Engineering", value="de"),
                                        dmc.TabsTab("Биннинг", value="binning"),
                                        dmc.TabsTab("Кластеризация", value="clustering"),
                                    ]),
                                    dmc.TabsPanel(
                                        [
                                            dmc.Text("Расчёт агрегатов по группам: выбираете ключ(и), столбцы и метрики — новые колонки добавляются в конец текущего датасета (после фильтров/кластеризации).", size="sm"),
                                            dmc.Space(h=10),
                                            dmc.Grid([
                                                dmc.GridCol([dmc.Text("Ключ(и) группировки", c="blue", fw=500, size="sm"), agg_keys_select], span=6, style={"minWidth": 0}),
                                                dmc.GridCol([dmc.Text("Столбцы для расчёта", c="blue", fw=500, size="sm"), agg_cols_select], span=6, style={"minWidth": 0}),
                                            ]),
                                            dmc.Space(h=8),
                                            agg_metrics_select,
                                            dmc.Space(h=10),
                                            dmc.Group([
                                                agg_exclude_zeros_switch,
                                                agg_exclude_empty_switch,
                                            ], gap="xl"),
                                            dmc.Space(h=10),
                                            dmc.Group([
                                                dmc.Button("Рассчитать", id="btn-agg", size="sm"),
                                            ], justify="flex-end"),
                                            dmc.Space(h=6),
                                            dmc.Text(id="de-agg-status", size="sm", c="dimmed"),

                                            dmc.Space(h=12),
                                            dmc.Divider(label="Текстовые копии (чтобы Plotly и фильтры видели как текст)"),
                                            dmc.Space(h=8),
                                            dmc.Grid([
                                                dmc.GridCol([
                                                    dmc.Text("Столбец(ы) для копирования в текст", c="blue", fw=500, size="sm"),
                                                    txtcopy_cols_select
                                                ], span=8, style={"minWidth": 0}),
                                                dmc.GridCol([
                                                    txtcopy_suffix_input
                                                ], span=4, style={"minWidth": 0}),
                                            ]),
                                            dmc.Space(h=8),
                                            dmc.Group([txtcopy_strip_switch], gap="xl"),
                                            dmc.Space(h=10),
                                            dmc.Group([
                                                dmc.Button("Создать текстовую копию", id="btn-txtcopy", size="sm", variant="light"),
                                            ], justify="flex-end"),
                                            dmc.Space(h=6),
                                            dmc.Text(id="de-txt-status", size="sm", c="dimmed"),

                                        ],
                                        value="de"
                                    ),
                                    dmc.TabsPanel(
                                        [
                                            dmc.Divider(label="Группировка численного столбца (биннинг)"),
                                            dmc.Grid([
                                                dmc.GridCol([html.Center(dmc.Text("Столбец для биннинга", c="blue", fw=500, size="sm")), bin_column_select], span=8, style={"minWidth": 0}),
                                                dmc.GridCol([html.Center(dmc.Text("Число групп", c="blue", fw=500, size="sm")), bin_k], span=2, style={"minWidth": 0}),
                                                dmc.GridCol([dmc.Button("Группировка", id="btn-grouping", size="xs")], span=2, style={"minWidth": 0, "marginTop": 23}),
                                            ]),
                                            dmc.Grid([
                                                dmc.GridCol([dmc.Text("Метод", c="blue", fw=500, size="sm"), bin_method], span=6, style={"minWidth": 0}),
                                                dmc.GridCol([dmc.Text("Метки", c="blue", fw=500, size="sm"), bin_label_style], span=6, style={"minWidth": 0}),
                                            ]),
                                        ],
                                        value="binning"
                                    ),
                                    dmc.TabsPanel(
                                        [
                                            dmc.Divider(label="Кластеризация (KMeans)"),
                                            dmc.Grid([
                                                dmc.GridCol([html.Center(dmc.Text("Числовые столбцы для кластеризации", c="blue", fw=500, size="sm")), dropdown_cluster_cols], span=8, style={"minWidth": 0}),
                                                dmc.GridCol([html.Center(dmc.Text("К(Кластеры)", c="blue", fw=500, size="sm")),
                                                             dmc.NumberInput(id="cluster-k", value=4, min=2, max=20, step=1, debounce=True)], span=2, style={"minWidth": 0}),
                                                dmc.GridCol([dmc.Button("Кластеризация", id="btn-cluster", size="xs")], span=2, style={"minWidth": 0, "marginTop": 23}),
                                            ]),
                                        ],
                                        value="clustering"
                                    ),
                                ]
                            )
                        ]
                    ),

html.Div(
                        dmc.Modal(
                            id="sheet-modal",
                            title="Выберите лист Excel",
                            opened=False,
                            centered=True,
                            zIndex=1000,
                            children=[]
                        ),
                        id="sheet-menu-wrapper"
                    ),
                    html.Div(id='status-message', style={'marginTop': 10}),
                                    ], style=STYLE_CARD, shadow="md", p="md", withBorder=True),

                dmc.Paper([
                    dmc.Grid([
                            dmc.GridCol([html.Center(dmc.Text("X", c="blue", fw=500, size="sm")), dropdown_x], span=4, style={"minWidth": 0}),
                            dmc.GridCol([html.Center(dmc.Text("Y", c="blue", fw=500, size="sm")), dropdown_y], span=4, style={"minWidth": 0}),
                            dmc.GridCol([html.Center(dmc.Text("Z", c="blue", fw=500, size="sm")), dropdown_z], span=4, style={"minWidth": 0})
                    ]),
                    
                    dmc.Grid([
                        dmc.GridCol([html.Center(dmc.Text("Группировка", c="blue", fw=500, size="sm")), dropdown_color], span=4, style={"minWidth": 0}),
                        dmc.GridCol([html.Center(dmc.Text("Размер пузыpя", c="blue", fw=500, size="sm")), dropdown_size], span=4, style={"minWidth": 0}),
                        dmc.GridCol([html.Center(dmc.Text("Подпись", c="blue", fw=500, size="sm")), dropdown_text], span=4, style={"minWidth": 0}),
                    ]),
                    dmc.Grid([
                        dmc.GridCol([html.Center(dmc.Text("Facet Row", c="blue", fw=500, size="sm")), dropdown_facet_row], span=6, style={"minWidth": 0}),
                        dmc.GridCol([html.Center(dmc.Text("Facet Col", c="blue", fw=500, size="sm")), dropdown_facet_col], span=6, style={"minWidth": 0}),
                    ]),
                    dmc.Grid([
                        dmc.GridCol([html.Center(dmc.Text("Hover Data", c="blue", fw=500, size="sm")), dropdown_hover_data], span=6, style={"minWidth": 0}),
                        dmc.GridCol([html.Center(dmc.Text("Корреляц. столбцы", c="blue", fw=500, size="sm")), dropdown_corr_columns], span=6, style={"minWidth": 0}),
                    ]),
                    # Принудительные текстовые столбцы
                    dmc.Grid([
                        dmc.GridCol([
                        ], span=12, style={"minWidth": 0}),
                    ]),
                    # === Блок группировки числового столбца ===
                    dmc.Divider(label="Data Engineering"),
                    dmc.Text("Биннинг, кластеризация и расчёт новых столбцов — в окне Data Engineering (иконка с инструментом).", size="xs", c="dimmed"),
                    dmc.Space(h=6),
                    html.Div([
                        dmc.Drawer(
                            title="Настройка графика",
                            id="drawer-simple",
                            padding="md",
                            position='right',
                            # Было withOverlay=False — включим оверлей, чтобы можно было закрыть кликом вне
                            withOverlay=True,
                            overlayProps={"opacity": 0.15},
                            size=600,
                            closeOnClickOutside=True,   # <-- Закрывать кликом вне дравера
                            closeOnEscape=True,         # <-- Закрывать по Esc
                            withCloseButton=True,
                            children=[
                                dmc.Grid([
                                    dmc.GridCol([SwitchBubble], span="content"),
                                    dmc.GridCol([
                                        dmc.NumberInput(
                                            id="InputMaxSizeBubble",
                                            label="Макс. размер бабла",
                                            value=30, min=1, max=100, debounce=True, step=5,
                                            persistence=True, persistence_type='local'
                                        )
                                    ], span="content"),
                                    dmc.Grid([
                                        dmc.GridCol([
                                            dmc.NumberInput(
                                                id="InputSizePlot",
                                                label="Высота графика",
                                                value=750, min=50, max=20000, debounce=True, step=50,
                                                persistence=True, persistence_type='local'
                                            )
                                        ], span="content"),
                                        dmc.GridCol([
                                            dmc.NumberInput(
                                                id="InputSizePlotW",
                                                label="Ширина графика",
                                                min=50, max=20000, debounce=True, step=50,
                                                persistence=True, persistence_type='local'
                                            )
                                        ], span="content")
                                    ]),
                                    dmc.Grid([
                                        dmc.GridCol([
                                            dmc.NumberInput(id="font-size-xaxis", label="Шрифт X оси", value=14, min=6, max=48, debounce=True, step=1)
                                        ], span=3),
                                        dmc.GridCol([
                                            dmc.NumberInput(id="font-size-yaxis", label="Шрифт Y оси", value=14, min=6, max=48, debounce=True, step=1)
                                        ], span=3),
                                        dmc.GridCol([
                                            dmc.NumberInput(id="font-size-title", label="Шрифт заголовка", value=16, min=6, max=48, debounce=True, step=1)
                                        ], span=3),
                                    ]),
                                    dmc.Grid([
                                        dmc.GridCol([dmc.NumberInput(id="font-size-ticks", label="Шрифт подписей", value=12, min=6, max=48, debounce=True, step=1)], span="content"),
                                        dmc.GridCol([dropdown_text_pozition], span="content"),
                                    ]),
                                    dmc.Grid([
                                        dmc.GridCol([dropdown_axes_category], span="content"),
                                        dmc.GridCol([dropdown_category_ascending], span="content"),
                                    ]),
                                    dmc.Grid([
                                        dmc.GridCol([dropdown_overlay], span="content"),
                                        dmc.GridCol([dropdown_legend], span="content"),
                                    ]),
                                    dmc.Grid([
                                        dmc.GridCol([dropdown_legend_order], span="content"),
                                        dmc.GridCol([input_legend_custom_order], span="content"),
                                    ]),
                                    dmc.Grid([dmc.GridCol([dropdown_style], span="content")]),
                                    dmc.Grid([
                                        dmc.GridCol([
                                            dmc.NumberInput(
                                                id="tick-step-xaxis", label="Шаг тиков X оси",
                                                value=0, min=0, step=0.1, decimalScale=2, debounce=True,
                                                persistence=True, persistence_type='local'
                                            )
                                        ], span="content"),
                                        dmc.GridCol([
                                            dmc.NumberInput(
                                                id="tick-step-yaxis", label="Шаг тиков Y оси",
                                                value=0, min=0, step=0.1, decimalScale=2, debounce=True,
                                                persistence=True, persistence_type='local'
                                            )
                                        ], span="content"),
                                    ]),
                                ])
                            ]
                        )
                    ]),
                ], style=STYLE_CARD, shadow="md", p="md", withBorder=True),

                dmc.Paper([
                    html.Center(dmc.Text("Фильтры", c="black", size="sm")),
                    html.Div(id="filters-container", children=[]),
                    dmc.Space(h=10),
                    add_filter_button
                ], style=STYLE_CARD, shadow="md", p="md", withBorder=True),
            ], span=4),

            dmc.GridCol([
                dmc.Grid([dmc.GridCol([segmentedcontrol], span="content")], align="center"),
                dmc.Grid([
                    dmc.GridCol([update_graf], span="content"),
                    dmc.GridCol([download_button, DownloadFile], span="content"),
                    dmc.GridCol([excel_download_button, DownloadExcel], span="content"),
                    
                    dmc.GridCol([copy_button], span="content"),
                    dmc.GridCol([
                        dmc.Tooltip(
                            label="Изменить цвета",
                            withArrow=True,
                            children=dmc.ActionIcon(
                                id="shuffle-button",          # тот же ID
                                variant="light",
                                size="xl",
                                radius="xl",
                                children=DashIconify(icon="tabler:palette", width=18)
                            )
                        )
                    ], span="content"),
dmc.GridCol([
    dmc.Tooltip(
        label="Data Engineering",
        withArrow=True,
        children=dmc.ActionIcon(
            id="de-button",
            variant="light",
            size="xl",
            radius="xl",
            children=DashIconify(icon="tabler:tools", width=18)
        )
    )
], span="content"),
                    dmc.GridCol([
                        dmc.Tooltip(
                            label="Настройка графика",
                            withArrow=True,
                            children=dmc.ActionIcon(
                                id="drawer-demo-button",             # тот же ID — колбэк не трогаем
                                variant="light",
                                size="xl",                           # можно xs/sm
                                radius="xl",
                                children=DashIconify(icon="lucide:settings", width=18)
                            )
                        )
                    ], span="content"),
                ], align="center"),
                dmc.Paper([
                    dcc.Loading(
                        dcc.Graph(figure={}, id="graph", config={
                            'displaylogo': False,
                            'modeBarButtonsToRemove': [],
                            'modeBarButtonsToAdd': ['fullscreen'],
                            'displayModeBar': True,
                            'scrollZoom': True
                        }),
                        type="default"
                    )
                ], style=STYLE_CARD, shadow="md", p="md", withBorder=True),

                # Заменили три DataTable на три графика столбцов корреляций
                html.Div(
                    id="corr-bars-section",
                    children=[
                        dmc.Paper([
                            dmc.Grid([
                                dmc.GridCol([dcc.Graph(id="corr-bar-x", config={'displaylogo': False, 'responsive': True})], span=6),
                                dmc.GridCol([dcc.Graph(id="corr-bar-y", config={'displaylogo': False, 'responsive': True})], span=6),
                            ])
                        ], style={**STYLE_CARD, "overflow": "visible"}, shadow="md", p="md", withBorder=True),
                    ],
                    style={**PAPER_BASE, "visibility": "hidden"}  # ← по умолчанию скрыто
                ),
            ], span=8)
        ])
    ]
)


@app.callback(
    Output("de-modal", "opened"),
    Input("de-button", "n_clicks"),
    State("de-modal", "opened"),
    prevent_initial_call=True
)
def toggle_de_modal(n, opened):
    if not n:
        raise PreventUpdate
    return not bool(opened)

@app.server.route("/_shutdown", methods=["POST"])
def _shutdown():
    _shutdown_server()
    return "Server shutting down..."

@app.callback(
    Output("cluster-metrics", "data"),
    Input("filtered-data", "data"),
    State("cluster-cols", "value"),
    State("meta-columns", "data"),
    prevent_initial_call=True
)
def compute_cluster_metrics(filtered_json, cluster_cols, meta):
    if not filtered_json or not cluster_cols or len(cluster_cols) < 2:
        raise PreventUpdate
    try:
        df = read_df_from_store(filtered_json, meta)
    except Exception:
        raise PreventUpdate
    use = [c for c in cluster_cols if c in df.columns]
    if len(use) < 2:
        raise PreventUpdate

    X = df[use].apply(pd.to_numeric, errors="coerce").dropna()
    if X.shape[0] < 5:
        # слишком мало данных для устойчивых метрик
        raise PreventUpdate

    Xs = StandardScaler().fit_transform(X.values)

    # разумный диапазон K
    n = Xs.shape[0]
    k_max = max(3, min(12, n - 1))
    ks = list(range(2, k_max + 1))

    inertias = []
    silhouettes = []
    for k in ks:
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
            inertias.append(float(km.inertia_))
            # silhouette требует k>=2 и >k наблюдений – соблюдается
            sil = silhouette_score(Xs, km.labels_) if n > k else float("nan")
            silhouettes.append(float(sil))
        except Exception:
            inertias.append(float("nan"))
            silhouettes.append(float("nan"))

    return {"ks": ks, "inertias": inertias, "silhouettes": silhouettes}

# =========================
# Инициализация первого фильтра
# =========================
@app.callback(
    Output("filters-container", "children"),
    Output("filter-count", "data"),
    Output("filters-initialized", "data"),
    Input("stored-data", "data"),
    State("filters-initialized", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=False
)
def init_first_filter(stored_json, inited, meta):
    if not stored_json:
        # первый рендер без данных — пусто и не инициализировано
        return [], 0, False

    # если уже инициализировали для текущего файла — ничего не делаем
    if inited:
        raise PreventUpdate

    # строим первую строку из ПОЛНОГО исходного набора
    try:
        dff0 = read_df_from_store(stored_json, meta)
    except Exception:
        dff0 = pd.DataFrame()

    num_cols, cat_cols, dt_cols = classify_simple(dff0)
    options = [''] + [*cat_cols, *num_cols, *dt_cols]

    row = dmc.Grid(
        id="filter_row_1",
        children=[
            dmc.GridCol([
                dmc.Group([
                    create_dropdown(
                        id={"type": "filter-column", "index": 1},
                        options=options,
                        value="",
                        persistence=False
                    ),
                    dmc.ActionIcon(
                        id={"type": "remove-filter", "index": 1},
                        children="×",
                        color="red",
                        variant="outline",
                        size="xs",
                        disabled=True
                    )
                ], gap="sm")
            ], span=5),
            dmc.GridCol([html.Div(id={"type": "filter-control", "index": 1})], span=6)
        ]
    )
    return [row], 1, True



@app.callback(
    Output("filters-initialized", "data", allow_duplicate=True),
    Input("stored-data", "data"),
    prevent_initial_call=True
)
def reset_filters_flag_on_new_file(_):
    return False

# ============ Диалог настройки (открыть дравер) ============
@callback(Output("drawer-simple", "opened"), Input("drawer-demo-button", "n_clicks"), prevent_initial_call=True)
def drawer_demo(n_clicks):
    return True



# ============ Локальный выбор файла (даёт полный путь) ============
# Важно: это работает ТОЛЬКО если вы запускаете Dash ЛОКАЛЬНО на своём ПК (сервер = ваш компьютер),
# и есть GUI. В браузере получить путь через dcc.Upload нельзя (ограничение безопасности).
@app.callback(
    Output("source-file-path", "data"),
    Output("source-file-name", "data"),
    Output("file-path-message", "children"),
    Input("pick-file-btn", "n_clicks"),
    prevent_initial_call=True
)
def pick_local_file(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

        path = filedialog.askopenfilename(
            title="Выберите файл данных",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("Pickle files", "*.pkl"),
                ("All files", "*.*"),
            ],
        )
        try:
            root.destroy()
        except Exception:
            pass
    except Exception as e:
        # headless/remote запуск: диалог не откроется
        return no_update, no_update, f"Локальный выбор файла недоступен: {e}"

    if not path:
        raise PreventUpdate

    return path, os.path.basename(path), f"Путь: {path}"


@app.callback(
    Output('status-message', 'children'),
    Output('sheet-menu-wrapper', 'children'),
    Output('stored-sheet-names', 'data'),
    Output('stored-data', 'data'),
    Output('selected-sheet', 'data'),
    Output('filtered-data', 'data', allow_duplicate=True),   # NEW
    Output('meta-columns', 'data', allow_duplicate=True),    # NEW
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),  # Добавлено для уведомлений
    Input('source-file-path', 'data'),
    State('source-file-name', 'data'),
    prevent_initial_call=True
)
def on_excel_upload(local_path, local_name):
    """Загрузка данных ТОЛЬКО с локального диска через выбор файла (tkinter).
    Сохранение пути делается отдельным callback по кнопке pick-file-btn.
    """
    if not local_path:
        raise PreventUpdate

    filename = local_name or os.path.basename(local_path)
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == '.xlsx':
            # ВАЖНО: ExcelFile нужно закрывать, иначе на Windows исходный .xlsx может оставаться 'занятым'.
            with pd.ExcelFile(local_path, engine='openpyxl') as xl:
                sheets = xl.sheet_names
                # один лист — читаем сразу
                if len(sheets) == 1:
                    sheet_name = sheets[0]
                    df = xl.parse(sheet_name)

            # после выхода из with файл гарантированно закрыт
            if len(sheets) == 1:
                meta = meta_from_df(df)
                js = df.to_json(date_format='iso', orient='split')
                msg = f"Загружен лист: {sheet_name}, кол-во строк: {len(df)}"
                return (
                    html.Div(msg, style={'color': 'green'}),
                    dash.no_update, sheets, js, sheet_name,
                    js, meta, []
                )

            # несколько листов — показываем модалку выбора
            modal = dmc.Modal(
                id="sheet-modal",
                title="Выберите лист Excel",
                opened=True,
                centered=True,
                zIndex=1000,
                children=[
                    dmc.Stack(
                        gap="sm",
                        children=[
                            dmc.Text("Выберите лист", c="blue", fw=500, size="sm"),
                            *[
                                dmc.Button(
                                    sheet,
                                    variant="light",
                                    fullWidth=True,
                                    id={"type": "sheet-select", "index": sheet},
                                    leftSection=DashIconify(icon="vscode-icons:file-type-excel2", width=20)
                                )
                                for sheet in sheets
                            ]
                        ]
                    )
                ]
            )
            msg = f"Загружен Excel: {filename}, листов: {len(sheets)}"
            return (
                html.Div(msg, style={'color': 'green'}),
                modal, sheets, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, []
            )
        elif ext == '.pkl':
            df = pd.read_pickle(local_path)
            meta = meta_from_df(df)
            js = df.to_json(date_format='iso', orient='split')
            msg = f"Загружен pkl: {filename}, строки: {len(df)}"
            return (
                html.Div(msg, style={'color': 'green'}),
                dash.no_update, None, js, None,
                js, meta, []
            )

    except Exception as e:
        notif = _make_error_notif(f"Ошибка загрузки файла: {str(e)}")
        return (
            html.Div(f"Ошибка: {e}", style={'color': 'red'}),
            dash.no_update, None, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, notif
        )

    notif = _make_error_notif("Неподдерживаемый формат (нужно .xlsx или .pkl)")
    return (
        html.Div("Неподдерживаемый формат", style={'color': 'red'}),
        dash.no_update, None, dash.no_update, dash.no_update,
        dash.no_update, dash.no_update, notif
    )



@app.callback(Output('sheet-modal', 'opened'), Input({'type': 'sheet-select', 'index': ALL}, 'n_clicks'), prevent_initial_call=True)
def close_modal(n_clicks):
    if not any(n_clicks):
        raise dash.exceptions.PreventUpdate
    return False
@app.callback(
    Output("filtered-data", "data", allow_duplicate=True),
    Output("meta-columns", "data", allow_duplicate=True),
    Output("cluster-metrics", "data", allow_duplicate=True),  # ← добавлено
    Output("notifications-container", "sendNotifications", allow_duplicate=True),  # ← добавлено (DE)

        Output("de-agg-status", "children", allow_duplicate=True),  # ← статус выполнения (DE)
# Авто-триггеры
    Input("stored-data", "data"),
    Input("filters-state", "data"),

    # Кнопки
    Input("btn-grouping", "n_clicks"),
    Input("btn-cluster",  "n_clicks"),
    Input("btn-agg",      "n_clicks"),

    # Параметры (State — не триггерят)
    State("bin-column", "value"),
    State("bin-method", "value"),
    State("bin-k", "value"),
    State("cluster-cols", "value"),
    State("cluster-k", "value"),

    # Data Engineering (агрегаты)
    State("agg-keys", "value"),
    State("agg-cols", "value"),
    State("agg-metrics", "value"),
    State("agg-exclude-zeros", "checked"),
    State("agg-exclude-empty", "checked"),

    # Текущий filtered (для кнопок, чтобы не терять ранее вычисленные столбцы)
    State("filtered-data", "data"),
    State("meta-columns", "data"),

    prevent_initial_call=True
)
def pipeline_graf_dataset(stored_json, filters_state,
                          n_group_btn, n_cluster_btn, n_agg_btn,
                          bin_col, bin_method, bin_k,
                          cluster_cols, cluster_k,
                          agg_keys, agg_cols, agg_metrics, agg_exclude_zeros, agg_exclude_empty,
                          filtered_json_prev, meta_state):
    notifications = []  # для dmc.Notifications.sendNotifications
    status_msg = dash.no_update  # текстовый статус в модалке (обновляем только по кнопке Рассчитать)


    # === 0) Исходные данные ===
    if not stored_json:
        raise PreventUpdate
    try:
        df0 = read_df_from_store(stored_json, meta_state)
    except Exception:
        return dash.no_update, dash.no_update, dash.no_update, _make_error_notif("Data Engineering: не удалось прочитать датасет."), dash.no_update
    if df0 is None or df0.empty:
        return dash.no_update, dash.no_update, dash.no_update, _make_error_notif("Data Engineering: не удалось прочитать датасет."), dash.no_update

    # === 1) База: кнопки → текущий filtered; stored/filters → от исходника ===
    trig = ctx.triggered_id
    if trig in ("btn-grouping", "btn-cluster", "btn-agg"):
        # Стартуем с текущего filtered, чтобы не терять уже рассчитанные колонки
        if filtered_json_prev:
            try:
                df = read_df_from_store(filtered_json_prev, meta_state)
            except Exception:
                df = df0.copy()
        else:
            df = df0.copy()
    else:
        # Чистый пересчёт от stored + фильтры
        df = df0.copy()
        meta0 = meta_from_df(df)
        fs = filters_state if isinstance(filters_state, dict) else {}

        def _apply_filters_simple(frame: pd.DataFrame, fs_dict: dict, meta: dict) -> pd.DataFrame:
            if not fs_dict or frame.empty:
                return frame
            mask = pd.Series(True, index=frame.index)
            for _, cfg in fs_dict.items():
                col = (cfg or {}).get("column")
                val = (cfg or {}).get("value")
                if not col or val in (None, [], '') or col not in frame.columns:
                    continue
                if col in (meta.get("numeric") or []):
                    lo, hi = (val or [None, None])
                    if lo is not None:
                        mask &= (pd.to_numeric(frame[col], errors="coerce") >= float(lo))
                    if hi is not None:
                        mask &= (pd.to_numeric(frame[col], errors="coerce") <= float(hi))
                else:
                    vs = val if isinstance(val, list) else [val]
                    mask &= frame[col].isin(vs)
            return frame.loc[mask]

        df = _apply_filters_simple(df, fs, meta0)

    # === 2) Флаги выполнения ===
    apply_binning    = (trig in ("btn-grouping", "filters-state", "stored-data"))
    apply_clustering = (trig in ("btn-cluster",  "filters-state", "stored-data"))
    apply_agg        = (trig == "btn-agg")  # только по кнопке, без автозапуска на загрузке/фильтрах

    # === 3) Биннинг (по кнопке/фильтрам/файлу) ===
    if apply_binning and bin_col and (bin_col in df.columns) and (bin_k is not None) and (int(bin_k) >= 2):
        # Удаляем только предыдущие "Группа(.)"
        for c in list(df.columns):
            if isinstance(c, str) and c.startswith("Группа(") and c.endswith(")"):
                df.drop(columns=[c], inplace=True, errors="ignore")

        ser = pd.to_numeric(df[bin_col], errors="coerce")
        valid = ser.dropna()
        if not valid.empty:
            try:
                if bin_method == "width":
                    cats = pd.cut(valid, bins=int(bin_k), include_lowest=True, duplicates="drop")
                else:
                    cats = pd.qcut(valid, q=int(bin_k), duplicates="drop")
                grp_name = f"Группа({bin_col})"
                df.loc[valid.index, grp_name] = cats.astype("string")
            except Exception:
                pass

    # === 4) Кластеризация (по кнопке/фильтрам/файлу) ===
    if apply_clustering and cluster_cols and len(cluster_cols) >= 2 and (cluster_k or 0) >= 2:
        use_cols = [c for c in cluster_cols if c in df.columns]
        if len(use_cols) >= 2:
            # Удаляем только свои служебные столбцы
            for col in ("PCA1", "PCA2", "Кластеры"):
                if col in df.columns:
                    df.drop(columns=[col], inplace=True, errors="ignore")

            num_df = df[use_cols].select_dtypes(include=[np.number]).dropna(how='any')
            if num_df.shape[0] >= 3 and num_df.shape[1] >= 2:
                Xs = StandardScaler().fit_transform(num_df.values)
                km = KMeans(n_clusters=cluster_k, n_init=10, random_state=42).fit(Xs)
                # Fix: Format labels as "Кластер N"
                labels = pd.Series([f"Кластер {i}" for i in km.labels_], index=num_df.index, dtype="string")
                df.loc[num_df.index, "Кластеры"] = labels

                try:
                    pca = PCA(n_components=2).fit_transform(Xs)
                    pca_df = pd.DataFrame(pca, index=num_df.index, columns=["PCA1", "PCA2"])
                    df = df.join(pca_df, how="left")
                except Exception:
                    pass

    # === 5) Data Engineering: агрегаты по группам (по кнопке/фильтрам/файлу) ===
    if apply_agg:
        # Понятный статус даже если уведомления не отображаются
        status_msg = "Data Engineering: ожидает параметров…"

        if not (agg_keys and agg_cols and agg_metrics):
            notifications.append({
                "id": "de-agg-missing",
                "title": "Data Engineering",
                "message": "Выберите ключ(и), столбцы и метрики, затем нажмите Рассчитать.",
                "color": "orange",
                "action": "show",
                "autoClose": 6000,
            })
            status_msg = "Выберите ключ(и), столбцы и метрики."

        else:
            # --- Нормализация выбранных значений (защита от пробелов/неразрывных пробелов/разного регистра) ---
            def _norm_token(x) -> str:
                # удаляем любые пробельные символы (включая NBSP), приводим к lower
                return re.sub(r"[\s\u00A0]+", "", str(x)).lower()

            # маппинг нормализованных имён колонок -> фактическое имя (с сохранением оригинальных пробелов)
            _col_map = {_norm_token(c): c for c in df.columns}

            def _resolve_columns(selected):
                out = []
                for v in (selected or []):
                    if v in df.columns:
                        out.append(v)
                        continue
                    nv = _norm_token(v)
                    if nv in _col_map:
                        out.append(_col_map[nv])
                # уникализируем, сохраняя порядок
                return list(dict.fromkeys(out))

            keys = _resolve_columns(agg_keys)
            cols = _resolve_columns(agg_cols)

            # Нормализуем метрики: допускаем значения с пробелами/русскими названиями/синонимами
            _metric_map = {
                "mean": "mean", "avg": "mean", "average": "mean", "среднее": "mean",
                "median": "median", "медиана": "median",
                "mode": "mode", "мода": "mode",
                "sum": "sum", "сумма": "sum",
                "cumsum": "cumsum", "cumulative": "cumsum", "кумулятивнаясумма": "cumsum", "накопительнаясумма": "cumsum",
                "min": "min", "мин": "min",
                "max": "max", "макс": "max",
                "std": "std", "stdev": "std", "sigma": "std", "стандартноеотклонение": "std",
                "count": "count", "количество": "count",
                "nunique": "nunique", "уникальных": "nunique", "числоуникальных": "nunique",
            }
            metrics = []
            for m in (agg_metrics or []):
                nm = _norm_token(m)
                nm = _metric_map.get(nm, nm)
                metrics.append(nm)
            metrics = list(dict.fromkeys([m for m in metrics if m]))

            if not keys or not cols or not metrics:
                notifications.append({
                    "id": "de-agg-invalid",
                    "title": "Data Engineering",
                    "message": "Ключи/столбцы не найдены в текущем датасете (после фильтров) или не выбраны метрики.",
                    "color": "orange",
                    "action": "show",
                    "autoClose": 7000,
                })
                status_msg = "Ключи/столбцы отсутствуют в текущем датасете или метрики не выбраны."
            else:
                exclude_zeros = bool(agg_exclude_zeros)
                exclude_empty = bool(agg_exclude_empty)

                added_cols = []
                skipped = []

                def _safe_tag(s: str) -> str:
                    return re.sub(r'[<>:"/\|?*]+', "_", str(s))

                keys_tag = _safe_tag("+".join(map(str, keys)))

                def _make_unique_col(name: str) -> str:
                    # Если имя уже существует (например, вы пересчитали метрику), добавим суффикс _2, _3 ...
                    base = str(name)
                    if base not in df.columns:
                        return base
                    i = 2
                    while f"{base}_{i}" in df.columns:
                        i += 1
                    return f"{base}_{i}"

                groupers_series = [df[k] for k in keys]

                # совместимость: dropna может отсутствовать в старых pandas
                def _gb_transform(series: pd.Series, func: str):
                    try:
                        return series.groupby(groupers_series, dropna=False, sort=False).transform(func)
                    except TypeError:
                        return series.groupby(groupers_series, sort=False).transform(func)

                def _gb_apply(series: pd.Series, fn):
                    try:
                        return series.groupby(groupers_series, dropna=False, sort=False).transform(fn)
                    except TypeError:
                        return series.groupby(groupers_series, sort=False).transform(fn)

                for col in cols:
                    is_numeric = pd.api.types.is_numeric_dtype(df[col])
                    x = pd.to_numeric(df[col], errors="coerce") if is_numeric else None

                    if is_numeric:
                        x_use = x.copy()
                        if not exclude_empty:
                            x_use = x_use.fillna(0)
                        if exclude_zeros:
                            x_use_no0 = x_use.mask(x_use == 0, np.nan)
                        else:
                            x_use_no0 = x_use

                    for met in metrics:
                        if met in ("mean", "median", "sum", "min", "max", "std"):
                            if not is_numeric:
                                skipped.append((col, met))
                                continue
                            try:
                                res = _gb_transform(x_use_no0, met)
                            except Exception:
                                skipped.append((col, met))
                                continue
                            out_col = _make_unique_col(f"{col}_{met}")
                            df[out_col] = res
                            added_cols.append(out_col)

                        elif met == "count":
                            try:
                                if is_numeric:
                                    if exclude_empty:
                                        res = _gb_transform(x_use_no0, "count")
                                    else:
                                        res = _gb_apply(df[col], lambda z: len(z))
                                else:
                                    s = df[col]
                                    if exclude_empty:
                                        res = _gb_transform(s, "count")
                                    else:
                                        res = _gb_apply(s, lambda z: len(z))
                            except Exception:
                                skipped.append((col, met))
                                continue
                            out_col = _make_unique_col(f"{col}_count")
                            df[out_col] = res
                            added_cols.append(out_col)

                        elif met == "nunique":
                            s = df[col]
                            try:
                                if not exclude_empty:
                                    s = s.astype("object").where(pd.notna(s), "<EMPTY>")
                                if exclude_zeros and is_numeric:
                                    s_num = pd.to_numeric(s, errors="coerce")
                                    s = s_num.mask(s_num == 0, np.nan)
                                res = _gb_apply(s, lambda z: z.nunique(dropna=exclude_empty))
                            except Exception:
                                skipped.append((col, met))
                                continue
                            out_col = _make_unique_col(f"{col}_nunique")
                            df[out_col] = res
                            added_cols.append(out_col)

                        elif met == "mode":
                            s = df[col]
                            try:
                                if not exclude_empty:
                                    s = s.astype("object").where(pd.notna(s), "<EMPTY>")
                                if exclude_zeros and is_numeric:
                                    s_num = pd.to_numeric(s, errors="coerce")
                                    s = s_num.mask(s_num == 0, np.nan)

                                def _mode_first(z: pd.Series):
                                    mm = z.mode(dropna=True)
                                    return mm.iloc[0] if len(mm) else np.nan

                                res = _gb_apply(s, _mode_first)
                            except Exception:
                                skipped.append((col, met))
                                continue
                            out_col = _make_unique_col(f"{col}_mode")
                            df[out_col] = res
                            added_cols.append(out_col)

                        elif met == "cumsum":
                            if not is_numeric:
                                skipped.append((col, met))
                                continue
                            try:
                                x_cum = x.copy().fillna(0)
                                res = x_cum.groupby(groupers_series, sort=False).cumsum()
                            except Exception:
                                skipped.append((col, met))
                                continue
                            out_col = _make_unique_col(f"{col}_cumsum")
                            df[out_col] = res
                            added_cols.append(out_col)

                        else:
                            skipped.append((col, met))
                            continue

                # убираем дубли
                added_cols = list(dict.fromkeys(added_cols))

                if added_cols:
                    notifications.append({
                        "id": "de-agg-ok",
                        "title": "Data Engineering",
                        "message": f"Добавлены столбцы: {len(added_cols)}",
                        "color": "green",
                        "action": "show",
                        "autoClose": 4000,
                    })
                    status_msg = f"Готово: добавлено {len(added_cols)} столбцов."
                else:
                    notifications.append({
                        "id": "de-agg-none",
                        "title": "Data Engineering",
                        "message": "Новые столбцы не добавлены (возможно, несовместимые метрики или типы данных).",
                        "color": "orange",
                        "action": "show",
                        "autoClose": 7000,
                    })
                    status_msg = "Ничего не добавлено (проверьте типы столбцов и метрики)."

                if skipped:
                    preview = ", ".join([f"{c}:{m}" for c, m in skipped[:5]])
                    more = "" if len(skipped) <= 5 else f" (+{len(skipped)-5})"
                    notifications.append({
                        "id": "de-agg-skip",
                        "title": "Data Engineering",
                        "message": f"Пропущены несовместимые пары (столбец/метрика): {preview}{more}",
                        "color": "yellow",
                        "action": "show",
                        "autoClose": 7000,
                    })
    # === 6) Финальный форс-типов ===

    # === 6) Серилизация и roundtrip-проверка ===
    js_filtered   = df.to_json(date_format='iso', orient='split')
    meta_filtered = meta_from_df(df)

    # Метрики кластера (отдельно, чтобы не ломать возврат)
    cluster_metrics = None
    try:
        if cluster_cols:
            X = df[cluster_cols].apply(pd.to_numeric, errors="coerce").dropna()
            if X.shape[0] >= 5:
                Xs = StandardScaler().fit_transform(X.values)
                n = Xs.shape[0]
                k_max = max(3, min(12, n - 1))
                ks = list(range(2, k_max + 1))
                inertias = []
                silhouettes = []
                for k in ks:
                    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
                    inertias.append(float(km.inertia_))
                    sil = silhouette_score(Xs, km.labels_) if n > k else float("nan")
                    silhouettes.append(float(sil))
                cluster_metrics = {"ks": ks, "inertias": inertias, "silhouettes": silhouettes}
    except Exception as e:
        logger.warning(f"[cluster-metrics] fail: {e}")

    return js_filtered, meta_filtered, cluster_metrics, notifications, status_msg



# ===================== Data Engineering: create text copies =====================
@app.callback(
    Output("filtered-data", "data", allow_duplicate=True),
    Output("meta-columns", "data", allow_duplicate=True),
    Output("de-txt-status", "children"),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("btn-txtcopy", "n_clicks"),
    State("filtered-data", "data"),
    State("meta-columns", "data"),
    State("txtcopy-cols", "value"),
    State("txtcopy-suffix", "value"),
    State("txtcopy-strip", "checked"),
    prevent_initial_call=True
)
def create_text_copies(n_clicks, filtered_json, meta, selected_cols, suffix, do_strip):
    if not n_clicks:
        raise PreventUpdate

    notifications = []

    if not filtered_json:
        notifications.append({
            "id": "de-txtcopy-nodata",
            "title": "Data Engineering",
            "message": "Нет данных: сначала загрузите файл и/или примените фильтры.",
            "color": "red",
            "action": "show",
            "autoClose": 6000,
        })
        return no_update, no_update, "Нет данных для преобразования.", notifications

    try:
        df = read_df_from_store(filtered_json, meta)
    except Exception as e:
        notifications.append({
            "id": "de-txtcopy-badjson",
            "title": "Data Engineering",
            "message": f"Не удалось прочитать текущий датасет: {e}",
            "color": "red",
            "action": "show",
            "autoClose": 7000,
        })
        return no_update, no_update, "Ошибка чтения данных.", notifications

    if df is None or df.empty:
        notifications.append({
            "id": "de-txtcopy-empty",
            "title": "Data Engineering",
            "message": "Текущий датасет пустой — нечего преобразовывать.",
            "color": "yellow",
            "action": "show",
            "autoClose": 6000,
        })
        return no_update, no_update, "Текущий датасет пустой.", notifications

    selected_cols = selected_cols or []
    if not selected_cols:
        notifications.append({
            "id": "de-txtcopy-noselect",
            "title": "Data Engineering",
            "message": "Выберите хотя бы один столбец для создания текстовой копии.",
            "color": "yellow",
            "action": "show",
            "autoClose": 6000,
        })
        return no_update, no_update, "Выберите столбец(ы).", notifications

    suffix = (suffix or "_txt")
    # нормализуем суффикс: если пользователь ввёл просто 'txt' — добавим '_'
    suf_clean = str(suffix).strip()
    if suf_clean and not suf_clean.startswith("_"):
        suf_clean = "_" + suf_clean
    if not suf_clean:
        suf_clean = "_txt"

    def _norm_token_local(x) -> str:
        return re.sub(r"[\s\u00A0]+", "", str(x)).lower()

    col_map = {_norm_token_local(c): c for c in df.columns}

    created = []
    skipped = []

    for col in selected_cols:
        actual = col_map.get(_norm_token_local(col))
        if not actual:
            skipped.append(str(col))
            continue

        base_new = f"{actual}{suf_clean}"
        new_name = base_new
        k = 2
        while new_name in df.columns:
            new_name = f"{base_new}_{k}"
            k += 1

        s = df[actual]

        # Превращаем в текстовый dtype, чтобы Plotly и фильтры воспринимали как категорию/текст
        try:
            new_s = s.astype("string")
        except Exception:
            new_s = pd.Series([None if pd.isna(v) else str(v) for v in s], dtype="string")

        if do_strip:
            try:
                new_s = new_s.str.strip()
            except Exception:
                pass

        df[new_name] = new_s
        created.append(new_name)

    if not created:
        msg = "Не создано ни одной колонки (проверьте выбор столбцов)."
        if skipped:
            msg += f" Не найдены: {', '.join(skipped[:8])}" + ("..." if len(skipped) > 8 else "")
        notifications.append({
            "id": "de-txtcopy-none",
            "title": "Data Engineering",
            "message": msg,
            "color": "yellow",
            "action": "show",
            "autoClose": 8000,
        })
        return no_update, no_update, "Ничего не создано.", notifications

    # обновляем meta и сериализуем
    meta = meta_from_df(df)
    out_json = df.to_json(date_format="iso", orient="split")

    notifications.append({
        "id": "de-txtcopy-ok",
        "title": "Data Engineering",
        "message": f"Созданы текстовые копии: {len(created)}",
        "color": "green",
        "action": "show",
        "autoClose": 4000,
    })

    status = f"Готово: создано {len(created)}: " + ", ".join(created[:6]) + ("..." if len(created) > 6 else "")
    if skipped:
        status += f" | не найдены: {', '.join(skipped[:4])}" + ("..." if len(skipped) > 4 else "")

    return out_json, meta, status, notifications

@app.callback(
    Output('selected-sheet', 'data', allow_duplicate=True),
    Input({'type': 'sheet-select', 'index': ALL}, 'n_clicks'),
    State({'type': 'sheet-select', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def on_sheet_selected(n_clicks, ids):
    if not any(n_clicks):
        raise dash.exceptions.PreventUpdate
    clicked_idx = [i for i, n in enumerate(n_clicks) if n]
    if not clicked_idx:
        raise dash.exceptions.PreventUpdate
    selected = ids[clicked_idx[0]]['index']
    return selected


@app.callback(
    Output('stored-data', 'data', allow_duplicate=True),
    Output('status-message', 'children', allow_duplicate=True),
    Output('sheet-modal', 'opened', allow_duplicate=True),
    Output('filtered-data', 'data', allow_duplicate=True),   # NEW
    Output('meta-columns', 'data', allow_duplicate=True),    # NEW
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),  # Добавлено
    Input('selected-sheet', 'data'),
    State('source-file-path', 'data'),
    prevent_initial_call=True
)
def load_selected_sheet(sheet_name, local_path):
    if not sheet_name:
        raise PreventUpdate

    if not local_path:
        notif = _make_error_notif("Нет пути к исходному файлу. Выберите файл заново.")
        return dash.no_update, html.Div("Ошибка: нет пути к файлу", style={'color': 'red'}), False, dash.no_update, dash.no_update, notif

    try:
        df = pd.read_excel(local_path, engine='openpyxl', sheet_name=sheet_name)
    except Exception as e:
        notif = _make_error_notif(f"Ошибка загрузки листа: {str(e)}")
        return dash.no_update, html.Div(f"Ошибка: {e}", style={'color': 'red'}), False, dash.no_update, dash.no_update, notif

    try:
        meta = meta_from_df(df)
        js = df.to_json(date_format='iso', orient='split')

        msg = f"Загружен лист: {sheet_name}, строки: {len(df)}"
        return js, html.Div(msg, style={'color': 'green'}), False, js, meta, []
    except Exception as e:
        notif = _make_error_notif(f"Ошибка загрузки листа: {str(e)}")
        return dash.no_update, html.Div(f"Ошибка: {e}", style={'color': 'red'}), False, dash.no_update, dash.no_update, notif






# === Управление фильтрами (через dash.ctx) ===
@app.callback(
    Output("filters-container", "children", allow_duplicate=True),
    Output("filter-count", "data", allow_duplicate=True),
    Output("filters-state", "data", allow_duplicate=True),
    Input("add-filter-btn", "n_clicks"),
    Input({"type": "remove-filter", "index": ALL}, "n_clicks"),
    State("filter-count", "data"),
    State("filters-container", "children"),
    State("filters-state", "data"),
    State("stored-data", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=True
)
def manage_filters(add_clicks, remove_clicks, filter_count, current_filters, filters_state, stored_json, meta):
    trig = ctx.triggered_id
    if not trig:
        raise PreventUpdate

    # формируем опции колонок из ИСХОДНОГО набора (stored)
    try:
        dff0 = read_df_from_store(stored_json, meta) if stored_json else pd.DataFrame()
    except Exception:
        dff0 = pd.DataFrame()
    num_cols, cat_cols, dt_cols = classify_simple(dff0)
    col_options = [''] + [*cat_cols, *num_cols, *dt_cols]

    def row(filter_id, state):
        current_column = (state or {}).get(str(filter_id), {}).get('column', '')
        return dmc.Grid(
            id=f"filter_row_{filter_id}",
            children=[
                dmc.GridCol([
                    dmc.Group([
                        create_dropdown(
                            id={"type": "filter-column", "index": filter_id},
                            options=col_options,
                            value=current_column,
                            persistence=False
                        ),
                        dmc.ActionIcon(
                            id={"type": "remove-filter", "index": filter_id},
                            children="×", color="red", variant="outline", size="xs",
                            disabled=(filter_id == 1)
                        )
                    ], gap="sm")
                ], span=5),
                dmc.GridCol([html.Div(id={"type": "filter-control", "index": filter_id})], span=6)
            ]
        )

    state = (filters_state or {}).copy()
    cur = list(current_filters or [])
    count = int(filter_count or 0)

    if trig == "add-filter-btn":
        new_id = count + 1
        cur.append(row(new_id, state))
        return cur, new_id, state

    if isinstance(trig, dict) and trig.get("type") == "remove-filter":
        idx = trig.get("index")
        cur = [c for c in cur if c["props"]["id"] != f"filter_row_{idx}"]
        state.pop(str(idx), None)
        return cur, len(cur), state

    # на прочие события контейнер не трогаем
    return no_update, no_update, state




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

@app.callback(
    Output({"type": "filter-control", "index": MATCH}, "children"),
    Input({"type": "filter-column", "index": MATCH}, "value"),
    State({"type": "filter-column", "index": MATCH}, "id"),
    State("filters-state", "data"),
    State("stored-data", "data"),          # ← берём исходный датасет
    State("meta-columns", "data"),
    prevent_initial_call=True
)
def update_filter_controls(column, column_id, filters_state, stored_json, meta):
    if not stored_json:
        return html.Div("Загрузите данные")

    try:
        full_dff = read_df_from_store(stored_json, meta)
    except Exception:
        return html.Div("Ошибка чтения данных")

    if full_dff.empty or not column:
        return html.Div("Выберите столбец")

    fid = str(column_id['index'])
    current_value = (filters_state or {}).get(fid, {}).get('value')

    # create_value_control должен уметь работать с полным df:
    # - категории: Multi/Dropdown со списком уникальных из full_dff[column]
    # - числовое: RangeSlider по min/max full_dff[column]
    # - datetime: DateRangePicker по min/max full_dff[column]
    return create_value_control(fid, column, current_value, full_dff)



# Сохранение состояния фильтров + применение к данным
@app.callback(
    Output("filters-state", "data"),
    Input({"type": "filter-column", "index": ALL}, "value"),
    Input({"type": "filter-value", "index": ALL}, "value"),
    State({"type": "filter-column", "index": ALL}, "id"),
    State("filters-state", "data"),
    State("filtered-data", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=True
)
def update_filters_state(columns, values, column_ids, filters_state, filtered_json, meta):
    # 1) Базовая защита: всегда возвращаем dict
    prev_state = filters_state if isinstance(filters_state, dict) else {}

    if not filtered_json:
        logger.warning("Данные не загружены")
        return prev_state

    try:
        dff = read_df_from_store(filtered_json, meta)
    except Exception:
        logger.warning("Не удалось прочитать filtered-data")
        return prev_state
    if dff.empty:
        logger.warning("Данные пусты")
        return prev_state

    updated = dict(prev_state)
    columns = columns or []
    values = values or []
    column_ids = column_ids or []

    for i, col_id in enumerate(column_ids):
        fid = str(col_id.get('index'))
        col_ok = i < len(columns) and columns[i]
        val_ok = i < len(values) and values[i] not in (None, [], '')
        if not col_ok or not val_ok:
            if fid in updated:
                del updated[fid]
            continue

        updated.setdefault(fid, {})
        updated[fid]['column'] = columns[i]
        updated[fid]['value'] = values[i]

    return updated

# Скачивание HTML
@app.callback(
    # Outputs first
    Output(DownloadFile, "data"),
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),
    # Then Inputs
    Input("download-button", "n_clicks"),
    # Then States
    State("graph", "figure"),
    State("dropdown_x", "value"),
    State("dropdown_y", "value"),
    State("segmented", "value"),
    prevent_initial_call=True
)
def download_html(n_clicks, figure, dropdown_x, dropdown_y, segmentedcontrol_value):
    if not n_clicks or not figure:
        raise PreventUpdate
    try:
        fig = go.Figure(figure)
        filename = (
            f'{dropdown_x} vs {dropdown_y} {segmentedcontrol_value}.html'
            if all([dropdown_x, dropdown_y, segmentedcontrol_value]) else "graph.html"
        )
        html_content = fig.to_html(include_plotlyjs='cdn')
        return {"content": html_content, "filename": filename, "type": "text/html"}, []  # mime type and empty notifications
    except Exception as e:
        notif = _make_error_notif(f"Ошибка скачивания: {str(e)}")
        return dash.no_update, notif


# =========================
# Сохранить текущий датасет (после фильтров/биннинга/кластеризации) в Excel
# =========================
# =========================
# Сохранить текущий датасет (после фильтров/биннинга/кластеризации) в Excel
# Сохраняем РЯДОМ с исходным файлом (по сохранённому локальному пути)
# =========================
@app.callback(
    Output(DownloadExcel, "data"),
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),
    Input("download-excel-button", "n_clicks"),
    State("filtered-data", "data"),
    State("source-file-path", "data"),
    State("source-file-name", "data"),
    State("selected-sheet", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=True
)
def download_excel_dataset(n_clicks, filtered_json, source_path, source_name, sheet_name, meta):
    if not n_clicks:
        raise PreventUpdate

    if not filtered_json:
        return dash.no_update, _make_error_notif(
            "Нет данных для сохранения. Загрузите файл и примените фильтры/кластеризацию."
        )

    if not source_path:
        return dash.no_update, _make_error_notif(
            "Неизвестен путь исходного файла. Выберите файл заново через кнопку выбора файла."
        )

    try:
        df = read_df_from_store(filtered_json, meta)
    except Exception as e:
        return dash.no_update, _make_error_notif(f"Не удалось прочитать текущий датасет: {e}")

    if df is None or df.empty:
        return dash.no_update, _make_error_notif("Текущий датасет пустой — сохранять нечего.")

    # имя файла
    stem = Path(source_name or source_path).stem if (source_name or source_path) else "dataset"
    sheet_suffix = f"_{sheet_name}" if sheet_name else ""
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{stem}{sheet_suffix}_filtered_{ts}.xlsx"

    # sanitize (Windows)
    out_name = re.sub(r'[<>:"/\\|?*]+', '_', out_name)

    out_path = Path(source_path).resolve().parent / out_name

    try:
        df.to_excel(out_path, index=False, engine="openpyxl")
        ok = [{
            "id": str(uuid.uuid4()),
            "title": "Excel сохранён",
            "message": f"Файл сохранён рядом с исходником: {out_path}",
            "color": "green",
            "loading": False,
            "action": "show",
            "autoClose": 7000,
            "style": {"fontSize": 18},
        }]
        return dash.no_update, ok
    except Exception as e:
        return dash.no_update, _make_error_notif(f"Ошибка сохранения Excel: {e}")



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
# ... (весь ваш код сверху остается без изменений, включая imports)

# =========================
# Приложение
# =========================
# ... (layout остается без изменений)

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



@app.callback(
    [
        # Осевые и прочие дропдауны
        Output("dropdown_x", "options"),
        Output("dropdown_y", "options"),
        Output("dropdown_z", "options"),
        Output("dropdown_color", "options"),
        Output("dropdown_size", "options"),
        Output("dropdown_hover_data",  "data"),
        Output("dropdown_corr_columns",  "data"),
        Output("dropdown_facet_row", "options"),
        Output("dropdown_facet_col", "options"),
        Output("dropdown_text", "options"),

        # Группировка/Кластеры
        Output("bin-column", "options"),
        Output("cluster-cols",  "data"),
        Output("agg-keys", "data"),
        Output("agg-cols", "data"),
        Output("txtcopy-cols", "data"),
    ],
    Input("filtered-data", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=False
)
def update_dropdown_options_all(filtered_json, meta):
    empty_axes = [[]] * 10
    empty_bin_options = []
    empty_cluster_opts = []
    empty_agg_data = []

    if not filtered_json:
        return (*empty_axes,
                empty_bin_options, empty_cluster_opts,
                empty_agg_data, empty_agg_data, empty_agg_data)

    try:
        dff = read_df_from_store(filtered_json, meta)
    except Exception:
        return (*empty_axes,
                empty_bin_options, empty_cluster_opts,
                empty_agg_data, empty_agg_data, empty_agg_data)

    if dff is None or dff.empty:
        return (*empty_axes,
                empty_bin_options, empty_cluster_opts,
                empty_agg_data, empty_agg_data, empty_agg_data)

    # простая классификация колонок
    numeric_cols, categorical_cols, datetime_cols = classify_simple(dff)

    def _opts(cols):
        return [{"label": str(c), "value": str(c)} for c in cols]

    all_cols     = [str(c) for c in dff.columns]
    all_options  = _opts(all_cols)
    color_options= [{"label": "Нет", "value": "Нет"}] + _opts(categorical_cols + numeric_cols)
    numeric_opts = _opts(numeric_cols)
    facet_options= [{"label": "Нет", "value": "Нет"}] + all_options

    # bin/cluster — числовые
    bin_options     = [{"label": c, "value": c} for c in numeric_cols]
    cluster_options = [{"label": c, "value": c} for c in numeric_cols]

    return [
        all_options,      # X
        all_options,      # Y
        all_options,      # Z
        color_options,    # Color
        numeric_opts,     # Size
        all_options,      # Hover
        numeric_opts,     # Corr
        facet_options,    # Facet row
        facet_options,    # Facet col
        all_options,      # Text
        bin_options,      # bin-column.options
        cluster_options,  # cluster-cols.options
        all_options,      # agg-keys.data
        all_options,      # agg-cols.data
        all_options       # txtcopy-cols.data
    ]

# Основной callback для главного графика (только Output("graph", "figure"))
@app.callback(
    Output("graph", "figure"),
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),  # Добавлено для уведомлений

    Input("update-graf", "n_clicks"),
    Input("dropdown_x", "value"),
    Input("dropdown_y", "value"),
    Input("dropdown_z", "value"),
    Input("dropdown_color", "value"),
    Input("dropdown_size", "value"),
    Input("dropdown_text", "value"),
    Input("dropdown_text_pozition", "value"),
    Input("segmented", "value"),
    Input("SwitchBubble", "checked"),
    Input("InputMaxSizeBubble", "value"),
    Input("InputSizePlot", "value"),
    Input("InputSizePlotW", "value"),
    Input("dropdown_style", "value"),

    # === ЕДИНСТВЕННЫЙ источник данных ===
    State("filtered-data", "data"),

    # Прочие состояния UI (как было)
    State("dropdown_hover_data", "value"),
    State("dropdown_corr_columns", "value"),
    Input("dropdown_facet_row", "value"),
    Input("dropdown_facet_col", "value"),
    State("filters-state", "data"),
    Input("font-size-xaxis", "value"),
    Input("font-size-yaxis", "value"),
    Input("font-size-ticks", "value"),
    Input("font-size-title", "value"),
    Input("dropdown_category_ascending", "value"),
    Input("dropdown_axes_category", "value"),
    Input("dropdown_overlay", "value"),
    Input("dropdown_legend", "value"),
    State("custom-colors", "data"),
    Input("tick-step-xaxis", "value"),
    Input("tick-step-yaxis", "value"),
    Input("dropdown_legend_order", "value"),
    State("input_legend_custom_order", "value"),
    State("meta-columns", "data"),

    prevent_initial_call=True
)
def update_main_graph(n_clicks, x_col, y_col, z_col, color_col, size_col, text_col, dropdown_text_pozition,
                      chart_type, bubble, MaxSizeBubble, height, width, selected_style,
                      filtered_json, hover_cols, corr_cols, facet_row, facet_col, filters_state,
                      xaxis_font_size, yaxis_font_size, font_size_ticks, title_font_size,
                      dropdown_sort_column, axes_category, dropdown_overlay, legend, custom_colors,
                      tick_step_x, tick_step_y, legend_order, legend_custom_order, meta):

    empty = _empty_fig()
    try:
        # ---- Чтение единственного источника ----
        if not filtered_json:
            return empty, []
        dff = read_df_from_store(filtered_json, meta)
        if dff is None or dff.empty:
            return empty, []

        # ---- Базовая валидация ----
        errors = []
        if not x_col or x_col not in dff.columns:
            errors.append(f"Не выбран или не существует столбец X: {x_col}")
        if chart_type == "3D_Scatter" and (not z_col or z_col not in dff.columns):
            errors.append("Для 3D требуется столбец Z")
        if errors:
            notif = _make_error_notif(" ".join(errors))
            return empty, notif

        # ---- Facet / Text ----
        facet_row = facet_row if (facet_row and facet_row in dff.columns) else None
        facet_col = facet_col if (facet_col and facet_col in dff.columns) else None
        text_data = dff[text_col] if (text_col and text_col in dff.columns and not dff.empty) else None

        plot_df = dff.copy()
        def _valid(col): 
            return bool(col) and (col in plot_df.columns)
        carg = color_col if _valid(color_col) else None
        sarg = size_col  if (bubble and _valid(size_col)) else None
        def _hide_top_facet_xlabels(fig: go.Figure):
            try:
                y0s = []
                for ya in fig.select_yaxes():
                    dom = getattr(ya, "domain", None)
                    if isinstance(dom, (list, tuple)) and len(dom) == 2:
                        y0s.append(dom[0])
                if not y0s:
                    return
                bottom = min(y0s)
                for xa in fig.select_xaxes():
                    anchor = getattr(xa, "anchor", None)
                    yaxis = getattr(fig.layout, anchor, None) if isinstance(anchor, str) else None
                    dom = getattr(yaxis, "domain", None) if yaxis is not None else None
                    if isinstance(dom, (list, tuple)) and len(dom) == 2:
                        if dom[0] > bottom + 1e-9:
                            xa.update(showticklabels=False)
                            xa.update(ticks="")
                            try:
                                xa.title.text = None
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"_hide_top_facet_xlabels failed: {e}")

        # ---- Тип осей (по новой схеме метаданных) ----
        meta = meta or {"numeric": [], "categorical": [], "datetime": []}
        def needs_text_axis(col: str) -> bool:
            if not col:
                return False
            if col in (meta.get("datetime") or []):
                return False
            return col not in (meta.get("numeric") or [])

        x_as_text = needs_text_axis(x_col)
        if x_as_text:
            plot_df[x_col] = plot_df[x_col].astype(str)

        fig = go.Figure()
        category_orders = {}

        # Category orders по первому фильтру (для facet) — сохраняем твою логику
        if isinstance(filters_state, dict) and len(filters_state) > 0:
            first_key = sorted(filters_state.keys())[0]
            first_filter = filters_state[first_key]
            filter_col = first_filter.get("column")
            filter_values = first_filter.get("value")
            if isinstance(filter_values, (int, float, str)):
                filter_values = [filter_values]
            if filter_col and isinstance(filter_values, list) and filter_values:
                if filter_col == facet_row or filter_col == facet_col:
                    category_orders = {filter_col: filter_values}
        if facet_row is None and facet_col is None:
            category_orders = None

        # ---------- Типы графиков ----------
        if chart_type == "Scatter":
            fig = px.scatter(
                plot_df, x=x_col, y=y_col,
                color=carg,
                size=sarg,
                size_max=MaxSizeBubble, height=height, width=width, hover_data=hover_cols,
                facet_row=facet_row, facet_col=facet_col, text=text_data,
                category_orders=category_orders, template=selected_style
            )
            if text_data is not None:
                fig.update_traces(textposition=dropdown_text_pozition, textfont=dict(size=font_size_ticks), selector=dict(mode='markers+text'))

        elif chart_type == "3D_Scatter":
            # 3D Scatter (фасеты в 3D не поддерживаются — честно игнорируем)
            fig = px.scatter_3d(
                plot_df, x=x_col, y=y_col, z=z_col,
                color=carg,
                size=sarg,
                size_max=MaxSizeBubble, height=height, width=width, hover_data=hover_cols,
                text=text_data, template=selected_style
            )
            if text_data is not None:
                fig.update_traces(textposition=dropdown_text_pozition, textfont=dict(size=font_size_ticks), selector=dict(mode='markers+text'))

        elif chart_type == "Box":
            fig = px.box(
                plot_df, x=x_col, y=y_col,
                color=carg,
                height=height, width=width, facet_row=facet_row, facet_col=facet_col,
                category_orders=category_orders, template=selected_style
            )
            fig.update_layout(boxmode="group")

        elif chart_type == "Bar":
            fig = px.bar(
                plot_df, x=x_col, y=y_col,
                color=carg,
                height=height, width=width, facet_row=facet_row, facet_col=facet_col,
                text_auto=True, category_orders=category_orders, template=selected_style
            )
            # overlay режим гистограмм/баров
            if dropdown_overlay in {'group', 'overlay', 'stack', 'relative'}:
                fig.update_layout(barmode=dropdown_overlay)
            if dropdown_overlay == 'overlay':
                fig.update_traces(opacity=0.85)

        elif chart_type == "Line":
            fig = px.line(
                plot_df, x=x_col, y=y_col,
                color=carg,
                height=height, width=width, facet_row=facet_row, facet_col=facet_col,
                category_orders=category_orders, template=selected_style
            )

        elif chart_type == "Hist":
            fig = px.histogram(
                plot_df, x=x_col,
                color=carg,
                height=height, width=width, facet_row=facet_row, facet_col=facet_col,
                category_orders=category_orders, template=selected_style
            )
            fig.update_layout(barmode=dropdown_overlay)
            if dropdown_overlay == 'overlay':
                fig.update_traces(opacity=0.75)

        elif chart_type == "Polar":
            fig = px.scatter_ternary(
                plot_df, a=x_col, b=y_col, c=z_col,
                color=carg,
                size=sarg,
                size_max=MaxSizeBubble, height=height, width=width, hover_data=hover_cols,
                text=text_data, template=selected_style
            )
            if text_data is not None:
                fig.update_traces(textposition=dropdown_text_pozition, textfont=dict(size=font_size_ticks), selector=dict(mode='markers+text'))

        elif chart_type == "Pie":
            if plot_df[x_col].dtypes != 'object':
                dff1 = plot_df[x_col].value_counts(dropna=False, bins=10).sort_values(ascending=False)
            else:
                dff1 = plot_df[x_col].value_counts(dropna=False).sort_values(ascending=False)
            dff1 = pd.DataFrame(dff1).reset_index()
            dff1.columns = [x_col, 'counts']
            dff1[x_col] = dff1[x_col].astype(str)
            fig = px.pie(dff1, values='counts', names=x_col, title=x_col, height=height, template=selected_style)
            fig.update_traces(textposition='inside', textinfo='percent+label+value', overwrite=True)

        elif chart_type == "Correlation":
            # Если corr_cols задан — рисуем heatmap; если нет — берём все числовые
            numeric_cols_all = (meta.get("numeric") or [])
            use_cols = [c for c in (corr_cols or numeric_cols_all) if c in numeric_cols_all]

            if use_cols:
                MAX_CORR_COLS = 50
                use_cols = use_cols[:MAX_CORR_COLS]
                corr_df = plot_df[use_cols].select_dtypes(include=[np.number]).dropna(how="all")
                if not corr_df.empty and corr_df.shape[1] >= 2:
                    corr_matrix = corr_df.corr().round(2)
                    # Heatmap
                    fig = go.Figure(go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu', zmin=-1, zmax=1,
                        text=corr_matrix.values, texttemplate="%{text}"
                    ))
                    fig.update_layout(
                        title='Корреляционная матрица',
                        xaxis=dict(tickangle=-45),
                        height=height, width=width,
                        template=selected_style
                    )

                    # автомаржины
                    fig.update_xaxes(automargin=True)
                    fig.update_yaxes(automargin=True)

                    # порядок легенды (heatmap её почти не использует — безопасно)
                    try:
                        _sort_legend_traces(fig, legend_order, legend_custom_order)
                    except Exception as _e:
                        logger.warning(f"Сортировка легенды пропущена: {_e}")

                else:
                    # не из чего считать матрицу
                    fig = _empty_fig()
            else:
                # нет колонок — пусто
                fig = _empty_fig()

        elif chart_type == "Violin":
            fig = px.violin(
                plot_df, x=x_col, y=y_col,
                color=carg,
                height=height, width=width, facet_row=facet_row, facet_col=facet_col,
                category_orders=category_orders, template=selected_style, box=False
            )

        elif chart_type == "Ridge":
            fig = go.Figure()
            if color_col == "Нет" or color_col not in plot_df.columns:
                fig.add_trace(go.Violin(
                    x=plot_df[x_col] if x_col in plot_df.columns else None,
                    y=plot_df[y_col] if y_col in plot_df.columns else None,
                    orientation='h', side='positive', width=3, points=False,
                    line_color=px.colors.qualitative.Plotly[0], name=y_col
                ))
            else:
                unique_values = plot_df[color_col].dropna().unique()
                colors = px.colors.qualitative.Plotly
                for i, val in enumerate(unique_values):
                    subset = plot_df[plot_df[color_col] == val]
                    fig.add_trace(go.Violin(
                        x=subset[x_col] if x_col in subset.columns else None,
                        y=subset[y_col] if y_col in subset.columns else None,
                        orientation='h', side='positive', width=3, points=False,
                        line_color=colors[i % len(colors)],
                        name=str(val)
                    ))
            fig.update_layout(height=height, width=width, template=selected_style)
        # === Scatter Matrix (X/Y/Z, только числовые; нужно >=2) ===
        elif chart_type == "ScatterMatrix":
            use_dims = []
            for c in [x_col, y_col, z_col]:
                if c and (c in plot_df.columns) and np.issubdtype(plot_df[c].dtype, np.number):
                    use_dims.append(c)
            use_dims = list(dict.fromkeys(use_dims))  # уникальные, в порядке
            if len(use_dims) < 2:
                notif = _make_error_notif("Для Scatter Matrix нужны ≥2 числовых столбца из X/Y/Z.")
                return empty, notif

            fig = px.scatter_matrix(
                plot_df, dimensions=use_dims, color=carg,
                height=height, width=width, template=selected_style
            )
            # фасеты scatter_matrix не поддерживаются напрямую — оставим без facet
            # подписи/стили подхватятся из template

        # === Parallel Coordinates (X/Y/Z, только числовые; нужно >=2). Цвет — по Color (кодируем, если категориальный) ===
        elif chart_type == "Parcoords":
            use_dims = []
            for c in [x_col, y_col, z_col]:
                if c and (c in plot_df.columns) and np.issubdtype(plot_df[c].dtype, np.number):
                    use_dims.append(c)
            use_dims = list(dict.fromkeys(use_dims))
            if len(use_dims) < 2:
                notif = _make_error_notif("Для Parallel Coordinates нужны ≥2 числовых столбца из X/Y/Z.")
                return empty, notif

            # Цвет линии
            line_color = None
            if carg:
                if carg in plot_df.columns:
                    if np.issubdtype(plot_df[carg].dtype, np.number):
                        line_color = plot_df[carg]
                    else:
                        codes, _ = pd.factorize(plot_df[carg].astype(str))
                        line_color = codes

            dims = [dict(label=c, values=plot_df[c].values) for c in use_dims]
            fig = go.Figure(data=go.Parcoords(
                dimensions=dims,
                line=dict(color=line_color) if line_color is not None else None
            ))
            fig.update_layout(height=height, width=width, template=selected_style)

        # === Иерархические: Sunburst / Treemap ===
        # Используем путь из [Color, X, Y] (что задано и существует). Значения — Y, если числовой, иначе count.
        elif chart_type in ("Sunburst", "Treemap"):
            path = [c for c in [color_col, x_col, y_col] if c and (c in plot_df.columns)]
            if not path:
                notif = _make_error_notif("Для Sunburst/Treemap нужен хотя бы один категориальный столбец из Color/X/Y.")
                return empty, notif

            values = None
            if y_col and (y_col in plot_df.columns) and np.issubdtype(plot_df[y_col].dtype, np.number):
                values = y_col  # суммируем по Y
            # Важно: color в px.* конфликтует, если уже в path — тогда не передаём color
            color_kw = {}
            if carg and (carg in plot_df.columns) and (carg not in path):
                color_kw["color"] = carg

            if chart_type == "Treemap":
                fig = px.treemap(
                    plot_df, path=path, values=values,
                    height=height, width=width, template=selected_style, **color_kw
                )
            else:
                fig = px.sunburst(
                    plot_df, path=path, values=values,
                    height=height, width=width, template=selected_style, **color_kw
                )

        # === Плотности 2D: Density Heatmap / Density Contour (по X,Y) ===
        elif chart_type in ("DensityHeat", "DensityContour"):
            if not x_col or not y_col or (x_col not in plot_df.columns) or (y_col not in plot_df.columns):
                notif = _make_error_notif("Для 2D-плотности нужны X и Y.")
                return empty, notif
            if (not np.issubdtype(plot_df[x_col].dtype, np.number)) or (not np.issubdtype(plot_df[y_col].dtype, np.number)):
                notif = _make_error_notif("Для 2D-плотности X и Y должны быть числовыми.")
                return empty, notif

            if chart_type == "DensityHeat":
                fig = px.density_heatmap(
                    plot_df, x=x_col, y=y_col, color_continuous_scale="Viridis",
                    height=height, width=width, template=selected_style,
                    facet_row=facet_row, facet_col=facet_col, category_orders=category_orders
                )
            else:
                fig = px.density_contour(
                    plot_df, x=x_col, y=y_col, color=carg,
                    height=height, width=width, template=selected_style,
                    facet_row=facet_row, facet_col=facet_col, category_orders=category_orders
                )
                fig.update_traces(contours_coloring="fill", contours_showlines=False)
            _hide_top_facet_xlabels(fig)

        # Пользовательские цвета (если есть мапа категорий → цвет)
        if isinstance(custom_colors, dict) and custom_colors:
            try:
                # если ключи — имена категорий, применим через colorway/legendgroup
                # (оставляю твоё упрощённое поведение — по индексам трейсев)
                for i, trace in enumerate(fig.data or []):
                    idx = str(i)
                    if idx in custom_colors:
                        trace.setdefault("marker", {})
                        if isinstance(trace["marker"], dict):
                            trace["marker"]["color"] = custom_colors[idx]
            except Exception as _e:
                logger.warning(f"custom_colors apply skipped: {_e}")

        # Оси и шрифты
        if x_as_text:
            fig.update_xaxes(type='category', categoryorder=dropdown_sort_column)
        else:
            fig.update_xaxes(tickfont_size=xaxis_font_size,
                             dtick=tick_step_x if tick_step_x and tick_step_x > 0 else None)

        def is_categorical_by_name(col):
            if col and col in plot_df.columns:
                return any(keyword.lower() in str(col).lower() for keyword in ['скважина', 'well', 'куст'])
            return False

        if is_categorical_by_name(y_col):
            fig.update_yaxes(type='category', categoryorder=dropdown_sort_column)
        else:
            fig.update_yaxes(tickfont_size=yaxis_font_size,
                             dtick=tick_step_y if tick_step_y and tick_step_y > 0 else None)

        if axes_category == "x" and x_as_text:
            fig.update_xaxes(categoryorder=dropdown_sort_column)
        elif axes_category == "y" and not is_categorical_by_name(y_col):
            fig.update_yaxes(categoryorder=dropdown_sort_column)

        fig.update_layout(
            legend=legend_config.get(legend, {}),
            legend_title_text=None,
            xaxis_title_font=dict(size=font_size_ticks),
            yaxis_title_font=dict(size=font_size_ticks),
            title_font_size=title_font_size,
            template=selected_style
        )

        # Порядок легенды
        try:
            _sort_legend_traces(fig, legend_order, legend_custom_order)
        except Exception as _e:
            logger.warning(f"Сортировка легенды пропущена: {_e}")

        # Автомаржины
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
        if chart_type != "3D_Scatter" and facet_row:
           _hide_top_facet_xlabels(fig)

        return fig, []

    except Exception as e:
        logger.error(f"Ошибка при построении графика: {e}", exc_info=True)
        notif = _make_error_notif(f"Ошибка отрисовки графика: {str(e)}. Попробуйте изменить параметры.")
        return empty, notif


# Новый callback для нижних графиков (независимый)
# Новый callback для нижних графиков (независимый)
@app.callback(
    Output("corr-bar-x", "figure"),
    Output("corr-bar-y", "figure"),
    Output("corr-bars-section", "style"),
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),  # Добавлено

    Input("update-graf", "n_clicks"),
    Input("segmented", "value"),
    State("dropdown_corr_columns", "value"),  # Input
    Input("dropdown_x", "value"),             # Input
    Input("dropdown_y", "value"),             # Input
    Input("cluster-metrics", "data"),

    State("filtered-data", "data"),
    State("dropdown_style", "value"),
    State("meta-columns", "data"),

    prevent_initial_call=True
)
def update_lower_graphs(n_clicks_graf, chart_type, corr_cols, x_col, y_col, cluster_metrics,
                        filtered_json, selected_style, meta):

    empty = _empty_fig()
    SHOW  = {"opacity": 1, "pointerEvents": "auto", "height": "auto", "overflow": "visible", "transition": "opacity 150ms ease"}
    HIDE  = {"opacity": 0, "pointerEvents": "none", "height": "auto", "overflow": "visible"}

    try:
        if not filtered_json:
            return empty, empty, HIDE, []

        dff = read_df_from_store(filtered_json, meta)
        if dff is None or dff.empty:
            return empty, empty, HIDE, []

        def build_corr_bar(corr_matrix: pd.DataFrame, target: str, title_text: str) -> go.Figure:
            if not target or target not in corr_matrix.columns:
                return empty
            s = corr_matrix[target].drop(labels=[target], errors="ignore").sort_values(ascending=False)
            if s.empty:
                return empty
            df_bar = s.reset_index().rename(columns={"index": "Параметр", target: "Корреляция"})
            # авто-высота
            n = len(df_bar); per_row = 26; padding = 140
            dyn_height = max(220, min(1400, per_row * max(1, n) + padding))
            bar = px.bar(
                                df_bar, x="Корреляция", y="Параметр", orientation="h",
                                title=title_text, template=selected_style, text="Корреляция",
                                height=dyn_height
                            )
            xmin = float(s.min()); xmax = float(s.max())
            if xmin == xmax:
                pad = max(0.1, abs(xmin) * 0.1)
                xmin -= pad; xmax += pad
            span = xmax - xmin
            pad = max(0.05 * span, 0.02)
            bar.update_xaxes(
                range=[xmin - pad, xmax + pad],
                showticklabels=True, ticks="outside",
                tickformat=".1f", automargin=True,
                zeroline=(xmin - pad < 0 < xmax + pad), zerolinewidth=1
            )
            bar.update_layout(yaxis_title=None)
            bar.update_yaxes(automargin=True)
            bar.update_traces(texttemplate="%{x:.2f}", textposition="auto", cliponaxis=False)
            return bar

        if chart_type == "Correlation":
            numeric_cols_all = (meta.get("numeric") or [])

            # 1) Формируем кандидатов для матрицы с приоритетом X, Y:
            #    - если corr_cols заданы → corr_cols ∪ {X, Y}
            #    - если corr_cols пусты → {X, Y} ∪ fallback (все числовые)
            if corr_cols and len(corr_cols) > 0:
                cand = list(dict.fromkeys(list(corr_cols) + ([x_col] if x_col else []) + ([y_col] if y_col else [])))
            else:
                cand = list(dict.fromkeys(([x_col] if x_col else []) + ([y_col] if y_col else []) + list(numeric_cols_all)))

            # оставляем только существующие в датафрейме
            cand = [c for c in cand if c and c in dff.columns]

            # и только числовые (по метаданным)
            use_cols = [c for c in cand if c in numeric_cols_all]

            # Если после фильтрации < 2 — добираем из всех числовых
            if len(use_cols) < 2:
                for c in numeric_cols_all:
                    if c in dff.columns and c not in use_cols:
                        use_cols.append(c)
                        if len(use_cols) >= 2:
                            break

            # Ограничение по ширине матрицы
            MAX_CORR_COLS = 50
            use_cols = use_cols[:MAX_CORR_COLS]

            if len(use_cols) >= 2:
                corr_df = dff[use_cols].select_dtypes(include=[np.number]).dropna(how="all")
                if not corr_df.empty and corr_df.shape[1] >= 2:
                    corr_matrix = corr_df.corr().round(2)

                    # 2) Строго целимся в X и Y. Если X/Y нет в матрице — берём фолбэк из use_cols в порядке приоритета.
                    targets_order = []
                    for cand_t in [x_col, y_col] + use_cols:
                        if cand_t and cand_t not in targets_order and cand_t in corr_matrix.columns:
                            targets_order.append(cand_t)

                    # два разных таргета для двух баров, если возможно
                    t0 = targets_order[0] if len(targets_order) >= 1 else None
                    t1 = targets_order[1] if len(targets_order) >= 2 else None

                    corr_bar_x_fig = build_corr_bar(corr_matrix, t0, f"Корреляции с {t0}" if t0 else "")
                    corr_bar_y_fig = build_corr_bar(corr_matrix, t1, f"Корреляции с {t1}" if t1 else "")

                    has_any = (len(corr_bar_x_fig.data or []) > 0) or (len(corr_bar_y_fig.data or []) > 0)
                    return corr_bar_x_fig if has_any else empty, corr_bar_y_fig if has_any else empty, (SHOW if has_any else HIDE), []

            # не из чего считать
            return empty, empty, HIDE, []

        # ---- НЕ "Correlation": показываем локоть/силуэт (если метрики есть) ----
        aux1, aux2 = empty, empty
        if isinstance(cluster_metrics, dict):
            try:
                ks = (cluster_metrics.get("ks") or cluster_metrics.get("K") or [])[:]
                inertias = cluster_metrics.get("inertias") or []
                sils = cluster_metrics.get("silhouettes") or []
                if ks and inertias:
                    df_in = pd.DataFrame({"K": ks, "Inertia": inertias})
                    aux1 = px.line(df_in, x="K", y="Inertia", template=selected_style, title="Метод локтя")
                    aux1.update_layout(height=400, margin=dict(l=50, r=20, t=40, b=40))
                if ks and sils:
                    df_s = pd.DataFrame({"K": ks, "Silhouette": sils})
                    aux2 = px.line(df_s, x="K", y="Silhouette", template=selected_style, title="Силуэтный метод")
                    aux2.update_layout(height=400, margin=dict(l=60, r=20, t=40, b=40))
            except Exception as e:
                logger.warning(f"Не удалось построить локоть/силуэт: {e}")

        has_any = (len(aux1.data or []) > 0) or (len(aux2.data or []) > 0)
        return aux1 if has_any else empty, aux2 if has_any else empty, (SHOW if has_any else HIDE), []

    except Exception as e:
        logger.error(f"Ошибка при построении нижних графиков: {e}", exc_info=True)
        notif = _make_error_notif(f"Ошибка в нижних графиках: {str(e)}")
        return empty, empty, HIDE, notif

# Диалог выбора цвета
@app.callback(
    Output("color-modal", "opened", allow_duplicate=True),
    Output("color-inputs", "children"),
    Input("shuffle-button", "n_clicks"),
    Input("color-mode-toggle", "checked"),
    State("graph", "figure"),
    State("dropdown_style", "value"),
    State("custom-colors", "data"),
    prevent_initial_call=True
)
def open_color_dialog(n_clicks, manual_mode, fig_dict, selected_style, custom_colors):
    if not fig_dict or "data" not in fig_dict:
        raise no_update

    traces = fig_dict["data"]
    use_dropdown = not manual_mode and len(traces) <= COLOR_THRESHOLD

    if selected_style == "seaborn_custom":
        style_colors = pio.templates[selected_style].layout.colorway
    else:
        style_colors = getattr(px.colors.qualitative, selected_style, px.colors.qualitative.Plotly)

    color_inputs = []
    for i, trace in enumerate(traces):
        index = str(i)
        name = trace.get("name", f"Категория {i+1}")
        current_color = (custom_colors or {}).get(index, trace.get("marker", {}).get("color", style_colors[i % len(style_colors)]))
        preview_id = {"type": "color-preview", "index": index}

        if use_dropdown:
            input_control = dmc.Group([
                dcc.Dropdown(
                    id={"type": "color-picker", "index": index},
                    value=current_color if current_color in style_colors else style_colors[i % len(style_colors)],
                    options=[{"label": f"Цвет {j+1} ({style_colors[j]})", "value": style_colors[j]} for j in range(len(style_colors))],
                    clearable=False,
                    style={"width": 300}
                ),
                html.Div(id=preview_id, style={"backgroundColor": current_color, "width": "20px", "height": "20px", "border": "1px solid #ccc", "marginLeft": "5px"})
            ])
        else:
            input_control = dmc.ColorInput(id={"type": "color-picker", "index": index}, value=current_color, format="hex")

        color_inputs.append(dmc.Group([dmc.Text(name, style={"width": 150}), input_control]))
    return True, color_inputs

# Исправленный apply_custom_colors (безопасный, с try-except)

@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("color-modal", "opened", allow_duplicate=True),
    Output("custom-colors", "data"),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),  # Добавлен для уведомлений
    Input("apply-colors", "n_clicks"),
    State("graph", "figure"),
    State({"type": "color-picker", "index": ALL}, "id"),
    State({"type": "color-picker", "index": ALL}, "value"),
    State("custom-colors", "data"),
    prevent_initial_call=True
)
def apply_custom_colors(n_clicks, fig_dict, ids, values, custom_colors):
    if not fig_dict or "data" not in fig_dict:
        raise no_update

    try:
        new_color_map = {item["index"]: val for item, val in zip(ids, values)}
        updated = (custom_colors or {}).copy()
        updated.update(new_color_map)
        
        # Конвертируем fig_dict обратно в go.Figure для безопасного применения
        fig = go.Figure(fig_dict)
        fig = apply_custom_colors_safely(fig, updated)

        return fig, False, updated, []  # Успех: очищаем уведомления
    except Exception as e:
        logger.error(f"Ошибка при применении цветов: {e}", exc_info=True)
        notif = _make_error_notif("Не удалось применить цвета. График остается без изменений.")
        return fig_dict, False, custom_colors, notif  # Возвращаем исходную fig и уведомление


@app.callback(Output({"type": "color-preview", "index": MATCH}, "style"),
              Input({"type": "color-picker", "index": MATCH}, "value"),
              prevent_initial_call=True)
def update_preview_color(selected_color):
    return {"backgroundColor": selected_color, "width": "20px", "height": "20px", "border": "1px solid #ccc", "marginLeft": "5px"}



app.clientside_callback(
    """
    function(n_clicks, figure) {
        if (!n_clicks || !figure) {
            throw window.dash_clientside.PreventUpdate;
        }

        // dataURL -> Blob без fetch (чтобы не терять user-gesture)
        function dataURLtoBlob(dataURL) {
            const [header, data] = dataURL.split(',');
            const mime = (header.match(/:(.*?);/) || [,'image/png'])[1];
            const binary = atob(data);
            const array = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) array[i] = binary.charCodeAt(i);
            return new Blob([array], { type: mime });
        }

        try {
            const host = document.getElementById('graph');
            const gd = host && host.getElementsByClassName('js-plotly-plot')[0];
            if (!gd) return 'График не найден в DOM.';

            // === WYSIWYG размеры: берём реальные видимые пиксели SVG ===
            const svg = gd.querySelector('svg.main-svg') || gd.querySelector('svg');
            let width, height;
            if (svg) {
                const r = svg.getBoundingClientRect(); // учитывает любые CSS-скейлы/zoom
                width  = Math.max(1, Math.round(r.width));
                height = Math.max(1, Math.round(r.height));
            } else {
                // фоллбэк, если svg не найден
                const r = gd.getBoundingClientRect();
                width  = Math.max(1, Math.round(r.width));
                height = Math.max(1, Math.round(r.height));
            }

            // Рисуем PNG ровно под видимые размеры (scale:1 для точного соответствия)
            return window.Plotly.toImage(gd, {
                format: 'png',
                width:  width,
                height: height,
                scale:  1
            })
            .then((dataUrl) => {
                // Пытаемся записать в буфер в рамках того же клика
                if (window.isSecureContext && window.ClipboardItem && navigator.clipboard && navigator.clipboard.write) {
                    try {
                        const blob = dataURLtoBlob(dataUrl);
                        const item = new ClipboardItem({ [blob.type]: blob });
                        return navigator.clipboard.write([item]).then(
                            () => {
                                // Параллельно сохраняем файл
                                const a = document.createElement('a');
                                a.href = dataUrl;
                                a.download = 'plotly_graph.png';
                                document.body.appendChild(a); a.click(); a.remove();
                                return 'PNG (как на экране) скопирован в буфер и сохранён как файл.';
                            },
                            (e) => {
                                console.warn('Clipboard write failed:', e);
                                const a = document.createElement('a');
                                a.href = dataUrl;
                                a.download = 'plotly_graph.png';
                                document.body.appendChild(a); a.click(); a.remove();
                                return 'Буфер недоступен — PNG (как на экране) сохранён как файл.';
                            }
                        );
                    } catch (e) {
                        console.warn('Clipboard exception:', e);
                    }
                }

                // Фоллбэк: только файл (например, не secure-контекст)
                const a = document.createElement('a');
                a.href = dataUrl;
                a.download = 'plotly_graph.png';
                document.body.appendChild(a); a.click(); a.remove();
                return 'Копирование в буфер недоступно — PNG (как на экране) сохранён как файл.';
            })
            .catch((err) => {
                console.error('toImage error:', err);
                return 'Ошибка генерации PNG. См. консоль.';
            });

        } catch (err) {
            console.error('Ошибка:', err);
            return 'Ошибка копирования/сохранения. См. консоль.';
        }
    }
    """,
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("copy-png-button", "n_clicks"),
    State("graph", "figure"),
    prevent_initial_call=True
)


# =========================
# Запуск (без сертификатов, буфер работает на localhost)
# =========================
if __name__ == "__main__":
    import threading, time, socket, webbrowser, contextlib, sys, os

    def get_lan_ip() -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return socket.gethostbyname(socket.gethostname())

    def pick_free_port(start_port: int) -> int:
        port = start_port
        while True:
            with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("", port))
                    return port
                except OSError:
                    port += 1

    # Базовый порт
    PORT = 8090
    PORT = pick_free_port(PORT)
    LAN_IP = get_lan_ip()

    # Для EXE (PyInstaller) – открываем localhost, чтобы Clipboard API работал
    # Для dev также можно принудительно через переменную окружения
    IS_FROZEN = getattr(sys, "frozen", False)
    FORCE_LOCALHOST = os.environ.get("OPEN_LOCALHOST", "1") == "1"  # по умолчанию да

    def wait_and_open():
        # Открываем локально (secure для Clipboard API в Chrome)
        url_local = f"http://127.0.0.1:{PORT}"
        # Ждём сервер
        for _ in range(50):
            try:
                with socket.create_connection(("127.0.0.1", PORT), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.1)

        # По умолчанию открываем localhost (и в EXE, и в dev)
        if FORCE_LOCALHOST or IS_FROZEN:
            webbrowser.open(url_local)
        else:
            # На всякий случай, но лучше не использовать для клипа
            webbrowser.open(url_local)

        # Подсказка в консоли: LAN-URL для коллег (у них буфер не сработает — это норма)
        print("\n================ DASH APP =================")
        print(f" Локально (буфер ОК): http://127.0.0.1:{PORT}")
        print(f" В сети LAN (без буфера): http://{LAN_IP}:{PORT}")
        print("==========================================\n")

    threading.Thread(target=wait_and_open, daemon=True).start()

    # Слушаем все интерфейсы — и себе, и коллегам
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False, threaded=True)