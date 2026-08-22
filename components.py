# -*- coding: utf-8 -*-
"""
UI-компоненты: dropdowns, selects, buttons, switches, segmented controls.
"""

import dash
from dash import dcc, html
from dash_iconify import DashIconify
import dash_mantine_components as dmc

from config import PLOTLY_STYLES, COLOR_THRESHOLD


# =========================
# Маркеры типов каналов
# =========================
COLUMN_TYPE_LABELS = {
    "numeric": ("123", "Числовой"),
    "categorical": ("Aa", "Категориальный"),
    "datetime": ("dt", "Дата и время"),
}


def make_column_badge(
    col_name: str,
    col_type: str,
    *,
    derived: bool = False,
    dataset_id: str | None = None,
) -> html.Div:
    """Compact draggable dataset channel with a restrained type marker."""
    type_mark, type_name = COLUMN_TYPE_LABELS.get(col_type, ("…", "Другой тип"))
    return html.Div(
        dmc.Tooltip(
            html.Div(
                [
                    html.Sup(
                        type_mark,
                        className=f"column-type-marker column-type-marker--{col_type}",
                        **{"aria-hidden": "true"},
                    ),
                    html.Sup(
                        "fx",
                        className="column-derived-marker",
                        title="Создан в Data Engineering",
                    ) if derived else None,
                    html.Span(str(col_name), className="column-channel-name"),
                    html.Span(
                        className="column-drag-handle",
                        **{"aria-hidden": "true"},
                    ),
                ],
                className="column-channel-row",
            ),
            label=f"{col_name} · {type_name}" + (" · производный канал" if derived else ""),
            openDelay=350,
            withArrow=True,
        ),
        className=f"column-badge column-badge--{col_type}",
        draggable="true",
        title=f"Перетащить канал «{col_name}»",
        **{
            "data-column-name": str(col_name),
            "data-column-type": col_type,
            "data-dataset-id": str(dataset_id or ""),
            "data-derived": "true" if derived else "false",
        },
    )


# =========================
# Вспомогательные функции создания компонентов
# =========================
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


# =========================
# Стиль графика
# =========================
graph_dataset_select = dmc.Select(
    id="graph-dataset-select",
    label="Датасет",
    data=[],
    value="source",
    allowDeselect=False,
    searchable=True,
    size="xs",
    comboboxProps={"shadow": "md"},
)

dropdown_style = dmc.Select(
    id="dropdown_style",
    label="Тема",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    data=[{"label": style.capitalize(), "value": style} for style in PLOTLY_STYLES],
    value="plotly",
    clearable=False,
    persistence=True
)

graph_render_mode = dmc.SegmentedControl(
    id="graph-render-mode",
    data=[
        {"label": "Гибрид", "value": "hybrid"},
        {"label": "SVG", "value": "svg"},
    ],
    value="hybrid",
    size="xs",
    fullWidth=True,
    persistence=True,
    persistence_type="local",
)

# =========================
# Основные дропдауны осей
# =========================
dropdown_x = create_dropdown("dropdown_x", [], None, clearable=True)
dropdown_y = create_dropdown("dropdown_y", [], None, clearable=True)
dropdown_z = create_dropdown("dropdown_z", [], None, clearable=True)
dropdown_color = create_dropdown("dropdown_color", [], None, clearable=True)
dropdown_size = create_dropdown("dropdown_size", [], None, clearable=True)
dropdown_text = create_dropdown("dropdown_text", [], None, clearable=True)
dropdown_facet_row = create_dropdown("dropdown_facet_row", [], None, clearable=True)
dropdown_facet_col = create_dropdown("dropdown_facet_col", [], None, clearable=True)

dropdown_hover_data = create_multiselect("dropdown_hover_data", [], value=[], clearable=True)
dropdown_corr_columns = create_multiselect("dropdown_corr_columns", [], value=[], clearable=True)
dropdown_corr_columns.className = "correlation-channels-select"
dropdown_corr_columns.style = {
    **dropdown_corr_columns.style,
    "width": "100%",
    "maxWidth": "100%",
}

mv_chart_type = dmc.Select(
    id="mv-chart-type",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    data=[
        {"value": "Correlogram", "label": "Коррелограмма"},
        {"value": "ScatterMatrix", "label": "Scatter Matrix"},
        {"value": "Parcoords", "label": "Parallel Coordinates"},
    ],
    value="Correlogram",
    clearable=False,
    persistence=True,
    style={"width": "200px", "fontSize": "13px"},
)

# =========================
# Cluster / Data Engineering колонки
# =========================
dropdown_cluster_cols = dmc.MultiSelect(
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
    label="Нули не учитывать",
    checked=False,
    onLabel="Да",
    offLabel="Нет",
    size="md"
)

agg_exclude_empty_switch = dmc.Switch(
    id="agg-exclude-empty",
    label="NaN не учитывать",
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

dropdown_chart_type = dmc.Select(
    id="segmented",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    data=[
        {"value": "Scatter", "label": "Scatter"},
        {"value": "3D_Scatter", "label": "3D Scatter"},
        {"value": "Bar", "label": "Bar"},
        {"value": "Box", "label": "Box"},
        {"value": "Line", "label": "Line"},
        {"value": "Polar", "label": "Треугольный график"},
        {"value": "Hist", "label": "Гистограмма"},
        {"value": "Pie", "label": "Круговая диаграмма"},
        {"value": "Violin", "label": "Violin"},
        {"value": "Ridge", "label": "Ridge Plot"},
        {"value": "Sunburst",        "label": "Sunburst"},
        {"value": "Treemap",         "label": "Treemap"},
        {"value": "DensityHeat",     "label": "Density Heatmap"},
        {"value": "DensityContour",  "label": "Density Contour"},
    ],
    value="Scatter",
    clearable=False,
    persistence=True,
    style={"width": "165px", "fontSize": "13px"}
)

SwitchBubble = dmc.Switch(
    id="SwitchBubble",
    size="xs",
    radius="sm",
    label="Bubbles",
    checked=True,
)

# =========================
# Вспомогательные дропдауны настроек
# =========================
dropdown_text_pozition = dmc.Select(
    id="dropdown_text_pozition",
    label="Положение подписей",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    value="middle center",
    data=['top left', 'top center', 'top right', 'middle left',
          'middle center', 'middle right', 'bottom left', 'bottom center', 'bottom right'],
    clearable=False, persistence=True
)
dropdown_category_ascending = dmc.Select(
    id="dropdown_category_ascending",
    label="Порядок категорий",
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
    label="Сортировать по оси",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    value="auto",               # было "y"
    data=["auto", "x", "y"],    # добавили auto
    clearable=False, persistence=True
)
dropdown_overlay = dmc.Select(
    id="dropdown_overlay",
    label="Режим столбцов",
    value="overlay",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    data=["overlay", "stack", "relative", "group"],
    clearable=False, persistence=True
)
dropdown_pie_aggregation = dmc.Select(
    id="dropdown_pie_aggregation",
    label="Круговая: агрегация",
    description="Для пары категория + число",
    value="sum",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    data=[
        {"value": "sum", "label": "Сумма"},
        {"value": "mean", "label": "Среднее"},
        {"value": "count", "label": "Количество"},
    ],
    clearable=False,
    persistence=True,
)
dropdown_legend = dmc.Select(
    id="dropdown_legend",
    label="Положение",
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
    label="Порядок",
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
    label="Свой порядок (через запятую)",
    placeholder="Пример: A, C, B",
    persistence=True,
)

# === Новый блок: ГРУППИРОВКА ЧИСЛОВОГО СТОЛБЦА ===
bin_column_select = create_dropdown("bin-column", [], None, clearable=True, persistence=False)
bin_method = dmc.SegmentedControl(
    id="bin-method",
    data=[{"label": "Равные интервалы", "value": "width"},
          {"label": "Равное число значений", "value": "count"}],
    value="count", size="xs", fullWidth=True
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

# Переключатель подписей на графиках Bar
bar_text_auto_switch = dmc.Switch(
    id="bar-text-auto",
    label="Значения на столбцах",
    checked=True,
    onLabel="Вкл",
    offLabel="Выкл",
    size="md"
)

bar_aggregation_select = dmc.Select(
    id="bar-aggregation",
    label="Bar: агрегация",
    value="sum",
    allowDeselect=False,
    comboboxProps={"shadow": "md"},
    data=[
        {"value": "none", "label": "Без агрегации (как есть)"},
        {"value": "sum", "label": "Сумма"},
        {"value": "mean", "label": "Среднее"},
        {"value": "count", "label": "Количество строк"},
    ],
    clearable=False,
    persistence=True,
)
