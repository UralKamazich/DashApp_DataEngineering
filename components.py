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
# Цвета плашек для типов колонок
# =========================
COLUMN_TYPE_COLORS = {
    "numeric": "#2196F3",      # синий
    "categorical": "#4CAF50",  # зелёный
    "datetime": "#795548",     # коричневый
}


def make_column_badge(col_name: str, col_type: str) -> html.Div:
    """Плашка-бейдж: название колонки, цвет по типу (c drag-and-drop)."""
    color = COLUMN_TYPE_COLORS.get(col_type, "#9E9E9E")
    return html.Div(
        dmc.Tooltip(
            dmc.Badge(
                col_name,
                color=color,
                variant="light",
                size="sm",
                fullWidth=True,
                className="column-badge-pill",
                style={"marginBottom": "3px", "textAlign": "left", "fontWeight": 400},
            ),
            label=str(col_name),
            openDelay=200,
            withArrow=True,
        ),
        className="column-badge",
        draggable="true",
        **{"data-column-name": str(col_name)},
        style={"cursor": "grab", "marginBottom": "2px"},
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

# =========================
# Кнопки и элементы управления
# =========================
add_filter_button = dmc.Button("Добавить фильтр", id="add-filter-btn", size="xs", variant="outline", leftSection=html.Div("+"))

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
        {"value": "Correlation", "label": "Коррелограмма"},
        {"value": "Violin", "label": "Violin"},
        {"value": "Ridge", "label": "Ridge Plot"},
        {"value": "ScatterMatrix",   "label": "Scatter Matrix"},
        {"value": "Parcoords",       "label": "Parallel Coordinates"},
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
    size="sm",
    radius="sm",
    label="Пузырьковый режим",
    checked=False,
)

update_graf = dmc.Tooltip(
    label="Обновить / Отобразить график",
    withArrow=True,
    children=dmc.ActionIcon(
        id="update-graf",                 # тот же ID — логика не меняется
        variant="light",
        size="xl",
        radius="xl",
        children=DashIconify(icon="lucide:refresh-ccw", width=18)
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
    )
)

copy_trigger = dcc.Clipboard(id="clipboard", style={"display": "none"})

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
