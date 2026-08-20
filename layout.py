# -*- coding: utf-8 -*-
"""
Layout приложения: MantineProvider, Store, Modal, Grid, Paper, Filters, Graph.
Многостраничное: График | Коррелограмма | Data Engineering | Кластеризация | ML.
Header + Footer общие на всех страницах.
"""

import dash
from dash import dcc, html
import dash_mantine_components as dmc

from config import APP_NAME, STYLE_CARD
from correlation_workspace import CorrelationWorkspace
from dataset_panel import create_dataset_drawer
from filter_panel import create_filter_drawer
from graph_settings import GraphSettingsPanel
from graph_workspace import GraphWorkspace, LEGACY_GRAPH_ACTION_IDS
from components import (
    dropdown_style,
    dropdown_x, dropdown_y, dropdown_z,
    dropdown_color, dropdown_size, dropdown_text,
    dropdown_hover_data, dropdown_corr_columns,
    dropdown_facet_row, dropdown_facet_col,
    dropdown_cluster_cols,
    agg_keys_select, agg_cols_select, agg_metrics_select,
    agg_exclude_zeros_switch, agg_exclude_empty_switch,
    txtcopy_cols_select, txtcopy_suffix_input, txtcopy_strip_switch,
    dropdown_chart_type,
    mv_chart_type, mv_dropdown_x, mv_dropdown_y, mv_dropdown_z, mv_dropdown_color,
    SwitchBubble,
    dropdown_text_pozition, dropdown_category_ascending,
    dropdown_axes_category, dropdown_overlay, dropdown_pie_aggregation,
    dropdown_legend, dropdown_legend_order, input_legend_custom_order,
    bin_column_select, bin_method, bin_k, bin_label_style,
    bar_text_auto_switch,
    bar_aggregation_select,
)

# Навигационные ссылки
NAV_LINKS = [
    {"label": "График", "href": "/"},
    {"label": "Корреляционный анализ", "href": "/correlation"},
    {"label": "Data Engineering", "href": "/data-engineering"},
    {"label": "Кластеризация", "href": "/clustering"},
    {"label": "ML", "href": "/ml"},
]


def make_nav_link(label, href):
    """Стилизованная ссылка навигации — dcc.Link, чтобы не перезагружать страницу."""
    return dcc.Link(
        label,
        href=href,
        className="nav-link",
        style={
            "color": "#aaa",
            "textDecoration": "none",
            "fontSize": "13px",
            "fontWeight": 500,
            "padding": "6px 14px",
            "borderRadius": "6px",
            "transition": "all 0.15s",
            "display": "inline-block",
        },
    )


GRAPH_SETTINGS_PANEL = GraphSettingsPanel(
    controls={
        "theme": dropdown_style,
        "bubble": SwitchBubble,
        "bar_labels": bar_text_auto_switch,
        "text_position": dropdown_text_pozition,
        "category_axis": dropdown_axes_category,
        "category_order": dropdown_category_ascending,
        "bar_mode": dropdown_overlay,
        "bar_aggregation": bar_aggregation_select,
        "pie_aggregation": dropdown_pie_aggregation,
        "legend_position": dropdown_legend,
        "legend_order": dropdown_legend_order,
        "legend_custom_order": input_legend_custom_order,
    }
)

GRAPH_WORKSPACE = GraphWorkspace(
    graph_id="graph",
    chart_type_control=dropdown_chart_type,
    field_controls={
        "dropdown_x": dropdown_x,
        "dropdown_y": dropdown_y,
        "dropdown_z": dropdown_z,
        "dropdown_color": dropdown_color,
        "dropdown_size": dropdown_size,
        "dropdown_text": dropdown_text,
        "dropdown_facet_row": dropdown_facet_row,
        "dropdown_facet_col": dropdown_facet_col,
        "dropdown_hover_data": dropdown_hover_data,
    },
    settings_panel=GRAPH_SETTINGS_PANEL,
    action_ids=LEGACY_GRAPH_ACTION_IDS,
)

CORRELATION_WORKSPACE = CorrelationWorkspace(dropdown_corr_columns)

# Многомерные графики (Scatter Matrix / Parallel Coordinates) живут на
# странице корреляционного анализа: им нужны только X/Y/Z и Цвет.
MULTIVARIATE_FIELDS = (
    {"key": "x", "label": "X", "target": "dropdown_x", "zone": "axis-x"},
    {"key": "y", "label": "Y", "target": "dropdown_y", "zone": "axis-y"},
    {"key": "z", "label": "Z", "target": "dropdown_z", "zone": "secondary"},
    {"key": "color", "label": "Цвет", "target": "dropdown_color", "zone": "secondary"},
)

MULTIVARIATE_WORKSPACE = GraphWorkspace(
    graph_id="mv-graph",
    chart_type_control=mv_chart_type,
    field_controls={
        "dropdown_x": mv_dropdown_x,
        "dropdown_y": mv_dropdown_y,
        "dropdown_z": mv_dropdown_z,
        "dropdown_color": mv_dropdown_color,
    },
    fields=MULTIVARIATE_FIELDS,
    include_color_controls=False,
)


def create_layout():
    graph_workspace = GRAPH_WORKSPACE.render()
    multivariate_workspace = MULTIVARIATE_WORKSPACE.render()
    # Мультиграфик занимает место бывшей корреляционной матрицы:
    # тип «Коррелограмма» строит её по тем же «коррелируемым каналам».
    correlation_workspace = CORRELATION_WORKSPACE.render(
        matrix_block=multivariate_workspace
    )
    filter_drawer = create_filter_drawer()
    dataset_drawer = create_dataset_drawer()

    return dmc.MantineProvider(
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "height": "100vh",
                    "overflow": "hidden",
                },
                children=[

            # --- URL Location ---
            dcc.Location(id="url", refresh=False),

            # --- Stores ---
            dcc.Store(id='stored-data', data=False, storage_type="memory"),
            dcc.Store(id='filtered-data'),
            dcc.Store(id='bin-applied-name'),
            dcc.Store(id='filter-count', data=0),
            dcc.Store(id='filters-state', data={}),
            dcc.Store(id='filters-applied-state', data={}),
            dcc.Store(id='stored-sheet-names'),
            dcc.Store(id='selected-sheet'),
            dcc.Store(id='sheet-modal-toggle', data=True),
            dcc.Store(id='meta-columns'),
            dcc.Store(id='cluster-metrics'),
            dcc.Store(id='source-file-path'),
            dcc.Store(id='source-file-name'),
            dcc.Store(id='right-panels-coordination'),

            dmc.NotificationContainer(id="notifications-container"),

            # ========================
            # HEADER (фиксированный сверху)
            # ========================
            dmc.Paper(
                children=[
                    dmc.Group([
                        dmc.Text(APP_NAME, fw=700, size="lg", c="white", style={"letterSpacing": "0.5px"}),
                        dmc.Divider(orientation="vertical", style={"borderColor": "rgba(255,255,255,0.2)"}),
                        dmc.Button(
                            "Выбрать файл (.xlsx, .pkl)",
                            id="pick-file-btn",
                            size="xs",
                            variant="outline",
                            color="gray",
                            style={"borderColor": "rgba(255,255,255,0.3)", "color": "#ccc"}
                        ),
                        dmc.Divider(orientation="vertical", style={"borderColor": "rgba(255,255,255,0.2)"}),
                        *[make_nav_link(item["label"], item["href"]) for item in NAV_LINKS],
                    ], gap="sm", align="center", wrap="nowrap", px="md", py=6),
                ],
                shadow="sm",
                p=0,
                style={
                    "backgroundColor": "#1A1B1E",
                    "borderBottom": "1px solid #2C2E33",
                    "flexShrink": "0",
                    "zIndex": "100",
                },
                withBorder=False,
            ),

            # --- Sheet modal ---
            html.Div(
                dmc.Modal(id="sheet-modal", title="Выберите лист Excel", opened=False, centered=True, zIndex=1000, children=[]),
                id="sheet-menu-wrapper"
            ),
            html.Div(id='status-message', style={'display': 'none'}),

            # --- de-modal (скрыт) ---
            html.Div(style={"display": "none"}, children=[dmc.Modal(id="de-modal", opened=False, children=[])]),

            # ========================
            # CONTENT: скроллируемая область между header и footer
            # ========================
            html.Div(
                style={
                    "flex": "1",
                    "display": "flex",
                    "overflow": "hidden",
                    "padding": "8px",
                    "gap": "8px",
                },
                children=[
                # ---------- ВЫКАТНАЯ ПАНЕЛЬ ДАТАСЕТА (слева) ----------
                dataset_drawer,

                # ---------- ПРАВАЯ ПАНЕЛЬ — график + страницы ----------------
                html.Div(
                    style={"flex": "1", "overflow": "auto", "minWidth": "0"},
                    children=[
                    html.Div(id="page-graph", children=[
                        graph_workspace,
                    ]),

                    html.Div(id="page-correlation", style={"display": "none"}, children=[
                        correlation_workspace,
                    ]),

                    html.Div(id="page-data-engineering", style={"display": "none"}, children=[
                        dmc.Paper([
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
                        ], style=STYLE_CARD, shadow="md", p="md", withBorder=True),

                        dmc.Paper([
                            dmc.Divider(label="Текстовые копии"),
                            dmc.Space(h=8),
                            dmc.Grid([
                                dmc.GridCol([dmc.Text("Столбец(ы)", c="blue", fw=500, size="sm"), txtcopy_cols_select], span=8, style={"minWidth": 0}),
                                dmc.GridCol([txtcopy_suffix_input], span=4, style={"minWidth": 0}),
                            ]),
                            dmc.Space(h=8),
                            dmc.Group([txtcopy_strip_switch], gap="xl"),
                            dmc.Space(h=10),
                            dmc.Group([dmc.Button("Создать текстовую копию", id="btn-txtcopy", size="sm", variant="light")], justify="flex-end"),
                            dmc.Space(h=6),
                            dmc.Text(id="de-txt-status", size="sm", c="dimmed"),
                        ], style=STYLE_CARD, shadow="md", p="md", withBorder=True),

                        dmc.Paper([
                            dmc.Divider(label="Расчёт агрегатов по группам"),
                            dmc.Text("Ключи, столбцы и метрики — новые колонки в конец датасета.", size="sm"),
                            dmc.Space(h=10),
                            dmc.Grid([
                                dmc.GridCol([dmc.Text("Ключ(и)", c="blue", fw=500, size="sm"), agg_keys_select], span=6, style={"minWidth": 0}),
                                dmc.GridCol([dmc.Text("Столбцы", c="blue", fw=500, size="sm"), agg_cols_select], span=6, style={"minWidth": 0}),
                            ]),
                            dmc.Space(h=8),
                            agg_metrics_select,
                            dmc.Space(h=10),
                            dmc.Group([agg_exclude_zeros_switch, agg_exclude_empty_switch], gap="xl"),
                            dmc.Space(h=10),
                            dmc.Group([dmc.Button("Рассчитать", id="btn-agg", size="sm")], justify="flex-end"),
                            dmc.Space(h=6),
                            dmc.Text(id="de-agg-status", size="sm", c="dimmed"),
                        ], style=STYLE_CARD, shadow="md", p="md", withBorder=True),
                    ]),

                    html.Div(id="page-clustering", style={"display": "none"}, children=[
                        dmc.Paper([
                            dmc.Divider(label="Кластеризация (KMeans)"),
                            dmc.Text("Кластеризация числовых столбцов с визуализацией PCA.", size="sm"),
                            dmc.Space(h=8),
                            dmc.Grid([
                                dmc.GridCol([html.Center(dmc.Text("Числовые столбцы", c="blue", fw=500, size="sm")), dropdown_cluster_cols], span=8, style={"minWidth": 0}),
                                dmc.GridCol([html.Center(dmc.Text("К", c="blue", fw=500, size="sm")),
                                              dmc.NumberInput(id="cluster-k", value=4, min=2, max=20, step=1, debounce=True)], span=2, style={"minWidth": 0}),
                                dmc.GridCol([dmc.Button("Кластеризация", id="btn-cluster", size="xs")], span=2, style={"minWidth": 0, "marginTop": 23}),
                            ]),
                        ], style=STYLE_CARD, shadow="md", p="md", withBorder=True),
                        html.Div(id="cluster-metrics-section", children=[
                            dmc.Paper([
                                dmc.Grid([
                                    dmc.GridCol([dcc.Graph(id="cluster-elbow-graph", config={'displaylogo': False, 'responsive': True})], span=6),
                                    dmc.GridCol([dcc.Graph(id="cluster-silhouette-graph", config={'displaylogo': False, 'responsive': True})], span=6),
                                ])
                            ], style={**STYLE_CARD, "overflow": "visible", "marginTop": "8px"}, shadow="md", p="md", withBorder=True),
                        ], style={"display": "none"}),
                    ]),

                    html.Div(id="page-ml", style={"display": "none"}, children=[
                        dmc.Paper([
                            dmc.Text("Machine Learning", fw=600, size="lg"),
                            dmc.Space(h=8),
                            dmc.Text("Раздел машинного обучения — в разработке.", size="md", c="dimmed"),
                            dmc.Space(h=16),
                            dmc.Text("Здесь будут: регрессия, классификация, подбор параметров, предсказания.", size="sm", c="dimmed"),
                        ], style=STYLE_CARD, shadow="md", p="md", withBorder=True),
                    ]),
                ]),  # конец правой панели

                # ---------- ВЫКАТНАЯ ПАНЕЛЬ ФИЛЬТРОВ (справа) ----------
                filter_drawer,
                ]  # конец flex-контейнера
            ),

            # ========================
            # FOOTER (фиксированный внизу)
            # ========================
            dmc.Paper(
                id="file-info-bar",
                children=["Файл не выбран"],
                shadow="sm",
                p="md",
                withBorder=False,
                style={
                    "backgroundColor": "#1A1B1E",
                    "borderTop": "1px solid #2C2E33",
                    "color": "#888",
                    "fontSize": "12px",
                    "fontFamily": "monospace",
                    "textAlign": "right",
                    "minHeight": "36px",
                    "flexShrink": "0",
                    "zIndex": "100",
                },
            ),

            dcc.Store(id="nav-active-store", data="/"),

            # --- Скрытые триггеры для контекстного меню графика ---
            html.Div(style={"display": "none"}, children=[
                html.Button("download-excel", id="download-excel-button"),
                dcc.Download(id="download-excel"),
            ]),

                ]  # конец children внешнего html.Div
            ),  # конец внешнего html.Div
        ]
    )
