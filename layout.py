# -*- coding: utf-8 -*-
"""
Layout приложения: MantineProvider, Store, Modal, Grid, Paper, Filters, Graph.
Многостраничное: График | Коррелограмма | Data Engineering | Кластеризация | ML.
Header + Footer общие на всех страницах.
"""

import dash
from dash import dcc, html
import dash_mantine_components as dmc
from dash_iconify import DashIconify

from config import APP_NAME, STYLE_CARD
from data_import import POPULAR_DATASETS
from correlation_workspace import CorrelationWorkspace
from data_engineering_workspace import create_data_engineering_workspace
from dashboard_workspace import DASHBOARD_WORKSPACES, create_dashboard_workspace
from clustering_workspace import create_clustering_workspace
from ml_workspace import create_ml_workspace
from dataset_panel import create_dataset_drawer
from filter_panel import create_filter_drawer
from graph_settings import GraphSettingsPanel
from graph_workspace import DEFAULT_FIELDS, GraphWorkspace, LEGACY_GRAPH_ACTION_IDS
from multi_axis_workspace import MultiYAxisWorkspace
from components import (
    dropdown_style, graph_render_mode, graph_dataset_select,
    dropdown_x, dropdown_y, dropdown_z,
    dropdown_color, dropdown_size, dropdown_text,
    dropdown_hover_data, dropdown_corr_columns,
    dropdown_hierarchy_levels, dropdown_hierarchy_value,
    dropdown_facet_row, dropdown_facet_col,
    agg_keys_select, agg_cols_select, agg_metrics_select,
    agg_exclude_zeros_switch, agg_exclude_empty_switch,
    txtcopy_cols_select, txtcopy_suffix_input, txtcopy_strip_switch,
    dropdown_chart_type,
    mv_chart_type,
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
    {"label": "Дашборд", "href": "/dashboard"},
    {"label": "Multi‑Y", "href": "/multi-y"},
    {"label": "Корреляционный анализ", "href": "/correlation"},
    {"label": "Data Engineering", "href": "/data-engineering"},
    {"label": "Кластеризация", "href": "/clustering"},
    {"label": "ML", "href": "/ml/experiments"},
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
        "dataset": graph_dataset_select,
        "theme": dropdown_style,
        "render_mode": graph_render_mode,
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
        "dropdown_hierarchy_levels": dropdown_hierarchy_levels,
        "dropdown_hierarchy_value": dropdown_hierarchy_value,
    },
    fields=(
        *DEFAULT_FIELDS,
        {
            "key": "hierarchy-levels",
            "label": "Уровни",
            "target": "dropdown_hierarchy_levels",
            "zone": "secondary",
            "mode": "append",
        },
        {
            "key": "hierarchy-value",
            "label": "Значение",
            "target": "dropdown_hierarchy_value",
            "zone": "secondary",
        },
    ),
    field_chart_types={
        "z": {"3D_Scatter", "Polar"},
        "color": {
            "Scatter", "3D_Scatter", "Box", "Bar", "Line", "Hist",
            "Polar", "Pie", "Violin", "Ridge", "Sunburst", "Treemap",
            "DensityContour",
        },
        "size": {"Scatter", "3D_Scatter", "Polar"},
        "text": {"Scatter", "3D_Scatter", "Polar"},
        "facet-row": {
            "Scatter", "Box", "Bar", "Line", "Hist", "Violin",
            "DensityHeat", "DensityContour",
        },
        "facet-col": {
            "Scatter", "Box", "Bar", "Line", "Hist", "Violin",
            "DensityHeat", "DensityContour",
        },
        "hover": {"Scatter", "3D_Scatter", "Polar", "Pie"},
        "hierarchy-levels": {"Sunburst", "Treemap"},
        "hierarchy-value": {"Sunburst", "Treemap"},
    },
    field_presentations={
        "3D_Scatter": {
            "x": {"label": "X", "placement": "top"},
            "y": {"label": "Y", "placement": "top"},
            "z": {"label": "Z", "placement": "top"},
        },
        "Polar": {
            "x": {"label": "A", "placement": "top"},
            "y": {"label": "B", "placement": "top"},
            "z": {"label": "C", "placement": "top"},
        },
        "Pie": {
            "x": {"label": "Категория", "placement": "top"},
            "y": {"label": "Значение", "placement": "top"},
        },
        "Sunburst": {
            "hierarchy-levels": {"label": "Уровни", "order": "0"},
            "hierarchy-value": {"label": "Значение", "order": "1"},
            "color": {"label": "Цвет", "order": "2"},
        },
        "Treemap": {
            "hierarchy-levels": {"label": "Уровни", "order": "0"},
            "hierarchy-value": {"label": "Значение", "order": "1"},
            "color": {"label": "Цвет", "order": "2"},
        },
    },
    settings_panel=GRAPH_SETTINGS_PANEL,
    action_ids=LEGACY_GRAPH_ACTION_IDS,
)

CORRELATION_WORKSPACE = CorrelationWorkspace(dropdown_corr_columns)

# Все многомерные графики используют единый список «Коррелируемые каналы»
# над рабочей областью. Собственные DnD-поля здесь намеренно отсутствуют.
MULTIVARIATE_WORKSPACE = GraphWorkspace(
    graph_id="mv-graph",
    chart_type_control=mv_chart_type,
    field_controls={},
    fields=(),
    include_color_controls=False,
)

MULTI_Y_WORKSPACE = MultiYAxisWorkspace(graph_id="multi-y-graph")


def create_layout():
    graph_workspace = GRAPH_WORKSPACE.render()
    multi_y_workspace = MULTI_Y_WORKSPACE.render()
    multivariate_workspace = MULTIVARIATE_WORKSPACE.render()
    dashboard_workspace = create_dashboard_workspace()
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
            dcc.Store(id='filter-applied-logic', data='and'),
            dcc.Store(id='stored-sheet-names'),
            dcc.Store(id='selected-sheet'),
            dcc.Store(id='sheet-modal-toggle', data=True),
            dcc.Store(id='meta-columns'),
            dcc.Store(id='source-file-path'),
            dcc.Store(id='source-file-name'),
            dcc.Store(id='dataset-registry', data={}),
            dcc.Store(id='active-dataset-id'),
            dcc.Store(id='active-dataset-data'),
            dcc.Store(id='de-draft-pipeline', data={"input_id": None, "scope": "base", "steps": []}),
            dcc.Store(id='de-auto-output-name'),

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
                            "Выбрать файл",
                            id="pick-file-btn",
                            className="dataset-file-picker dataset-file-drop-target",
                            size="xs",
                            variant="outline",
                            color="gray",
                            attributes={"title": "Excel, CSV, TXT, TSV, ZIP или Pickle"},
                            leftSection=DashIconify(icon="tabler:file-upload", width=15),
                            style={"borderColor": "rgba(255,255,255,0.3)", "color": "#ccc"}
                        ),
                        dmc.Menu(
                            id="online-dataset-menu",
                            trigger="click",
                            position="bottom-start",
                            width=390,
                            shadow="md",
                            closeOnItemClick=True,
                            withinPortal=True,
                            zIndex=1200,
                            children=[
                                dmc.MenuTarget(
                                    dmc.Tooltip(
                                        label="Загрузить dataset из интернета",
                                        position="bottom",
                                        withArrow=True,
                                        openDelay=350,
                                        children=dmc.ActionIcon(
                                            DashIconify(
                                                icon="tabler:world-download", width=16
                                            ),
                                            id="online-dataset-open",
                                            size=26,
                                            variant="subtle",
                                            color="gray",
                                            radius="sm",
                                            attributes={
                                                "aria-label": "Загрузить dataset из интернета"
                                            },
                                            style={"color": "#ccc"},
                                        ),
                                    )
                                ),
                                dmc.MenuDropdown(
                                    p="sm",
                                    children=[
                                        dmc.Text("Готовые datasets", size="xs", fw=700),
                                        dmc.Select(
                                            id="online-dataset-catalog",
                                            data=[
                                                {
                                                    "label": item["label"],
                                                    "value": item["value"],
                                                }
                                                for item in POPULAR_DATASETS
                                            ],
                                            value=POPULAR_DATASETS[0]["value"],
                                            size="xs",
                                            searchable=True,
                                            allowDeselect=False,
                                            mt=5,
                                        ),
                                        dmc.Text(
                                            POPULAR_DATASETS[0]["description"],
                                            id="online-dataset-description",
                                            size="10px",
                                            c="dimmed",
                                            mt=4,
                                            lh=1.25,
                                        ),
                                        dmc.Divider(
                                            label="или прямая ссылка",
                                            labelPosition="center",
                                            my="xs",
                                        ),
                                        dmc.TextInput(
                                            id="online-dataset-url",
                                            placeholder="https://…/dataset.csv или .zip",
                                            size="xs",
                                            leftSection=DashIconify(
                                                icon="tabler:link", width=14
                                            ),
                                        ),
                                        dmc.Text(
                                            "CSV, TXT, TSV или ZIP · ссылка имеет приоритет над каталогом",
                                            size="9px",
                                            c="dimmed",
                                            mt=3,
                                        ),
                                        dmc.MenuItem(
                                            "Загрузить dataset",
                                            id="online-dataset-load",
                                            color="blue",
                                            mt="xs",
                                            leftSection=DashIconify(
                                                icon="tabler:download", width=15
                                            ),
                                        ),
                                    ],
                                ),
                            ],
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

                    html.Div(id="page-dashboard", style={"display": "none"}, children=[
                        dashboard_workspace,
                    ]),

                    html.Div(id="page-multi-y", style={"display": "none"}, children=[
                        multi_y_workspace,
                    ]),

                    html.Div(id="page-correlation", style={"display": "none"}, children=[
                        correlation_workspace,
                    ]),

                    html.Div(
                        id="page-data-engineering",
                        style={
                            "display": "none",
                            "width": "100%",
                            "maxWidth": "100%",
                            "minWidth": "0",
                            "overflowX": "hidden",
                        },
                        children=[create_data_engineering_workspace()],
                    ),

                    html.Div(
                        id="page-clustering",
                        style={"display": "none", "minWidth": "0", "width": "100%"},
                        children=[create_clustering_workspace()],
                    ),

                    html.Div(
                        id="page-ml",
                        style={"display": "none", "minWidth": "0", "width": "100%"},
                        children=[create_ml_workspace()],
                    ),
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
                className="app-file-footer",
                children=["Файл не выбран"],
                shadow="sm",
                withBorder=False,
                style={"padding": "0 20px 0 12px"},
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
