# -*- coding: utf-8 -*-
"""
Layout приложения: MantineProvider, Store, Modal, Grid, Paper, Filters, Graph.
Многостраничное: График | Коррелограмма | Data Engineering | Кластеризация | ML.
Header + Footer общие на всех страницах.
"""

import dash
from dash import dcc, html
from dash_iconify import DashIconify
import dash_mantine_components as dmc

from config import STYLE_CARD, PAPER_BASE, initial_fig
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
    add_filter_button,
    dropdown_chart_type,
    SwitchBubble, update_graf,
    download_button, DownloadFile,
    excel_download_button, DownloadExcel,
    copy_button, copy_trigger,
    dropdown_text_pozition, dropdown_category_ascending,
    dropdown_axes_category, dropdown_overlay,
    dropdown_legend, dropdown_legend_order, input_legend_custom_order,
    bin_column_select, bin_method, bin_k, bin_label_style,
    bar_text_auto_switch,
)

# Навигационные ссылки
NAV_LINKS = [
    {"label": "График", "href": "/"},
    {"label": "Коррелограмма", "href": "/correlation"},
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


def create_layout():
    return dmc.MantineProvider(
        children=[
            # ============================================
            # FLEX-КОНТЕЙНЕР: header (фикс) + content (flex:1) + footer (фикс)
            # ============================================
            html.Div(
                style={"display": "flex", "flexDirection": "column", "minHeight": "100vh"},
                children=[

            # --- URL Location ---
            dcc.Location(id="url", refresh=False),

            # --- Stores ---
            dcc.Store(id='stored-data', data=False, storage_type="memory"),
            dcc.Store(id='filtered-data'),
            dcc.Store(id='bin-applied-name'),
            dcc.Store(id='filter-count', data=1),
            dcc.Store(id='filters-state', data={}),
            dcc.Store(id='stored-sheet-names'),
            dcc.Store(id='selected-sheet'),
            dcc.Store(id='sheet-modal-toggle', data=True),
            dcc.Store(id='custom-colors', data={}),
            dcc.Store(id='meta-columns'),
            dcc.Store(id='cluster-metrics'),
            dcc.Store(id='source-file-path'),
            dcc.Store(id='source-file-name'),
            dcc.Store(id='filters-initialized', data=False),

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

            dmc.NotificationContainer(id="notifications-container"),
            copy_trigger,

            # ========================
            # HEADER (общий, фиксированный сверху)
            # ========================
            dmc.Paper(
                children=[
                    dmc.Group([
                        dmc.Text("DataAnalize", fw=700, size="lg", c="white", style={"letterSpacing": "0.5px"}),
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
                style={"backgroundColor": "#1A1B1E", "borderBottom": "1px solid #2C2E33"},
                withBorder=False,
            ),

            # --- Sheet modal ---
            html.Div(
                dmc.Modal(id="sheet-modal", title="Выберите лист Excel", opened=False, centered=True, zIndex=1000, children=[]),
                id="sheet-menu-wrapper"
            ),
            html.Div(id='status-message', style={'display': 'none'}),

            # --- de-modal (скрыт, контент перенесён на страницу) ---
            html.Div(style={"display": "none"}, children=[dmc.Modal(id="de-modal", opened=False, children=[])]),

            # ========================
            # CONTENT: левая панель (меняется) + правая (общая) — растягивается до низа
            # ========================
            dmc.Grid([
                # ---------- САЙДБАР ПЛАШЕК КОЛОНОК ----------
                dmc.GridCol([
                    dmc.Paper(
                        id="columns-sidebar",
                        children=[
                            dmc.Text("Колонки датасета", size="xs", fw=600, c="dimmed"),
                            dmc.Space(h=6),
                            dmc.Divider(label="Исходный", labelPosition="center", size="xs"),
                            dmc.Stack(
                                id="columns-original-badges",
                                children=[],
                                gap="2px",
                            ),
                            dmc.Space(h=8),
                            dmc.Divider(
                                id="columns-filtered-label",
                                label="Фильтрованный датасет",
                                labelPosition="center",
                                size="xs",
                            ),
                            dmc.Stack(
                                id="columns-filtered-badges",
                                children=[],
                                gap="2px",
                            ),
                        ],
                        shadow="sm",
                        p="xs",
                        withBorder=True,
                        style={
                            "overflowY": "auto",
                            "maxHeight": "calc(100vh - 100px)",
                            "padding": "8px",
                        },
                    ),
                ], span=2),

                # ---------- ЛЕВАЯ ПАНЕЛЬ (меняется от страницы) ----------
                dmc.GridCol([

                    # --- Стр. 1: ГРАФИК — оси + фильтры ---
                    html.Div(id="page-graph", children=[
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
                                dmc.GridCol([html.Center(dmc.Text("Hover Data", c="blue", fw=500, size="sm")), dropdown_hover_data], span=12, style={"minWidth": 0}),
                            ]),
                            dmc.Space(h=6),
                            dmc.Divider(label="Настройки графика"),
                            dmc.Text("Bubble, подписи, шрифты и стили — в панели настроек (иконка шестерёнки).", size="xs", c="dimmed"),
                            dmc.Space(h=6),
                            html.Div([
                                dmc.Drawer(
                                    title="Настройка графика", id="drawer-simple", padding="md", position='right',
                                    withOverlay=True, overlayProps={"opacity": 0.15}, size=600,
                                    closeOnClickOutside=True, closeOnEscape=True, withCloseButton=True,
                                    children=[
                                        dmc.Grid([
                                            dmc.GridCol([SwitchBubble], span="content"),
                                            dmc.GridCol([dmc.NumberInput(id="InputMaxSizeBubble", label="Макс. размер бабла",
                                                value=30, min=1, max=100, debounce=True, step=5,
                                                persistence=True, persistence_type='local')], span="content"),
                                            dmc.GridCol([bar_text_auto_switch], span="content"),
                                            dmc.Grid([
                                                dmc.GridCol([dmc.NumberInput(id="InputSizePlot", label="Высота графика",
                                                    value=750, min=50, max=20000, debounce=True, step=50,
                                                    persistence=True, persistence_type='local')], span="content"),
                                                dmc.GridCol([dmc.NumberInput(id="InputSizePlotW", label="Ширина графика",
                                                    min=50, max=20000, debounce=True, step=50,
                                                    persistence=True, persistence_type='local')], span="content")
                                            ]),
                                            dmc.Grid([
                                                dmc.GridCol([dmc.NumberInput(id="font-size-xaxis", label="Шрифт X оси", value=14, min=6, max=48, debounce=True, step=1)], span=3),
                                                dmc.GridCol([dmc.NumberInput(id="font-size-yaxis", label="Шрифт Y оси", value=14, min=6, max=48, debounce=True, step=1)], span=3),
                                                dmc.GridCol([dmc.NumberInput(id="font-size-title", label="Шрифт заголовка", value=16, min=6, max=48, debounce=True, step=1)], span=3),
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
                                                dmc.GridCol([dmc.NumberInput(id="tick-step-xaxis", label="Шаг тиков X оси",
                                                    value=0, min=0, step=0.1, decimalScale=2, debounce=True,
                                                    persistence=True, persistence_type='local')], span="content"),
                                                dmc.GridCol([dmc.NumberInput(id="tick-step-yaxis", label="Шаг тиков Y оси",
                                                    value=0, min=0, step=0.1, decimalScale=2, debounce=True,
                                                    persistence=True, persistence_type='local')], span="content"),
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
                    ]),

                    # --- Стр. 2: КОРРЕЛОГРАММА ---
                    html.Div(id="page-correlation", style={"display": "none"}, children=[
                        dmc.Paper([
                            dmc.Text("Корреляционный анализ", fw=600, size="md"),
                            dmc.Space(h=8),
                            dmc.Text("Корреляц. столбцы", c="blue", fw=500, size="sm"),
                            dropdown_corr_columns,
                            dmc.Space(h=10),
                            dmc.Text("Выберите столбцы для расчёта корреляционной матрицы. Нажмите «Обновить» чтобы построить график.", size="xs", c="dimmed"),
                        ], style=STYLE_CARD, shadow="md", p="md", withBorder=True),
                    ]),

                    # --- Стр. 3: DATA ENGINEERING ---
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
                            dmc.Divider(label="Текстовые копии (чтобы Plotly и фильтры видели как текст)"),
                            dmc.Space(h=8),
                            dmc.Grid([
                                dmc.GridCol([dmc.Text("Столбец(ы) для копирования в текст", c="blue", fw=500, size="sm"), txtcopy_cols_select], span=8, style={"minWidth": 0}),
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
                            dmc.Text("Выбираете ключ(и), столбцы и метрики — новые колонки добавляются в конец датасета.", size="sm"),
                            dmc.Space(h=10),
                            dmc.Grid([
                                dmc.GridCol([dmc.Text("Ключ(и) группировки", c="blue", fw=500, size="sm"), agg_keys_select], span=6, style={"minWidth": 0}),
                                dmc.GridCol([dmc.Text("Столбцы для расчёта", c="blue", fw=500, size="sm"), agg_cols_select], span=6, style={"minWidth": 0}),
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

                    # --- Стр. 4: КЛАСТЕРИЗАЦИЯ ---
                    html.Div(id="page-clustering", style={"display": "none"}, children=[
                        dmc.Paper([
                            dmc.Divider(label="Кластеризация (KMeans)"),
                            dmc.Text("Кластеризация числовых столбцов методом KMeans с визуализацией PCA и метриками качества.", size="sm"),
                            dmc.Space(h=8),
                            dmc.Grid([
                                dmc.GridCol([html.Center(dmc.Text("Числовые столбцы для кластеризации", c="blue", fw=500, size="sm")), dropdown_cluster_cols], span=8, style={"minWidth": 0}),
                                dmc.GridCol([html.Center(dmc.Text("К (Кластеры)", c="blue", fw=500, size="sm")),
                                              dmc.NumberInput(id="cluster-k", value=4, min=2, max=20, step=1, debounce=True)], span=2, style={"minWidth": 0}),
                                dmc.GridCol([dmc.Button("Кластеризация", id="btn-cluster", size="xs")], span=2, style={"minWidth": 0, "marginTop": 23}),
                            ]),
                        ], style=STYLE_CARD, shadow="md", p="md", withBorder=True),
                    ]),

                    # --- Стр. 5: ML ---
                    html.Div(id="page-ml", style={"display": "none"}, children=[
                        dmc.Paper([
                            dmc.Text("Machine Learning", fw=600, size="lg"),
                            dmc.Space(h=8),
                            dmc.Text("Раздел машинного обучения — в разработке.", size="md", c="dimmed"),
                            dmc.Space(h=16),
                            dmc.Text("Здесь будут: обучение моделей регрессии/классификации, подбор гиперпараметров, оценка метрик, предсказания.", size="sm", c="dimmed"),
                        ], style=STYLE_CARD, shadow="md", p="md", withBorder=True),
                    ]),
                ], span=3),

                # ---------- ПРАВАЯ ПАНЕЛЬ (общая: график + тулбар) ----------
                dmc.GridCol([
                    dmc.Grid([
                        dmc.GridCol([
                            dmc.Group([
                                dmc.Text("Тип графика", size="sm", fw=500, c="dimmed"),
                                dropdown_chart_type,
                            ], gap="xs", align="center")
                        ], span="content"),
                        dmc.GridCol([update_graf], span="content"),
                        dmc.GridCol([download_button, DownloadFile], span="content"),
                        dmc.GridCol([excel_download_button, DownloadExcel], span="content"),
                        dmc.GridCol([copy_button], span="content"),
                        dmc.GridCol([
                            dmc.Tooltip(label="Изменить цвета", withArrow=True,
                                children=dmc.ActionIcon(id="shuffle-button", variant="light", size="xl", radius="xl",
                                    children=DashIconify(icon="tabler:palette", width=18)))
                        ], span="content"),
                        dmc.GridCol([
                            dmc.Tooltip(label="Настройка графика", withArrow=True,
                                children=dmc.ActionIcon(id="drawer-demo-button", variant="light", size="xl", radius="xl",
                                    children=DashIconify(icon="lucide:settings", width=18)))
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

                    # Графики корреляций / метрик кластеризации
                    html.Div(id="corr-bars-section", children=[
                        dmc.Paper([
                            dmc.Grid([
                                dmc.GridCol([dcc.Graph(id="corr-bar-x", config={'displaylogo': False, 'responsive': True})], span=6),
                                dmc.GridCol([dcc.Graph(id="corr-bar-y", config={'displaylogo': False, 'responsive': True})], span=6),
                            ])
                        ], style={**STYLE_CARD, "overflow": "visible"}, shadow="md", p="md", withBorder=True),
                    ], style={**PAPER_BASE, "visibility": "hidden"}),

                ], span=7)
            ], style={"flex": 1, "margin": 0}),

            # ========================
            # FOOTER (общий, фиксированный внизу) — информация о файле
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
                },
            ),

            dcc.Store(id="nav-active-store", data="/"),

                ]  # конец flex-контейнера
            ),
        ]
    )