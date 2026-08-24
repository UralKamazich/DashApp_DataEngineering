# -*- coding: utf-8 -*-
"""Multi-page ML workspace with a CatBoost implementation and shared shell."""

from dash import dcc, html, dash_table
import dash_mantine_components as dmc

from ml_models import MODEL_ADAPTERS


GRAPH_CONFIG = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "modeBarButtonsToRemove": [
        "zoom2d", "pan2d", "zoomIn2d", "zoomOut2d", "autoScale2d",
        "select2d", "lasso2d",
    ],
    "responsive": True,
    "scrollZoom": True,
}


def _field(label, control, class_name=""):
    return html.Div(
        [dmc.Text(label, size="10px", fw=650, c="dimmed"), control],
        className=f"ml-field {class_name}".strip(),
    )


def _metric(label, component_id, note):
    return dmc.Paper(
        [
            dmc.Text(label, size="9px", c="dimmed", tt="uppercase", fw=700),
            dmc.Text("—", id=component_id, className="ml-metric-value"),
            dmc.Text(note, size="9px", c="dimmed"),
        ],
        p="xs", withBorder=True, shadow="xs", className="ml-metric-card",
    )


def _number(component_id, value, **kwargs):
    return dmc.NumberInput(id=component_id, value=value, size="xs", debounce=True, **kwargs)


def create_catboost_workspace():
    routing = dmc.Paper(
        [
            dmc.Group(
                [
                    html.Div([
                        dmc.Text("CatBoost · регрессия", fw=700, size="sm"),
                        dmc.Text(
                            "Обучение, проверка качества и запись прогноза в dataset",
                            size="10px", c="dimmed",
                        ),
                    ]),
                    dmc.Group([
                        dmc.Badge("Нет данных", id="ml-dataset-badge", size="sm",
                                  variant="light", color="gray"),
                        dmc.Badge("Ожидает запуска", id="ml-run-status", size="sm",
                                  variant="light", color="gray"),
                    ], gap=6),
                ],
                justify="space-between", align="flex-start",
            ),
            dmc.SimpleGrid(
                [
                    _field("Входной dataset", dmc.Select(
                        id="ml-input-dataset", data=[], searchable=True,
                        allowDeselect=False, size="xs",
                    )),
                    _field("Слой данных", dmc.SegmentedControl(
                        id="ml-input-scope",
                        data=[
                            {"label": "До фильтров", "value": "base"},
                            {"label": "После фильтров", "value": "filtered"},
                        ],
                        value="base", size="xs", fullWidth=True,
                    )),
                    _field("Новый dataset", dmc.TextInput(
                        id="ml-output-name", placeholder="CatBoost_До фильтров_1", size="xs",
                    )),
                ],
                cols=3, spacing="xs", mt="sm", className="ml-routing-grid",
            ),
        ],
        p="sm", withBorder=True, shadow="xs", className="ml-routing",
    )

    feature_select = dmc.MultiSelect(
        id="ml-features", data=[], value=[], searchable=True, clearable=True,
        nothingFoundMessage="Ничего не найдено", maxDropdownHeight=300,
        comboboxProps={"shadow": "md"}, size="xs",
    )
    controls = dmc.Paper(
        [
            dmc.Group([
                dmc.Text("Данные модели", fw=700, size="xs"),
                dmc.Button("Все числовые", id="ml-select-numeric", size="compact-xs",
                           variant="subtle"),
            ], justify="space-between"),
            dmc.SimpleGrid([
                _field("Целевой канал · число", dmc.Select(
                    id="ml-target", data=[], searchable=True, clearable=True, size="xs",
                )),
                _field("ID / подпись · необязательно", dmc.Select(
                    id="ml-id-column", data=[], searchable=True, clearable=True, size="xs",
                )),
            ], cols=2, spacing="xs", mt="xs"),
            html.Div([
                dmc.Group([
                    dmc.Text("Признаки", size="10px", fw=650, c="dimmed"),
                    dmc.Text("числа и категории", size="9px", c="dimmed"),
                ], justify="space-between"),
                feature_select,
            ], id="ml-features-drop", className="ml-features-drop",
               **{"data-drop-target": "ml-features", "data-drop-mode": "append",
                  "data-current-value": "[]"}),

            dmc.Divider(label="Проверка качества", labelPosition="left", my="xs"),
            dmc.Select(
                id="ml-method",
                data=[
                    {"label": "Train / test · случайное", "value": "split"},
                    {"label": "KFold · случайные фолды", "value": "cv"},
                    {"label": "GroupKFold · группы не смешиваются", "value": "group_cv"},
                    {"label": "TimeSeriesSplit · прошлое → будущее", "value": "time_cv"},
                ],
                value="split", size="xs", allowDeselect=False,
            ),
            dmc.SimpleGrid([
                _field("Доля test", _number("ml-test-size", 0.2, min=0.05, max=0.5, step=0.05),
                       "ml-split-option"),
                _field("Фолды", _number("ml-folds", 5, min=2, max=20, step=1),
                       "ml-cv-option"),
            ], cols=2, spacing="xs", mt="xs"),
            dmc.SimpleGrid([
                _field("Канал группы", dmc.Select(
                    id="ml-group-column", data=[], searchable=True, clearable=True,
                    size="xs", disabled=True,
                )),
                _field("Время / порядок", dmc.Select(
                    id="ml-time-column", data=[], searchable=True, clearable=True,
                    size="xs", disabled=True,
                )),
            ], cols=2, spacing="xs", mt="xs"),
            dmc.Text(
                "Для скважин и месторождений используйте GroupKFold: одна группа не попадёт одновременно в обучение и проверку.",
                id="ml-validation-hint", size="9px", c="dimmed", mt=4,
            ),

            dmc.Divider(label="Параметры CatBoost", labelPosition="left", my="xs"),
            _field("Пресет", dmc.Select(
                id="ml-preset", value="balanced", allowDeselect=False, size="xs",
                data=[
                    {"label": "Быстрый черновик", "value": "draft"},
                    {"label": "Баланс", "value": "balanced"},
                    {"label": "Высокое качество", "value": "quality"},
                    {"label": "Вручную", "value": "custom"},
                ],
            )),
            dmc.SimpleGrid([
                _field("Деревья", _number("ml-iterations", 800, min=1, max=20000, step=50)),
                _field("Глубина", _number("ml-depth", 6, min=1, max=16, step=1)),
                _field("Learning rate", _number("ml-learning-rate", 0.05, min=0.001, max=1, step=0.01)),
                _field("L2", _number("ml-l2", 3.0, min=0, step=0.5)),
            ], cols=4, spacing="xs", mt="xs", className="ml-param-grid"),
            html.Details([
                html.Summary("Расширенные параметры"),
                dmc.SimpleGrid([
                    _field("Функция потерь", dmc.Select(
                        id="ml-loss", value="RMSE", allowDeselect=False, size="xs",
                        data=[value for value in ["RMSE", "MAE", "MAPE", "Quantile"]],
                    )),
                    _field("Early stopping", _number("ml-early-stopping", 80, min=0, step=10)),
                    _field("Random strength", _number("ml-random-strength", 1.0, min=0, step=0.1)),
                    _field("Bagging temperature", _number("ml-bagging-temperature", 1.0, min=0, step=0.1)),
                    _field("Random seed", _number("ml-random-seed", 42, min=0, step=1)),
                ], cols=5, spacing="xs", mt="xs"),
            ], className="ml-advanced"),

            dmc.Divider(label="Выходные каналы", labelPosition="left", my="xs"),
            dmc.SimpleGrid([
                _field("Имя прогноза", dmc.TextInput(
                    id="ml-prediction-column", value="Прогноз CatBoost", size="xs",
                )),
                dmc.Switch(id="ml-include-residual", label="Добавить остаток", checked=True, size="xs"),
                dmc.Switch(id="ml-compute-shap", label="Рассчитать SHAP", checked=True, size="xs"),
            ], cols=3, spacing="xs"),
            dmc.Button("Обучить модель", id="ml-run", size="xs", fullWidth=True, mt="sm"),
            dmc.SimpleGrid([
                dmc.Button("Создать dataset", id="ml-commit", size="xs",
                           variant="light", disabled=True),
                dmc.Button("Выгрузить Excel", id="ml-export-excel", size="xs",
                           variant="light", color="violet", disabled=True),
                dmc.Button("Сохранить модель", id="ml-save-model", size="xs",
                           variant="light", color="grape", disabled=True),
            ], cols=3, spacing="xs", mt=5),
            dmc.Text(id="ml-row-status", size="9px", c="dimmed", mt=4),
        ],
        p="sm", withBorder=True, shadow="xs", className="ml-controls",
    )

    metrics = html.Div([
        _metric("MAE", "ml-metric-mae", "ниже — лучше"),
        _metric("RMSE", "ml-metric-rmse", "ниже — лучше"),
        _metric("MAPE", "ml-metric-mape", "% · нули исключены"),
        _metric("R²", "ml-metric-r2", "выше — лучше"),
        _metric("Baseline MAE", "ml-metric-baseline", "прогноз средним"),
    ], className="ml-metrics-grid")

    empty_graph = lambda graph_id: dcc.Graph(id=graph_id, config=GRAPH_CONFIG)
    results = dmc.Paper([
        dmc.Group([
            html.Div([
                dmc.Text("Результаты модели", fw=700, size="xs"),
                dmc.Text(id="ml-evaluation-note", size="9px", c="dimmed"),
            ]),
            dmc.Text(id="ml-shap-note", size="9px", c="dimmed"),
        ], justify="space-between"),
        dmc.Tabs([
            dmc.TabsList([
                dmc.TabsTab("Прогноз", value="prediction"),
                dmc.TabsTab("Обучение", value="learning"),
                dmc.TabsTab("Важность", value="importance"),
                dmc.TabsTab("SHAP", value="shap"),
                dmc.TabsTab("Диагностика", value="diagnostics"),
                dmc.TabsTab("Таблица", value="table"),
                dmc.TabsTab("Протокол", value="log"),
            ]),
            dmc.TabsPanel(empty_graph("ml-prediction-graph"), value="prediction"),
            dmc.TabsPanel(empty_graph("ml-learning-graph"), value="learning"),
            dmc.TabsPanel(empty_graph("ml-importance-graph"), value="importance"),
            dmc.TabsPanel(empty_graph("ml-shap-graph"), value="shap"),
            dmc.TabsPanel(empty_graph("ml-diagnostics-graph"), value="diagnostics"),
            dmc.TabsPanel(dash_table.DataTable(
                id="ml-prediction-table", data=[], columns=[], page_size=20,
                sort_action="native", filter_action="native", style_table={"overflowX": "auto"},
                style_cell={"fontSize": "11px", "padding": "5px", "maxWidth": "180px",
                            "overflow": "hidden", "textOverflow": "ellipsis"},
            ), value="table"),
            dmc.TabsPanel(html.Pre(id="ml-log", className="ml-log"), value="log"),
        ], value="prediction", mt="xs", keepMounted=True),
    ], p="xs", withBorder=True, shadow="xs", className="ml-results")

    return html.Div([
        routing,
        html.Div([controls, html.Div([metrics, results], className="ml-output")],
                 className="ml-main-grid"),
    ], className="ml-catboost-workspace")


def _subnav_link(label, href, icon):
    return dcc.Link(
        [html.Span(icon, className="ml-subnav-icon"), html.Span(label)],
        href=href,
        className="ml-subnav-link",
    )


def _empty_experiments_graph():
    return {
        "data": [],
        "layout": {
            "height": 330,
            "margin": {"l": 42, "r": 18, "t": 30, "b": 45},
            "annotations": [{
                "text": "После обучения здесь появится сравнение запусков",
                "x": .5, "y": .5, "xref": "paper", "yref": "paper",
                "showarrow": False, "font": {"color": "#868e96", "size": 12},
            }],
            "xaxis": {"visible": False}, "yaxis": {"visible": False},
        },
    }


def create_experiments_workspace():
    return html.Div([
        dmc.Paper([
            dmc.Group([
                html.Div([
                    dmc.Text("Эксперименты", fw=700, size="sm"),
                    dmc.Text(
                        "Единый журнал качества, разбиения и артефактов всех моделей",
                        size="10px", c="dimmed",
                    ),
                ]),
                dmc.Group([
                    dmc.Badge("0 запусков", id="ml-history-count", variant="light", color="gray"),
                    dmc.Badge("Лучший MAE: —", id="ml-history-best", variant="light", color="green"),
                ], gap=6),
            ], justify="space-between"),
        ], p="sm", withBorder=True, shadow="xs"),
        html.Div([
            dmc.Paper([
                dmc.Text("Сравнение MAE", fw=700, size="xs", mb=4),
                dcc.Graph(
                    id="ml-experiments-graph", figure=_empty_experiments_graph(),
                    config=GRAPH_CONFIG,
                ),
            ], p="xs", withBorder=True, shadow="xs", className="ml-experiment-chart"),
            dmc.Paper([
                dmc.Text("История запусков", fw=700, size="xs", mb=6),
                dash_table.DataTable(
                    id="ml-experiments-table", data=[], columns=[], page_size=12,
                    sort_action="native", filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "fontSize": "10px", "padding": "5px", "maxWidth": "190px",
                        "overflow": "hidden", "textOverflow": "ellipsis",
                    },
                ),
            ], p="xs", withBorder=True, shadow="xs", className="ml-experiment-table"),
        ], className="ml-experiments-grid"),
    ], className="ml-page-body")


def create_future_model_workspace(model_key):
    descriptor = MODEL_ADAPTERS[model_key].descriptor
    return dmc.Paper([
        dmc.Group([
            html.Div([
                dmc.Text(descriptor.title, fw=750, size="lg"),
                dmc.Text(descriptor.family, size="10px", c="dimmed"),
            ]),
            dmc.Badge("Подготовлено к подключению", color="gray", variant="light"),
        ], justify="space-between"),
        dmc.Text(descriptor.description, size="sm", mt="md"),
        dmc.Divider(my="md"),
        dmc.Text(
            "Подлист уже находится в общем ML-контуре. Он получит тот же паспорт данных, стратегии проверки, журнал экспериментов и формат результатов.",
            size="xs", c="dimmed",
        ),
    ], p="lg", withBorder=True, shadow="xs", className="ml-future-model")


def create_ml_workspace():
    """Render the stable shell; individual models occupy separate subpages."""
    return html.Div([
        dmc.Paper([
            dmc.Group([
                html.Div([
                    dmc.Text("ML Studio", fw=750, size="md"),
                    dmc.Text("Модели, эксперименты и производные dataset", size="10px", c="dimmed"),
                ]),
                dmc.Badge("Регрессия", variant="dot", color="violet"),
            ], justify="space-between", align="flex-start"),
            html.Nav([
                _subnav_link("Эксперименты", "/ml/experiments", "◫"),
                _subnav_link("CatBoost", "/ml/catboost", "CB"),
                _subnav_link("Random Forest", "/ml/random-forest", "RF"),
                _subnav_link("Нейросети", "/ml/neural-networks", "NN"),
            ], className="ml-subnav"),
        ], p="sm", withBorder=True, shadow="xs", className="ml-shell-header"),
        html.Div(id="ml-page-experiments", children=[create_experiments_workspace()]),
        html.Div(id="ml-page-catboost", style={"display": "none"},
                 children=[create_catboost_workspace()]),
        html.Div(id="ml-page-random-forest", style={"display": "none"},
                 children=[create_future_model_workspace("random-forest")]),
        html.Div(id="ml-page-neural-networks", style={"display": "none"},
                 children=[create_future_model_workspace("neural-networks")]),
        dcc.Store(id="ml-analysis"),
        dcc.Store(id="ml-auto-output-name"),
        dcc.Store(id="ml-columns-sync"),
        dcc.Store(id="ml-experiment-history", data=[], storage_type="local"),
    ], className="ml-workspace")
