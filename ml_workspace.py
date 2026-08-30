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

COMPACT_SEGMENTED_STYLES = {
    "root": {"height": 23, "minHeight": 23, "padding": 1},
    "control": {"height": 21},
    "label": {
        "display": "flex", "alignItems": "center", "justifyContent": "center",
        "height": 21, "minHeight": 21, "padding": "0 5px", "fontSize": "8px",
        "lineHeight": 1,
    },
}

COMPACT_SWITCH_STYLES = {
    "label": {"fontSize": "8px", "lineHeight": 1.05, "paddingLeft": 5},
}

COMPACT_NUMBER_STYLES = {
    "input": {"height": 23, "minHeight": 23},
    "controls": {"top": 1, "right": 1, "bottom": 1, "height": 21},
    "control": {"height": "50%", "minHeight": 0},
}


def _field(label, control, class_name="", label_id=None):
    label_props = {"size": "9px", "fw": 650, "c": "dimmed"}
    if label_id:
        label_props["id"] = label_id
    return html.Div(
        [dmc.Text(label, **label_props), control],
        className=f"ml-field {class_name}".strip(),
    )


def _metric(label, component_id, note):
    return dmc.Paper(
        [
            dmc.Text(label, id=f"{component_id}-label", size="9px", c="dimmed",
                     tt="uppercase", fw=700),
            dmc.Text("—", id=component_id, className="ml-metric-value"),
            dmc.Text(note, id=f"{component_id}-note", size="9px", c="dimmed"),
        ],
        p="xs", withBorder=True, shadow="xs", className="ml-metric-card",
    )


def _profile_metric(label, component_id):
    return dmc.Paper(
        [
            dmc.Text(label, size="8px", c="dimmed", tt="uppercase", fw=700),
            dmc.Text("—", id=component_id, className="ml-profile-metric-value"),
        ],
        p=6, withBorder=True, shadow="xs", className="ml-profile-metric",
    )


def _number(component_id, value, **kwargs):
    return dmc.NumberInput(
        id=component_id, value=value, size="xs", debounce=True,
        styles=COMPACT_NUMBER_STYLES, **kwargs,
    )


def create_catboost_workspace():
    routing = dmc.Paper(
        [
            dmc.Group(
                [
                    dmc.Text("CatBoost · регрессия", id="ml-workspace-title",
                             fw=700, size="sm"),
                    dmc.Group([
                        dmc.Badge("Нет данных", id="ml-dataset-badge", size="sm",
                                  variant="light", color="gray"),
                        dmc.Badge("Ожидает запуска", id="ml-run-status", size="sm",
                                  variant="light", color="gray"),
                    ], gap=6),
                ],
                justify="space-between", align="center",
            ),
            dmc.SimpleGrid(
                [
                    _field("Задача", dmc.SegmentedControl(
                        id="ml-task",
                        data=[
                            {"label": "Регрессия", "value": "regression"},
                            {"label": "Классификация", "value": "classification"},
                        ],
                        value="regression", size="xs", fullWidth=True,
                    )),
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
                cols=4, spacing="xs", mt=6, className="ml-routing-grid",
            ),
        ],
        p="xs", withBorder=True, shadow="xs", className="ml-routing",
    )

    feature_select = dmc.MultiSelect(
        id="ml-features", data=[], value=[], searchable=True, clearable=True,
        nothingFoundMessage="Ничего не найдено", maxDropdownHeight=300,
        comboboxProps={"shadow": "md"}, size="xs",
    )
    controls = dmc.Paper(
        [
            dmc.Group([
                dmc.Text("Данные модели", fw=700, size="11px"),
                dmc.Button("Все числовые", id="ml-select-numeric", size="compact-xs",
                           variant="subtle"),
            ], justify="space-between"),
            dmc.SimpleGrid([
                _field("Целевой канал · число", dmc.Select(
                    id="ml-target", data=[], searchable=True, clearable=True, size="xs",
                ), label_id="ml-target-label"),
                _field("ID / подпись · необязательно", dmc.Select(
                    id="ml-id-column", data=[], searchable=True, clearable=True, size="xs",
                )),
            ], cols=2, spacing="xs", mt=4),
            html.Div([
                dmc.Group([
                    dmc.Text("Признаки", size="9px", fw=650, c="dimmed"),
                    dmc.Text("числа и категории", size="8px", c="dimmed"),
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
            ], cols=2, spacing="xs", mt=4),
            dmc.SimpleGrid([
                _field("Канал группы", dmc.Select(
                    id="ml-group-column", data=[], searchable=True, clearable=True,
                    size="xs", disabled=True,
                )),
                _field("Время / порядок", dmc.Select(
                    id="ml-time-column", data=[], searchable=True, clearable=True,
                    size="xs", disabled=True,
                )),
            ], cols=2, spacing="xs", mt=4),
            dmc.Text(
                "Для скважин и месторождений используйте GroupKFold: одна группа не попадёт одновременно в обучение и проверку.",
                id="ml-validation-hint", size="9px", c="dimmed", mt=4,
            ),

            dmc.Divider(label="Параметры CatBoost", labelPosition="left", my="xs"),
            _field("Режим обучения", dmc.SegmentedControl(
                id="ml-run-mode", value="single", size="xs", fullWidth=True,
                styles=COMPACT_SEGMENTED_STYLES,
                data=[
                    {"label": "Один запуск", "value": "single"},
                    {"label": "Автоподбор параметров", "value": "tune"},
                ],
            )),
            dmc.SimpleGrid([
                _field("Пресет / исходная точка", dmc.Select(
                    id="ml-preset", value="balanced", allowDeselect=False, size="xs",
                    data=[
                        {"label": "Быстрый черновик", "value": "draft"},
                        {"label": "Баланс", "value": "balanced"},
                        {"label": "Высокое качество", "value": "quality"},
                        {"label": "Вручную", "value": "custom"},
                    ],
                )),
                html.Div(_field("Попытки", _number(
                    "ml-tuning-trials", 12, min=3, max=60, step=1,
                )), id="ml-tuning-trials-wrap", style={"display": "none"}),
                _field("Вычислитель", dmc.Select(
                    id="ml-compute-device", value="auto", allowDeselect=False,
                    size="xs", data=[
                        {"label": "Авто", "value": "auto"},
                        {"label": "CPU", "value": "cpu"},
                        {"label": "GPU", "value": "gpu"},
                    ],
                )),
            ], cols=3, spacing="xs", mt=4),
            dmc.Text(id="ml-compute-hint", size="8px", c="dimmed", mt=2),
            html.Div(id="ml-tuning-hint", style={"display": "none"}),
            dmc.SimpleGrid([
                _field("Деревья", _number("ml-iterations", 800, min=1, max=20000, step=50)),
                _field("Глубина", _number("ml-depth", 6, min=1, max=16, step=1)),
                _field("Learning rate", _number("ml-learning-rate", 0.05, min=0.001, max=1, step=0.01)),
                _field("L2", _number("ml-l2", 3.0, min=0, step=0.5)),
            ], cols=4, spacing="xs", mt=4, className="ml-param-grid"),
            html.Details([
                html.Summary("Контроль переобучения"),
                dmc.SimpleGrid([
                    _field("Early stopping · 0 = выкл", _number(
                        "ml-early-stopping", 80, min=0, max=5000, step=10,
                    )),
                    html.Div(dmc.Switch(
                        id="ml-use-best-iteration",
                        label="Финал по лучшей итерации",
                        checked=True, size="xs", styles=COMPACT_SWITCH_STYLES,
                    ), className="ml-overfit-switch"),
                ], cols=2, spacing="xs", mt=4),
                dmc.Text(
                    "Early stopping прекращает рост деревьев без улучшения validation. "
                    "Финальная модель использует среднюю лучшую итерацию по проверкам.",
                    size="8px", c="dimmed", mt=3,
                ),
            ], open=True, className="ml-advanced ml-overfit-controls"),
            html.Details([
                html.Summary("Расширенные параметры"),
                dmc.SimpleGrid([
                    _field("Функция потерь", dmc.Select(
                        id="ml-loss", value="RMSE", allowDeselect=False, size="xs",
                        data=[value for value in ["RMSE", "MAE", "MAPE", "Quantile"]],
                    )),
                    html.Div(_field("Баланс классов", dmc.Select(
                        id="ml-class-weights", value="none", allowDeselect=False,
                        size="xs", data=[
                            {"label": "Без балансировки", "value": "none"},
                            {"label": "Автоматически", "value": "balanced"},
                        ],
                    )), id="ml-class-weights-wrap", style={"display": "none"}),
                    _field("Random strength", _number("ml-random-strength", 1.0, min=0, step=0.1)),
                    _field("Bagging temperature", _number("ml-bagging-temperature", 1.0, min=0, step=0.1)),
                    _field("Random seed", _number("ml-random-seed", 42, min=0, step=1)),
                ], cols=3, spacing="xs", mt="xs", className="ml-advanced-grid"),
            ], className="ml-advanced"),

            dmc.Divider(label="Выходные каналы", labelPosition="left", my="xs"),
            dmc.SimpleGrid([
                _field("Имя прогноза", dmc.TextInput(
                    id="ml-prediction-column", value="Прогноз CatBoost", size="xs",
                )),
                html.Div(dmc.Switch(
                    id="ml-include-residual", label="Добавить остаток", checked=True,
                    size="xs", styles=COMPACT_SWITCH_STYLES,
                ), id="ml-residual-wrap"),
                html.Div(dmc.Switch(
                    id="ml-include-confidence", label="Добавить уверенность",
                    checked=True, size="xs", styles=COMPACT_SWITCH_STYLES,
                ), id="ml-confidence-wrap", style={"display": "none"}),
                dmc.Switch(
                    id="ml-compute-shap", label="Рассчитать SHAP", checked=True,
                    size="xs", styles=COMPACT_SWITCH_STYLES,
                ),
            ], cols=4, spacing="xs", className="ml-output-channel-grid"),
            dmc.SimpleGrid([
                dmc.Button("Обучить модель", id="ml-run", size="xs", fullWidth=True),
                dmc.Button(
                    "Отменить", id="ml-cancel", size="xs", fullWidth=True,
                    variant="light", color="red", disabled=True,
                ),
            ], cols=2, spacing="xs", mt=6),
            dmc.Progress(
                id="ml-job-progress", value=0, size="xs", animated=True,
                striped=True, mt=6, className="ml-job-progress",
            ),
            dmc.Text(id="ml-job-message", size="9px", c="dimmed", mt=3),
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
        p="xs", withBorder=True, shadow="xs", className="ml-controls",
    )

    metrics = html.Div([
        _metric("MAE", "ml-metric-mae", "ниже — лучше"),
        _metric("RMSE", "ml-metric-rmse", "ниже — лучше"),
        _metric("MAPE", "ml-metric-mape", "% · нули исключены"),
        _metric("R²", "ml-metric-r2", "выше — лучше"),
        _metric("Baseline MAE", "ml-metric-baseline", "прогноз средним"),
        _metric("Train ↔ validation", "ml-metric-gap", "контроль переобучения"),
    ], className="ml-metrics-grid")

    empty_graph = lambda graph_id: dcc.Graph(id=graph_id, config=GRAPH_CONFIG)
    results = dmc.Paper([
        dmc.Group([
            html.Div([
                dmc.Text("Результаты модели", fw=700, size="xs"),
                dmc.Text(id="ml-evaluation-note", size="9px", c="dimmed"),
            ]),
            dmc.Group([
                dmc.Badge(
                    "Нет оценки", id="ml-overfit-status", size="sm",
                    variant="light", color="gray",
                ),
                dmc.Text(id="ml-shap-note", size="9px", c="dimmed"),
            ], gap=6),
        ], justify="space-between"),
        dmc.Tabs([
            dmc.TabsList([
                dmc.TabsTab("Прогноз", value="prediction"),
                dmc.TabsTab("Обучение", value="learning"),
                dmc.TabsTab("Важность", value="importance"),
                dmc.TabsTab("SHAP", value="shap"),
                dmc.TabsTab("Диагностика", value="diagnostics"),
                dmc.TabsTab("Автоподбор", value="tuning"),
                dmc.TabsTab("Таблица", value="table"),
                dmc.TabsTab("Протокол", value="log"),
            ]),
            dmc.TabsPanel(empty_graph("ml-prediction-graph"), value="prediction"),
            dmc.TabsPanel(empty_graph("ml-learning-graph"), value="learning"),
            dmc.TabsPanel(empty_graph("ml-importance-graph"), value="importance"),
            dmc.TabsPanel(empty_graph("ml-shap-graph"), value="shap"),
            dmc.TabsPanel(empty_graph("ml-diagnostics-graph"), value="diagnostics"),
            dmc.TabsPanel(empty_graph("ml-tuning-graph"), value="tuning"),
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


def create_random_forest_workspace():
    routing = dmc.Paper([
        dmc.Group([
            dmc.Text("Random Forest · регрессия", id="rf-workspace-title",
                     fw=700, size="sm"),
            dmc.Group([
                dmc.Badge("Нет данных", id="rf-dataset-badge", size="sm",
                          variant="light", color="gray"),
                dmc.Badge("Ожидает запуска", id="rf-run-status", size="sm",
                          variant="light", color="gray"),
            ], gap=6),
        ], justify="space-between", align="center"),
        dmc.SimpleGrid([
            _field("Задача", dmc.SegmentedControl(
                id="rf-task", value="regression", size="xs", fullWidth=True,
                data=[
                    {"label": "Регрессия", "value": "regression"},
                    {"label": "Классификация", "value": "classification"},
                ],
            )),
            _field("Входной dataset", dmc.Select(
                id="rf-input-dataset", data=[], searchable=True,
                allowDeselect=False, size="xs",
            )),
            _field("Слой данных", dmc.SegmentedControl(
                id="rf-input-scope", value="base", size="xs", fullWidth=True,
                data=[
                    {"label": "До фильтров", "value": "base"},
                    {"label": "После фильтров", "value": "filtered"},
                ],
            )),
            _field("Новый dataset", dmc.TextInput(
                id="rf-output-name", placeholder="RandomForest_До фильтров_1", size="xs",
            )),
        ], cols=4, spacing="xs", mt=6, className="ml-routing-grid"),
    ], p="xs", withBorder=True, shadow="xs", className="ml-routing")

    controls = dmc.Paper([
        dmc.Group([
            dmc.Text("Данные модели", fw=700, size="11px"),
            dmc.Button("Все числовые", id="rf-select-numeric", size="compact-xs",
                       variant="subtle"),
        ], justify="space-between"),
        dmc.SimpleGrid([
            _field("Целевой канал · число", dmc.Select(
                id="rf-target", data=[], searchable=True, clearable=True, size="xs",
            ), label_id="rf-target-label"),
            _field("ID / подпись · необязательно", dmc.Select(
                id="rf-id-column", data=[], searchable=True, clearable=True, size="xs",
            )),
        ], cols=2, spacing="xs", mt=4),
        html.Div([
            dmc.Group([
                dmc.Text("Признаки", size="9px", fw=650, c="dimmed"),
                dmc.Text("числа и категории", size="8px", c="dimmed"),
            ], justify="space-between"),
            dmc.MultiSelect(
                id="rf-features", data=[], value=[], searchable=True, clearable=True,
                nothingFoundMessage="Ничего не найдено", maxDropdownHeight=300,
                comboboxProps={"shadow": "md"}, size="xs",
            ),
        ], id="rf-features-drop", className="ml-features-drop",
           **{"data-drop-target": "rf-features", "data-drop-mode": "append",
              "data-current-value": "[]"}),

        dmc.Divider(label="Проверка качества", labelPosition="left", my="xs"),
        dmc.Select(
            id="rf-method", value="split", size="xs", allowDeselect=False,
            data=[
                {"label": "Train / test · случайное", "value": "split"},
                {"label": "KFold · случайные фолды", "value": "cv"},
                {"label": "GroupKFold · группы не смешиваются", "value": "group_cv"},
                {"label": "TimeSeriesSplit · прошлое → будущее", "value": "time_cv"},
            ],
        ),
        dmc.SimpleGrid([
            _field("Доля test", _number("rf-test-size", .2, min=.05, max=.5, step=.05)),
            _field("Фолды", _number("rf-folds", 5, min=2, max=20, step=1)),
        ], cols=2, spacing="xs", mt=4),
        dmc.SimpleGrid([
            _field("Канал группы", dmc.Select(
                id="rf-group-column", data=[], searchable=True, clearable=True,
                size="xs", disabled=True,
            )),
            _field("Время / порядок", dmc.Select(
                id="rf-time-column", data=[], searchable=True, clearable=True,
                size="xs", disabled=True,
            )),
        ], cols=2, spacing="xs", mt=4),
        dmc.Text(id="rf-validation-hint", size="9px", c="dimmed", mt=4),

        dmc.Divider(label="Параметры Random Forest", labelPosition="left", my="xs"),
        dmc.SimpleGrid([
            _field("Пресет", dmc.Select(
                id="rf-preset", value="balanced", allowDeselect=False, size="xs",
                data=[
                    {"label": "Быстрый черновик", "value": "draft"},
                    {"label": "Баланс", "value": "balanced"},
                    {"label": "Высокое качество", "value": "quality"},
                    {"label": "Вручную", "value": "custom"},
                ],
            )),
            _field("Вычислитель", dmc.TextInput(
                value="CPU · все ядра", size="xs", disabled=True,
            )),
        ], cols=2, spacing="xs", mt=4),
        dmc.SimpleGrid([
            _field("Деревья", _number("rf-n-estimators", 600, min=10, max=5000, step=50)),
            _field("Макс. глубина · 0 = ∞", _number("rf-max-depth", 0, min=0, max=200, step=1)),
            _field("Мин. строк в листе", _number("rf-min-samples-leaf", 2, min=1, step=1)),
            _field("Мин. строк для разбиения", _number("rf-min-samples-split", 2, min=2, step=1)),
        ], cols=4, spacing="xs", mt=4, className="ml-param-grid"),
        html.Details([
            html.Summary("Контроль переобучения"),
            dmc.SimpleGrid([
                _field("Признаков на дерево", dmc.Select(
                    id="rf-max-features", value="sqrt", allowDeselect=False, size="xs",
                    data=[
                        {"label": "√ числа признаков", "value": "sqrt"},
                        {"label": "log₂ признаков", "value": "log2"},
                        {"label": "50% признаков", "value": "0.5"},
                        {"label": "80% признаков", "value": "0.8"},
                        {"label": "Все признаки", "value": "all"},
                    ],
                )),
                _field("Доля строк на дерево", _number(
                    "rf-max-samples", .85, min=.1, max=1, step=.05,
                )),
                html.Div(dmc.Switch(
                    id="rf-bootstrap", label="Bootstrap", checked=True,
                    size="xs", styles=COMPACT_SWITCH_STYLES,
                ), className="ml-overfit-switch"),
                html.Div(dmc.Switch(
                    id="rf-oob-score", label="OOB-оценка", checked=True,
                    size="xs", styles=COMPACT_SWITCH_STYLES,
                ), className="ml-overfit-switch"),
            ], cols=2, spacing="xs", mt=4),
            dmc.Text(
                "Глубина и размер листа ограничивают сложность. Bootstrap и случайный "
                "набор признаков уменьшают сходство деревьев; OOB даёт дополнительную "
                "оценку без отдельного test.",
                size="8px", c="dimmed", mt=3,
            ),
        ], open=True, className="ml-advanced ml-overfit-controls"),
        html.Details([
            html.Summary("Расширенные параметры"),
            dmc.SimpleGrid([
                _field("Критерий", dmc.Select(
                    id="rf-criterion", value="squared_error", allowDeselect=False,
                    size="xs", data=[
                        {"label": "Squared error", "value": "squared_error"},
                        {"label": "Absolute error", "value": "absolute_error"},
                        {"label": "Poisson", "value": "poisson"},
                    ],
                )),
                html.Div(_field("Баланс классов", dmc.Select(
                    id="rf-class-weight", value="balanced", allowDeselect=False,
                    size="xs", data=[
                        {"label": "Без балансировки", "value": "none"},
                        {"label": "Balanced", "value": "balanced"},
                        {"label": "Balanced subsample", "value": "balanced_subsample"},
                    ],
                )), id="rf-class-weight-wrap", style={"display": "none"}),
                _field("Random seed", _number("rf-random-seed", 42, min=0, step=1)),
            ], cols=3, spacing="xs", mt="xs", className="ml-advanced-grid"),
        ], className="ml-advanced"),

        dmc.Divider(label="Выходные каналы", labelPosition="left", my="xs"),
        dmc.SimpleGrid([
            _field("Имя прогноза", dmc.TextInput(
                id="rf-prediction-column", value="Прогноз Random Forest", size="xs",
            )),
            html.Div(dmc.Switch(
                id="rf-include-residual", label="Добавить остаток", checked=True,
                size="xs", styles=COMPACT_SWITCH_STYLES,
            ), id="rf-residual-wrap"),
            html.Div(dmc.Switch(
                id="rf-include-confidence", label="Добавить уверенность", checked=True,
                size="xs", styles=COMPACT_SWITCH_STYLES,
            ), id="rf-confidence-wrap", style={"display": "none"}),
        ], cols=3, spacing="xs"),
        dmc.SimpleGrid([
            dmc.Button("Обучить модель", id="rf-run", size="xs", fullWidth=True),
            dmc.Button("Отменить", id="rf-cancel", size="xs", fullWidth=True,
                       variant="light", color="red", disabled=True),
        ], cols=2, spacing="xs", mt=6),
        dmc.Progress(id="rf-job-progress", value=0, size="xs", animated=True,
                     striped=True, mt=6, className="ml-job-progress"),
        dmc.Text(id="rf-job-message", size="9px", c="dimmed", mt=3),
        dmc.SimpleGrid([
            dmc.Button("Создать dataset", id="rf-commit", size="xs",
                       variant="light", disabled=True),
            dmc.Button("Выгрузить Excel", id="rf-export-excel", size="xs",
                       variant="light", color="violet", disabled=True),
            dmc.Button("Сохранить модель", id="rf-save-model", size="xs",
                       variant="light", color="grape", disabled=True),
        ], cols=3, spacing="xs", mt=5),
        dmc.Text(id="rf-row-status", size="9px", c="dimmed", mt=4),
    ], p="xs", withBorder=True, shadow="xs", className="ml-controls")

    metrics = html.Div([
        _metric("MAE", "rf-metric-mae", "ниже — лучше"),
        _metric("RMSE", "rf-metric-rmse", "ниже — лучше"),
        _metric("R²", "rf-metric-r2", "выше — лучше"),
        _metric("Baseline MAE", "rf-metric-baseline", "прогноз средним"),
        _metric("OOB", "rf-metric-oob", "вне bootstrap-выборки"),
        _metric("Train ↔ validation", "rf-metric-gap", "контроль переобучения"),
    ], className="ml-metrics-grid")
    empty_graph = lambda graph_id: dcc.Graph(id=graph_id, config=GRAPH_CONFIG)
    results = dmc.Paper([
        dmc.Group([
            html.Div([
                dmc.Text("Результаты модели", fw=700, size="xs"),
                dmc.Text(id="rf-evaluation-note", size="9px", c="dimmed"),
            ]),
            dmc.Badge("Нет оценки", id="rf-overfit-status", size="sm",
                      variant="light", color="gray"),
        ], justify="space-between"),
        dmc.Tabs([
            dmc.TabsList([
                dmc.TabsTab("Прогноз", value="prediction"),
                dmc.TabsTab("Валидация", value="validation"),
                dmc.TabsTab("Важность", value="importance"),
                dmc.TabsTab("Диагностика", value="diagnostics"),
                dmc.TabsTab("Таблица", value="table"),
                dmc.TabsTab("Протокол", value="log"),
            ]),
            dmc.TabsPanel(empty_graph("rf-prediction-graph"), value="prediction"),
            dmc.TabsPanel(empty_graph("rf-validation-graph"), value="validation"),
            dmc.TabsPanel(empty_graph("rf-importance-graph"), value="importance"),
            dmc.TabsPanel(empty_graph("rf-diagnostics-graph"), value="diagnostics"),
            dmc.TabsPanel(dash_table.DataTable(
                id="rf-prediction-table", data=[], columns=[], page_size=20,
                sort_action="native", filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": "11px", "padding": "5px", "maxWidth": "180px",
                            "overflow": "hidden", "textOverflow": "ellipsis"},
            ), value="table"),
            dmc.TabsPanel(html.Pre(id="rf-log", className="ml-log"), value="log"),
        ], value="prediction", mt="xs", keepMounted=True),
    ], p="xs", withBorder=True, shadow="xs", className="ml-results")
    return html.Div([
        routing,
        html.Div([controls, html.Div([metrics, results], className="ml-output")],
                 className="ml-main-grid"),
    ], className="ml-catboost-workspace ml-rf-workspace")


def create_neural_network_workspace():
    routing = dmc.Paper([
        dmc.Group([
            dmc.Text("Neural Network · регрессия", id="nn-workspace-title",
                     fw=700, size="sm"),
            dmc.Group([
                dmc.Badge("Нет данных", id="nn-dataset-badge", size="sm",
                          variant="light", color="gray"),
                dmc.Badge("Ожидает запуска", id="nn-run-status", size="sm",
                          variant="light", color="gray"),
            ], gap=6),
        ], justify="space-between", align="center"),
        dmc.SimpleGrid([
            _field("Задача", dmc.SegmentedControl(
                id="nn-task", value="regression", size="xs", fullWidth=True,
                data=[
                    {"label": "Регрессия", "value": "regression"},
                    {"label": "Классификация", "value": "classification"},
                ],
            )),
            _field("Входной dataset", dmc.Select(
                id="nn-input-dataset", data=[], searchable=True,
                allowDeselect=False, size="xs",
            )),
            _field("Слой данных", dmc.SegmentedControl(
                id="nn-input-scope", value="base", size="xs", fullWidth=True,
                data=[
                    {"label": "До фильтров", "value": "base"},
                    {"label": "После фильтров", "value": "filtered"},
                ],
            )),
            _field("Новый dataset", dmc.TextInput(
                id="nn-output-name", placeholder="NeuralNetwork_До фильтров_1", size="xs",
            )),
        ], cols=4, spacing="xs", mt=6, className="ml-routing-grid"),
    ], p="xs", withBorder=True, shadow="xs", className="ml-routing")

    controls = dmc.Paper([
        dmc.Group([
            dmc.Text("Данные модели", fw=700, size="11px"),
            dmc.Button("Все числовые", id="nn-select-numeric", size="compact-xs",
                       variant="subtle"),
        ], justify="space-between"),
        dmc.SimpleGrid([
            _field("Целевой канал · число", dmc.Select(
                id="nn-target", data=[], searchable=True, clearable=True, size="xs",
            ), label_id="nn-target-label"),
            _field("ID / подпись · необязательно", dmc.Select(
                id="nn-id-column", data=[], searchable=True, clearable=True, size="xs",
            )),
        ], cols=2, spacing="xs", mt=4),
        html.Div([
            dmc.Group([
                dmc.Text("Признаки", size="9px", fw=650, c="dimmed"),
                dmc.Text("числа и категории", size="8px", c="dimmed"),
            ], justify="space-between"),
            dmc.MultiSelect(
                id="nn-features", data=[], value=[], searchable=True, clearable=True,
                nothingFoundMessage="Ничего не найдено", maxDropdownHeight=300,
                comboboxProps={"shadow": "md"}, size="xs",
            ),
        ], id="nn-features-drop", className="ml-features-drop",
           **{"data-drop-target": "nn-features", "data-drop-mode": "append",
              "data-current-value": "[]"}),

        dmc.Divider(label="Проверка качества", labelPosition="left", my="xs"),
        dmc.Select(
            id="nn-method", value="split", size="xs", allowDeselect=False,
            data=[
                {"label": "Train / test · случайное", "value": "split"},
                {"label": "KFold · случайные фолды", "value": "cv"},
                {"label": "GroupKFold · группы не смешиваются", "value": "group_cv"},
                {"label": "TimeSeriesSplit · прошлое → будущее", "value": "time_cv"},
            ],
        ),
        dmc.SimpleGrid([
            _field("Доля test", _number("nn-test-size", .2, min=.05, max=.5, step=.05)),
            _field("Фолды", _number("nn-folds", 5, min=2, max=20, step=1)),
        ], cols=2, spacing="xs", mt=4),
        dmc.SimpleGrid([
            _field("Канал группы", dmc.Select(
                id="nn-group-column", data=[], searchable=True, clearable=True,
                size="xs", disabled=True,
            )),
            _field("Время / порядок", dmc.Select(
                id="nn-time-column", data=[], searchable=True, clearable=True,
                size="xs", disabled=True,
            )),
        ], cols=2, spacing="xs", mt=4),
        dmc.Text(id="nn-validation-hint", size="9px", c="dimmed", mt=4),

        dmc.Divider(label="Параметры Neural Network", labelPosition="left", my="xs"),
        dmc.SimpleGrid([
            _field("Пресет", dmc.Select(
                id="nn-preset", value="balanced", allowDeselect=False, size="xs",
                data=[
                    {"label": "Быстрый черновик", "value": "draft"},
                    {"label": "Баланс", "value": "balanced"},
                    {"label": "Глубокая сеть", "value": "deep"},
                    {"label": "Вручную", "value": "custom"},
                ],
            )),
            _field("Движок", dmc.Select(
                id="nn-engine", value="pytorch", allowDeselect=False, size="xs",
                data=[
                    {"label": "PyTorch", "value": "pytorch"},
                    {"label": "sklearn MLP", "value": "sklearn"},
                ],
            )),
            _field("Вычислитель", dmc.Select(
                id="nn-compute-device", value="auto", allowDeselect=False, size="xs",
                data=[
                    {"label": "Авто", "value": "auto"},
                    {"label": "CPU", "value": "cpu"},
                    {"label": "GPU · Metal/MPS", "value": "mps"},
                ],
            )),
        ], cols=3, spacing="xs", mt=4),
        dmc.Text(id="nn-compute-hint", size="8px", c="dimmed", mt=2),
        dmc.SimpleGrid([
            _field("Скрытые слои", dmc.TextInput(
                id="nn-hidden-layers", value="64, 32", size="xs",
                placeholder="64, 32",
            )),
            _field("Активация", dmc.Select(
                id="nn-activation", value="relu", allowDeselect=False, size="xs",
                data=[
                    {"label": "ReLU", "value": "relu"},
                    {"label": "Tanh", "value": "tanh"},
                    {"label": "Logistic", "value": "logistic"},
                    {"label": "Linear", "value": "identity"},
                ],
            )),
            _field("Оптимизатор", dmc.Select(
                id="nn-solver", value="adam", allowDeselect=False, size="xs",
                data=[
                    {"label": "Adam", "value": "adam"},
                    {"label": "SGD", "value": "sgd"},
                    {"label": "L-BFGS · малые данные", "value": "lbfgs"},
                ],
            )),
            _field("Эпохи", _number("nn-max-iter", 500, min=20, max=5000, step=50)),
        ], cols=4, spacing="xs", mt=4, className="ml-param-grid"),
        dmc.SimpleGrid([
            _field("Learning rate", _number(
                "nn-learning-rate", .001, min=.000001, max=1, step=.0005,
            )),
            _field("L2 · alpha", _number(
                "nn-alpha", .0001, min=0, max=100, step=.0001,
            )),
            _field("Batch", _number("nn-batch-size", 64, min=1, max=65536, step=16)),
        ], cols=3, spacing="xs", mt=4, className="ml-param-grid"),
        html.Details([
            html.Summary("Контроль переобучения"),
            dmc.SimpleGrid([
                html.Div(dmc.Switch(
                    id="nn-early-stopping", label="Early stopping", checked=True,
                    size="xs", styles=COMPACT_SWITCH_STYLES,
                ), className="ml-overfit-switch"),
                _field("Patience · эпох", _number(
                    "nn-patience", 30, min=2, max=500, step=5,
                )),
                _field("Validation внутри train", _number(
                    "nn-validation-fraction", .15, min=.05, max=.4, step=.05,
                )),
                _field("Tolerance", _number(
                    "nn-tolerance", .0001, min=.0000001, max=1, step=.0001,
                )),
            ], cols=2, spacing="xs", mt=4),
            dmc.Text(
                "Внешний test/OOF оценивает качество. Внутренняя validation только "
                "останавливает обучение, когда улучшение прекращается.",
                size="8px", c="dimmed", mt=3,
            ),
        ], open=True, className="ml-advanced ml-overfit-controls"),
        html.Details([
            html.Summary("Расширенные параметры"),
            dmc.SimpleGrid([
                _field("Мин. частота категории", _number(
                    "nn-min-category-frequency", 2, min=1, max=10000, step=1,
                )),
                _field("Permutation repeats · 0 = выкл", _number(
                    "nn-permutation-repeats", 3, min=0, max=20, step=1,
                )),
                html.Div(_field("Баланс классов", dmc.Select(
                    id="nn-class-balance", value="balanced", allowDeselect=False,
                    size="xs", data=[
                        {"label": "Без балансировки", "value": "none"},
                        {"label": "Balanced", "value": "balanced"},
                    ],
                )), id="nn-class-balance-wrap", style={"display": "none"}),
                _field("Random seed", _number("nn-random-seed", 42, min=0, step=1)),
            ], cols=2, spacing="xs", mt="xs", className="ml-advanced-grid"),
        ], className="ml-advanced"),

        dmc.Divider(label="Выходные каналы", labelPosition="left", my="xs"),
        dmc.SimpleGrid([
            _field("Имя прогноза", dmc.TextInput(
                id="nn-prediction-column", value="Прогноз Neural Network", size="xs",
            )),
            html.Div(dmc.Switch(
                id="nn-include-residual", label="Добавить остаток", checked=True,
                size="xs", styles=COMPACT_SWITCH_STYLES,
            ), id="nn-residual-wrap"),
            html.Div(dmc.Switch(
                id="nn-include-confidence", label="Добавить уверенность", checked=True,
                size="xs", styles=COMPACT_SWITCH_STYLES,
            ), id="nn-confidence-wrap", style={"display": "none"}),
        ], cols=3, spacing="xs"),
        dmc.SimpleGrid([
            dmc.Button("Обучить модель", id="nn-run", size="xs", fullWidth=True),
            dmc.Button("Отменить", id="nn-cancel", size="xs", fullWidth=True,
                       variant="light", color="red", disabled=True),
        ], cols=2, spacing="xs", mt=6),
        dmc.Progress(id="nn-job-progress", value=0, size="xs", animated=True,
                     striped=True, mt=6, className="ml-job-progress"),
        dmc.Text(id="nn-job-message", size="9px", c="dimmed", mt=3),
        dmc.SimpleGrid([
            dmc.Button("Создать dataset", id="nn-commit", size="xs",
                       variant="light", disabled=True),
            dmc.Button("Выгрузить Excel", id="nn-export-excel", size="xs",
                       variant="light", color="violet", disabled=True),
            dmc.Button("Сохранить модель", id="nn-save-model", size="xs",
                       variant="light", color="grape", disabled=True),
        ], cols=3, spacing="xs", mt=5),
        dmc.Text(id="nn-row-status", size="9px", c="dimmed", mt=4),
    ], p="xs", withBorder=True, shadow="xs", className="ml-controls")

    metrics = html.Div([
        _metric("MAE", "nn-metric-mae", "ниже — лучше"),
        _metric("RMSE", "nn-metric-rmse", "ниже — лучше"),
        _metric("R²", "nn-metric-r2", "выше — лучше"),
        _metric("Baseline MAE", "nn-metric-baseline", "прогноз средним"),
        _metric("Эпохи", "nn-metric-epochs", "финальная модель"),
        _metric("Train ↔ validation", "nn-metric-gap", "контроль переобучения"),
    ], className="ml-metrics-grid")
    empty_graph = lambda graph_id: dcc.Graph(id=graph_id, config=GRAPH_CONFIG)
    results = dmc.Paper([
        dmc.Group([
            html.Div([
                dmc.Text("Результаты модели", fw=700, size="xs"),
                dmc.Text(id="nn-evaluation-note", size="9px", c="dimmed"),
            ]),
            dmc.Badge("Нет оценки", id="nn-overfit-status", size="sm",
                      variant="light", color="gray"),
        ], justify="space-between"),
        dmc.Tabs([
            dmc.TabsList([
                dmc.TabsTab("Прогноз", value="prediction"),
                dmc.TabsTab("Обучение", value="learning"),
                dmc.TabsTab("Валидация", value="validation"),
                dmc.TabsTab("Важность", value="importance"),
                dmc.TabsTab("Диагностика", value="diagnostics"),
                dmc.TabsTab("Таблица", value="table"),
                dmc.TabsTab("Протокол", value="log"),
            ]),
            dmc.TabsPanel(empty_graph("nn-prediction-graph"), value="prediction"),
            dmc.TabsPanel(empty_graph("nn-learning-graph"), value="learning"),
            dmc.TabsPanel(empty_graph("nn-validation-graph"), value="validation"),
            dmc.TabsPanel(empty_graph("nn-importance-graph"), value="importance"),
            dmc.TabsPanel(empty_graph("nn-diagnostics-graph"), value="diagnostics"),
            dmc.TabsPanel(dash_table.DataTable(
                id="nn-prediction-table", data=[], columns=[], page_size=20,
                sort_action="native", filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": "11px", "padding": "5px", "maxWidth": "180px",
                            "overflow": "hidden", "textOverflow": "ellipsis"},
            ), value="table"),
            dmc.TabsPanel(html.Pre(id="nn-log", className="ml-log"), value="log"),
        ], value="prediction", mt="xs", keepMounted=True),
    ], p="xs", withBorder=True, shadow="xs", className="ml-results")
    return html.Div([
        routing,
        html.Div([controls, html.Div([metrics, results], className="ml-output")],
                 className="ml-main-grid"),
    ], className="ml-catboost-workspace ml-nn-workspace")


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


def create_data_profile_workspace():
    controls = dmc.Paper([
        dmc.Group([
            dmc.Text("Паспорт данных", fw=700, size="sm"),
            dmc.Badge(
                "Нет данных", id="ml-profile-status", size="sm",
                variant="light", color="gray",
            ),
        ], justify="space-between", align="center"),
        dmc.SimpleGrid([
            _field("Dataset", dmc.Select(
                id="ml-profile-dataset", data=[], searchable=True,
                allowDeselect=False, size="xs",
            )),
            _field("Слой данных", dmc.SegmentedControl(
                id="ml-profile-scope", value="base", size="xs", fullWidth=True,
                data=[
                    {"label": "До фильтров", "value": "base"},
                    {"label": "После фильтров", "value": "filtered"},
                ],
            )),
            _field("Целевой канал", dmc.Select(
                id="ml-profile-target", data=[], searchable=True,
                clearable=True, size="xs",
            )),
            _field("Интерпретация цели", dmc.SegmentedControl(
                id="ml-profile-task", value="auto", size="xs", fullWidth=True,
                data=[
                    {"label": "Авто", "value": "auto"},
                    {"label": "Регрессия", "value": "regression"},
                    {"label": "Классы", "value": "classification"},
                ],
            )),
        ], cols=4, spacing="xs", mt=5, className="ml-profile-routing-grid"),
    ], p="xs", withBorder=True, shadow="xs", className="ml-profile-routing")

    metrics = html.Div([
        _profile_metric("Строки", "ml-profile-rows"),
        _profile_metric("Каналы", "ml-profile-columns"),
        _profile_metric("Числовые", "ml-profile-numeric"),
        _profile_metric("Категории", "ml-profile-categorical"),
        _profile_metric("Пропуски", "ml-profile-missing"),
        _profile_metric("Память", "ml-profile-memory"),
    ], className="ml-profile-metrics")

    insights = html.Div([
        dmc.Paper([
            dmc.Text("Что требует внимания", fw=700, size="xs", mb=5),
            html.Div(id="ml-profile-issues", className="ml-profile-list"),
        ], p="xs", withBorder=True, shadow="xs"),
        dmc.Paper([
            dmc.Text("Целевой канал", fw=700, size="xs", mb=5),
            html.Div(id="ml-profile-target-summary", className="ml-profile-list"),
        ], p="xs", withBorder=True, shadow="xs"),
        dmc.Paper([
            dmc.Text("Рекомендации", fw=700, size="xs", mb=5),
            html.Div(id="ml-profile-recommendations", className="ml-profile-list"),
        ], p="xs", withBorder=True, shadow="xs"),
    ], className="ml-profile-insights")

    charts = html.Div([
        dmc.Paper([
            dmc.Text("Пропуски · топ 20", fw=700, size="xs"),
            dcc.Graph(
                id="ml-profile-missing-graph", config=GRAPH_CONFIG,
                style={"height": "270px", "maxHeight": "270px"},
            ),
        ], p="xs", withBorder=True, shadow="xs", className="ml-profile-chart-card"),
        dmc.Paper([
            dmc.Text("Распределение цели", fw=700, size="xs"),
            dcc.Graph(
                id="ml-profile-target-graph", config=GRAPH_CONFIG,
                style={"height": "270px", "maxHeight": "270px"},
            ),
        ], p="xs", withBorder=True, shadow="xs", className="ml-profile-chart-card"),
    ], className="ml-profile-charts")

    table = dmc.Paper([
        dmc.Group([
            dmc.Text("Каналы и сигналы качества", fw=700, size="xs"),
            dmc.Text(id="ml-profile-sample-note", size="8px", c="dimmed"),
        ], justify="space-between"),
        dash_table.DataTable(
            id="ml-profile-table", data=[], columns=[], page_size=18,
            sort_action="native", filter_action="native",
            style_table={"overflowX": "auto", "marginTop": "5px"},
            style_cell={
                "fontSize": "9px", "padding": "4px", "maxWidth": "230px",
                "overflow": "hidden", "textOverflow": "ellipsis",
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": "{Сигналы} contains 'утечка'"},
                    "backgroundColor": "#fff5f5", "color": "#c92a2a",
                },
                {
                    "if": {"filter_query": "{Сигналы} contains 'пропусков'"},
                    "backgroundColor": "#fff9db",
                },
            ],
        ),
    ], p="xs", withBorder=True, shadow="xs", className="ml-profile-table")

    return html.Div([
        controls,
        dcc.Loading(
            html.Div([metrics, insights, charts, table], className="ml-profile-content"),
            type="dot", color="#7950f2",
        ),
        dcc.Store(id="ml-profile-analysis"),
    ], className="ml-page-body ml-data-profile")


def create_experiments_workspace():
    return html.Div([
        dmc.Paper([
            dmc.Group([
                html.Div([
                    dmc.Text("Эксперименты", fw=700, size="sm"),
                ]),
                dmc.Group([
                    dmc.Badge("0 запусков", id="ml-history-count", variant="light", color="gray"),
                    dmc.Badge("Лучший MAE: —", id="ml-history-best", variant="light", color="green"),
                ], gap=6),
            ], justify="space-between"),
        ], p="xs", withBorder=True, shadow="xs"),
        html.Div([
            dmc.Paper([
                dmc.Text("Сравнение MAE", id="ml-history-chart-title",
                         fw=700, size="xs", mb=4),
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
                dmc.Text("ML Studio", fw=750, size="sm", className="ml-shell-title"),
                html.Nav([
                    _subnav_link("Паспорт", "/ml/data-profile", "DP"),
                    _subnav_link("Эксперименты", "/ml/experiments", "◫"),
                    _subnav_link("CatBoost", "/ml/catboost", "CB"),
                    _subnav_link("Random Forest", "/ml/random-forest", "RF"),
                    _subnav_link("Нейросети", "/ml/neural-networks", "NN"),
                ], className="ml-subnav"),
                dmc.Badge("Регрессия", id="ml-shell-task-badge",
                          variant="dot", color="violet"),
            ], gap="xs", wrap="nowrap", align="center"),
        ], p=6, withBorder=True, shadow="xs", className="ml-shell-header"),
        html.Div(id="ml-page-data-profile", style={"display": "none"},
                 children=[create_data_profile_workspace()]),
        html.Div(id="ml-page-experiments", children=[create_experiments_workspace()]),
        html.Div(id="ml-page-catboost", style={"display": "none"},
                 children=[create_catboost_workspace()]),
        html.Div(id="ml-page-random-forest", style={"display": "none"},
                 children=[create_random_forest_workspace()]),
        html.Div(id="ml-page-neural-networks", style={"display": "none"},
                 children=[create_neural_network_workspace()]),
        dcc.Store(id="ml-analysis"),
        dcc.Store(id="ml-job-state", data={"status": "idle", "progress": 0}),
        dcc.Store(id="rf-analysis"),
        dcc.Store(id="rf-job-state", data={"status": "idle", "progress": 0}),
        dcc.Store(id="rf-auto-output-name"),
        dcc.Store(id="rf-columns-sync"),
        dcc.Store(id="nn-analysis"),
        dcc.Store(id="nn-job-state", data={"status": "idle", "progress": 0}),
        dcc.Store(id="nn-auto-output-name"),
        dcc.Store(id="nn-columns-sync"),
        dcc.Store(id="ml-auto-output-name"),
        dcc.Store(id="ml-tuning-presets", data={}, storage_type="local"),
        dcc.Store(id="ml-columns-sync"),
        dcc.Store(id="ml-experiment-history", data=[], storage_type="local"),
        dcc.Interval(id="ml-job-poll", interval=650, disabled=True, n_intervals=0),
        dcc.Interval(id="rf-job-poll", interval=650, disabled=True, n_intervals=0),
        dcc.Interval(id="nn-job-poll", interval=650, disabled=True, n_intervals=0),
    ], className="ml-workspace")
