# -*- coding: utf-8 -*-
"""Dataset-aware clustering laboratory layout."""

from dash import dcc, html
import dash_mantine_components as dmc


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
        className=f"cluster-field {class_name}".strip(),
    )


def _metric_card(title, value_id, hint):
    return dmc.Paper(
        [
            dmc.Text(title, size="9px", c="dimmed"),
            dmc.Text("—", id=value_id, size="lg", fw=750),
            dmc.Text(hint, size="8px", c="dimmed"),
        ],
        p="xs",
        withBorder=True,
        className="cluster-metric-card",
    )


def create_clustering_workspace():
    features = dmc.MultiSelect(
        id="cluster-cols",
        data=[],
        value=[],
        searchable=True,
        clearable=True,
        nothingFoundMessage="Ничего не найдено",
        maxDropdownHeight=320,
        comboboxProps={"shadow": "md"},
        size="xs",
    )
    feature_drop = html.Div(
        [
            dmc.Group(
                [
                    dmc.Text("Признаки модели", size="10px", fw=650, c="dimmed"),
                    dmc.Text("DnD из датасета", size="9px", c="dimmed"),
                ],
                justify="space-between",
                gap="xs",
                mb=3,
            ),
            features,
        ],
        id="cluster-columns-drop",
        className="correlation-channel-drop cluster-feature-drop",
        **{
            "data-drop-target": "cluster-cols",
            "data-drop-mode": "append",
            "data-accept-type": "numeric",
            "data-current-value": "[]",
        },
    )

    routing = dmc.Paper(
        [
            dmc.Group(
                [
                    html.Div(
                        [
                            dmc.Text("Лаборатория кластеризации", fw=700, size="sm"),
                            dmc.Text(
                                "Исследуйте модель, затем запишите новые каналы в dataset",
                                size="10px",
                                c="dimmed",
                            ),
                        ]
                    ),
                    dmc.Badge(
                        "Нет данных",
                        id="cluster-dataset-badge",
                        size="sm",
                        variant="light",
                        color="gray",
                    ),
                ],
                justify="space-between",
            ),
            dmc.SimpleGrid(
                [
                    _field(
                        "Входной dataset",
                        dmc.Select(
                            id="cluster-input-dataset",
                            data=[],
                            allowDeselect=False,
                            searchable=True,
                            size="xs",
                        ),
                    ),
                    _field(
                        "Слой данных",
                        dmc.SegmentedControl(
                            id="cluster-input-scope",
                            data=[
                                {"label": "До фильтров", "value": "base"},
                                {"label": "После фильтров", "value": "filtered"},
                            ],
                            value="base",
                            size="xs",
                            fullWidth=True,
                        ),
                    ),
                    _field(
                        "Результат",
                        dmc.SegmentedControl(
                            id="cluster-output-mode",
                            data=[
                                {"label": "В текущий", "value": "current"},
                                {"label": "Новый dataset", "value": "new"},
                            ],
                            value="new",
                            size="xs",
                            fullWidth=True,
                        ),
                    ),
                    _field(
                        "Название результата",
                        dmc.TextInput(
                            id="cluster-output-name",
                            placeholder="Кластеризация_До фильтров_1",
                            size="xs",
                        ),
                    ),
                ],
                cols=4,
                spacing="xs",
                mt="sm",
                className="cluster-routing-grid",
            ),
            dmc.Text(id="cluster-scope-note", size="9px", c="dimmed", mt=4),
        ],
        p="sm",
        withBorder=True,
        shadow="xs",
        className="cluster-routing",
    )

    controls = dmc.Paper(
        [
            dmc.Text("Параметры модели", fw=700, size="xs"),
            html.Div(feature_drop, className="cluster-feature-wrap"),
            dmc.SimpleGrid(
                [
                    _field(
                        "Алгоритм",
                        dmc.Select(
                            id="cluster-algorithm",
                            data=[
                                {"label": "KMeans", "value": "kmeans"},
                                {"label": "MiniBatch KMeans", "value": "minibatch"},
                            ],
                            value="kmeans",
                            allowDeselect=False,
                            size="xs",
                        ),
                    ),
                    _field(
                        "Количество кластеров K",
                        dmc.NumberInput(
                            id="cluster-k",
                            value=4,
                            min=2,
                            max=50,
                            step=1,
                            debounce=True,
                            size="xs",
                        ),
                    ),
                ],
                cols=2,
                spacing="xs",
                mt="xs",
            ),
            dmc.SimpleGrid(
                [
                    _field(
                        "Масштабирование",
                        dmc.Select(
                            id="cluster-scaling",
                            data=[
                                {"label": "Standard", "value": "standard"},
                                {"label": "Robust", "value": "robust"},
                                {"label": "MinMax", "value": "minmax"},
                                {"label": "Без масштаба", "value": "none"},
                            ],
                            value="standard",
                            allowDeselect=False,
                            size="xs",
                        ),
                    ),
                    _field(
                        "Пропуски",
                        dmc.Select(
                            id="cluster-missing-policy",
                            data=[
                                {"label": "Исключить строки", "value": "drop"},
                                {"label": "Заполнить медианой", "value": "median"},
                                {"label": "Заполнить средним", "value": "mean"},
                                {"label": "Заполнить нулём", "value": "zero"},
                            ],
                            value="drop",
                            allowDeselect=False,
                            size="xs",
                        ),
                    ),
                ],
                cols=2,
                spacing="xs",
                mt="xs",
            ),
            _field(
                "Имя канала кластеров",
                dmc.TextInput(id="cluster-output-column", value="Кластер", size="xs"),
                "cluster-output-column",
            ),
            dmc.SimpleGrid(
                [
                    dmc.Switch(id="cluster-include-id", label="ID кластера", checked=True, size="xs"),
                    dmc.Switch(id="cluster-include-distance", label="Расстояние", checked=True, size="xs"),
                    dmc.Switch(id="cluster-include-pca", label="PCA1 / PCA2", checked=True, size="xs"),
                ],
                cols=3,
                spacing="xs",
                mt="xs",
                className="cluster-output-switches",
            ),
            dmc.Group(
                [
                    dmc.Button("Рассчитать", id="cluster-run", size="xs"),
                    dmc.Button(
                        "Записать результат",
                        id="cluster-commit",
                        size="xs",
                        variant="light",
                        disabled=True,
                    ),
                ],
                grow=True,
                mt="sm",
            ),
            dmc.Group(
                [
                    dmc.Badge(
                        "Ожидает расчёта",
                        id="cluster-run-status",
                        color="gray",
                        variant="light",
                        size="sm",
                    ),
                    dmc.Text(id="cluster-row-status", size="9px", c="dimmed"),
                ],
                justify="space-between",
                mt="xs",
                gap="xs",
            ),
        ],
        p="sm",
        withBorder=True,
        shadow="xs",
        className="cluster-controls",
    )

    overview = dmc.Paper(
        [
            dmc.Group(
                [
                    dmc.Text("Проекция PCA", fw=700, size="xs"),
                    dmc.Text(id="cluster-pca-note", size="9px", c="dimmed"),
                ],
                justify="space-between",
            ),
            dcc.Graph(id="cluster-projection-graph", config=GRAPH_CONFIG, className="cluster-main-graph"),
        ],
        p="xs",
        withBorder=True,
        shadow="xs",
        className="cluster-overview",
    )

    metrics = html.Div(
        [
            _metric_card("Silhouette", "cluster-metric-silhouette", "выше — лучше"),
            _metric_card("Davies–Bouldin", "cluster-metric-db", "ниже — лучше"),
            _metric_card("Calinski–Harabasz", "cluster-metric-ch", "выше — лучше"),
            _metric_card("Рекомендованное K", "cluster-metric-k", "по silhouette"),
        ],
        className="cluster-metrics-grid",
    )

    diagnostics = html.Div(
        [
            dmc.Paper(
                [dmc.Text("Подбор K", fw=700, size="xs"), dcc.Graph(id="cluster-diagnostics-graph", config=GRAPH_CONFIG)],
                p="xs", withBorder=True, shadow="xs",
            ),
            dmc.Paper(
                [dmc.Text("Размер кластеров", fw=700, size="xs"), dcc.Graph(id="cluster-sizes-graph", config=GRAPH_CONFIG)],
                p="xs", withBorder=True, shadow="xs",
            ),
            dmc.Paper(
                [dmc.Text("Профили кластеров", fw=700, size="xs"), dcc.Graph(id="cluster-profile-graph", config=GRAPH_CONFIG)],
                p="xs", withBorder=True, shadow="xs", className="cluster-profile-paper",
            ),
        ],
        id="cluster-analysis-section",
        className="cluster-analysis-grid",
        style={"display": "none"},
    )

    return html.Div(
        [
            routing,
            html.Div([controls, overview], className="cluster-main-grid"),
            metrics,
            diagnostics,
            dcc.Store(id="cluster-analysis"),
            dcc.Store(id="cluster-auto-output-name"),
            dcc.Store(id="cluster-columns-sync"),
        ],
        className="cluster-workspace",
    )
