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


def _help_button(target_id, label):
    return html.Button(
        "?",
        type="button",
        className="cluster-help-button",
        title=label,
        **{
            "aria-label": label,
            "data-help-window-target": target_id,
        },
    )


def _card_header(title, help_id, *, extra=None, size="xs"):
    right = []
    if extra is not None:
        right.append(extra)
    right.append(_help_button(help_id, f"Справка: {title}"))
    return dmc.Group(
        [
            dmc.Text(title, fw=700, size=size),
            dmc.Group(right, gap=6, wrap="nowrap"),
        ],
        justify="space-between",
        align="flex-start",
        gap="xs",
        wrap="nowrap",
        className="cluster-card-header",
    )


def _help_window(window_id, title, purpose, sections):
    section_nodes = []
    for heading, items in sections:
        section_nodes.extend([
            dmc.Text(heading, fw=700, size="11px", mt="xs"),
            html.Ul(
                [html.Li(item) for item in items],
                className="cluster-help-list",
            ),
        ])
    return html.Section(
        [
            html.Header(
                [
                    html.Div(
                        [
                            dmc.Text("Справка по кластеризации", size="9px", c="dimmed"),
                            dmc.Text(title, fw=700, size="sm"),
                        ],
                        className="graph-help-window-heading",
                    ),
                    html.Button(
                        "×",
                        type="button",
                        className="graph-help-window-close",
                        title="Закрыть справку",
                        **{"aria-label": "Закрыть справку"},
                    ),
                ],
                className="graph-help-window-header",
            ),
            html.Div(
                [
                    dmc.Text(purpose, size="11px"),
                    *section_nodes,
                ],
                className="graph-help-window-body cluster-help-window-body",
            ),
        ],
        id=window_id,
        className="graph-help-window cluster-help-window",
        role="dialog",
        **{"aria-hidden": "true", "aria-modal": "false"},
    )


CLUSTER_HELP = {
    "routing": (
        "Вход и сохранение результата",
        "Здесь задаётся, какие строки анализировать и куда записать рассчитанные каналы.",
        [
            ("Как настроить", [
                "Входной dataset — исходный набор либо ранее созданный D1, D2 и т.д.",
                "«До фильтров» использует базовый слой выбранного dataset; «После фильтров» — только строки, прошедшие применённые фильтры.",
                "«В текущий» дописывает каналы в выбранный dataset; «Новый dataset» сохраняет отдельный производный набор.",
                "Название нового dataset формируется автоматически, но его можно отредактировать.",
            ]),
            ("Порядок работы", [
                "«Рассчитать» строит модель и графики, не изменяя dataset.",
                "«Записать результат» становится доступной только для актуального расчёта.",
                "После изменения параметров требуется пересчёт — это защищает от записи устаревшего результата.",
            ]),
        ],
    ),
    "parameters": (
        "Параметры модели",
        "Настройки определяют геометрию пространства, в котором алгоритм ищет группы.",
        [
            ("Признаки и алгоритм", [
                "Выберите минимум два числовых признака. Их можно перетащить из панели dataset.",
                "KMeans подходит для компактных примерно сферических кластеров сопоставимого размера.",
                "MiniBatch KMeans быстрее на больших данных, но результат может быть немного менее точным.",
                "K — предполагаемое число кластеров. Начните с рекомендации и сравните соседние значения.",
            ]),
            ("Подготовка данных", [
                "Standard — универсальный вариант: среднее 0, стандартное отклонение 1.",
                "Robust устойчивее к выбросам; MinMax приводит признаки к диапазону 0–1.",
                "«Без масштаба» используйте только когда единицы измерения и диапазоны признаков уже сопоставимы.",
                "Исключение пропусков не удаляет строки из результата: у таких строк кластер останется пустым. Заполнение позволяет кластеризовать все строки.",
            ]),
            ("Выходные каналы", [
                "Основной канал содержит подписи «Кластер 1», «Кластер 2» и пригоден для цвета и фильтров.",
                "ID — числовой номер для дальнейшей обработки и ML.",
                "Расстояние — удалённость строки от ближайшего центра: большие значения могут указывать на пограничные точки или выбросы.",
                "PCA1/PCA2 сохраняют координаты проекции. Существующие каналы не перезаписываются — имя будет дополнено номером.",
            ]),
        ],
    ),
    "pca": (
        "Проекция PCA",
        "Двумерная карта показывает многомерные объекты в координатах двух главных компонент.",
        [
            ("Как читать", [
                "Каждая точка — строка dataset; цвет — назначенный кластер.",
                "Близкие точки имеют похожие значения выбранных признаков в пространстве модели.",
                "Раздельные цветовые облака подтверждают хорошее разделение; сильное перекрытие говорит о слабой структуре или неподходящем K.",
                "Одиночные точки вдали от облаков стоит проверить как возможные выбросы.",
            ]),
            ("Ограничения", [
                "PCA — только проекция: расстояния и разделение могут искажаться при переходе из многих измерений в два.",
                "Процент объяснённой дисперсии показывает, сколько информации удерживают PCA1 и PCA2. Чем он ниже, тем осторожнее интерпретация.",
                "Для производительности на графике показывается не более 6 000 точек, но модель рассчитывается по всем подготовленным строкам.",
            ]),
        ],
    ),
    "silhouette": (
        "Silhouette",
        "Средний силуэт оценивает одновременно компактность кластеров и их отделённость друг от друга.",
        [
            ("Интерпретация", [
                "Диапазон примерно от −1 до 1; больше — лучше.",
                "Значения около 1 означают хорошо отделённые группы, около 0 — перекрывающиеся границы, отрицательные — вероятно неверное назначение части точек.",
                "Ориентир 0,5 часто считается хорошим, но универсального порога нет: учитывайте предметную область и размер выборки.",
                "Сравнивайте варианты на одном dataset с одинаковыми признаками и подготовкой.",
            ]),
        ],
    ),
    "davies": (
        "Davies–Bouldin",
        "Индекс сравнивает внутрикластерный разброс с расстоянием между кластерами.",
        [
            ("Интерпретация", [
                "Минимальное возможное значение — 0; ниже — лучше.",
                "Малое значение означает компактные и хорошо разнесённые кластеры.",
                "У метрики нет универсальной границы качества. Используйте её для сравнения нескольких запусков на одинаковых данных.",
                "Метрика благоприятствует выпуклым кластерам и поэтому особенно естественна для KMeans.",
            ]),
        ],
    ),
    "calinski": (
        "Calinski–Harabasz",
        "Индекс сопоставляет разброс между кластерами с разбросом внутри кластеров.",
        [
            ("Интерпретация", [
                "Выше — лучше: группы дальше друг от друга и компактнее внутри.",
                "Абсолютное значение зависит от числа строк, признаков и масштаба.",
                "Сравнивайте только модели, построенные на одном и том же наборе подготовленных строк и признаков.",
                "Рост индекса сам по себе не гарантирует содержательно полезных кластеров — проверьте профили и размеры.",
            ]),
        ],
    ),
    "recommended_k": (
        "Рекомендованное K",
        "Автоматическая подсказка выбирает K с максимальным средним Silhouette среди проверенных вариантов.",
        [
            ("Как использовать", [
                "Проверяется диапазон K от 2 до 12 либо до допустимого числа строк.",
                "На больших данных диагностика рассчитывается по воспроизводимой выборке, чтобы не блокировать интерфейс.",
                "Рекомендация — стартовая точка, а не окончательный ответ.",
                "Сравните соседние K, форму PCA, размеры и смысл профилей. Иногда чуть меньший K проще и полезнее интерпретировать.",
            ]),
        ],
    ),
    "diagnostics": (
        "Подбор количества кластеров K",
        "График объединяет метод локтя и средний Silhouette для нескольких K.",
        [
            ("Как читать", [
                "Inertia — сумма квадратов расстояний до центров; она всегда уменьшается при росте K.",
                "Ищите «локоть»: точку, после которой уменьшение Inertia заметно замедляется.",
                "Silhouette должен быть как можно выше; пунктир отмечает его лучший вариант.",
                "Хороший выбор обычно находится рядом с локтем и одновременно имеет высокий Silhouette.",
            ]),
            ("Предостережение", [
                "Если кривые не дают выраженного выбора, в данных может не быть естественной кластерной структуры.",
                "Попробуйте изменить признаки, масштабирование, обработать выбросы или проверить другой класс алгоритмов.",
            ]),
        ],
    ),
    "sizes": (
        "Размер кластеров",
        "Столбцы показывают количество строк, назначенных каждому кластеру.",
        [
            ("Интерпретация", [
                "Сопоставимые размеры удобны, но не обязательны: реальная структура данных может быть несбалансированной.",
                "Очень маленький кластер может быть редкой значимой группой, набором выбросов или следствием слишком большого K.",
                "Один доминирующий кластер и несколько малых требуют проверки признаков и масштабирования.",
                "Сопоставьте размеры с профилями и точками на PCA, прежде чем менять K.",
            ]),
        ],
    ),
    "profiles": (
        "Профили кластеров",
        "Тепловая карта объясняет, какими признаками каждый кластер отличается от общей выборки.",
        [
            ("Как читать", [
                "Строки — кластеры, столбцы — выбранные признаки.",
                "Ячейка — среднее стандартизованное отклонение признака внутри кластера; 0 соответствует среднему по анализируемой выборке.",
                "Положительное значение означает уровень выше среднего, отрицательное — ниже; точное число видно в Hover и по цветовой шкале.",
                "Ищите устойчивые сочетания признаков, которые дают кластеру понятное предметное описание.",
            ]),
            ("Важно", [
                "Профили всегда стандартизованы отдельно для интерпретации и сопоставимы между признаками, даже если модель рассчитана с Robust, MinMax или без масштаба.",
                "Номер кластера не означает порядок или качество: «Кластер 1» не лучше «Кластера 2».",
                "Если профили почти одинаковы, разделение может быть слабым или основанным на комбинации, незаметной по средним значениям.",
            ]),
        ],
    ),
}


def _metric_card(title, value_id, hint, help_key):
    help_id = f"cluster-help-{help_key}"
    return dmc.Paper(
        [
            dmc.Group(
                [
                    dmc.Text(title, size="9px", c="dimmed"),
                    _help_button(help_id, f"Справка: {title}"),
                ],
                justify="space-between",
                align="flex-start",
                gap=4,
                wrap="nowrap",
            ),
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
                    dmc.Group(
                        [
                            dmc.Badge(
                                "Нет данных",
                                id="cluster-dataset-badge",
                                size="sm",
                                variant="light",
                                color="gray",
                            ),
                            _help_button(
                                "cluster-help-routing",
                                "Справка: вход и сохранение результата",
                            ),
                        ],
                        gap=6,
                        wrap="nowrap",
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
            _card_header("Параметры модели", "cluster-help-parameters"),
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
            _card_header(
                "Проекция PCA",
                "cluster-help-pca",
                extra=dmc.Text(id="cluster-pca-note", size="9px", c="dimmed"),
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
            _metric_card("Silhouette", "cluster-metric-silhouette", "выше — лучше", "silhouette"),
            _metric_card("Davies–Bouldin", "cluster-metric-db", "ниже — лучше", "davies"),
            _metric_card("Calinski–Harabasz", "cluster-metric-ch", "выше — лучше", "calinski"),
            _metric_card("Рекомендованное K", "cluster-metric-k", "по silhouette", "recommended_k"),
        ],
        className="cluster-metrics-grid",
    )

    diagnostics = html.Div(
        [
            dmc.Paper(
                [
                    _card_header("Подбор K", "cluster-help-diagnostics"),
                    dcc.Graph(id="cluster-diagnostics-graph", config=GRAPH_CONFIG),
                ],
                p="xs", withBorder=True, shadow="xs",
            ),
            dmc.Paper(
                [
                    _card_header("Размер кластеров", "cluster-help-sizes"),
                    dcc.Graph(id="cluster-sizes-graph", config=GRAPH_CONFIG),
                ],
                p="xs", withBorder=True, shadow="xs",
            ),
            dmc.Paper(
                [
                    _card_header("Профили кластеров", "cluster-help-profiles"),
                    dcc.Graph(id="cluster-profile-graph", config=GRAPH_CONFIG),
                ],
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
            *[
                _help_window(
                    f"cluster-help-{key}",
                    content[0],
                    content[1],
                    content[2],
                )
                for key, content in CLUSTER_HELP.items()
            ],
        ],
        className="cluster-workspace",
    )
