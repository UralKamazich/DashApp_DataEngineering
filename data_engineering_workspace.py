# -*- coding: utf-8 -*-
"""Compact feature-engineering workspace."""

from dash import html
import dash_mantine_components as dmc

from components import (
    agg_cols_select,
    agg_exclude_empty_switch,
    agg_exclude_zeros_switch,
    agg_keys_select,
    agg_metrics_select,
    bin_column_select,
    bin_k,
    bin_label_style,
    bin_method,
    txtcopy_cols_select,
    txtcopy_strip_switch,
    txtcopy_suffix_input,
)


def _field(label, control, class_name=""):
    return html.Div(
        [dmc.Text(label, size="10px", fw=650, c="dimmed"), control],
        className=f"de-field {class_name}".strip(),
    )


def _method_help_button(method):
    target_id = f"de-help-{method}"
    return html.Button(
        "?",
        type="button",
        className="cluster-help-button de-method-help-button",
        title="Справка по методике",
        **{
            "aria-label": "Справка по методике",
            "data-help-window-target": target_id,
        },
    )


def _method_help_window(method, title, purpose, sections):
    content = []
    for heading, paragraphs in sections:
        content.extend([
            dmc.Text(heading, fw=700, size="11px", mt="xs"),
            html.Ul(
                [html.Li(paragraph) for paragraph in paragraphs],
                className="cluster-help-list de-help-list",
            ),
        ])
    return html.Section(
        [
            html.Header(
                [
                    html.Div(
                        [
                            dmc.Text("Справка Data Engineering", size="9px", c="dimmed"),
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
                [dmc.Text(purpose, size="11px"), *content],
                className="graph-help-window-body cluster-help-window-body",
            ),
        ],
        id=f"de-help-{method}",
        className="graph-help-window cluster-help-window de-help-window",
        role="dialog",
        **{"aria-hidden": "true", "aria-modal": "false"},
    )


DATA_ENGINEERING_HELP = {
    "binning": (
        "Биннинг числового канала",
        "Биннинг заменяет непрерывный числовой диапазон компактными категориальными интервалами и добавляет новый канал, не изменяя исходный.",
        [
            ("Когда применять", [
                "Для цветовых групп, фильтров, сводных графиков и моделей, которым полезны диапазоны вместо точных значений.",
                "Для инженерной интерпретации: глубины, давления или возраста можно представить понятными классами.",
            ]),
            ("Методы", [
                "По количеству наблюдений (quantile): каждая группа получает примерно одинаковое число строк. Границы интервалов могут быть неравными.",
                "По ширине диапазона: числовой диапазон делится на равные интервалы. Число строк в группах может сильно отличаться.",
                "Если уникальных значений мало, фактическое число групп может оказаться меньше заданного — повторяющиеся границы удаляются.",
            ]),
            ("Метки и пропуски", [
                "Интервалы сохраняют реальные границы; метки «Группа 1…N» удобнее для презентации.",
                "NaN остаются пустыми в новом канале. Исходный числовой канал сохраняется без изменений.",
            ]),
        ],
    ),
    "text": (
        "Текстовая копия каналов",
        "Метод создаёт временно независимые текстовые версии выбранных каналов — полезно, когда числовые значения нужно трактовать как категории или маркеры.",
        [
            ("Как работает", [
                "Каждый выбранный канал приводится к строковому типу и записывается в новый канал с указанным суффиксом.",
                "Если имя уже занято, добавляется номер; существующие данные никогда не перезаписываются.",
                "Опция strip удаляет только пробелы по краям текста, но не меняет внутренние пробелы и регистр.",
            ]),
            ("Когда применять", [
                "Чтобы числовые коды 1, 2, 3 использовались как категории, а не как непрерывная шкала.",
                "Чтобы сохранить исходный тип для расчётов и одновременно получить отдельный канал для подписей, цвета или фильтров.",
            ]),
        ],
    ),
    "aggregate": (
        "Групповые агрегаты",
        "Агрегаты рассчитывают статистики внутри групп и добавляют их обратно к каждой исходной строке. Число строк не меняется.",
        [
            ("Настройка", [
                "Ключи группировки определяют группы, например Месторождение + Скважина.",
                "Каналы — величины, по которым считаются выбранные метрики.",
                "Среднее, медиана, сумма, минимум, максимум и стандартное отклонение применимы к числовым данным; count, nunique и mode также полезны для категорий.",
                "Кумулятивная сумма зависит от текущего порядка строк внутри каждой группы — заранее отсортируйте данные, если порядок имеет смысл.",
            ]),
            ("Нули и NaN", [
                "«Исключить нули» убирает нулевые значения только из расчёта метрики, но не удаляет строки.",
                "Если пустые значения не исключаются, в числовых расчётах они считаются нулём; для категорий учитываются как отдельное пустое значение.",
            ]),
            ("Интерпретация", [
                "Одинаковое агрегированное значение повторяется у всех строк одной группы — это удобно для графиков и последующих формул.",
                "Это не сводная таблица: для уменьшения числа строк используйте отдельную агрегацию или Long → Wide.",
            ]),
        ],
    ),
    "reshape": (
        "Преобразование Long ↔ Wide",
        "Метод меняет структуру таблицы: Long хранит измерения в строках, Wide раскладывает значения по отдельным каналам. Исходный dataset не изменяется до выполнения конвейера.",
        [
            ("Long → Wide", [
                "Идентификаторы строк задают будущую уникальную строку, например Скважина + Дата.",
                "Значения канала заголовков становятся именами новых каналов, например тип измерения Oil/Gas/Water.",
                "Каналы значений содержат числа или текст, которые попадут в ячейки Wide. При нескольких каналах формируются составные имена через выбранный разделитель.",
                "Комбинация идентификаторов и заголовка должна быть уникальной. Безопасный режим сообщает о дубликатах; first/last выбирают одно значение, sum/mean/min/max/count агрегируют повторения.",
                "Защита ограничивает результат примерно 5 000 создаваемыми каналами; перед очень широким pivot сузьте данные фильтром.",
            ]),
            ("Wide → Long", [
                "Идентификаторы повторяются для каждого разворачиваемого канала и сохраняют связь с исходным объектом.",
                "Имена исходных каналов записываются в новый канал «Переменная», их содержимое — в «Значение»; оба имени можно изменить.",
                "Если список разворачиваемых каналов пуст, используются все каналы кроме идентификаторов.",
                "Удаление пустых значений сокращает результат, но является потерей информации о явно пустых ячейках.",
            ]),
            ("Пример", [
                "Long: Скважина=A, Показатель=Oil, Значение=10 и Скважина=A, Показатель=Gas, Значение=2.",
                "Wide: одна строка Скважина=A и отдельные каналы Oil=10, Gas=2. Обратная операция снова создаёт две строки.",
            ]),
            ("Важно для конвейера", [
                "Преобразование меняет число строк и схему каналов. Следующие шаги должны ссылаться уже на имена, существующие после reshape.",
                "При записи результата в текущий dataset старые фильтры очищаются, потому что их каналы или строки могли исчезнуть.",
                "Для экспериментов безопаснее выбрать «Новый dataset»: тогда Long и Wide версии останутся доступны параллельно.",
            ]),
        ],
    ),
}


def _binning_panel():
    return dmc.TabsPanel(
        [
            _method_help_button("binning"),
            dmc.SimpleGrid(
                [
                    _field("Числовой канал", bin_column_select),
                    _field("Количество групп", bin_k),
                ],
                cols=2,
                spacing="xs",
            ),
            dmc.SimpleGrid(
                [_field("Метод", bin_method), _field("Метки", bin_label_style)],
                cols=2,
                spacing="xs",
                mt="xs",
            ),
            dmc.Group(
                dmc.Button("Добавить шаг", id="btn-grouping", size="xs"),
                justify="flex-end",
                mt="sm",
            ),
        ],
        value="binning",
        className="de-operation-panel",
    )


def _text_panel():
    return dmc.TabsPanel(
        [
            _method_help_button("text"),
            dmc.SimpleGrid(
                [
                    _field("Каналы", txtcopy_cols_select),
                    _field("Суффикс", txtcopy_suffix_input),
                ],
                cols=2,
                spacing="xs",
            ),
            html.Div(txtcopy_strip_switch, className="de-inline-option"),
            dmc.Group(
                dmc.Button(
                    "Добавить шаг",
                    id="btn-txtcopy",
                    size="xs",
                    variant="light",
                ),
                justify="flex-end",
            ),
            html.Div(id="de-txt-status", style={"display": "none"}),
        ],
        value="text",
        className="de-operation-panel",
    )


def _aggregate_panel():
    return dmc.TabsPanel(
        [
            _method_help_button("aggregate"),
            dmc.SimpleGrid(
                [_field("Ключи группировки", agg_keys_select), _field("Каналы", agg_cols_select)],
                cols=2,
                spacing="xs",
            ),
            _field("Метрики", agg_metrics_select, "de-field--metrics"),
            dmc.Group(
                [agg_exclude_zeros_switch, agg_exclude_empty_switch],
                gap="md",
                mt="xs",
            ),
            dmc.Text(
                "Только для расчёта: строки сохраняются; если NaN учитываются, "
                "в числовых метриках они считаются нулём.",
                size="9px",
                c="dimmed",
                mt=3,
            ),
            dmc.Group(
                dmc.Button("Добавить шаг", id="btn-agg", size="xs"),
                justify="flex-end",
                mt="xs",
            ),
            html.Div(id="de-agg-status", style={"display": "none"}),
        ],
        value="aggregate",
        className="de-operation-panel",
    )


def _reshape_panel():
    common_select = {
        "data": [],
        "searchable": True,
        "clearable": True,
        "nothingFoundMessage": "Ничего не найдено",
        "maxDropdownHeight": 420,
        "comboboxProps": {"shadow": "md"},
        "size": "xs",
    }
    return dmc.TabsPanel(
        [
            _method_help_button("reshape"),
            dmc.SegmentedControl(
                id="reshape-direction",
                data=[
                    {"label": "Long → Wide", "value": "long_to_wide"},
                    {"label": "Wide → Long", "value": "wide_to_long"},
                ],
                value="long_to_wide",
                size="xs",
                fullWidth=True,
            ),
            html.Div(
                [
                    dmc.SimpleGrid(
                        [
                            _field(
                                "Идентификаторы строк",
                                dmc.MultiSelect(id="reshape-wide-index", value=[], **common_select),
                            ),
                            _field(
                                "Заголовки из канала",
                                dmc.Select(id="reshape-wide-names", value=None, **common_select),
                            ),
                        ],
                        cols=2,
                        spacing="xs",
                    ),
                    dmc.SimpleGrid(
                        [
                            _field(
                                "Каналы значений",
                                dmc.MultiSelect(id="reshape-wide-values", value=[], **common_select),
                            ),
                            _field(
                                "Если строки повторяются",
                                dmc.Select(
                                    id="reshape-wide-aggregation",
                                    data=[
                                        {"label": "Сообщить об ошибке", "value": "error"},
                                        {"label": "Первое значение", "value": "first"},
                                        {"label": "Последнее значение", "value": "last"},
                                        {"label": "Сумма", "value": "sum"},
                                        {"label": "Среднее", "value": "mean"},
                                        {"label": "Минимум", "value": "min"},
                                        {"label": "Максимум", "value": "max"},
                                        {"label": "Количество", "value": "count"},
                                    ],
                                    value="error",
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
                        "Разделитель составных заголовков",
                        dmc.TextInput(id="reshape-wide-separator", value="__", size="xs"),
                        "de-field--reshape-separator",
                    ),
                ],
                id="reshape-long-to-wide-panel",
                className="de-reshape-direction-panel",
            ),
            html.Div(
                [
                    dmc.SimpleGrid(
                        [
                            _field(
                                "Идентификаторы строк",
                                dmc.MultiSelect(id="reshape-long-id", value=[], **common_select),
                            ),
                            _field(
                                "Разворачиваемые каналы",
                                dmc.MultiSelect(id="reshape-long-values", value=[], **common_select),
                            ),
                        ],
                        cols=2,
                        spacing="xs",
                    ),
                    dmc.SimpleGrid(
                        [
                            _field(
                                "Имя канала переменной",
                                dmc.TextInput(
                                    id="reshape-long-variable-name",
                                    value="Переменная",
                                    size="xs",
                                ),
                            ),
                            _field(
                                "Имя канала значения",
                                dmc.TextInput(
                                    id="reshape-long-value-name",
                                    value="Значение",
                                    size="xs",
                                ),
                            ),
                        ],
                        cols=2,
                        spacing="xs",
                        mt="xs",
                    ),
                    dmc.Switch(
                        id="reshape-long-dropna",
                        label="Удалить строки с пустыми значениями",
                        checked=False,
                        size="xs",
                        mt="xs",
                    ),
                    dmc.Text(
                        "Если разворачиваемые каналы не выбраны, используются все каналы кроме идентификаторов.",
                        size="9px",
                        c="dimmed",
                        mt=3,
                    ),
                ],
                id="reshape-wide-to-long-panel",
                className="de-reshape-direction-panel",
                style={"display": "none"},
            ),
            dmc.Group(
                dmc.Button("Добавить шаг", id="btn-reshape", size="xs"),
                justify="flex-end",
                mt="xs",
            ),
        ],
        value="reshape",
        className="de-operation-panel de-operation-panel--reshape",
    )


def create_data_engineering_workspace():
    return html.Div(
        [
            dmc.Paper(
                [
                    dmc.Group(
                        [
                            html.Div(
                                [
                                    dmc.Text("Конструктор признаков", fw=700, size="sm"),
                                    dmc.Text(
                                        "Новые каналы для графиков, дашбордов и ML",
                                        size="10px",
                                        c="dimmed",
                                    ),
                                ]
                            ),
                            dmc.Badge(
                                "Рабочий слой",
                                id="de-active-dataset-badge",
                                size="sm",
                                variant="light",
                                color="blue",
                            ),
                        ],
                        justify="space-between",
                    ),
                    dmc.SimpleGrid(
                        [
                            _field(
                                "Входной dataset",
                                dmc.Select(
                                    id="de-input-dataset",
                                    data=[],
                                    value="source",
                                    allowDeselect=False,
                                    searchable=True,
                                    size="xs",
                                ),
                            ),
                            _field(
                                "Слой данных",
                                dmc.SegmentedControl(
                                    id="de-input-scope",
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
                                    id="de-output-mode",
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
                                "Название нового dataset",
                                dmc.TextInput(
                                    id="de-output-name",
                                    placeholder="Например: Агрегаты",
                                    size="xs",
                                ),
                            ),
                        ],
                        cols=4,
                        spacing="xs",
                        mt="sm",
                        className="de-routing-grid",
                    ),
                    dmc.Text(id="de-scope-note", size="10px", c="dimmed", mt=5),
                ],
                p="sm",
                withBorder=True,
                shadow="xs",
                className="de-toolbar",
            ),
            html.Div(
                [
                    dmc.Paper(
                        [
                            dmc.Tabs(
                                [
                                    dmc.TabsList(
                                        [
                                            dmc.TabsTab("Биннинг", value="binning"),
                                            dmc.TabsTab("Текст", value="text"),
                                            dmc.TabsTab("Агрегаты", value="aggregate"),
                                            dmc.TabsTab("Long ↔ Wide", value="reshape"),
                                        ]
                                    ),
                                    _binning_panel(),
                                    _text_panel(),
                                    _aggregate_panel(),
                                    _reshape_panel(),
                                ],
                                value="binning",
                                id="de-operation-tabs",
                            )
                        ],
                        p="sm",
                        withBorder=True,
                        shadow="xs",
                        className="de-builder",
                    ),
                    dmc.Paper(
                        [
                            dmc.Group(
                                [
                                    dmc.Text("Конвейер", fw=700, size="xs"),
                                    dmc.Text(id="de-dataset-summary", size="10px", c="dimmed"),
                                ],
                                justify="space-between",
                            ),
                            dmc.Text(id="de-queue-context", size="9px", c="dimmed", mt=3),
                            html.Div(id="de-pipeline-list", className="de-pipeline-list"),
                            dmc.Group(
                                [
                                    dmc.Button(
                                        "Очистить",
                                        id="de-clear-pipeline",
                                        size="compact-xs",
                                        variant="subtle",
                                        color="gray",
                                    ),
                                    dmc.Button(
                                        "Выполнить",
                                        id="de-run-pipeline",
                                        size="compact-xs",
                                        disabled=True,
                                    ),
                                ],
                                justify="space-between",
                                mt=7,
                            ),
                        ],
                        p="sm",
                        withBorder=True,
                        shadow="xs",
                        className="de-pipeline",
                    ),
                ],
                className="de-main-grid",
            ),
            dmc.Paper(
                [
                    dmc.Group(
                        [
                            dmc.Text("Предпросмотр активного dataset", fw=700, size="xs"),
                            dmc.Text(
                                "Первые 8 строк · первые 3 и последние 7 каналов",
                                size="10px",
                                c="dimmed",
                            ),
                        ],
                        justify="space-between",
                    ),
                    html.Div(id="de-preview", className="de-preview"),
                ],
                p="sm",
                withBorder=True,
                shadow="xs",
                className="de-preview-paper",
            ),
            *[
                _method_help_window(method, content[0], content[1], content[2])
                for method, content in DATA_ENGINEERING_HELP.items()
            ],
        ],
        className="de-workspace",
    )
