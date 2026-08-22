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


def _binning_panel():
    return dmc.TabsPanel(
        [
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
                                    value="current",
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
                                        ]
                                    ),
                                    _binning_panel(),
                                    _text_panel(),
                                    _aggregate_panel(),
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
        ],
        className="de-workspace",
    )
