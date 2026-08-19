# -*- coding: utf-8 -*-
"""Global overlay filter panel shared by every analysis page."""

from dash import dcc, html
from dash_iconify import DashIconify
import dash_mantine_components as dmc


def create_filter_trigger():
    return html.Div(
        dmc.Button(
            [
                DashIconify(icon="tabler:filter", width=15),
                html.Span("Фильтры"),
                html.Span("0", id="filters-active-count", className="filter-trigger-count"),
            ],
            id="filters-panel-toggle",
            variant="subtle",
            color="gray",
            size="xs",
            className="filter-panel-trigger",
        ),
        id="filter-drop-target",
        className="filter-drop-target filter-header-drop-target",
        title="Открыть фильтры или перетащить сюда канал",
    )


def create_filter_tab():
    return html.Div(
        [
            DashIconify(icon="tabler:filter", width=15),
            html.Span("Фильтры", className="filter-side-tab-label"),
            html.Span("0", id="filters-side-tab-count", className="filter-side-tab-count"),
        ],
        id="filters-side-tab",
        className="filter-side-tab",
        title="Открыть панель фильтров",
    )


def create_filter_drawer():
    header = html.Div(
        [
            dmc.Group(
                [
                    DashIconify(icon="tabler:filter", width=16),
                    dmc.Text("Фильтры", fw=700, size="sm"),
                    dmc.Badge("0 активных", id="filters-drawer-count", variant="light", size="sm"),
                ],
                gap="xs",
            ),
            dmc.ActionIcon(
                DashIconify(icon="tabler:x", width=16),
                id="filters-drawer-close",
                variant="subtle",
                color="gray",
                size="sm",
                **{"aria-label": "Закрыть панель фильтров"},
            ),
        ],
        className="filters-drawer-header",
    )

    return html.Div(
        [
            create_filter_tab(),
            html.Div(
                [
                    header,
                    html.Div(
                        [
                dcc.Store(id="filter-drop-store"),
                html.Div(
                    [
                        html.Div(
                            [
                                dmc.Group(
                                    [
                                        dmc.Text("Связь условий", size="xs", fw=600),
                                        dmc.SegmentedControl(
                                            id="filter-logic-mode",
                                            value="and",
                                            data=[
                                                {"label": "И", "value": "and"},
                                                {"label": "ИЛИ", "value": "or"},
                                            ],
                                            size="xs",
                                            persistence=True,
                                        ),
                                    ],
                                    justify="space-between",
                                ),
                                dmc.Text(
                                    "Перетащите канал из датасета или добавьте условие вручную",
                                    size="xs",
                                    c="dimmed",
                                    mt=3,
                                ),
                            ],
                            className="filter-panel-intro",
                        ),
                        html.Div(
                            [
                                DashIconify(icon="tabler:drag-drop", width=18),
                                html.Span("Перетащить канал сюда"),
                            ],
                            id="filter-drawer-drop-zone",
                            className="filter-drop-target filter-drawer-drop-zone",
                        ),
                        html.Div(id="filters-container", children=[], className="filter-cards"),
                        html.Div(
                            [
                                DashIconify(icon="tabler:filter-plus", width=24),
                                dmc.Text("Активных условий пока нет", size="sm", fw=600),
                                dmc.Text(
                                    "Добавьте фильтр или перетащите канал из списка датасета.",
                                    size="xs",
                                    c="dimmed",
                                    ta="center",
                                ),
                            ],
                            className="filter-empty-state",
                        ),
                        dmc.Button(
                            "Добавить фильтр",
                            id="add-filter-btn",
                            size="xs",
                            variant="light",
                            leftSection=DashIconify(icon="tabler:plus", width=14),
                            fullWidth=True,
                            mt="sm",
                        ),
                    ],
                    className="filter-panel-scroll",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dmc.Text("Результат", size="xs", c="dimmed"),
                                        dmc.Text(
                                            "Загрузите данные",
                                            id="filter-results-summary",
                                            size="xs",
                                            fw=600,
                                        ),
                                    ],
                                    className="filter-results-summary",
                                ),
                                dmc.Group(
                                    [
                                        dmc.Button(
                                            "Сбросить",
                                            id="reset-filters-btn",
                                            variant="subtle",
                                            color="gray",
                                            size="xs",
                                            leftSection=DashIconify(icon="tabler:rotate", width=14),
                                        ),
                                        dmc.Button(
                                            "Применить",
                                            id="apply-filters-btn",
                                            size="xs",
                                            leftSection=DashIconify(icon="tabler:check", width=14),
                                        ),
                                    ],
                                    gap="xs",
                                ),
                            ],
                            className="filter-panel-footer-row",
                        ),
                        dmc.Group(
                            [
                                dmc.Checkbox(
                                    id="filter-close-on-apply",
                                    label="Закрывать при «Применить»",
                                    checked=False,
                                    size="xs",
                                    persistence=True,
                                ),
                                dmc.Checkbox(
                                    id="filter-close-on-outside",
                                    label="Закрывать при клике вне",
                                    checked=False,
                                    size="xs",
                                    persistence=True,
                                ),
                            ],
                            gap="md",
                            wrap="wrap",
                            className="filter-panel-options",
                        ),
                    ],
                    className="filter-panel-footer",
                ),
                        ],
                        className="filter-panel-shell",
                    ),
                ],
                className="filters-drawer-body",
            ),
            dcc.Store(id="filters-outside-close-store", data=False),
            dcc.Store(id="filters-drawer-open-state", data=False),
        ],
        id="filters-drawer",
        className="filters-drawer-panel",
    )
