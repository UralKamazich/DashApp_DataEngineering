# -*- coding: utf-8 -*-
"""Slide-out filter panel on the right edge — instance of the unified SlidePanel."""

from dash import dcc, html
from dash_iconify import DashIconify
import dash_mantine_components as dmc

from slide_panel import SlidePanel

def _panel_content():
    content = dmc.Paper(
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
        shadow="sm",
        p="xs",
        withBorder=True,
        style={
            "overflowY": "auto",
            "overflowX": "hidden",
            "height": "100%",
            "padding": "8px",
            "fontSize": "10px",
        },
    )

    footer = html.Div(
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
    )

    return [
        dcc.Store(id="filter-drop-store"),
        html.Div(content, className="filter-panel-columns"),
        footer,
    ]


FILTERS_PANEL = SlidePanel(
    root_id="filters-drawer",
    tab_id="filters-side-tab",
    state_id="filters-drawer-open-state",
    side="right",
    mode="reflow",
    width=299,
    tab_icon="tabler:filter",
    tab_label="Фильтры",
    tab_title="Открыть панель фильтров",
    tab_style={"top": "66%", "transform": "translateY(-50%)"},
    tab_extra_children=[
        html.Span("0", id="filters-side-tab-count", className="filter-side-tab-count"),
    ],
    extra_tab_classes="filter-side-tab",
    content=_panel_content,
    extra_stores=[dcc.Store(id="filters-outside-close-store", data=False)],
)


def create_filter_drawer():
    return FILTERS_PANEL.render()
