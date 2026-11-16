# -*- coding: utf-8 -*-
"""Page «Экспресс корр. анализ»: correlation rating of one channel vs all.

The rating table and the floating cross-plot windows are drawn by
``assets/express_correlation.js`` (plain DOM + Plotly), so thousands of rows
do not add Dash components.  See ``callbacks/express_correlation.py``.
"""

from __future__ import annotations

from dash import dcc, html
from dash_iconify import DashIconify
import dash_mantine_components as dmc



PAGE_PATH = "/express-correlation"


DROPDOWN_LABELS = {
    "search": "Поиск…",
    "clear_search": "Очистить поиск",
    "clear_selection": "Очистить",
    "no_options_found": "Ничего не найдено",
}


def _labeled(label, control, width):
    # dcc.Dropdown is virtualized: every column stays reachable by scrolling,
    # even with thousands of channels (Mantine Select renders all options).
    return dmc.Stack(
        gap=3,
        w=width,
        children=[dmc.Text(label, size="xs", fw=500), control],
    )


def create_express_correlation_workspace():
    controls = dmc.Paper(
        withBorder=True,
        p="sm",
        radius="md",
        className="xcorr-controls",
        children=dmc.Group(
            align="flex-end",
            gap="md",
            wrap="wrap",
            children=[
                _labeled("Канал", dcc.Dropdown(
                    id="xcorr-target",
                    options=[],
                    placeholder="Выберите числовой канал",
                    searchable=True,
                    clearable=False,
                    maxHeight=420,
                    persistence=True,
                    className="xcorr-dropdown",
                    labels=DROPDOWN_LABELS,
                ), width=260),
                _labeled("Начать правее столбца", dcc.Dropdown(
                    id="xcorr-start-after",
                    options=[],
                    placeholder="Все столбцы",
                    searchable=True,
                    clearable=True,
                    maxHeight=420,
                    persistence=True,
                    className="xcorr-dropdown",
                    labels=DROPDOWN_LABELS,
                ), width=260),
                dmc.NumberInput(
                    id="xcorr-min-periods",
                    label="Мин. пар",
                    value=10,
                    min=3,
                    step=1,
                    size="xs",
                    w=90,
                    persistence=True,
                ),
                dmc.Checkbox(
                    id="xcorr-exclude-zero",
                    label="Исключать 0",
                    checked=True,
                    size="xs",
                    mb=6,
                    persistence=True,
                ),
                dmc.Button(
                    "Рассчитать",
                    id="xcorr-run",
                    size="xs",
                    leftSection=DashIconify(icon="tabler:calculator", width=15),
                ),
            ],
        ),
    )

    table_card = dmc.Paper(
        withBorder=True,
        p="sm",
        radius="md",
        mt="sm",
        className="xcorr-table-card",
        children=[
            dmc.Group(
                justify="space-between",
                align="center",
                mb=6,
                children=[
                    dmc.Text(
                        "Нажмите на строку — справа от курсора откроется кросс-плот.",
                        id="xcorr-status",
                        size="xs",
                        c="dimmed",
                    ),
                    dmc.TextInput(
                        id="xcorr-filter",
                        placeholder="Фильтр каналов…",
                        size="xs",
                        w=220,
                        leftSection=DashIconify(icon="tabler:search", width=14),
                    ),
                ],
            ),
            # Filled by assets/express_correlation.js; React never owns it.
            html.Div(id="xcorr-table", className="xcorr-table-host"),
        ],
    )

    return html.Div(
        id="express-correlation-workspace",
        className="xcorr-workspace",
        children=[
            controls,
            table_card,
            dcc.Store(id="xcorr-result"),
            dcc.Store(id="xcorr-rendered"),
        ],
    )
