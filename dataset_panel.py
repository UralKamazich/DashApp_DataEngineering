# -*- coding: utf-8 -*-
"""Slide-out dataset panel on the left edge with an attached handle tab."""

from dash import dcc, html
from dash_iconify import DashIconify
import dash_mantine_components as dmc


def create_dataset_drawer():
    tab = html.Div(
        [
            DashIconify(icon="tabler:database", width=15),
            html.Span("Датасет", className="dataset-side-tab-label"),
        ],
        id="dataset-side-tab",
        className="dataset-side-tab",
        title="Показать или скрыть датасет",
    )

    footer = html.Div(
        dmc.Checkbox(
            id="dataset-close-on-outside",
            label="Закрывать при клике вне",
            checked=False,
            size="xs",
            persistence=True,
        ),
        className="dataset-drawer-footer",
    )

    return html.Div(
        [
            tab,
            html.Div(
                [
                    html.Div(
                        dmc.Paper(
                            id="columns-sidebar",
                            children=[
                                dmc.Stack(
                                    id="columns-badges",
                                    children=[],
                                    gap="2px",
                                    style={"maxWidth": "100%", "overflow": "hidden"},
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
                        ),
                        className="dataset-drawer-columns",
                    ),
                    footer,
                ],
                className="dataset-drawer-body",
            ),
            dcc.Store(id="dataset-outside-close-store", data=False),
            dcc.Store(id="dataset-drawer-open-state", data=False),
        ],
        id="dataset-drawer",
        className="dataset-drawer-panel",
    )
