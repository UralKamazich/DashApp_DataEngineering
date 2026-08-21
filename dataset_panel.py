# -*- coding: utf-8 -*-
"""Slide-out dataset panel on the left edge — instance of the unified SlidePanel."""

from dash import dcc, html
import dash_mantine_components as dmc

from slide_panel import SlidePanel


def _panel_content():
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

    return [
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
                className="dataset-panel-paper dataset-file-drop-target",
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
    ]


DATASET_PANEL = SlidePanel(
    root_id="dataset-drawer",
    tab_id="dataset-side-tab",
    state_id="dataset-drawer-open-state",
    side="left",
    mode="reflow",
    width=299,
    tab_icon="tabler:database",
    tab_label="Датасет",
    tab_title="Показать или скрыть датасет",
    tab_style={"top": "10px"},
    extra_tab_classes="dataset-side-tab dataset-file-drop-target",
    content=_panel_content,
    extra_stores=[
        dcc.Store(id="dataset-outside-close-store", data=False),
        dcc.Store(id="dataset-file-drop-store"),
    ],
)


def create_dataset_drawer():
    return DATASET_PANEL.render()
