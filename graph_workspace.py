# -*- coding: utf-8 -*-
"""Reusable visual shell around a Plotly graph.

The shell owns only presentation and interaction targets. Plot construction and
Dash callbacks remain outside, so the component can later be reused by other
dashboards with a different graph id and field mapping.
"""

from __future__ import annotations

from collections.abc import Mapping

from dash import dcc, html
import dash_mantine_components as dmc


DEFAULT_FIELDS = (
    {"key": "x", "label": "X", "target": "dropdown_x", "zone": "axis-x"},
    {"key": "y", "label": "Y", "target": "dropdown_y", "zone": "axis-y"},
    {"key": "z", "label": "Z", "target": "dropdown_z", "zone": "secondary"},
    {"key": "color", "label": "Цвет", "target": "dropdown_color", "zone": "secondary"},
    {"key": "size", "label": "Размер", "target": "dropdown_size", "zone": "secondary"},
    {"key": "text", "label": "Подпись", "target": "dropdown_text", "zone": "secondary"},
    {"key": "facet-row", "label": "Facet row", "target": "dropdown_facet_row", "zone": "secondary"},
    {"key": "facet-col", "label": "Facet col", "target": "dropdown_facet_col", "zone": "secondary"},
    {
        "key": "hover",
        "label": "Hover",
        "target": "dropdown_hover_data",
        "zone": "secondary",
        "mode": "append",
    },
)


class GraphWorkspace:
    """Build a sized Paper containing a graph, controls and drop targets."""

    def __init__(
        self,
        *,
        graph_id: str,
        chart_type_control,
        field_controls: Mapping[str, object],
        fields=DEFAULT_FIELDS,
        initial_height: int = 750,
        initial_width: int | None = None,
    ):
        self.graph_id = graph_id
        self.paper_id = f"{graph_id}-paper"
        self.chart_type_control = chart_type_control
        self.field_controls = field_controls
        self.fields = tuple(fields)
        self.initial_height = initial_height
        self.initial_width = initial_width

        required = {field["target"] for field in self.fields}
        missing = required.difference(field_controls)
        if missing:
            raise ValueError(f"Missing graph field controls: {sorted(missing)}")

    @staticmethod
    def _pixel_size(value: int | None, fallback: str) -> str:
        return f"{value}px" if value is not None else fallback

    @staticmethod
    def _drop_zone(field: Mapping[str, str]):
        return html.Div(
            [
                html.Span(field["label"], className="graph-drop-zone-name"),
                html.Span("Не выбрано", className="graph-drop-zone-value"),
            ],
            className=f"graph-drop-zone graph-drop-zone--{field['zone']}",
            **{
                "data-drop-target": field["target"],
                "data-drop-mode": field.get("mode", "replace"),
                "data-default-label": field["label"],
            },
        )

    @staticmethod
    def _action_button(component_id: str, symbol: str, label: str):
        return dmc.Tooltip(
            label=label,
            withArrow=True,
            openDelay=300,
            children=dmc.ActionIcon(
                html.Span(symbol, className="graph-toolbar-symbol"),
                id=component_id,
                variant="subtle",
                color="gray",
                size="lg",
                radius="md",
            ),
        )

    def render(self):
        secondary = [
            self._drop_zone(field)
            for field in self.fields
            if field["zone"] == "secondary"
        ]
        axes = [
            self._drop_zone(field)
            for field in self.fields
            if field["zone"] != "secondary"
        ]

        return dmc.Paper(
            id=self.paper_id,
            className="graph-workspace-shell",
            shadow="md",
            withBorder=True,
            style={
                "height": self._pixel_size(self.initial_height, "750px"),
                "width": self._pixel_size(self.initial_width, "100%"),
            },
            children=html.Div(
                id=f"{self.graph_id}-workspace",
                className="graph-workspace",
                children=[
                    # This toolbar is a sibling of dcc.Graph, so Plotly.toImage
                    # and fig.to_html export only the figure itself.
                    html.Div(
                        className="graph-workspace-toolbar",
                        children=[
                            html.Div(self.chart_type_control, className="graph-type-control"),
                            html.Div(className="graph-workspace-toolbar-separator"),
                            self._action_button("update-graf", "↻", "Обновить график"),
                            self._action_button("copy-png-button", "⧉", "Копировать PNG в буфер"),
                            self._action_button("context-menu-btn", "⚙", "Настройки графика"),
                        ],
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            figure={},
                            id=self.graph_id,
                            className="graph-workspace-plot",
                            config={
                                "displaylogo": False,
                                "modeBarButtonsToRemove": [],
                                "modeBarButtonsToAdd": ["fullscreen"],
                                "displayModeBar": True,
                                "scrollZoom": True,
                                "responsive": True,
                            },
                            style={"height": "100%", "width": "100%"},
                        ),
                        type="default",
                        className="graph-workspace-spinner",
                        parent_className="graph-workspace-loading",
                        parent_style={"height": "100%", "width": "100%"},
                    ),
                    html.Div(
                        className="graph-drop-layer",
                        children=[
                            *axes,
                            html.Div(secondary, className="graph-drop-secondary"),
                        ],
                    ),
                    html.Div(
                        list(self.field_controls.values()),
                        className="graph-field-state",
                        **{"aria-hidden": "true"},
                    ),
                ],
            ),
        )
