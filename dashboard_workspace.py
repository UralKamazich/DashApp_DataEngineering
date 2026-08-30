# -*- coding: utf-8 -*-
"""A dashboard page composed from independent reusable graph workspaces."""

from __future__ import annotations

from copy import deepcopy

from dash import html

from components import (
    SwitchBubble,
    bar_aggregation_select,
    bar_text_auto_switch,
    dropdown_axes_category,
    dropdown_category_ascending,
    dropdown_chart_type,
    dropdown_legend,
    dropdown_legend_order,
    dropdown_overlay,
    dropdown_pie_aggregation,
    dropdown_style,
    dropdown_text_pozition,
    graph_render_mode,
    input_legend_custom_order,
    create_dropdown,
    create_multiselect,
)
from graph_settings import GraphSettingsPanel
from graph_workspace import DEFAULT_FIELDS, GraphWorkspace


DASHBOARD_GRAPH_COUNT = 4
DASHBOARD_GRAPH_HEIGHT = 360


FIELD_CHART_TYPES = {
    "z": {"3D_Scatter", "Polar"},
    "color": {
        "Scatter", "3D_Scatter", "Box", "Bar", "Line", "Hist", "Polar",
        "Pie", "Violin", "Ridge", "Sunburst", "Treemap", "DensityContour",
    },
    "size": {"Scatter", "3D_Scatter", "Polar"},
    "text": {"Scatter", "3D_Scatter", "Polar"},
    "facet-row": {
        "Scatter", "Box", "Bar", "Line", "Hist", "Violin",
        "DensityHeat", "DensityContour",
    },
    "facet-col": {
        "Scatter", "Box", "Bar", "Line", "Hist", "Violin",
        "DensityHeat", "DensityContour",
    },
    "hover": {"Scatter", "3D_Scatter", "Polar", "Pie"},
    "hierarchy-levels": {"Sunburst", "Treemap"},
    "hierarchy-value": {"Sunburst", "Treemap"},
}


FIELD_PRESENTATIONS = {
    "3D_Scatter": {
        "x": {"label": "X", "placement": "top"},
        "y": {"label": "Y", "placement": "top"},
        "z": {"label": "Z", "placement": "top"},
    },
    "Polar": {
        "x": {"label": "A", "placement": "top"},
        "y": {"label": "B", "placement": "top"},
        "z": {"label": "C", "placement": "top"},
    },
    "Pie": {
        "x": {"label": "Категория", "placement": "top"},
        "y": {"label": "Значение", "placement": "top"},
    },
    "Sunburst": {
        "hierarchy-levels": {"label": "Уровни", "order": "0"},
        "hierarchy-value": {"label": "Значение", "order": "1"},
        "color": {"label": "Цвет", "order": "2"},
    },
    "Treemap": {
        "hierarchy-levels": {"label": "Уровни", "order": "0"},
        "hierarchy-value": {"label": "Значение", "order": "1"},
        "color": {"label": "Цвет", "order": "2"},
    },
}


def _clone(component, component_id: str):
    cloned = deepcopy(component)
    cloned.id = component_id
    return cloned


def _make_workspace(index: int) -> GraphWorkspace:
    namespace = f"dashboard-graph-{index}"
    chart_type = _clone(dropdown_chart_type, f"{namespace}-chart-type")

    field_controls = {
        "x": create_dropdown(f"{namespace}-x", [], None, clearable=True),
        "y": create_dropdown(f"{namespace}-y", [], None, clearable=True),
        "z": create_dropdown(f"{namespace}-z", [], None, clearable=True),
        "color": create_dropdown(f"{namespace}-color", [], None, clearable=True),
        "size": create_dropdown(f"{namespace}-size", [], None, clearable=True),
        "text": create_dropdown(f"{namespace}-text", [], None, clearable=True),
        "facet-row": create_dropdown(f"{namespace}-facet-row", [], None, clearable=True),
        "facet-col": create_dropdown(f"{namespace}-facet-col", [], None, clearable=True),
        "hover": create_multiselect(f"{namespace}-hover", [], value=[], clearable=True),
        "hierarchy-levels": create_multiselect(
            f"{namespace}-hierarchy-levels", [], value=[], clearable=True
        ),
        "hierarchy-value": create_dropdown(
            f"{namespace}-hierarchy-value", [], None, clearable=True
        ),
    }
    fields = (
        *(
            {**field, "target": field["key"]}
            for field in DEFAULT_FIELDS
        ),
        {
            "key": "hierarchy-levels", "label": "Уровни",
            "target": "hierarchy-levels", "zone": "secondary", "mode": "append",
        },
        {
            "key": "hierarchy-value", "label": "Значение",
            "target": "hierarchy-value", "zone": "secondary",
        },
    )

    settings = GraphSettingsPanel(
        controls={
            "theme": _clone(dropdown_style, f"{namespace}-theme"),
            "render_mode": _clone(graph_render_mode, f"{namespace}-render-mode"),
            "bubble": _clone(SwitchBubble, f"{namespace}-bubble"),
            "bar_labels": _clone(bar_text_auto_switch, f"{namespace}-bar-labels"),
            "text_position": _clone(dropdown_text_pozition, f"{namespace}-text-position"),
            "category_axis": _clone(dropdown_axes_category, f"{namespace}-category-axis"),
            "category_order": _clone(dropdown_category_ascending, f"{namespace}-category-order"),
            "bar_mode": _clone(dropdown_overlay, f"{namespace}-bar-mode"),
            "bar_aggregation": _clone(bar_aggregation_select, f"{namespace}-bar-aggregation"),
            "pie_aggregation": _clone(dropdown_pie_aggregation, f"{namespace}-pie-aggregation"),
            "legend_position": _clone(dropdown_legend, f"{namespace}-legend-position"),
            "legend_order": _clone(dropdown_legend_order, f"{namespace}-legend-order"),
            "legend_custom_order": _clone(
                input_legend_custom_order, f"{namespace}-legend-custom-order"
            ),
        },
        initial_height=DASHBOARD_GRAPH_HEIGHT,
    )
    return GraphWorkspace(
        graph_id=namespace,
        chart_type_control=chart_type,
        field_controls=field_controls,
        fields=fields,
        field_chart_types=FIELD_CHART_TYPES,
        field_presentations=FIELD_PRESENTATIONS,
        settings_panel=settings,
        initial_height=DASHBOARD_GRAPH_HEIGHT,
    )


DASHBOARD_WORKSPACES = tuple(
    _make_workspace(index) for index in range(1, DASHBOARD_GRAPH_COUNT + 1)
)


def create_dashboard_workspace():
    return html.Div(
        [
            html.Div(
                workspace.render(),
                className="dashboard-graph-cell",
                **{"data-dashboard-graph": str(index)},
            )
            for index, workspace in enumerate(DASHBOARD_WORKSPACES, 1)
        ],
        className="dashboard-grid",
    )
