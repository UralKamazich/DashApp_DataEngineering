# -*- coding: utf-8 -*-
"""Callbacks for the four independent graph workspaces on Dashboard."""

from dash import Input, Output, State

from callbacks.dropdowns import update_dropdown_options_all
from callbacks.graph import build_main_figure
from dash_app import app
from dashboard_workspace import DASHBOARD_WORKSPACES


def _register_option_callback(workspace):
    @app.callback(
        Output(workspace.field_id("x"), "options"),
        Output(workspace.field_id("y"), "options"),
        Output(workspace.field_id("z"), "options"),
        Output(workspace.field_id("color"), "options"),
        Output(workspace.field_id("size"), "options"),
        Output(workspace.field_id("hover"), "data"),
        Output(workspace.field_id("facet-row"), "options"),
        Output(workspace.field_id("facet-col"), "options"),
        Output(workspace.field_id("text"), "options"),
        Output(workspace.field_id("hierarchy-levels"), "data"),
        Output(workspace.field_id("hierarchy-value"), "options"),
        Input("filtered-data", "data"),
        State("meta-columns", "data"),
    )
    def update_options(filtered_json, meta):
        options = update_dropdown_options_all(filtered_json, meta)
        return (*options[:6], *options[7:10], options[0], options[1])


def _register_figure_callback(workspace):
    @workspace.figure_callback(
        app,
        Input(workspace.ids["update"], "n_clicks"),
        Input(workspace.field_id("x"), "value"),
        Input(workspace.field_id("y"), "value"),
        Input(workspace.field_id("z"), "value"),
        Input(workspace.field_id("color"), "value"),
        Input(workspace.field_id("size"), "value"),
        Input(workspace.field_id("text"), "value"),
        Input(workspace.settings_control_id("text_position"), "value"),
        Input(workspace.chart_type_id, "value"),
        Input(workspace.settings_control_id("bubble"), "checked"),
        Input(workspace.settings_id("InputMaxSizeBubble"), "value"),
        Input(workspace.settings_id("InputSizePlot"), "value"),
        Input(workspace.settings_id("InputSizePlotW"), "value"),
        Input(workspace.settings_control_id("theme"), "value"),
        Input(workspace.settings_control_id("bar_labels"), "checked"),
        Input(workspace.ids["view_revision"], "data"),
        Input("filtered-data", "data"),
        Input(workspace.field_id("hover"), "value"),
        Input(workspace.field_id("facet-row"), "value"),
        Input(workspace.field_id("facet-col"), "value"),
        State("filters-applied-state", "data"),
        Input(workspace.settings_id("font-size-xaxis"), "value"),
        Input(workspace.settings_id("font-size-yaxis"), "value"),
        Input(workspace.settings_id("font-size-ticks"), "value"),
        Input(workspace.settings_id("font-size-title"), "value"),
        Input(workspace.settings_control_id("category_order"), "value"),
        Input(workspace.settings_control_id("category_axis"), "value"),
        Input(workspace.settings_control_id("bar_mode"), "value"),
        Input(workspace.settings_control_id("legend_position"), "value"),
        State(workspace.ids["custom_colors"], "data"),
        Input(workspace.settings_id("tick-step-xaxis"), "value"),
        Input(workspace.settings_id("tick-step-yaxis"), "value"),
        Input(workspace.settings_control_id("legend_order"), "value"),
        State(workspace.settings_control_id("legend_custom_order"), "value"),
        State("meta-columns", "data"),
        Input(workspace.settings_control_id("pie_aggregation"), "value"),
        Input(workspace.settings_control_id("bar_aggregation"), "value"),
        Input(workspace.ids["field_modes"], "data"),
        Input(workspace.settings_control_id("render_mode"), "value"),
        Input(workspace.settings_id("InputMarkerSize"), "value"),
        Input(workspace.field_id("hierarchy-levels"), "value"),
        Input(workspace.field_id("hierarchy-value"), "value"),
        Input(workspace.settings_id("font-size-legend"), "value"),
        prevent_initial_call=True,
    )
    def update_dashboard_graph(*args, **kwargs):
        return build_main_figure(*args, **kwargs)


for _workspace in DASHBOARD_WORKSPACES:
    _register_option_callback(_workspace)
    _register_figure_callback(_workspace)
