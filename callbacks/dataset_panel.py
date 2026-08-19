# -*- coding: utf-8 -*-
"""Callbacks for the left slide-out dataset panel."""

from dash import Input, Output, State, clientside_callback

from dash_app import app


@app.callback(
    Output("dataset-drawer", "className"),
    Output("dataset-drawer-open-state", "data"),
    Input("dataset-side-tab", "n_clicks"),
    State("dataset-drawer-open-state", "data"),
    prevent_initial_call=True,
)
def toggle_dataset_drawer(_tab_clicks, opened):
    should_open = not bool(opened)
    panel_class = "dataset-drawer-panel open" if should_open else "dataset-drawer-panel"
    return panel_class, should_open


@app.callback(
    Output("dataset-outside-close-store", "data"),
    Input("dataset-close-on-outside", "checked"),
)
def sync_dataset_close_on_outside(close_on_outside):
    return bool(close_on_outside)


clientside_callback(
    """
    function (enabled) {
        if (window.__datasetOutsideAbort) {
            window.__datasetOutsideAbort.abort();
            window.__datasetOutsideAbort = null;
        }
        if (!enabled) {
            return window.dash_clientside.no_update;
        }
        var controller = new AbortController();
        window.__datasetOutsideAbort = controller;
        document.addEventListener("mousedown", function (event) {
            var panel = document.getElementById("dataset-drawer");
            if (!panel || !panel.classList.contains("open")) return;
            if (panel.contains(event.target)) return;
            window.dash_clientside.set_props("dataset-drawer", {className: "dataset-drawer-panel"});
            window.dash_clientside.set_props("dataset-drawer-open-state", {data: false});
        }, {signal: controller.signal});
        return window.dash_clientside.no_update;
    }
    """,
    Output("dataset-drawer-open-state", "data", allow_duplicate=True),
    Input("dataset-outside-close-store", "data"),
    prevent_initial_call=True,
)
