# -*- coding: utf-8 -*-
"""
Callbacks: открытие/закрытие модалок и drawer.
"""

from dash import callback, Output, Input, State, no_update, dcc
from dash.exceptions import PreventUpdate
from app import app


@app.callback(
    Output("de-modal", "opened"),
    Input("de-button", "n_clicks"),
    State("de-modal", "opened"),
    prevent_initial_call=True
)
def toggle_de_modal(n, opened):
    if not n:
        raise PreventUpdate
    return not bool(opened)


# ============ Диалог настройки (открыть дравер) ============
@callback(
    Output("drawer-simple", "opened"),
    Input("drawer-demo-button", "n_clicks"),
    prevent_initial_call=True
)
def drawer_demo(n_clicks):
    return True