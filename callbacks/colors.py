# -*- coding: utf-8 -*-
"""
Callbacks: диалог выбора цвета, применение цветов, preview.
"""

import logging
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import dash_mantine_components as dmc
from dash import callback, Output, Input, State, no_update, html, dcc, MATCH, ALL

COLOR_THRESHOLD = 10

from app import app
from utils import _make_error_notif, apply_custom_colors_safely

logger = logging.getLogger(__name__)



# Диалог выбора цвета
@app.callback(
    Output("color-modal", "opened", allow_duplicate=True),
    Output("color-inputs", "children"),
    Input("shuffle-button", "n_clicks"),
    Input("color-mode-toggle", "checked"),
    State("graph", "figure"),
    State("dropdown_style", "value"),
    State("custom-colors", "data"),
    prevent_initial_call=True
)
def open_color_dialog(n_clicks, manual_mode, fig_dict, selected_style, custom_colors):
    if not fig_dict or "data" not in fig_dict:
        raise no_update

    traces = fig_dict["data"]
    use_dropdown = not manual_mode and len(traces) <= COLOR_THRESHOLD

    if selected_style == "seaborn_custom":
        style_colors = pio.templates[selected_style].layout.colorway
    else:
        style_colors = getattr(px.colors.qualitative, selected_style, px.colors.qualitative.Plotly)

    color_inputs = []
    for i, trace in enumerate(traces):
        index = str(i)
        name = trace.get("name", f"Категория {i+1}")
        current_color = (custom_colors or {}).get(index, trace.get("marker", {}).get("color", style_colors[i % len(style_colors)]))
        preview_id = {"type": "color-preview", "index": index}

        if use_dropdown:
            input_control = dmc.Group([
                dcc.Dropdown(
                    id={"type": "color-picker", "index": index},
                    value=current_color if current_color in style_colors else style_colors[i % len(style_colors)],
                    options=[{"label": f"Цвет {j+1} ({style_colors[j]})", "value": style_colors[j]} for j in range(len(style_colors))],
                    clearable=False,
                    style={"width": 300}
                ),
                html.Div(id=preview_id, style={"backgroundColor": current_color, "width": "20px", "height": "20px", "border": "1px solid #ccc", "marginLeft": "5px"})
            ])
        else:
            input_control = dmc.ColorInput(id={"type": "color-picker", "index": index}, value=current_color, format="hex")

        color_inputs.append(dmc.Group([dmc.Text(name, style={"width": 150}), input_control]))
    return True, color_inputs


# Применение цветов
@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("color-modal", "opened", allow_duplicate=True),
    Output("custom-colors", "data"),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Input("apply-colors", "n_clicks"),
    State("graph", "figure"),
    State({"type": "color-picker", "index": ALL}, "id"),
    State({"type": "color-picker", "index": ALL}, "value"),
    State("custom-colors", "data"),
    prevent_initial_call=True
)
def apply_custom_colors(n_clicks, fig_dict, ids, values, custom_colors):
    if not fig_dict or "data" not in fig_dict:
        raise no_update

    try:
        new_color_map = {item["index"]: val for item, val in zip(ids, values)}
        updated = (custom_colors or {}).copy()
        updated.update(new_color_map)

        fig = go.Figure(fig_dict)
        fig = apply_custom_colors_safely(fig, updated)

        return fig, False, updated, []
    except Exception as e:
        logger.error(f"Ошибка при применении цветов: {e}", exc_info=True)
        notif = _make_error_notif("Не удалось применить цвета. График остается без изменений.")
        return fig_dict, False, custom_colors, notif


# Preview цвета
@app.callback(
    Output({"type": "color-preview", "index": MATCH}, "style"),
    Input({"type": "color-picker", "index": MATCH}, "value"),
    prevent_initial_call=True
)
def update_preview_color(selected_color):
    return {"backgroundColor": selected_color, "width": "20px", "height": "20px", "border": "1px solid #ccc", "marginLeft": "5px"}