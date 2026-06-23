# -*- coding: utf-8 -*-
"""
Константы, стили и настройки приложения.
"""

import plotly.graph_objects as go
import plotly.io as pio

# =========================
# Константы и настройки
# =========================
COLOR_THRESHOLD = 10
MAX_FILTERS = 10
PORT = 8090
DEFAULT_X = "Дата"
DEFAULT_Y = "Добыча"
DEFAULT_Z = "Забойное давление"
DEFAULT_BUBBLE_SIZE = "Дебит"
DEFAULT_HOVER_COLS = ["Скважина", "Пластовое давление"]
STYLE = {"margin": 10}

legend_config = {
    "top-left-inside": dict(x=0.01, y=0.99, xanchor="left", yanchor="top", orientation="v"),
    "top-center-outside": dict(x=0.5, y=1.01, xanchor="center", yanchor="bottom", orientation="h"),
    "top-right-inside": dict(x=0.99, y=0.99, xanchor="right", yanchor="top", orientation="v"),
    "top-right-outside": dict(x=1.02, y=1, xanchor="left", yanchor="top", orientation="v"),
    "bottom-outside": dict(x=0.5, y=-0.12, xanchor="center", yanchor="top", orientation="h"),
}

# Пользовательский шаблон
seaborn_custom = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Arial, sans-serif", size=12, color="#000000"),
        title_font=dict(family="Arial, sans-serif", size=16, color="#000000"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(
            gridcolor="#D3D3D3",
            gridwidth=1,
            zerolinecolor="#D3D3D3",
            zerolinewidth=1,
            showline=True,
            linecolor="#000000",
            linewidth=2,
            mirror=True
        ),
        yaxis=dict(
            gridcolor="#D3D3D3",
            gridwidth=1,
            zerolinecolor="#D3D3D3",
            zerolinewidth=1,
            showline=True,
            linecolor="#000000",
            linewidth=2,
            mirror=True
        ),
        colorway=[
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
    )
)
pio.templates["seaborn_custom"] = seaborn_custom

PLOTLY_STYLES = [
    "plotly", "ggplot2", "seaborn", "seaborn_custom", "simple_white", "plotly_white", "plotly_dark"
]

NOTIF_POSITION = "bottom-right"

STYLE_CARD = {
    'maxWidth': '100%',
    'padding': '20px',
    'boxSizing': 'border-box',
    "margin": "10px"
}
PAPER_BASE = {"height": "auto", "overflow": "visible"}
initial_fig = go.Figure()