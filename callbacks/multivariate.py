# -*- coding: utf-8 -*-
"""
Callbacks: многомерные графики (Коррелограмма, Scatter Matrix, Parallel Coordinates).

Живут в отдельном воркспейсе на странице корреляционного анализа
(layout.MULTIVARIATE_WORKSPACE). Коррелограмма строится по «коррелируемым
каналам» (dropdown_corr_columns) с учётом метода и минимума наблюдений.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, State, no_update

from correlation_workspace import (
    _empty_analysis_figure,
    build_correlogram,
    compute_correlation,
)
from dash_app import app
from layout import CORRELATION_WORKSPACE, MULTIVARIATE_WORKSPACE
from utils import _make_error_notif, read_df_from_store


def _numeric_dimensions(plot_df, columns):
    use_dims = []
    for column in columns:
        if column and (column in plot_df.columns) and pd.api.types.is_numeric_dtype(plot_df[column]):
            use_dims.append(column)
    return list(dict.fromkeys(use_dims))


@MULTIVARIATE_WORKSPACE.figure_callback(
    app,
    Input(MULTIVARIATE_WORKSPACE.chart_type_id, "value"),
    Input("dropdown_corr_columns", "value"),
    Input(CORRELATION_WORKSPACE.ids["method"], "value"),
    Input(CORRELATION_WORKSPACE.ids["min_periods"], "value"),
    Input("filtered-data", "data"),
    Input("url", "pathname"),
    Input("dropdown_style", "value"),
    State("meta-columns", "data"),
)
def build_multivariate_figure(chart_type, corr_columns, corr_method, min_periods,
                              filtered_json, pathname, selected_style, meta):
    if pathname != "/correlation":
        return no_update, no_update
    empty = go.Figure().update_layout(template=selected_style or "plotly")
    if not filtered_json:
        return empty, []
    if len(corr_columns or []) < 2:
        return _empty_analysis_figure(
            "Выберите минимум два числовых канала", selected_style
        ), []
    try:
        plot_df = read_df_from_store(filtered_json, meta)
    except Exception:
        return empty, []
    if plot_df is None or plot_df.empty:
        return empty, []

    if chart_type == "Correlogram":
        correlation, pair_counts, _status, error = compute_correlation(
            plot_df, corr_columns, corr_method, min_periods
        )
        if error:
            return _empty_analysis_figure(error, selected_style), []
        return build_correlogram(correlation, pair_counts, selected_style or "plotly"), []

    use_dims = _numeric_dimensions(plot_df, corr_columns)
    if len(use_dims) < 2:
        return empty, _make_error_notif(
            f"Для {chart_type or 'многомерного графика'} нужны ≥2 числовых "
            "канала в списке «Коррелируемые каналы»."
        )

    if chart_type == "Parcoords":
        dims = [dict(label=c, values=plot_df[c].values) for c in use_dims]
        fig = go.Figure(data=go.Parcoords(
            dimensions=dims,
        ))
        fig.update_layout(template=selected_style)
    else:
        fig = px.scatter_matrix(
            plot_df, dimensions=use_dims, template=selected_style
        )

    return fig, []
