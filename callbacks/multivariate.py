# -*- coding: utf-8 -*-
"""
Callbacks: многомерные графики (Коррелограмма, Scatter Matrix, Parallel Coordinates).

Живут в отдельном воркспейсе на странице корреляционного анализа
(layout.MULTIVARIATE_WORKSPACE). Коррелограмма строится по «коррелируемым
каналам» (dropdown_corr_columns) с учётом метода и минимума наблюдений.
"""

from io import StringIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, State, no_update

from correlation_workspace import (
    CORRELATION_WORK_LOCK,
    _empty_analysis_figure,
    build_correlogram,
)
from dash_app import app
from layout import CORRELATION_WORKSPACE, MULTIVARIATE_WORKSPACE
from utils import _make_error_notif, read_df_from_store


MAX_SCATTER_MATRIX_DIMENSIONS = 10
MAX_PARCOORDS_DIMENSIONS = 24
MAX_SCATTER_MATRIX_ROWS = 5_000
MAX_PARCOORDS_ROWS = 10_000


def _persistent_warning(notification_id, title, message):
    """A safety warning that remains visible until explicitly closed."""
    return [{
        "id": notification_id,
        "title": title,
        "message": message,
        "color": "yellow",
        "action": "show",
        "position": "bottom-right",
        "autoClose": False,
        "withCloseButton": True,
    }]


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
    Input(CORRELATION_WORKSPACE.ids["result"], "data"),
    State("meta-columns", "data"),
)
def build_multivariate_figure(chart_type, corr_columns, corr_method, min_periods,
                              filtered_json, pathname, selected_style,
                              correlation_result, meta):
    if pathname != "/correlation":
        return no_update, no_update
    empty = go.Figure().update_layout(template=selected_style or "plotly")
    if not filtered_json:
        return empty, []
    if len(corr_columns or []) < 2:
        return _empty_analysis_figure(
            "Выберите минимум два числовых канала", selected_style
        ), []
    if chart_type == "Correlogram":
        result = correlation_result or {}
        request_matches = (
            result.get("columns") == list(corr_columns or [])
            and result.get("method") == (
                corr_method if corr_method in {"pearson", "spearman"} else "pearson"
            )
            and result.get("min_periods") == max(int(min_periods or 2), 2)
        )
        if not request_matches:
            return _empty_analysis_figure(
                "Расчёт корреляционной матрицы…", selected_style
            ), []
        error = result.get("error")
        if error:
            return _empty_analysis_figure(error, selected_style), _persistent_warning(
                "correlation-safety-warning",
                "Корреляционный анализ ограничен",
                error,
            )
        try:
            correlation = pd.read_json(
                StringIO(result["correlation"]), orient="split"
            )
            pair_counts = pd.read_json(
                StringIO(result["pair_counts"]), orient="split"
            )
        except Exception:
            return _empty_analysis_figure(
                "Не удалось прочитать результат корреляционного анализа",
                selected_style,
            ), []
        return (
            build_correlogram(correlation, pair_counts, selected_style or "plotly"),
            [],
        )

    dimension_limit = (
        MAX_PARCOORDS_DIMENSIONS
        if chart_type == "Parcoords"
        else MAX_SCATTER_MATRIX_DIMENSIONS
    )
    if len(corr_columns or []) > dimension_limit:
        message = (
            f"Для {chart_type} выбрано {len(corr_columns or [])} каналов. "
            f"Безопасный максимум — {dimension_limit}. Уменьшите число каналов."
        )
        return empty, _persistent_warning(
            "multivariate-dimension-limit",
            f"{chart_type}: слишком много каналов",
            message,
        )

    try:
        with CORRELATION_WORK_LOCK:
            source_frame = read_df_from_store(filtered_json, meta)
            if source_frame is None or source_frame.empty:
                return empty, []

            use_dims = _numeric_dimensions(source_frame, corr_columns)
            if len(use_dims) < 2:
                return empty, _make_error_notif(
                    f"Для {chart_type or 'многомерного графика'} нужны ≥2 числовых "
                    "канала в списке «Коррелируемые каналы»."
                )

            original_rows = len(source_frame)
            row_limit = (
                MAX_PARCOORDS_ROWS
                if chart_type == "Parcoords"
                else MAX_SCATTER_MATRIX_ROWS
            )
            plot_df = source_frame[use_dims]
            if original_rows > row_limit:
                plot_df = plot_df.sample(n=row_limit, random_state=42).sort_index()
            else:
                plot_df = plot_df.copy(deep=False)
    except Exception:
        return empty, []
    del source_frame

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

    notifications = []
    if original_rows > len(plot_df):
        message = (
            f"Отображается {len(plot_df):,} из {original_rows:,} строк. "
            "Корреляции при этом рассчитываются по всем строкам."
        ).replace(",", " ")
        notifications.append({
            "id": "multivariate-visual-sample",
            "title": f"{chart_type}: визуальная выборка",
            "message": message,
            "color": "blue",
            "action": "show",
            "position": "bottom-right",
            "autoClose": False,
            "withCloseButton": True,
        })
    return fig, notifications
