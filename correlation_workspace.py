# -*- coding: utf-8 -*-
"""Страница корреляционного анализа: рейтинги корреляций.

Сама коррелограмма (матрица) живёт в мультиграфике
(callbacks/multivariate.py, тип «Correlogram»); отсюда берутся
compute_correlation и build_correlogram.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update
import dash_mantine_components as dmc

from utils import read_df_from_store


MAX_CORRELATION_COLUMNS = 50
DEFAULT_CORRELATION_COLUMNS = 12


def _empty_analysis_figure(
    message: str = "Выберите минимум два числовых столбца",
    template: str = "plotly",
):
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=15, color="#868E96"),
    )
    figure.update_layout(
        template=template or "plotly",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=30, r=30, t=55, b=30),
        height=420,
    )
    return figure


def _correlation_bar(correlation, pair_counts, target, template):
    values = correlation[target].drop(labels=[target], errors="ignore").dropna()
    if values.empty:
        return _empty_analysis_figure(f"Нет корреляций для «{target}»", template)

    order = values.abs().sort_values(ascending=True, kind="stable").index
    values = values.loc[order]
    counts = pair_counts.loc[order, target].astype(int)
    height = max(300, min(1200, 28 * len(values) + 130))
    figure = go.Figure(go.Bar(
        x=values.to_numpy(),
        y=[str(column) for column in values.index],
        orientation="h",
        customdata=counts.to_numpy(),
        marker=dict(
            color=values.to_numpy(),
            colorscale="RdBu",
            reversescale=True,
            cmin=-1,
            cmax=1,
            showscale=False,
        ),
        text=[f"{value:.2f}" for value in values],
        textposition="auto",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>Корреляция: %{x:.4f}"
            "<br>Совместных наблюдений: %{customdata}<extra></extra>"
        ),
    ))
    figure.update_layout(
        title=f"Корреляции с «{target}»",
        template=template,
        height=height,
        margin=dict(l=30, r=25, t=60, b=45),
        xaxis=dict(
            title="Коэффициент корреляции",
            range=[-1.05, 1.05],
            zeroline=True,
            zerolinewidth=1,
            tickformat=".1f",
            automargin=True,
        ),
        yaxis=dict(title=None, automargin=True),
    )
    return figure


def compute_correlation(frame, columns, method="pearson", min_periods=10):
    """Посчитать матрицу корреляций по выбранным столбцам.

    Возвращает (correlation, pair_counts, status, error): при ошибке
    correlation/pair_counts равны None, error содержит текст сообщения.
    """
    frame = frame.copy()
    frame.columns = [str(column) for column in frame.columns]
    method = method if method in {"pearson", "spearman"} else "pearson"
    min_periods = max(int(min_periods or 2), 2)
    selected = list(dict.fromkeys(
        str(column) for column in (columns or []) if str(column) in frame.columns
    ))
    if len(selected) > MAX_CORRELATION_COLUMNS:
        message = (
            f"Выбрано {len(selected)} столбцов. Максимум для одного анализа — "
            f"{MAX_CORRELATION_COLUMNS}."
        )
        return None, None, message, message
    if len(selected) < 2:
        message = "Выберите минимум два числовых столбца."
        return None, None, message, message

    numeric = frame[selected].apply(pd.to_numeric, errors="coerce")
    excluded = [
        column for column in selected
        if numeric[column].count() < min_periods or numeric[column].nunique(dropna=True) <= 1
    ]
    usable = [column for column in selected if column not in excluded]
    if len(usable) < 2:
        message = (
            "После исключения константных и недостаточно заполненных столбцов "
            "осталось меньше двух полей."
        )
        return None, None, message, message

    numeric = numeric[usable]
    correlation = numeric.corr(method=method, min_periods=min_periods)
    valid = numeric.notna().astype(int)
    pair_counts = valid.T.dot(valid)
    correlation = correlation.mask(pair_counts < min_periods)

    method_label = "Пирсон" if method == "pearson" else "Спирмен"
    status = (
        f"Метод: {method_label} · Столбцов: {len(usable)} · "
        f"Строк: {len(frame)} · Минимум наблюдений: {min_periods}"
    )
    if excluded:
        status += f" · Исключено столбцов: {len(excluded)}"
    return correlation, pair_counts, status, None


def build_correlogram(correlation, pair_counts, template="plotly"):
    """Тепловая карта матрицы корреляций (коррелограмма)."""
    count = len(correlation)
    matrix_height = max(620, min(1500, 38 * count + 260))
    show_values = count <= 15
    heatmap = go.Heatmap(
        z=correlation.to_numpy(),
        x=[str(column) for column in correlation.columns],
        y=[str(column) for column in correlation.index],
        customdata=pair_counts.to_numpy(),
        colorscale="RdBu",
        reversescale=True,
        zmin=-1,
        zmax=1,
        zmid=0,
        colorbar=dict(title="r", thickness=14),
        texttemplate="%{z:.2f}" if show_values else None,
        hoverongaps=False,
        hovertemplate=(
            "<b>%{y} ↔ %{x}</b><br>Корреляция: %{z:.4f}"
            "<br>Совместных наблюдений: %{customdata}<extra></extra>"
        ),
    )
    matrix = go.Figure(heatmap)
    matrix.update_layout(
        title=dict(
            text="Корреляционная матрица",
            x=0.5,
            xanchor="center",
        ),
        template=template,
        height=matrix_height,
        margin=dict(l=35, r=40, t=65, b=35),
        xaxis=dict(tickangle=-45, automargin=True, side="bottom"),
        yaxis=dict(automargin=True, autorange="reversed"),
    )
    return matrix


def _build_correlation_figures(frame, columns, method="pearson", min_periods=10,
                               template="plotly"):
    """Return matrix, two focused bars, status and an optional error."""
    correlation, pair_counts, status, error = compute_correlation(
        frame, columns, method, min_periods
    )
    if error:
        empty = _empty_analysis_figure(error, template)
        return empty, empty, empty, error, error

    first_target = correlation.columns[0]
    second_target = correlation.columns[1]
    first_bar = _correlation_bar(correlation, pair_counts, first_target, template)
    second_bar = _correlation_bar(correlation, pair_counts, second_target, template)

    matrix = build_correlogram(correlation, pair_counts, template)
    return matrix, first_bar, second_bar, status, None


class CorrelationWorkspace:
    """Own the correlation page layout and its callbacks."""

    def __init__(self, columns_control, *, prefix="correlation"):
        self.columns_control = columns_control
        self.ids = {
            "method": f"{prefix}-method",
            "min_periods": f"{prefix}-min-periods",
            "bar_primary": f"{prefix}-bar-primary",
            "bar_secondary": f"{prefix}-bar-secondary",
            "status": f"{prefix}-status",
            "columns_drop": f"{prefix}-columns-drop",
            "columns_sync": f"{prefix}-columns-sync",
        }
        self._callbacks_registered = False

    def render(self, matrix_block=None):
        """Слот матрицы занимает matrix_block — мультиграфик
        (Scatter Matrix / Parallel Coordinates / Коррелограмма)."""
        graph_config = {"displaylogo": False, "responsive": True}
        columns_drop_target = html.Div(
            [
                dmc.Group(
                    [
                        dmc.Text("Коррелируемые каналы", size="xs", fw=600),
                        dmc.Text("Можно перетащить из датасета", size="xs", c="dimmed"),
                    ],
                    justify="space-between",
                    gap="xs",
                    mb=4,
                ),
                self.columns_control,
            ],
            id=self.ids["columns_drop"],
            className="correlation-channel-drop",
            **{
                "data-drop-target": self.columns_control.id,
                "data-drop-mode": "append",
                "data-accept-type": "numeric",
                "data-current-value": "[]",
            },
        )
        controls = dmc.Paper(
            [
                dmc.Group(
                    [
                        dmc.Text("Анализ корреляционных зависимостей", fw=700, size="md"),
                        dmc.Group(
                            [
                                dmc.SegmentedControl(
                                    id=self.ids["method"],
                                    value="pearson",
                                    size="xs",
                                    data=[
                                        {"label": "Пирсон", "value": "pearson"},
                                        {"label": "Спирмен", "value": "spearman"},
                                    ],
                                    persistence=True,
                                ),
                                dmc.Group(
                                    [
                                        dmc.Text("Мин. N", size="xs", c="dimmed"),
                                        dmc.NumberInput(
                                            id=self.ids["min_periods"],
                                            value=10,
                                            min=2,
                                            step=1,
                                            size="xs",
                                            debounce=True,
                                            persistence=True,
                                            style={"width": "82px"},
                                        ),
                                    ],
                                    gap=6,
                                    wrap="nowrap",
                                ),
                            ],
                            gap="sm",
                            wrap="nowrap",
                        ),
                    ],
                    justify="space-between",
                    align="center",
                    wrap="wrap",
                    gap="xs",
                ),
                html.Div(columns_drop_target, style={"marginTop": "6px"}),
                dmc.Text(id=self.ids["status"], size="xs", c="dimmed", mt=4),
                dcc.Store(id=self.ids["columns_sync"]),
            ],
            p="xs",
            radius="md",
            withBorder=True,
            shadow="sm",
        )
        lower = dmc.Grid(
            [
                dmc.GridCol(
                    dmc.Paper(
                        dcc.Graph(
                            id=self.ids["bar_primary"],
                            figure=_empty_analysis_figure(),
                            config=graph_config,
                        ),
                        p="xs", radius="md", withBorder=True, shadow="sm",
                    ),
                    span=6,
                ),
                dmc.GridCol(
                    dmc.Paper(
                        dcc.Graph(
                            id=self.ids["bar_secondary"],
                            figure=_empty_analysis_figure(),
                            config=graph_config,
                        ),
                        p="xs", radius="md", withBorder=True, shadow="sm",
                    ),
                    span=6,
                ),
            ],
            mt="sm",
            gutter="sm",
        )
        blocks = [controls]
        if matrix_block is not None:
            blocks.append(html.Div(matrix_block, style={"marginTop": "10px"}))
        blocks.append(lower)
        return html.Div(
            blocks,
            id="correlation-workspace",
            # ``hidden`` creates an implicit vertical scroll container when it
            # is used only on one axis. ``clip`` trims long Plotly labels
            # without adding a second scrollbar beside the page scrollbar.
            style={"minWidth": 0, "overflowX": "clip"},
        )

    def register_callbacks(self, app):
        if self._callbacks_registered:
            return

        app.clientside_callback(
            f"""
            function(value) {{
                var target = document.getElementById('{self.ids["columns_drop"]}');
                if (!target) return window.dash_clientside.no_update;
                target.setAttribute('data-current-value', JSON.stringify(value || []));
                target.classList.toggle('has-value', Array.isArray(value) && value.length > 0);
                return Date.now();
            }}
            """,
            Output(self.ids["columns_sync"], "data"),
            Input(self.columns_control.id, "value"),
            prevent_initial_call="initial_duplicate",
        )

        @app.callback(
            Output(self.columns_control.id, "value"),
            Input("filtered-data", "data"),
            Input("url", "pathname"),
            State("meta-columns", "data"),
            State(self.columns_control.id, "value"),
        )
        def initialize_columns(filtered_json, pathname, meta, current):
            # Pages remain mounted and are only hidden with CSS. Do not scan a
            # newly loaded Excel dataset while correlation analysis is hidden.
            if pathname != "/correlation":
                return no_update
            if not filtered_json:
                return []
            all_columns = {
                str(column) for column in (meta or {}).get("columns", [])
            }
            numeric_columns = [
                str(column) for column in (meta or {}).get("numeric", [])
                if str(column) in all_columns
            ]
            if not numeric_columns:
                try:
                    frame = read_df_from_store(filtered_json, meta)
                    frame.columns = [str(column) for column in frame.columns]
                    numeric_columns = [
                        str(column) for column in frame.select_dtypes(include="number").columns
                    ]
                except Exception:
                    return []
            current = [column for column in (current or []) if column in numeric_columns]
            if len(current) >= 2:
                return no_update
            return numeric_columns[:DEFAULT_CORRELATION_COLUMNS]

        @app.callback(
            Output(self.ids["bar_primary"], "figure"),
            Output(self.ids["bar_secondary"], "figure"),
            Output(self.ids["status"], "children"),
            Output(self.ids["status"], "c"),
            Input("filtered-data", "data"),
            Input(self.columns_control.id, "value"),
            Input(self.ids["method"], "value"),
            Input(self.ids["min_periods"], "value"),
            Input("url", "pathname"),
            Input("dropdown_style", "value"),
            State("meta-columns", "data"),
        )
        def update_analysis(filtered_json, columns, method, min_periods, pathname, template, meta):
            if pathname != "/correlation":
                return no_update, no_update, no_update, no_update
            if not filtered_json:
                empty = _empty_analysis_figure("Сначала загрузите датасет", template)
                return empty, empty, "Сначала загрузите датасет.", "dimmed"
            if len(columns or []) < 2:
                empty = _empty_analysis_figure(
                    "Выберите минимум два числовых канала", template
                )
                return empty, empty, "Выберите минимум два числовых канала.", "dimmed"
            try:
                frame = read_df_from_store(filtered_json, meta)
                # Матрица строится мультиграфиком (тип «Correlogram»);
                # рейтинги корреляций используют тот же расчёт.
                _matrix, first, second, status, error = _build_correlation_figures(
                    frame, columns, method, min_periods, template or "plotly"
                )
                return first, second, status, "red" if error else "dimmed"
            except Exception as error:
                message = f"Не удалось рассчитать корреляции: {error}"
                empty = _empty_analysis_figure(message, template)
                return empty, empty, message, "red"

        self._callbacks_registered = True
