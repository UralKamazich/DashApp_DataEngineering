# -*- coding: utf-8 -*-
"""
Callbacks: основной график и диагностические графики кластеризации.
"""

import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import callback, Output, Input, State, no_update
from dash.exceptions import PreventUpdate

from dash_app import app
from utils import (
    read_df_from_store, _empty_fig, _make_error_notif,
    _sort_legend_traces, hide_xlabels_on_upper_facets,
    needs_text_axis,
)
from config import legend_config
from layout import GRAPH_WORKSPACE

logger = logging.getLogger(__name__)


Y_ONLY_CHART_TYPES = {
    "Scatter",
    "Box",
    "Bar",
    "Line",
    "Hist",
    "Pie",
    "Violin",
    "Ridge",
}

# Plotly renders Bar traces as SVG paths. Thousands of raw bars can leave
# Plotly.react in a state where subsequent figures no longer reach the canvas.
# Keep the raw mode useful for small datasets and guard the renderer for large
# ones. Labels have a lower limit because every value adds another SVG node.
MAX_BAR_POINTS = 800
MAX_BAR_LABELS = 250

# Иерархические графики собирают путь из Цвет → X → Y, поэтому X для
# них не обязателен — достаточно любого из трёх полей.
HIERARCHY_CHART_TYPES = {"Sunburst", "Treemap"}

RIDGE_GRID_SIZE = 180
MAX_RIDGE_GROUPS = 60
MAX_PIE_SLICES = 30


def _color_with_alpha(color, alpha=0.34):
    """Return a Plotly/CSS color with alpha when its format is known."""
    if not isinstance(color, str):
        return color
    value = color.strip()
    if value.startswith("#") and len(value) in (4, 7):
        if len(value) == 4:
            value = "#" + "".join(channel * 2 for channel in value[1:])
        red, green, blue = (int(value[index:index + 2], 16) for index in (1, 3, 5))
        return f"rgba({red},{green},{blue},{alpha})"
    if value.startswith("rgb("):
        return value.replace("rgb(", "rgba(").replace(")", f",{alpha})")
    return value


def _ridge_density(values, grid, fallback_bandwidth):
    """Calculate a small Gaussian KDE without a scipy runtime dependency."""
    values = np.asarray(values, dtype=float)
    sample_size = values.size
    if sample_size > 2500:
        indices = np.linspace(0, sample_size - 1, 2500, dtype=int)
        values = np.sort(values)[indices]
        sample_size = values.size

    if sample_size > 1:
        std = float(np.std(values, ddof=1))
        q25, q75 = np.percentile(values, [25, 75])
        robust_std = float(q75 - q25) / 1.34
        scales = [scale for scale in (std, robust_std) if np.isfinite(scale) and scale > 0]
        scale = min(scales) if scales else fallback_bandwidth
        bandwidth = 0.9 * scale * sample_size ** (-0.2)
    else:
        bandwidth = fallback_bandwidth

    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = fallback_bandwidth
    bandwidth = max(bandwidth, fallback_bandwidth * 0.08)

    distances = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * distances ** 2).mean(axis=1) / (
        bandwidth * np.sqrt(2 * np.pi)
    )


def _build_ridge_figure(plot_df, x_col, y_col, color_col, height, width,
                        template, category_order="trace"):
    """Build a stable ridgeline from SVG scatter polygons.

    One of X/Y is the numeric value axis. The other axis, when supplied, is
    the ridge category. Color can add a second categorical split.
    """
    axis_columns = [column for column in (x_col, y_col) if column in plot_df.columns]
    numeric_columns = [
        column for column in axis_columns
        if pd.api.types.is_numeric_dtype(plot_df[column])
    ]

    if len(numeric_columns) != 1:
        return None, (
            "Для Ridge Plot выберите ровно один числовой столбец на X или Y. "
            "Вторая ось может содержать категории."
        )

    value_col = numeric_columns[0]
    group_col = next((column for column in axis_columns if column != value_col), None)
    orientation = "h" if value_col == x_col else "v"
    color_col = color_col if color_col in plot_df.columns else None

    working_columns = list(dict.fromkeys(
        column for column in (value_col, group_col, color_col) if column
    ))
    ridge_df = plot_df[working_columns].copy()
    ridge_df[value_col] = pd.to_numeric(ridge_df[value_col], errors="coerce")
    ridge_df = ridge_df[np.isfinite(ridge_df[value_col])]
    if ridge_df.empty:
        return None, f"В столбце «{value_col}» нет числовых значений для Ridge Plot."

    split_columns = list(dict.fromkeys(
        column for column in (group_col, color_col) if column
    ))
    if split_columns:
        grouped = list(ridge_df.groupby(split_columns, sort=False, dropna=False))
    else:
        grouped = [((), ridge_df)]

    if len(grouped) > MAX_RIDGE_GROUPS:
        return None, (
            f"Ridge Plot получил {len(grouped)} групп. Оставьте не более "
            f"{MAX_RIDGE_GROUPS} с помощью фильтра или выберите столбец с меньшим числом категорий."
        )

    def display_value(value):
        return "(пусто)" if pd.isna(value) else str(value)

    records = []
    for key, subset in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        key_by_column = dict(zip(split_columns, key))
        group_value = key_by_column.get(group_col) if group_col else None
        color_value = key_by_column.get(color_col) if color_col else None
        if group_col and color_col and group_col != color_col:
            label = f"{display_value(group_value)} · {display_value(color_value)}"
        elif group_col:
            label = display_value(group_value)
        elif color_col:
            label = display_value(color_value)
        else:
            label = str(value_col)
        records.append({
            "label": label,
            "color_value": display_value(color_value) if color_col else None,
            "values": subset[value_col].to_numpy(dtype=float),
        })

    if category_order in ("category ascending", "category descending"):
        records.sort(
            key=lambda item: item["label"].casefold(),
            reverse=category_order.endswith("descending"),
        )
    elif category_order in ("total ascending", "total descending"):
        records.sort(
            key=lambda item: len(item["values"]),
            reverse=category_order.endswith("descending"),
        )

    all_values = ridge_df[value_col].to_numpy(dtype=float)
    value_min = float(np.min(all_values))
    value_max = float(np.max(all_values))
    value_span = value_max - value_min
    fallback_bandwidth = max(value_span * 0.03, abs(value_min) * 0.01, 1e-9)
    padding = max(value_span * 0.04, fallback_bandwidth * 2)
    grid = np.linspace(value_min - padding, value_max + padding, RIDGE_GRID_SIZE)

    palette = px.colors.qualitative.Plotly
    color_values = list(dict.fromkeys(
        record["color_value"] for record in records if record["color_value"] is not None
    ))
    color_map = {
        value: palette[index % len(palette)]
        for index, value in enumerate(color_values)
    }
    shown_legend_values = set()
    fig = go.Figure()

    for baseline, record in enumerate(records):
        density = _ridge_density(record["values"], grid, fallback_bandwidth)
        density_peak = float(np.max(density))
        ridge_height = density / density_peak * 0.82 if density_peak > 0 else density
        color_value = record["color_value"]
        color = color_map.get(color_value, palette[0])
        show_legend = color_value is not None and color_value not in shown_legend_values
        if show_legend:
            shown_legend_values.add(color_value)

        baseline_values = np.full(grid.shape, baseline, dtype=float)
        if orientation == "h":
            trace_x = np.concatenate(([grid[0]], grid, [grid[-1]]))
            trace_y = np.concatenate(([baseline], baseline_values + ridge_height, [baseline]))
            hover_template = (
                f"<b>{record['label']}</b><br>{value_col}: %{{x:.4g}}"
                "<br>Плотность: %{customdata:.4g}<extra></extra>"
            )
        else:
            trace_x = np.concatenate(([baseline], baseline_values + ridge_height, [baseline]))
            trace_y = np.concatenate(([grid[0]], grid, [grid[-1]]))
            hover_template = (
                f"<b>{record['label']}</b><br>{value_col}: %{{y:.4g}}"
                "<br>Плотность: %{customdata:.4g}<extra></extra>"
            )

        fig.add_trace(go.Scatter(
            x=trace_x,
            y=trace_y,
            customdata=np.concatenate(([0.0], density, [0.0])),
            mode="lines",
            fill="toself",
            fillcolor=_color_with_alpha(color),
            line=dict(color=color, width=1.5),
            name=color_value or record["label"],
            legendgroup=color_value or record["label"],
            showlegend=show_legend,
            hovertemplate=hover_template,
        ))

    tick_values = list(range(len(records)))
    tick_labels = [record["label"] for record in records]
    category_axis = dict(
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_labels,
        range=[-0.12, max(len(records) - 1 + 0.95, 0.95)],
        showgrid=False,
        zeroline=False,
        title=group_col or color_col or "",
    )
    value_axis = dict(title=value_col, zeroline=False)
    fig.update_layout(
        height=height,
        width=width,
        template=template,
        hovermode="closest",
        xaxis=value_axis if orientation == "h" else category_axis,
        yaxis=category_axis if orientation == "h" else value_axis,
    )
    return fig, None


def _pie_hover_summary(series):
    """Produce a compact group summary for an extra Pie hover column."""
    values = series.dropna()
    if values.empty:
        return "—"
    if pd.api.types.is_numeric_dtype(values):
        minimum = float(values.min())
        maximum = float(values.max())
        return f"{minimum:.4g}" if minimum == maximum else f"{minimum:.4g} … {maximum:.4g}"
    unique_values = list(dict.fromkeys(str(value) for value in values))
    summary = ", ".join(unique_values[:3])
    return summary + (" …" if len(unique_values) > 3 else "")


def _build_pie_figure(plot_df, x_col, y_col, color_col, aggregation,
                      height, width, template, hover_cols=None):
    """Build a Pie using counts or an explicit category/value aggregation."""
    axis_columns = list(dict.fromkeys(
        column for column in (x_col, y_col) if column in plot_df.columns
    ))
    numeric_columns = [
        column for column in axis_columns
        if pd.api.types.is_numeric_dtype(plot_df[column])
    ]
    if len(numeric_columns) > 1:
        return None, (
            "Для круговой диаграммы выберите не более одного числового столбца "
            "на осях X/Y. Второй столбец должен содержать категории."
        )

    value_col = numeric_columns[0] if numeric_columns else None
    category_cols = [column for column in axis_columns if column != value_col]
    color_col = color_col if color_col in plot_df.columns else None
    if color_col and color_col != value_col and color_col not in category_cols:
        category_cols.append(color_col)

    aggregation = aggregation if aggregation in {"sum", "mean", "count"} else "sum"
    aggregation_labels = {
        "sum": "Сумма",
        "mean": "Среднее",
        "count": "Количество",
    }
    hover_cols = [hover_cols] if isinstance(hover_cols, str) else (hover_cols or [])
    requested_hover = [
        column for column in (hover_cols or [])
        if column in plot_df.columns and column not in category_cols and column != value_col
    ]
    requested_hover = list(dict.fromkeys(requested_hover))

    working_columns = list(dict.fromkeys(
        column for column in [*category_cols, value_col, *requested_hover] if column
    ))
    pie_df = plot_df[working_columns].copy()
    for column in category_cols:
        pie_df[column] = pie_df[column].astype(object).where(
            pie_df[column].notna(), "(пусто)"
        ).astype(str)

    numeric_only = value_col is not None and not category_cols
    if numeric_only:
        pie_df[value_col] = pd.to_numeric(pie_df[value_col], errors="coerce")
        pie_df = pie_df[np.isfinite(pie_df[value_col])]
        if pie_df.empty:
            return None, f"В столбце «{value_col}» нет числовых значений."
        bin_count = min(10, max(int(pie_df[value_col].nunique()), 1))
        pie_df["__pie_bin__"] = pd.cut(
            pie_df[value_col], bins=bin_count, include_lowest=True, duplicates="drop"
        ).astype(str)
        category_cols = ["__pie_bin__"]
        grouped = pie_df.groupby(category_cols, sort=False, dropna=False)
        metric = grouped.size().astype(float)
        records = grouped.size()
        metric_weights = records
        metric_label = "Количество значений"
        title = f"Распределение «{value_col}»"
    else:
        if not category_cols:
            return None, "Для круговой диаграммы нужен категориальный столбец."
        grouped = pie_df.groupby(category_cols, sort=False, dropna=False)
        records = grouped.size()
        if value_col:
            pie_df[value_col] = pd.to_numeric(pie_df[value_col], errors="coerce")
            grouped = pie_df.groupby(category_cols, sort=False, dropna=False)
            if aggregation == "mean":
                metric = grouped[value_col].mean()
                metric_weights = grouped[value_col].count()
            elif aggregation == "count":
                metric = grouped[value_col].count().astype(float)
                metric_weights = metric
            else:
                metric = grouped[value_col].sum(min_count=1)
                metric_weights = grouped[value_col].count()
            metric_label = f"{aggregation_labels[aggregation]} «{value_col}»"
            title = f"{metric_label} по «{' · '.join(category_cols)}»"
        else:
            metric = records.astype(float)
            metric_weights = records
            metric_label = "Количество записей"
            title = f"Количество записей по «{' · '.join(category_cols)}»"

    aggregated = pd.DataFrame({
        "__value__": metric,
        "__records__": records,
        "__metric_weight__": metric_weights,
    })
    hover_aliases = []
    for index, column in enumerate(requested_hover):
        alias = f"__hover_{index}__"
        aggregated[alias] = grouped[column].agg(_pie_hover_summary)
        hover_aliases.append((alias, column))
    aggregated = aggregated.reset_index()
    aggregated = aggregated[np.isfinite(aggregated["__value__"])]
    if aggregated.empty:
        return None, "После агрегации не осталось значений для круговой диаграммы."
    if (aggregated["__value__"] < 0).any():
        return None, "Круговая диаграмма не поддерживает отрицательные итоговые значения."
    aggregated = aggregated[aggregated["__value__"] > 0]
    if aggregated.empty:
        return None, "Сумма значений круговой диаграммы должна быть больше нуля."

    def make_label(row):
        return " · ".join(str(row[column]) for column in category_cols)

    aggregated["__label__"] = aggregated.apply(make_label, axis=1)
    aggregated = aggregated.sort_values("__value__", ascending=False, kind="stable")
    if len(aggregated) > MAX_PIE_SLICES:
        visible = aggregated.iloc[:MAX_PIE_SLICES - 1].copy()
        remainder = aggregated.iloc[MAX_PIE_SLICES - 1:]
        remainder_records = int(remainder["__records__"].sum())
        if value_col and aggregation == "mean":
            remainder_weight = int(remainder["__metric_weight__"].sum())
            remainder_value = float(np.average(
                remainder["__value__"], weights=remainder["__metric_weight__"]
            )) if remainder_weight else 0.0
        else:
            remainder_weight = int(remainder["__metric_weight__"].sum())
            remainder_value = float(remainder["__value__"].sum())
        other = {
            "__label__": "Остальные",
            "__value__": remainder_value,
            "__records__": remainder_records,
            "__metric_weight__": remainder_weight,
        }
        for column in category_cols:
            other[column] = "Остальные" if column == category_cols[0] else ""
        for alias, _column in hover_aliases:
            other[alias] = "—"
        aggregated = pd.concat([visible, pd.DataFrame([other])], ignore_index=True)

    custom_columns = ["__records__", *(alias for alias, _column in hover_aliases)]
    color_argument = color_col if color_col in aggregated.columns else None
    fig = px.pie(
        aggregated,
        values="__value__",
        names="__label__",
        color=color_argument,
        custom_data=custom_columns,
        title=title,
        height=height,
        width=width,
        template=template,
    )
    hover_template = (
        f"<b>%{{label}}</b><br>{metric_label}: %{{value:,.4g}}"
        "<br>Доля: %{percent}<br>Строк: %{customdata[0]}"
    )
    for index, (_alias, column) in enumerate(hover_aliases, start=1):
        hover_template += f"<br>{column}: %{{customdata[{index}]}}"
    hover_template += "<extra></extra>"
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label+value",
        insidetextorientation="auto",
        hovertemplate=hover_template,
    )
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")
    return fig, None


def _aggregate_bar_frame(plot_df, x_col, y_col, color_col, facet_row, facet_col,
                         aggregation):
    """Схлопнуть повторяющиеся категории Bar в один столбец.

    Группировка идёт по X + Цвет + фасетам; пропуски категориальных ключей
    помечаются «(пусто)», как в круговой диаграмме. Если агрегация
    неприменима (нет ключей/режим «как есть»), кадр возвращается без изменений.
    """
    if aggregation not in {"sum", "mean", "count"}:
        return plot_df
    keys = [c for c in [x_col, color_col, facet_row, facet_col] if c and c in plot_df.columns]
    if not keys:
        return plot_df
    if not y_col or y_col not in plot_df.columns or y_col in keys:
        return plot_df
    if aggregation != "count" and not pd.api.types.is_numeric_dtype(plot_df[y_col]):
        return plot_df

    working = plot_df[list(dict.fromkeys(keys + [y_col]))].copy()
    for column in keys:
        if not pd.api.types.is_numeric_dtype(working[column]):
            working[column] = (
                working[column]
                .where(pd.notna(working[column]), "(пусто)")
                .astype(str)
                .str.strip()
            )
            working.loc[working[column] == "", column] = "(пусто)"

    if aggregation == "count":
        grouped = working.groupby(keys, sort=False, dropna=False).size()
    else:
        values = working[y_col]
        series = values.groupby([working[key] for key in keys], sort=False)
        grouped = series.sum() if aggregation == "sum" else series.mean()
    result = grouped.reset_index()
    if aggregation == "count":
        # size() создаёт безымянный столбец 0 — возвращаем ему имя Y,
        # чтобы px.bar продолжал работать без изменений.
        result = result.rename(columns={0: y_col})
    return result


def _prepare_bar_frame(plot_df, x_col, y_col, color_col, facet_row, facet_col,
                       aggregation):
    """Return a renderer-safe Bar frame and an optional user notification."""
    effective_aggregation = aggregation
    automatic = False
    notice = []

    if aggregation == "none" and len(plot_df) > MAX_BAR_POINTS:
        automatic = True
        effective_aggregation = (
            "sum"
            if y_col and y_col in plot_df.columns
            and pd.api.types.is_numeric_dtype(plot_df[y_col])
            else "count"
        )
        notice = [{
            "id": "bar-auto-aggregation",
            "title": "Bar: безопасная отрисовка",
            "message": (
                f"Режим «как есть» создаёт {len(plot_df)} столбцов. "
                f"Применена агрегация: {effective_aggregation}."
            ),
            "color": "yellow",
            "action": "show",
            "autoClose": 6500,
        }]

    bar_df = _aggregate_bar_frame(
        plot_df, x_col, y_col, color_col, facet_row, facet_col,
        effective_aggregation,
    )
    if len(bar_df) > MAX_BAR_POINTS:
        return None, _make_error_notif(
            f"Bar содержит {len(bar_df)} столбцов — это слишком много для "
            "стабильной отрисовки. Выберите агрегацию, другой X или сузьте данные фильтром."
        ), None, automatic
    applied_aggregation = effective_aggregation if bar_df is not plot_df else None
    return bar_df, notice, applied_aggregation, automatic


def _bar_aggregation_caption(aggregation, y_col, group_columns, automatic=False):
    """Explain the statistical meaning of an aggregated Bar directly on it."""
    if aggregation not in {"sum", "mean", "count"}:
        return None
    groups = [str(column) for column in group_columns if column]
    grouped_by = ", ".join(f"«{column}»" for column in groups)
    prefix = "Автоагрегация" if automatic else "Агрегация"
    if aggregation == "count":
        meaning = "количество строк"
    else:
        operation = "сумма" if aggregation == "sum" else "среднее"
        meaning = f"{operation} «{y_col}»"
    suffix = f" по {grouped_by}" if grouped_by else ""
    return f"<b>{prefix}:</b> {meaning}{suffix}"


def _prepare_hierarchy_frame(plot_df, path, values):
    """Рабочий кадр для Sunburst/Treemap без пропусков в уровнях пути.

    Plotly превращает NaN в пустые метки и отказывается строить иерархию
    («Non-leaves rows are not permitted»), поэтому пропуски каждого уровня
    помечаются «(пусто)» — как в круговой диаграмме.
    """
    frame = plot_df[list(dict.fromkeys([*path, *( [values] if values else [])]))].copy()
    for column in path:
        frame[column] = (
            frame[column]
            .where(pd.notna(frame[column]), "(пусто)")
            .astype(str)
            .str.strip()
        )
        frame.loc[frame[column] == "", column] = "(пусто)"
    return frame


def _graph_uirevision(chart_type, x_col, y_col, z_col, facet_row, facet_col, view_revision=0):
    """Keep user zoom while the chart keeps the same coordinate system."""
    parts = (chart_type, x_col, y_col, z_col, facet_row, facet_col, view_revision or 0)
    return "graph-view:" + "|".join("" if value is None else str(value) for value in parts)


def _primary_axis_errors(chart_type, x_col, y_col, color_col, columns):
    """Validate X/Y while allowing ordinary charts to use only Y."""
    errors = []
    x_valid = bool(x_col) and x_col in columns
    y_valid = bool(y_col) and y_col in columns

    if x_col and not x_valid:
        errors.append(f"Не существует столбец X: {x_col}")
    if y_col and not y_valid:
        errors.append(f"Не существует столбец Y: {y_col}")
    if chart_type in HIERARCHY_CHART_TYPES:
        if not any((color_col, x_col, y_col)):
            errors.append("Для Sunburst/Treemap выберите хотя бы один столбец из Цвет/X/Y")
    elif not x_col and not (chart_type in Y_ONLY_CHART_TYPES and y_valid):
        errors.append("Не выбран столбец X")

    return errors


@GRAPH_WORKSPACE.figure_callback(
    app,
    Input(GRAPH_WORKSPACE.ids["update"], "n_clicks"),
    Input(GRAPH_WORKSPACE.field_ids["dropdown_x"], "value"),
    Input(GRAPH_WORKSPACE.field_ids["dropdown_y"], "value"),
    Input(GRAPH_WORKSPACE.field_ids["dropdown_z"], "value"),
    Input(GRAPH_WORKSPACE.field_ids["dropdown_color"], "value"),
    Input(GRAPH_WORKSPACE.field_ids["dropdown_size"], "value"),
    Input(GRAPH_WORKSPACE.field_ids["dropdown_text"], "value"),
    Input("dropdown_text_pozition", "value"),
    Input(GRAPH_WORKSPACE.chart_type_id, "value"),
    Input("SwitchBubble", "checked"),
    Input("InputMaxSizeBubble", "value"),
    Input("InputSizePlot", "value"),
    Input("InputSizePlotW", "value"),
    Input("dropdown_style", "value"),
    Input("bar-text-auto", "checked"),
    Input(GRAPH_WORKSPACE.ids["view_revision"], "data"),

    Input("filtered-data", "data"),
    Input(GRAPH_WORKSPACE.field_ids["dropdown_hover_data"], "value"),
    Input(GRAPH_WORKSPACE.field_ids["dropdown_facet_row"], "value"),
    Input(GRAPH_WORKSPACE.field_ids["dropdown_facet_col"], "value"),
    State("filters-applied-state", "data"),
    Input("font-size-xaxis", "value"),
    Input("font-size-yaxis", "value"),
    Input("font-size-ticks", "value"),
    Input("font-size-title", "value"),
    Input("dropdown_category_ascending", "value"),
    Input("dropdown_axes_category", "value"),
    Input("dropdown_overlay", "value"),
    Input("dropdown_legend", "value"),
    State(GRAPH_WORKSPACE.ids["custom_colors"], "data"),
    Input("tick-step-xaxis", "value"),
    Input("tick-step-yaxis", "value"),
    Input("dropdown_legend_order", "value"),
    State("input_legend_custom_order", "value"),
    State("meta-columns", "data"),
    Input("dropdown_pie_aggregation", "value"),
    Input("bar-aggregation", "value"),

    prevent_initial_call=True,
)
def update_main_graph(n_clicks, x_col, y_col, z_col, color_col, size_col, text_col, dropdown_text_pozition,
                      chart_type, bubble, MaxSizeBubble, height, width, selected_style, bar_text_auto,
                      view_revision,
                      filtered_json, hover_cols, facet_row, facet_col, filters_state,
                      xaxis_font_size, yaxis_font_size, font_size_ticks, title_font_size,
                      dropdown_sort_column, axes_category, dropdown_overlay, legend, custom_colors,
                      tick_step_x, tick_step_y, legend_order, legend_custom_order, meta,
                      pie_aggregation="sum",
                      bar_aggregation="sum"):

    empty = _empty_fig()
    try:
        if not filtered_json:
            return empty, []
        dff = read_df_from_store(filtered_json, meta)
        if dff is None or dff.empty:
            return empty, []

        # A deliberately cleared workspace is a valid state. Keep the loaded
        # dataframe in its stores and show a clean canvas without an error.
        assigned_fields = (
            x_col, y_col, z_col, color_col, size_col, text_col,
            facet_row, facet_col,
        )
        if not any(assigned_fields) and not hover_cols:
            return empty, []

        errors = _primary_axis_errors(chart_type, x_col, y_col, color_col, dff.columns)
        if chart_type == "3D_Scatter" and (not z_col or z_col not in dff.columns):
            errors.append("Для 3D требуется столбец Z")
        if errors:
            notif = _make_error_notif(" ".join(errors))
            return empty, notif

        facet_row = facet_row if (facet_row and facet_row in dff.columns) else None
        facet_col = facet_col if (facet_col and facet_col in dff.columns) else None
        text_data = dff[text_col] if (text_col and text_col in dff.columns and not dff.empty) else None

        plot_df = dff.copy()
        def _valid(col):
            return bool(col) and (col in plot_df.columns)
        carg = color_col if _valid(color_col) else None
        sarg = size_col  if (bubble and _valid(size_col)) else None

        meta = meta or {"numeric": [], "categorical": [], "datetime": []}
        x_as_text = needs_text_axis(x_col, meta)
        y_as_text = needs_text_axis(y_col, meta)
        if x_as_text:
            plot_df[x_col] = plot_df[x_col].astype(str)
        if y_as_text:
            plot_df[y_col] = plot_df[y_col].astype(str)

        fig = go.Figure()
        category_orders = {}

        if isinstance(filters_state, dict) and len(filters_state) > 0:
            first_key = sorted(filters_state.keys())[0]
            first_filter = filters_state[first_key]
            filter_col = first_filter.get("column")
            filter_values = first_filter.get("value")
            if isinstance(filter_values, (int, float, str)):
                filter_values = [filter_values]
            if filter_col and isinstance(filter_values, list) and filter_values:
                if filter_col == facet_row or filter_col == facet_col:
                    category_orders = {filter_col: filter_values}
        if facet_row is None and facet_col is None:
            category_orders = None

        # ---- ТИПЫ ГРАФИКОВ ----
        if chart_type == "Scatter":
            fig = px.scatter(
                plot_df, x=x_col, y=y_col, color=carg, size=sarg,
                size_max=MaxSizeBubble, height=height, width=width, hover_data=hover_cols,
                facet_row=facet_row, facet_col=facet_col, text=text_data,
                category_orders=category_orders, template=selected_style,
                # Plotly's WebGL trace can remain in a broken pending state
                # when an existing numeric axis is replaced by a categorical
                # one. SVG is stable for categorical axes and data labels;
                # fully numeric point clouds can still use WebGL automatically.
                render_mode=(
                    "svg"
                    if text_data is not None or x_as_text or y_as_text
                    else "auto"
                ),
            )
            if text_data is not None:
                fig.update_traces(textposition=dropdown_text_pozition, textfont=dict(size=font_size_ticks), selector=dict(mode='markers+text'))

        elif chart_type == "3D_Scatter":
            fig = px.scatter_3d(
                plot_df, x=x_col, y=y_col, z=z_col, color=carg, size=sarg,
                size_max=MaxSizeBubble, height=height, width=width, hover_data=hover_cols,
                text=text_data, template=selected_style
            )
            if text_data is not None:
                fig.update_traces(textposition=dropdown_text_pozition, textfont=dict(size=font_size_ticks), selector=dict(mode='markers+text'))

        elif chart_type == "Box":
            fig = px.box(
                plot_df, x=x_col, y=y_col, color=carg,
                height=height, width=width, facet_row=facet_row, facet_col=facet_col,
                category_orders=category_orders, template=selected_style
            )
            fig.update_layout(boxmode="group")

        elif chart_type == "Bar":
            bar_df, bar_notifications, applied_bar_aggregation, automatic_bar_aggregation = _prepare_bar_frame(
                plot_df, x_col, y_col, carg, facet_row, facet_col, bar_aggregation
            )
            if bar_df is None:
                return empty, bar_notifications
            fig = px.bar(
                bar_df, x=x_col, y=y_col, color=carg,
                height=height, width=width, facet_row=facet_row, facet_col=facet_col,
                text_auto=bool(bar_text_auto and len(bar_df) <= MAX_BAR_LABELS),
                category_orders=category_orders, template=selected_style
            )
            if dropdown_overlay in {'group', 'overlay', 'stack', 'relative'}:
                fig.update_layout(barmode=dropdown_overlay)
            if dropdown_overlay == 'overlay':
                fig.update_traces(opacity=0.85)
            aggregation_caption = _bar_aggregation_caption(
                applied_bar_aggregation,
                y_col,
                [x_col, carg, facet_row, facet_col],
                automatic=automatic_bar_aggregation,
            )
            if aggregation_caption:
                fig.add_annotation(
                    text=aggregation_caption,
                    x=0.5,
                    y=1.025,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    yanchor="bottom",
                    showarrow=False,
                    font=dict(size=11, color="#364152"),
                    bgcolor="rgba(248, 250, 252, 0.94)",
                    bordercolor="rgba(134, 142, 150, 0.35)",
                    borderwidth=1,
                    borderpad=4,
                )

        elif chart_type == "Line":
            fig = px.line(
                plot_df, x=x_col, y=y_col, color=carg,
                height=height, width=width, facet_row=facet_row, facet_col=facet_col,
                category_orders=category_orders, template=selected_style
            )

        elif chart_type == "Hist":
            fig = px.histogram(
                plot_df, x=x_col, y=y_col if not x_col else None, color=carg,
                height=height, width=width, facet_row=facet_row, facet_col=facet_col,
                category_orders=category_orders, template=selected_style
            )
            fig.update_layout(barmode=dropdown_overlay)
            if dropdown_overlay == 'overlay':
                fig.update_traces(opacity=0.75)

        elif chart_type == "Polar":
            fig = px.scatter_ternary(
                plot_df, a=x_col, b=y_col, c=z_col, color=carg, size=sarg,
                size_max=MaxSizeBubble, height=height, width=width, hover_data=hover_cols,
                text=text_data, template=selected_style
            )
            if text_data is not None:
                fig.update_traces(textposition=dropdown_text_pozition, textfont=dict(size=font_size_ticks), selector=dict(mode='markers+text'))

        elif chart_type == "Pie":
            fig, pie_error = _build_pie_figure(
                plot_df, x_col, y_col, carg, pie_aggregation,
                height, width, selected_style, hover_cols,
            )
            if pie_error:
                return empty, _make_error_notif(pie_error)

        elif chart_type == "Violin":
            fig = px.violin(
                plot_df, x=x_col, y=y_col, color=carg,
                height=height, width=width, facet_row=facet_row, facet_col=facet_col,
                category_orders=category_orders, template=selected_style, box=False
            )

        elif chart_type == "Ridge":
            fig, ridge_error = _build_ridge_figure(
                plot_df, x_col, y_col, carg, height, width, selected_style,
                dropdown_sort_column,
            )
            if ridge_error:
                return empty, _make_error_notif(ridge_error)

        # ScatterMatrix и Parcoords перенесены на страницу корреляционного
        # анализа: их строит callbacks/multivariate.py в MULTIVARIATE_WORKSPACE.

        elif chart_type in ("Sunburst", "Treemap"):
            path = [c for c in [color_col, x_col, y_col] if c and (c in plot_df.columns)]
            if not path:
                notif = _make_error_notif("Для Sunburst/Treemap нужен хотя бы один категориальный столбец из Color/X/Y.")
                return empty, notif
            values = None
            if y_col and (y_col in plot_df.columns) and pd.api.types.is_numeric_dtype(plot_df[y_col]):
                values = y_col
            color_kw = {}
            if carg and (carg in plot_df.columns) and (carg not in path):
                color_kw["color"] = carg
            hierarchy_df = _prepare_hierarchy_frame(plot_df, path, values)
            if chart_type == "Treemap":
                fig = px.treemap(
                    hierarchy_df, path=path, values=values,
                    height=height, width=width, template=selected_style, **color_kw
                )
            else:
                fig = px.sunburst(
                    hierarchy_df, path=path, values=values,
                    height=height, width=width, template=selected_style, **color_kw
                )

        elif chart_type in ("DensityHeat", "DensityContour"):
            if not x_col or not y_col or (x_col not in plot_df.columns) or (y_col not in plot_df.columns):
                notif = _make_error_notif("Для 2D-плотности нужны X и Y.")
                return empty, notif
            if (not pd.api.types.is_numeric_dtype(plot_df[x_col])) or (not pd.api.types.is_numeric_dtype(plot_df[y_col])):
                notif = _make_error_notif("Для 2D-плотности X и Y должны быть числовыми.")
                return empty, notif
            if chart_type == "DensityHeat":
                fig = px.density_heatmap(
                    plot_df, x=x_col, y=y_col, color_continuous_scale="Viridis",
                    height=height, width=width, template=selected_style,
                    facet_row=facet_row, facet_col=facet_col, category_orders=category_orders
                )
            else:
                fig = px.density_contour(
                    plot_df, x=x_col, y=y_col, color=carg,
                    height=height, width=width, template=selected_style,
                    facet_row=facet_row, facet_col=facet_col, category_orders=category_orders
                )
                fig.update_traces(contours_coloring="fill", contours_showlines=False)
            hide_xlabels_on_upper_facets(fig)

        # Пользовательские цвета
        if isinstance(custom_colors, dict) and custom_colors:
            try:
                for i, trace in enumerate(fig.data or []):
                    idx = str(i)
                    if idx in custom_colors:
                        selected_color = custom_colors[idx]
                        if chart_type == "Ridge" and isinstance(trace, go.Scatter):
                            trace.line.color = selected_color
                            trace.fillcolor = _color_with_alpha(selected_color)
                        else:
                            trace.setdefault("marker", {})
                            if isinstance(trace["marker"], dict):
                                trace["marker"]["color"] = selected_color
            except Exception as _e:
                logger.warning(f"custom_colors apply skipped: {_e}")

        # Применяем шрифт тиков через полный объект tickfont (семейство + размер)
        if chart_type == "Ridge":
            fig.update_xaxes(tickfont=dict(size=xaxis_font_size))
            fig.update_yaxes(tickfont=dict(size=yaxis_font_size))
        elif x_as_text:
            fig.update_xaxes(type='category', categoryorder=dropdown_sort_column,
                             tickfont=dict(size=xaxis_font_size))
        else:
            fig.update_xaxes(tickfont=dict(size=xaxis_font_size),
                             dtick=tick_step_x if tick_step_x and tick_step_x > 0 else None)

        if chart_type == "Ridge":
            pass
        elif y_as_text:
            fig.update_yaxes(type='category', categoryorder=dropdown_sort_column,
                             tickfont=dict(size=yaxis_font_size))
        else:
            fig.update_yaxes(tickfont=dict(size=yaxis_font_size),
                             dtick=tick_step_y if tick_step_y and tick_step_y > 0 else None)

        if axes_category == "x" and x_as_text:
            fig.update_xaxes(categoryorder=dropdown_sort_column)
        elif axes_category == "y" and y_as_text:
            fig.update_yaxes(categoryorder=dropdown_sort_column)

        fig.update_layout(
            legend=legend_config.get(legend, {}),
            legend_title_text=None,
            xaxis_title_font=dict(size=font_size_ticks),
            yaxis_title_font=dict(size=font_size_ticks),
            title_font=dict(size=title_font_size),
            template=selected_style,
            # Plotly preserves axis ranges, 3D camera and other direct user
            # interactions while this key stays unchanged. Labels, colors and
            # styling intentionally do not participate in the key.
            uirevision=_graph_uirevision(
                chart_type, x_col, y_col, z_col, facet_row, facet_col,
                view_revision,
            ),
        )

        try:
            _sort_legend_traces(fig, legend_order, legend_custom_order)
        except Exception as _e:
            logger.warning(f"Сортировка легенды пропущена: {_e}")

        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
        if chart_type != "3D_Scatter" and facet_row:
           hide_xlabels_on_upper_facets(fig)

        return fig, bar_notifications if chart_type == "Bar" else []

    except Exception as e:
        logger.error(f"Ошибка при построении графика: {e}", exc_info=True)
        notif = _make_error_notif(f"Ошибка отрисовки графика: {str(e)}. Попробуйте изменить параметры.")
        return empty, notif


# ============ Метрики кластеризации ============
@app.callback(
    Output("cluster-elbow-graph", "figure"),
    Output("cluster-silhouette-graph", "figure"),
    Output("cluster-metrics-section", "style"),
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),
    Input("cluster-metrics", "data"),
    State("dropdown_style", "value"),
    prevent_initial_call=True
)
def update_cluster_metric_graphs(cluster_metrics, selected_style):

    empty = _empty_fig()
    SHOW = {"display": "block"}
    HIDE = {"display": "none"}

    try:
        aux1, aux2 = empty, empty
        if isinstance(cluster_metrics, dict):
            try:
                ks = (cluster_metrics.get("ks") or cluster_metrics.get("K") or [])[:]
                inertias = cluster_metrics.get("inertias") or []
                sils = cluster_metrics.get("silhouettes") or []
                if ks and inertias:
                    df_in = pd.DataFrame({"K": ks, "Inertia": inertias})
                    aux1 = px.line(df_in, x="K", y="Inertia", template=selected_style, title="Метод локтя")
                    aux1.update_layout(height=400, margin=dict(l=50, r=20, t=40, b=40))
                if ks and sils:
                    df_s = pd.DataFrame({"K": ks, "Silhouette": sils})
                    aux2 = px.line(df_s, x="K", y="Silhouette", template=selected_style, title="Силуэтный метод")
                    aux2.update_layout(height=400, margin=dict(l=60, r=20, t=40, b=40))
            except Exception as e:
                logger.warning(f"Не удалось построить локоть/силуэт: {e}")

        has_any = (len(aux1.data or []) > 0) or (len(aux2.data or []) > 0)
        return aux1 if has_any else empty, aux2 if has_any else empty, (SHOW if has_any else HIDE), []

    except Exception as e:
        logger.error(f"Ошибка при построении метрик кластеризации: {e}", exc_info=True)
        notif = _make_error_notif(f"Ошибка в метриках кластеризации: {str(e)}")
        return empty, empty, HIDE, notif
