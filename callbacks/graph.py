# -*- coding: utf-8 -*-
"""
Callbacks: основной график (update_main_graph) и нижние графики (update_lower_graphs).
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


def _graph_uirevision(chart_type, x_col, y_col, z_col, facet_row, facet_col, view_revision=0):
    """Keep user zoom while the chart keeps the same coordinate system."""
    parts = (chart_type, x_col, y_col, z_col, facet_row, facet_col, view_revision or 0)
    return "graph-view:" + "|".join("" if value is None else str(value) for value in parts)


def _primary_axis_errors(chart_type, x_col, y_col, columns):
    """Validate X/Y while allowing ordinary charts to use only Y."""
    errors = []
    x_valid = bool(x_col) and x_col in columns
    y_valid = bool(y_col) and y_col in columns

    if x_col and not x_valid:
        errors.append(f"Не существует столбец X: {x_col}")
    if y_col and not y_valid:
        errors.append(f"Не существует столбец Y: {y_col}")
    if not x_col and not (chart_type in Y_ONLY_CHART_TYPES and y_valid):
        errors.append("Не выбран столбец X")

    return errors


@app.callback(
    Output("graph", "figure"),
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),

    Input("update-graf", "n_clicks"),
    Input("dropdown_x", "value"),
    Input("dropdown_y", "value"),
    Input("dropdown_z", "value"),
    Input("dropdown_color", "value"),
    Input("dropdown_size", "value"),
    Input("dropdown_text", "value"),
    Input("dropdown_text_pozition", "value"),
    Input("segmented", "value"),
    Input("SwitchBubble", "checked"),
    Input("InputMaxSizeBubble", "value"),
    Input("InputSizePlot", "value"),
    Input("InputSizePlotW", "value"),
    Input("dropdown_style", "value"),
    Input("bar-text-auto", "checked"),
    Input("graph-view-revision", "data"),

    State("filtered-data", "data"),
    State("dropdown_hover_data", "value"),
    State("dropdown_corr_columns", "value"),
    Input("dropdown_facet_row", "value"),
    Input("dropdown_facet_col", "value"),
    State("filters-state", "data"),
    Input("font-size-xaxis", "value"),
    Input("font-size-yaxis", "value"),
    Input("font-size-ticks", "value"),
    Input("font-size-title", "value"),
    Input("dropdown_category_ascending", "value"),
    Input("dropdown_axes_category", "value"),
    Input("dropdown_overlay", "value"),
    Input("dropdown_legend", "value"),
    State("custom-colors", "data"),
    Input("tick-step-xaxis", "value"),
    Input("tick-step-yaxis", "value"),
    Input("dropdown_legend_order", "value"),
    State("input_legend_custom_order", "value"),
    State("meta-columns", "data"),

    prevent_initial_call=True
)
def update_main_graph(n_clicks, x_col, y_col, z_col, color_col, size_col, text_col, dropdown_text_pozition,
                      chart_type, bubble, MaxSizeBubble, height, width, selected_style, bar_text_auto,
                      view_revision,
                      filtered_json, hover_cols, corr_cols, facet_row, facet_col, filters_state,
                      xaxis_font_size, yaxis_font_size, font_size_ticks, title_font_size,
                      dropdown_sort_column, axes_category, dropdown_overlay, legend, custom_colors,
                      tick_step_x, tick_step_y, legend_order, legend_custom_order, meta):

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

        errors = _primary_axis_errors(chart_type, x_col, y_col, dff.columns)
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
        if x_as_text:
            plot_df[x_col] = plot_df[x_col].astype(str)

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
                # Plotly Express switches large datasets to scattergl. Its
                # WebGL layer does not render point labels, so keep SVG only
                # while the user explicitly requests labels.
                render_mode="svg" if text_data is not None else "auto",
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
            fig = px.bar(
                plot_df, x=x_col, y=y_col, color=carg,
                height=height, width=width, facet_row=facet_row, facet_col=facet_col,
                text_auto=bar_text_auto, category_orders=category_orders, template=selected_style
            )
            if dropdown_overlay in {'group', 'overlay', 'stack', 'relative'}:
                fig.update_layout(barmode=dropdown_overlay)
            if dropdown_overlay == 'overlay':
                fig.update_traces(opacity=0.85)

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
            pie_col = x_col or y_col
            if plot_df[pie_col].dtypes != 'object':
                dff1 = plot_df[pie_col].value_counts(dropna=False, bins=10).sort_values(ascending=False)
            else:
                dff1 = plot_df[pie_col].value_counts(dropna=False).sort_values(ascending=False)
            dff1 = pd.DataFrame(dff1).reset_index()
            dff1.columns = [pie_col, 'counts']
            dff1[pie_col] = dff1[pie_col].astype(str)
            fig = px.pie(dff1, values='counts', names=pie_col, title=pie_col, height=height, template=selected_style)
            fig.update_traces(textposition='inside', textinfo='percent+label+value', overwrite=True)

        elif chart_type == "Correlation":
            numeric_cols_all = (meta.get("numeric") or [])
            use_cols = [c for c in (corr_cols or numeric_cols_all) if c in numeric_cols_all]
            if use_cols:
                MAX_CORR_COLS = 50
                use_cols = use_cols[:MAX_CORR_COLS]
                corr_df = plot_df[use_cols].select_dtypes(include=[np.number]).dropna(how="all")
                if not corr_df.empty and corr_df.shape[1] >= 2:
                    corr_matrix = corr_df.corr().round(2)
                    fig = go.Figure(go.Heatmap(
                        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                        colorscale='RdBu', zmin=-1, zmax=1,
                        text=corr_matrix.values, texttemplate="%{text}"
                    ))
                    fig.update_layout(
                        title='Корреляционная матрица',
                        xaxis=dict(tickangle=-45),
                        height=height, width=width, template=selected_style
                    )
                    fig.update_xaxes(automargin=True)
                    fig.update_yaxes(automargin=True)
                    try:
                        _sort_legend_traces(fig, legend_order, legend_custom_order)
                    except Exception as _e:
                        logger.warning(f"Сортировка легенды пропущена: {_e}")
                else:
                    fig = _empty_fig()
            else:
                fig = _empty_fig()

        elif chart_type == "Violin":
            fig = px.violin(
                plot_df, x=x_col, y=y_col, color=carg,
                height=height, width=width, facet_row=facet_row, facet_col=facet_col,
                category_orders=category_orders, template=selected_style, box=False
            )

        elif chart_type == "Ridge":
            fig = go.Figure()
            ridge_orientation = 'h' if x_col else 'v'
            if color_col == "Нет" or color_col not in plot_df.columns:
                fig.add_trace(go.Violin(
                    x=plot_df[x_col] if x_col in plot_df.columns else None,
                    y=plot_df[y_col] if y_col in plot_df.columns else None,
                    orientation=ridge_orientation, side='positive', width=3, points=False,
                    line_color=px.colors.qualitative.Plotly[0], name=y_col
                ))
            else:
                unique_values = plot_df[color_col].dropna().unique()
                colors = px.colors.qualitative.Plotly
                for i, val in enumerate(unique_values):
                    subset = plot_df[plot_df[color_col] == val]
                    fig.add_trace(go.Violin(
                        x=subset[x_col] if x_col in subset.columns else None,
                        y=subset[y_col] if y_col in subset.columns else None,
                        orientation=ridge_orientation, side='positive', width=3, points=False,
                        line_color=colors[i % len(colors)],
                        name=str(val)
                    ))
            fig.update_layout(height=height, width=width, template=selected_style)

        elif chart_type == "ScatterMatrix":
            use_dims = []
            for c in [x_col, y_col, z_col]:
                if c and (c in plot_df.columns) and np.issubdtype(plot_df[c].dtype, np.number):
                    use_dims.append(c)
            use_dims = list(dict.fromkeys(use_dims))
            if len(use_dims) < 2:
                notif = _make_error_notif("Для Scatter Matrix нужны ≥2 числовых столбца из X/Y/Z.")
                return empty, notif
            fig = px.scatter_matrix(
                plot_df, dimensions=use_dims, color=carg,
                height=height, width=width, template=selected_style
            )

        elif chart_type == "Parcoords":
            use_dims = []
            for c in [x_col, y_col, z_col]:
                if c and (c in plot_df.columns) and np.issubdtype(plot_df[c].dtype, np.number):
                    use_dims.append(c)
            use_dims = list(dict.fromkeys(use_dims))
            if len(use_dims) < 2:
                notif = _make_error_notif("Для Parallel Coordinates нужны ≥2 числовых столбца из X/Y/Z.")
                return empty, notif
            line_color = None
            if carg:
                if carg in plot_df.columns:
                    if np.issubdtype(plot_df[carg].dtype, np.number):
                        line_color = plot_df[carg]
                    else:
                        codes, _ = pd.factorize(plot_df[carg].astype(str))
                        line_color = codes
            dims = [dict(label=c, values=plot_df[c].values) for c in use_dims]
            fig = go.Figure(data=go.Parcoords(
                dimensions=dims,
                line=dict(color=line_color) if line_color is not None else None
            ))
            fig.update_layout(height=height, width=width, template=selected_style)

        elif chart_type in ("Sunburst", "Treemap"):
            path = [c for c in [color_col, x_col, y_col] if c and (c in plot_df.columns)]
            if not path:
                notif = _make_error_notif("Для Sunburst/Treemap нужен хотя бы один категориальный столбец из Color/X/Y.")
                return empty, notif
            values = None
            if y_col and (y_col in plot_df.columns) and np.issubdtype(plot_df[y_col].dtype, np.number):
                values = y_col
            color_kw = {}
            if carg and (carg in plot_df.columns) and (carg not in path):
                color_kw["color"] = carg
            if chart_type == "Treemap":
                fig = px.treemap(
                    plot_df, path=path, values=values,
                    height=height, width=width, template=selected_style, **color_kw
                )
            else:
                fig = px.sunburst(
                    plot_df, path=path, values=values,
                    height=height, width=width, template=selected_style, **color_kw
                )

        elif chart_type in ("DensityHeat", "DensityContour"):
            if not x_col or not y_col or (x_col not in plot_df.columns) or (y_col not in plot_df.columns):
                notif = _make_error_notif("Для 2D-плотности нужны X и Y.")
                return empty, notif
            if (not np.issubdtype(plot_df[x_col].dtype, np.number)) or (not np.issubdtype(plot_df[y_col].dtype, np.number)):
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
                        trace.setdefault("marker", {})
                        if isinstance(trace["marker"], dict):
                            trace["marker"]["color"] = custom_colors[idx]
            except Exception as _e:
                logger.warning(f"custom_colors apply skipped: {_e}")

        # Применяем шрифт тиков через полный объект tickfont (семейство + размер)
        if x_as_text:
            fig.update_xaxes(type='category', categoryorder=dropdown_sort_column,
                             tickfont=dict(size=xaxis_font_size))
        else:
            fig.update_xaxes(tickfont=dict(size=xaxis_font_size),
                             dtick=tick_step_x if tick_step_x and tick_step_x > 0 else None)

        def is_categorical_by_name(col):
            if col and col in plot_df.columns:
                return any(keyword.lower() in str(col).lower() for keyword in ['скважина', 'well', 'куст'])
            return False

        if is_categorical_by_name(y_col):
            fig.update_yaxes(type='category', categoryorder=dropdown_sort_column,
                             tickfont=dict(size=yaxis_font_size))
        else:
            fig.update_yaxes(tickfont=dict(size=yaxis_font_size),
                             dtick=tick_step_y if tick_step_y and tick_step_y > 0 else None)

        if axes_category == "x" and x_as_text:
            fig.update_xaxes(categoryorder=dropdown_sort_column)
        elif axes_category == "y" and not is_categorical_by_name(y_col):
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

        return fig, []

    except Exception as e:
        logger.error(f"Ошибка при построении графика: {e}", exc_info=True)
        notif = _make_error_notif(f"Ошибка отрисовки графика: {str(e)}. Попробуйте изменить параметры.")
        return empty, notif


# ============ Нижние графики ============
@app.callback(
    Output("corr-bar-x", "figure"),
    Output("corr-bar-y", "figure"),
    Output("corr-bars-section", "style"),
    Output('notifications-container', 'sendNotifications', allow_duplicate=True),

    Input("update-graf", "n_clicks"),
    Input("segmented", "value"),
    State("dropdown_corr_columns", "value"),
    Input("dropdown_x", "value"),
    Input("dropdown_y", "value"),
    Input("cluster-metrics", "data"),

    State("filtered-data", "data"),
    State("dropdown_style", "value"),
    State("meta-columns", "data"),

    prevent_initial_call=True
)
def update_lower_graphs(n_clicks_graf, chart_type, corr_cols, x_col, y_col, cluster_metrics,
                        filtered_json, selected_style, meta):

    empty = _empty_fig()
    SHOW  = {"opacity": 1, "pointerEvents": "auto", "height": "auto", "overflow": "visible", "transition": "opacity 150ms ease"}
    HIDE  = {"opacity": 0, "pointerEvents": "none", "height": "auto", "overflow": "visible"}

    try:
        if not filtered_json:
            return empty, empty, HIDE, []

        dff = read_df_from_store(filtered_json, meta)
        if dff is None or dff.empty:
            return empty, empty, HIDE, []

        def build_corr_bar(corr_matrix: pd.DataFrame, target: str, title_text: str) -> go.Figure:
            if not target or target not in corr_matrix.columns:
                return empty
            s = corr_matrix[target].drop(labels=[target], errors="ignore").sort_values(ascending=False)
            if s.empty:
                return empty
            df_bar = s.reset_index().rename(columns={"index": "Параметр", target: "Корреляция"})
            n = len(df_bar); per_row = 26; padding = 140
            dyn_height = max(220, min(1400, per_row * max(1, n) + padding))
            bar = px.bar(
                df_bar, x="Корреляция", y="Параметр", orientation="h",
                title=title_text, template=selected_style, text="Корреляция",
                height=dyn_height
            )
            xmin = float(s.min()); xmax = float(s.max())
            if xmin == xmax:
                pad = max(0.1, abs(xmin) * 0.1)
                xmin -= pad; xmax += pad
            span = xmax - xmin
            pad = max(0.05 * span, 0.02)
            bar.update_xaxes(
                range=[xmin - pad, xmax + pad],
                showticklabels=True, ticks="outside",
                tickformat=".1f", automargin=True,
                zeroline=(xmin - pad < 0 < xmax + pad), zerolinewidth=1
            )
            bar.update_layout(yaxis_title=None)
            bar.update_yaxes(automargin=True)
            bar.update_traces(texttemplate="%{x:.2f}", textposition="auto", cliponaxis=False)
            return bar

        if chart_type == "Correlation":
            numeric_cols_all = (meta.get("numeric") or [])
            if corr_cols and len(corr_cols) > 0:
                cand = list(dict.fromkeys(list(corr_cols) + ([x_col] if x_col else []) + ([y_col] if y_col else [])))
            else:
                cand = list(dict.fromkeys(([x_col] if x_col else []) + ([y_col] if y_col else []) + list(numeric_cols_all)))

            cand = [c for c in cand if c and c in dff.columns]
            use_cols = [c for c in cand if c in numeric_cols_all]

            if len(use_cols) < 2:
                for c in numeric_cols_all:
                    if c in dff.columns and c not in use_cols:
                        use_cols.append(c)
                        if len(use_cols) >= 2:
                            break

            MAX_CORR_COLS = 50
            use_cols = use_cols[:MAX_CORR_COLS]

            if len(use_cols) >= 2:
                corr_df = dff[use_cols].select_dtypes(include=[np.number]).dropna(how="all")
                if not corr_df.empty and corr_df.shape[1] >= 2:
                    corr_matrix = corr_df.corr().round(2)

                    targets_order = []
                    for cand_t in [x_col, y_col] + use_cols:
                        if cand_t and cand_t not in targets_order and cand_t in corr_matrix.columns:
                            targets_order.append(cand_t)

                    t0 = targets_order[0] if len(targets_order) >= 1 else None
                    t1 = targets_order[1] if len(targets_order) >= 2 else None

                    corr_bar_x_fig = build_corr_bar(corr_matrix, t0, f"Корреляции с {t0}" if t0 else "")
                    corr_bar_y_fig = build_corr_bar(corr_matrix, t1, f"Корреляции с {t1}" if t1 else "")

                    has_any = (len(corr_bar_x_fig.data or []) > 0) or (len(corr_bar_y_fig.data or []) > 0)
                    return corr_bar_x_fig if has_any else empty, corr_bar_y_fig if has_any else empty, (SHOW if has_any else HIDE), []

            return empty, empty, HIDE, []

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
        logger.error(f"Ошибка при построении нижних графиков: {e}", exc_info=True)
        notif = _make_error_notif(f"Ошибка в нижних графиках: {str(e)}")
        return empty, empty, HIDE, notif
