"""Fast smoke tests for application startup and core data serialization."""

import json
import unittest
from pathlib import Path

import pandas as pd
from dash import Dash, Input, dcc, html

from app import app
from callbacks.dropdowns import _select_options
from callbacks.modals import _normalize_main_chart_type
from callbacks.filters import (
    _clean_filter_state,
    _default_filter_value,
    _filter_card,
)
from callbacks.graph import (
    Y_ONLY_CHART_TYPES,
    _build_pie_figure,
    _build_ridge_figure,
    _graph_uirevision,
    _primary_axis_errors,
    update_main_graph,
)
from callbacks.pipeline import apply_filters
from components import dropdown_chart_type, make_column_badge
from config import APP_NAME, APP_TITLE, APP_VERSION
from correlation_workspace import _build_correlation_figures
from graph_help import GRAPH_HELP_ORDER, GRAPH_INSTRUCTIONS, render_instruction
from graph_settings import GraphSettingsPanel, REQUIRED_CONTROLS
from graph_workspace import DEFAULT_FIELDS, GraphWorkspace, _plotly_recovery_script
from utils import create_value_control, meta_from_df, read_df_from_store


def walk_components(root):
    """Yield a Dash component tree without depending on renderer internals."""
    stack = [root]
    while stack:
        component = stack.pop()
        yield component
        children = getattr(component, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(reversed(children))
        elif children is not None and not isinstance(children, (str, int, float, bool)):
            stack.append(children)


class DashApplicationSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.server.test_client()

    def test_callbacks_are_registered(self):
        self.assertGreater(len(app.callback_map), 0)

    def test_application_pages_are_available(self):
        for path in (
            "/",
            "/correlation",
            "/data-engineering",
            "/clustering",
            "/ml",
        ):
            with self.subTest(path=path):
                with self.client.get(path) as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("text/html", response.content_type)

    def test_dash_metadata_endpoints_return_json(self):
        for path in ("/_dash-layout", "/_dash-dependencies"):
            with self.subTest(path=path):
                with self.client.get(path) as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("application/json", response.content_type)
                    self.assertIsNotNone(json.loads(response.data))

    def test_custom_assets_are_available(self):
        assets = {
            "/assets/context_menu.css": "text/css",
            "/assets/graph_settings.css": "text/css",
            "/assets/graph_context_menu.js": "text/javascript",
            "/assets/graph_field_picker.js": "text/javascript",
            "/assets/graph_png.js": "text/javascript",
            "/assets/graph_modebar.js": "text/javascript",
            "/assets/filter_panel.css": "text/css",
            "/assets/slide_panel.css": "text/css",
            "/assets/filter_panel.js": "text/javascript",
        }
        for path, content_type in assets.items():
            with self.subTest(path=path):
                with self.client.get(path) as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertIn(content_type, response.content_type)

    def test_application_branding_uses_current_version_only_in_window_title(self):
        self.assertEqual(APP_VERSION, "2.0.0")
        self.assertEqual(app.title, APP_TITLE)
        self.assertNotIn("collapsed panel", app.title)

        header_labels = [
            component.children
            for component in walk_components(app.layout)
            if component.__class__.__name__ == "Text"
            and getattr(component, "children", None) == APP_NAME
        ]
        self.assertEqual(header_labels, [APP_NAME])

    def test_application_icon_sources_exist(self):
        assets = Path(__file__).resolve().parents[1] / "assets"
        for filename in ("icon.svg", "icon.png", "icon.icns", "favicon.ico"):
            with self.subTest(filename=filename):
                self.assertTrue((assets / filename).is_file())

    def test_correlation_is_a_separate_analysis_page(self):
        chart_values = {item["value"] for item in dropdown_chart_type.data}
        self.assertNotIn("Correlation", chart_values)

        components = list(walk_components(app.layout))
        correlation_page = next(
            component for component in components
            if getattr(component, "id", None) == "page-correlation"
        )
        correlation_ids = {
            getattr(component, "id", None)
            for component in walk_components(correlation_page)
        }
        self.assertIn("correlation-matrix", correlation_ids)
        self.assertIn("correlation-bar-primary", correlation_ids)
        self.assertIn("correlation-bar-secondary", correlation_ids)
        self.assertIn("correlation-columns-drop", correlation_ids)
        self.assertNotIn("graph", correlation_ids)
        drop_target = next(
            component for component in walk_components(correlation_page)
            if getattr(component, "id", None) == "correlation-columns-drop"
        )
        drop_props = drop_target.to_plotly_json()["props"]
        self.assertEqual(drop_props["data-drop-target"], "dropdown_corr_columns")
        self.assertEqual(drop_props["data-drop-mode"], "append")
        self.assertEqual(drop_props["data-accept-type"], "numeric")
        self.assertEqual(_normalize_main_chart_type("Correlation"), "Scatter")
        self.assertEqual(_normalize_main_chart_type("Pie"), "Pie")

    def test_filter_panel_is_global_and_not_below_the_graph(self):
        components = list(walk_components(app.layout))
        component_ids = {getattr(component, "id", None) for component in components}
        for component_id in (
            "filters-panel-toggle",
            "filters-side-tab",
            "filters-drawer",
            "filters-outside-close-store",
            "filters-container",
            "filters-applied-state",
            "filter-logic-mode",
            "apply-filters-btn",
            "reset-filters-btn",
            "filter-close-on-apply",
            "filter-close-on-outside",
            "dataset-drawer",
            "dataset-side-tab",
            "dataset-close-on-outside",
            "dataset-drawer-open-state",
            "dataset-outside-close-store",
            "columns-sidebar",
            "columns-badges",
        ):
            self.assertIn(component_id, component_ids)

        graph_page = next(
            component for component in components
            if getattr(component, "id", None) == "page-graph"
        )
        graph_page_ids = {
            getattr(component, "id", None)
            for component in walk_components(graph_page)
        }
        self.assertNotIn("filters-container", graph_page_ids)

    def test_side_panels_share_unified_slide_panel_class(self):
        panels = {
            component.id: component
            for component in walk_components(app.layout)
            if getattr(component, "id", None) in {
                "dataset-drawer",
                "filters-drawer",
                "drawer-simple",
            }
        }
        self.assertEqual(set(panels), {"dataset-drawer", "filters-drawer", "drawer-simple"})
        for panel in panels.values():
            classes = panel.className.split()
            self.assertIn("slide-panel", classes)
        self.assertIn("slide-panel--left", panels["dataset-drawer"].className.split())
        self.assertIn("slide-panel--reflow", panels["dataset-drawer"].className.split())
        self.assertIn("slide-panel--right", panels["filters-drawer"].className.split())
        self.assertIn("slide-panel--reflow", panels["filters-drawer"].className.split())
        self.assertIn("slide-panel--right", panels["drawer-simple"].className.split())
        self.assertIn("slide-panel--overlay", panels["drawer-simple"].className.split())


class DataStoreRoundTripTests(unittest.TestCase):
    def test_dataframe_metadata_and_datetime_round_trip(self):
        source = pd.DataFrame(
            {
                "value": [1.5, 2.5],
                "category": ["A", "B"],
                "date": pd.to_datetime(["2026-08-17", "2026-08-18"]),
            }
        )

        meta = meta_from_df(source)
        restored = read_df_from_store(
            source.to_json(date_format="iso", orient="split"),
            meta,
        )

        self.assertEqual(meta["numeric"], ["value"])
        self.assertEqual(meta["categorical"], ["category"])
        self.assertEqual(meta["datetime"], ["date"])
        self.assertEqual(restored["value"].tolist(), [1.5, 2.5])
        self.assertEqual(restored["category"].tolist(), ["A", "B"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(restored["date"]))


class ColumnBadgeTests(unittest.TestCase):
    def test_badge_uses_width_constrained_shell(self):
        badge_shell = make_column_badge("Очень длинное название столбца", "numeric")
        self.assertIn("column-badge", badge_shell.className)
        self.assertIn("column-badge--numeric", badge_shell.className)
        self.assertEqual(
            badge_shell.to_plotly_json()["props"]["data-column-type"],
            "numeric",
        )

        row = next(
            component
            for component in walk_components(badge_shell)
            if "column-channel-row" in (getattr(component, "className", "") or "")
        )
        row_classes = {
            getattr(component, "className", "")
            for component in walk_components(row)
        }
        self.assertIn("column-type-marker column-type-marker--numeric", row_classes)
        self.assertIn("column-channel-name", row_classes)
        self.assertIn("column-drag-handle", row_classes)
        type_marker = next(
            component
            for component in walk_components(row)
            if "column-type-marker" in (getattr(component, "className", "") or "")
        )
        self.assertEqual(type_marker.__class__.__name__, "Sup")


class DropdownOptionTests(unittest.TestCase):
    def test_numeric_column_labels_are_safe_for_mantine_controls(self):
        self.assertEqual(
            _select_options(["depth", 1]),
            [
                {"label": "depth", "value": "depth"},
                {"label": "1", "value": "1"},
            ],
        )


class FilterPanelTests(unittest.TestCase):
    def setUp(self):
        self.source = pd.DataFrame({
            "value": [1.0, 2.0, 3.0, 4.0],
            "category": ["A", "A", "B", "C"],
            "date": pd.to_datetime(["2026-08-15", "2026-08-16", "2026-08-17", "2026-08-18"]),
        })
        self.meta = meta_from_df(self.source)
        self.payload = self.source.to_json(date_format="iso", orient="split")

    def test_filter_cards_use_type_aware_compact_controls(self):
        state = {"1": {"column": "value", "value": [1.0, 4.0]}}
        card = _filter_card(1, "value", state, self.source)
        classes = {
            getattr(component, "className", "")
            for component in walk_components(card)
        }
        names = {component.__class__.__name__ for component in walk_components(card)}
        self.assertIn("filter-card filter-card--numeric", classes)
        self.assertIn("RangeSlider", names)

        date_control = create_value_control("date", "date", None, self.source)
        category_control = create_value_control("category", "category", None, self.source)
        self.assertIn("DatePickerInput", {
            component.__class__.__name__ for component in walk_components(date_control)
        })
        self.assertIn("MultiSelect", {
            component.__class__.__name__ for component in walk_components(category_control)
        })

    def test_filter_state_is_cleaned_before_apply(self):
        state = {
            "1": {"column": "category", "value": ["A"]},
            "2": {"column": "value", "value": None},
            "3": {"column": "date", "value": ["2026-08-15", None]},
        }
        self.assertEqual(
            _clean_filter_state(state),
            {"1": {"column": "category", "value": ["A"]}},
        )
        self.assertEqual(_default_filter_value(self.source, "value"), [1.0, 4.0])

    def test_pipeline_supports_and_or_and_date_ranges(self):
        filters = {
            "1": {"column": "category", "value": ["A"]},
            "2": {"column": "value", "value": [3.0, 4.0]},
        }
        and_payload, _ = apply_filters(filters, "and", self.payload, self.meta)
        or_payload, _ = apply_filters(filters, "or", self.payload, self.meta)
        and_frame = read_df_from_store(and_payload, self.meta)
        or_frame = read_df_from_store(or_payload, self.meta)
        self.assertTrue(and_frame.empty)
        self.assertEqual(or_frame["value"].tolist(), [1.0, 2.0, 3.0, 4.0])

        date_payload, _ = apply_filters(
            {"1": {"column": "date", "value": ["2026-08-16", "2026-08-17"]}},
            "and",
            self.payload,
            self.meta,
        )
        date_frame = read_df_from_store(date_payload, self.meta)
        self.assertEqual(date_frame["value"].tolist(), [2.0, 3.0])


class CorrelationAnalysisTests(unittest.TestCase):
    def test_spearman_detects_monotonic_relationship(self):
        source = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6],
            "curved": [1, 4, 9, 16, 25, 36],
        })

        pearson, *_ = _build_correlation_figures(
            source, ["x", "curved"], "pearson", 2, "plotly"
        )
        spearman, first_bar, second_bar, status, error = _build_correlation_figures(
            source, ["x", "curved"], "spearman", 2, "plotly"
        )

        self.assertIsNone(error)
        self.assertLess(pearson.data[0].z[0][1], 1.0)
        self.assertAlmostEqual(spearman.data[0].z[0][1], 1.0)
        self.assertIn("Спирмен", status)
        self.assertEqual(list(first_bar.layout.xaxis.range), [-1.05, 1.05])
        self.assertEqual(list(second_bar.layout.xaxis.range), [-1.05, 1.05])

    def test_matrix_hover_contains_pairwise_observation_counts(self):
        source = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, None],
            "y": [1.0, None, 3.0, 4.0],
        })

        matrix, *_ = _build_correlation_figures(
            source, ["x", "y"], "pearson", 2, "plotly"
        )

        self.assertEqual(matrix.data[0].customdata[0][1], 2)
        self.assertIn("Совместных наблюдений", matrix.data[0].hovertemplate)

    def test_constant_columns_are_excluded_from_analysis(self):
        source = pd.DataFrame({
            "x": [1, 2, 3, 4],
            "y": [4, 3, 2, 1],
            "constant": [7, 7, 7, 7],
        })

        matrix, _first, _second, status, error = _build_correlation_figures(
            source, ["x", "constant", "y"], "pearson", 2, "plotly"
        )

        self.assertIsNone(error)
        self.assertEqual(list(matrix.data[0].x), ["x", "y"])
        self.assertIn("Исключено столбцов: 1", status)


class GraphAxisValidationTests(unittest.TestCase):
    def test_pie_counts_a_single_categorical_column(self):
        source = pd.DataFrame({"layer": ["A", "A", "B", None]})

        figure, error = _build_pie_figure(
            source, "layer", None, None, "sum", 550, 900, "plotly"
        )

        self.assertIsNone(error)
        self.assertEqual(list(figure.data[0].labels), ["A", "B", "(пусто)"])
        self.assertEqual(list(figure.data[0].values), [2.0, 1.0, 1.0])
        self.assertEqual(figure.layout.width, 900)

    def test_pie_aggregates_a_numeric_value_by_category(self):
        source = pd.DataFrame({
            "layer": ["A", "A", "B"],
            "production": [10.0, 15.0, 7.0],
        })

        figure, error = _build_pie_figure(
            source, "layer", "production", None, "sum", 550, None, "plotly"
        )

        self.assertIsNone(error)
        self.assertEqual(list(figure.data[0].labels), ["A", "B"])
        self.assertEqual(list(figure.data[0].values), [25.0, 7.0])
        self.assertIn("Сумма «production»", figure.layout.title.text)

    def test_pie_supports_mean_and_extra_hover_columns(self):
        source = pd.DataFrame({
            "layer": ["A", "A", "B"],
            "production": [10.0, 20.0, 9.0],
            "well": ["W-1", "W-2", "W-3"],
        })

        figure, error = _build_pie_figure(
            source, "production", "layer", None, "mean", 550, None,
            "plotly", ["well"],
        )

        self.assertIsNone(error)
        self.assertEqual(list(figure.data[0].values), [15.0, 9.0])
        self.assertIn("well: %{customdata[1]}", figure.data[0].hovertemplate)

    def test_pie_count_omits_categories_without_numeric_values(self):
        source = pd.DataFrame({
            "layer": ["A", "A", "B"],
            "production": [10.0, None, None],
        })

        figure, error = _build_pie_figure(
            source, "layer", "production", None, "count", 550, None, "plotly"
        )

        self.assertIsNone(error)
        self.assertEqual(list(figure.data[0].labels), ["A"])
        self.assertEqual(list(figure.data[0].values), [1.0])

    def test_pie_bins_a_single_numeric_column(self):
        source = pd.DataFrame({"value": range(20)})

        figure, error = _build_pie_figure(
            source, "value", None, None, "sum", 550, None, "plotly"
        )

        self.assertIsNone(error)
        self.assertEqual(len(figure.data[0].labels), 10)
        self.assertEqual(sum(figure.data[0].values), 20)

    def test_pie_rejects_two_numeric_axes(self):
        source = pd.DataFrame({"first": [1, 2], "second": [3, 4]})

        figure, error = _build_pie_figure(
            source, "first", "second", None, "sum", 550, None, "plotly"
        )

        self.assertIsNone(figure)
        self.assertIn("не более одного числового столбца", error)

    def test_ridge_builds_one_svg_density_per_category(self):
        source = pd.DataFrame({
            "depth": [100, 110, 120, 200, 210, 220],
            "layer": ["A", "A", "A", "B", "B", "B"],
        })

        figure, error = _build_ridge_figure(
            source, "depth", "layer", None, 550, None, "plotly"
        )

        self.assertIsNone(error)
        self.assertEqual(len(figure.data), 2)
        self.assertTrue(all(trace.type == "scatter" for trace in figure.data))
        self.assertTrue(all(trace.fill == "toself" for trace in figure.data))
        self.assertEqual(list(figure.layout.yaxis.ticktext), ["A", "B"])
        self.assertEqual(figure.layout.xaxis.title.text, "depth")
        self.assertEqual(figure.layout.yaxis.title.text, "layer")

    def test_ridge_supports_numeric_y_without_x(self):
        source = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})

        figure, error = _build_ridge_figure(
            source, None, "value", None, 550, None, "plotly"
        )

        self.assertIsNone(error)
        self.assertEqual(len(figure.data), 1)
        self.assertEqual(figure.layout.yaxis.title.text, "value")
        self.assertEqual(list(figure.layout.xaxis.ticktext), ["value"])

    def test_ridge_rejects_ambiguous_axis_types(self):
        source = pd.DataFrame({
            "first": [1, 2, 3],
            "second": [4, 5, 6],
        })

        figure, error = _build_ridge_figure(
            source, "first", "second", None, 550, None, "plotly"
        )

        self.assertIsNone(figure)
        self.assertIn("ровно один числовой столбец", error)

    def test_ridge_color_creates_a_stable_legend(self):
        source = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 6, 7, 8],
            "layer": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "kind": ["one", "one", "two", "two", "one", "one", "two", "two"],
        })

        figure, error = _build_ridge_figure(
            source, "value", "layer", "kind", 550, None, "plotly"
        )

        self.assertIsNone(error)
        self.assertEqual(len(figure.data), 4)
        self.assertEqual(
            [trace.name for trace in figure.data if trace.showlegend],
            ["one", "two"],
        )

    def test_view_revision_ignores_labels_and_styling_but_changes_with_axes(self):
        original = _graph_uirevision("Scatter", "x", "y", None, None, None, 0)

        self.assertEqual(
            original,
            _graph_uirevision("Scatter", "x", "y", None, None, None, 0),
        )
        self.assertNotEqual(
            original,
            _graph_uirevision("Scatter", "other_x", "y", None, None, None, 0),
        )
        self.assertNotEqual(
            original,
            _graph_uirevision("Line", "x", "y", None, None, None, 0),
        )
        self.assertNotEqual(
            original,
            _graph_uirevision("Scatter", "x", "y", None, None, None, 1),
        )

    def test_y_only_is_allowed_for_regular_charts(self):
        for chart_type in Y_ONLY_CHART_TYPES:
            with self.subTest(chart_type=chart_type):
                self.assertEqual(
                    _primary_axis_errors(chart_type, None, "value", ["value"]),
                    [],
                )

    def test_chart_without_x_or_y_is_rejected(self):
        self.assertEqual(
            _primary_axis_errors("Line", None, None, ["value"]),
            ["Не выбран столбец X"],
        )

    def test_density_chart_still_requires_x(self):
        self.assertEqual(
            _primary_axis_errors("DensityHeat", None, "value", ["value"]),
            ["Не выбран столбец X"],
        )

    def test_large_scatter_with_labels_uses_svg_trace(self):
        row_count = 1200
        source = pd.DataFrame(
            {
                "x": range(row_count),
                "y": range(row_count),
                "label": [f"Точка {index}" for index in range(row_count)],
            }
        )
        figure, notifications = update_main_graph(
            n_clicks=1,
            x_col="x",
            y_col="y",
            z_col=None,
            color_col=None,
            size_col=None,
            text_col="label",
            dropdown_text_pozition="top right",
            chart_type="Scatter",
            bubble=False,
            MaxSizeBubble=30,
            height=550,
            width=None,
            selected_style="plotly",
            bar_text_auto=True,
            view_revision=0,
            filtered_json=source.to_json(date_format="iso", orient="split"),
            hover_cols=None,
            facet_row=None,
            facet_col=None,
            filters_state={},
            xaxis_font_size=14,
            yaxis_font_size=14,
            font_size_ticks=12,
            title_font_size=16,
            dropdown_sort_column="trace",
            axes_category="auto",
            dropdown_overlay="overlay",
            legend="top-right-outside",
            custom_colors={},
            tick_step_x=0,
            tick_step_y=0,
            legend_order="original",
            legend_custom_order=None,
            meta=meta_from_df(source),
        )

        self.assertEqual(notifications, [])
        self.assertEqual(figure.data[0].type, "scatter")
        self.assertEqual(figure.data[0].mode, "markers+text")
        self.assertEqual(len(figure.data[0].text), row_count)
        self.assertEqual(
            figure.layout.uirevision,
            _graph_uirevision("Scatter", "x", "y", None, None, None, 0),
        )

    def test_categorical_scatter_axis_uses_svg_and_explicit_axis_type(self):
        row_count = 1200
        source = pd.DataFrame(
            {
                "category": [f"Группа {index % 5}" for index in range(row_count)],
                "x": range(row_count),
                "y": range(row_count, row_count * 2),
            }
        )
        meta = meta_from_df(source)

        def build(x_col, y_col):
            return update_main_graph(
                n_clicks=1,
                x_col=x_col,
                y_col=y_col,
                z_col=None,
                color_col=None,
                size_col=None,
                text_col=None,
                dropdown_text_pozition="middle center",
                chart_type="Scatter",
                bubble=False,
                MaxSizeBubble=30,
                height=550,
                width=None,
                selected_style="plotly",
                bar_text_auto=True,
                view_revision=0,
                filtered_json=source.to_json(date_format="iso", orient="split"),
                hover_cols=[],
                facet_row=None,
                facet_col=None,
                filters_state={},
                xaxis_font_size=14,
                yaxis_font_size=14,
                font_size_ticks=12,
                title_font_size=16,
                dropdown_sort_column="trace",
                axes_category="auto",
                dropdown_overlay="overlay",
                legend="top-right-outside",
                custom_colors={},
                tick_step_x=0,
                tick_step_y=0,
                legend_order="original",
                legend_custom_order=None,
                meta=meta,
            )

        x_figure, x_notifications = build("category", "y")
        y_figure, y_notifications = build("x", "category")

        self.assertEqual(x_notifications, [])
        self.assertEqual(y_notifications, [])
        self.assertEqual(x_figure.data[0].type, "scatter")
        self.assertEqual(y_figure.data[0].type, "scatter")
        self.assertEqual(x_figure.layout.xaxis.type, "category")
        self.assertEqual(y_figure.layout.yaxis.type, "category")

    def test_fully_cleared_graph_is_empty_without_notification(self):
        source = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        figure, notifications = update_main_graph(
            n_clicks=1,
            x_col=None,
            y_col=None,
            z_col=None,
            color_col=None,
            size_col=None,
            text_col=None,
            dropdown_text_pozition="middle center",
            chart_type="Scatter",
            bubble=False,
            MaxSizeBubble=30,
            height=550,
            width=None,
            selected_style="plotly",
            bar_text_auto=True,
            view_revision=1,
            filtered_json=source.to_json(date_format="iso", orient="split"),
            hover_cols=[],
            facet_row=None,
            facet_col=None,
            filters_state={},
            xaxis_font_size=14,
            yaxis_font_size=14,
            font_size_ticks=12,
            title_font_size=16,
            dropdown_sort_column="trace",
            axes_category="auto",
            dropdown_overlay="overlay",
            legend="top-right-outside",
            custom_colors={},
            tick_step_x=0,
            tick_step_y=0,
            legend_order="original",
            legend_custom_order=None,
            meta=meta_from_df(source),
        )

        self.assertEqual(len(figure.data), 0)
        self.assertEqual(notifications, [])


class GraphWorkspaceTests(unittest.TestCase):
    def setUp(self):
        controls = {
            field["target"]: dcc.Store(id=f"test-{field['target']}")
            for field in DEFAULT_FIELDS
        }
        self.component = GraphWorkspace(
            graph_id="test-graph",
            chart_type_control=html.Div(id="test-chart-type"),
            field_controls=controls,
        )
        self.workspace = self.component.render()
        self.components = list(walk_components(self.workspace))

    def test_all_drop_targets_are_rendered(self):
        targets = {
            component.to_plotly_json()["props"].get("data-drop-target")
            for component in self.components
            if hasattr(component, "to_plotly_json")
        }
        targets.discard(None)
        self.assertEqual(targets, set(self.component.field_ids.values()))

    def test_each_drop_zone_has_a_clear_button(self):
        clear_buttons = [
            component
            for component in self.components
            if "graph-zone-clear" in (getattr(component, "className", "") or "")
        ]
        self.assertEqual(len(clear_buttons), len(DEFAULT_FIELDS))

    def test_drop_zones_have_role_specific_classes(self):
        classes = {
            getattr(component, "className", "")
            for component in self.components
            if "graph-drop-zone" in (getattr(component, "className", "") or "")
        }
        self.assertTrue(any("graph-drop-zone--x" in value for value in classes))
        self.assertTrue(any("graph-drop-zone--hover" in value for value in classes))

    def test_overlay_controls_are_outside_plotly_graph(self):
        graph = next(component for component in self.components if getattr(component, "id", None) == "test-graph")
        graph_descendants = {
            getattr(component, "id", None)
            for component in walk_components(graph)
        }
        self.assertNotIn(self.component.ids["update"], graph_descendants)
        self.assertNotIn(self.component.ids["copy_png"], graph_descendants)
        self.assertNotIn(self.component.ids["open_settings"], graph_descendants)

    def test_paper_is_owned_and_sized_by_workspace(self):
        paper = next(
            component
            for component in self.components
            if getattr(component, "id", None) == "test-graph-paper"
        )
        self.assertEqual(paper.style["height"], "750px")
        self.assertEqual(paper.style["width"], "100%")

        descendant_ids = {
            getattr(component, "id", None)
            for component in walk_components(self.workspace)
        }
        self.assertIn("test-graph", descendant_ids)
        self.assertIn("test-graph-workspace", descendant_ids)

    def test_component_owns_namespaced_actions_and_state(self):
        descendant_ids = {
            getattr(component, "id", None)
            for component in self.components
        }
        expected = {
            self.component.ids["update"],
            self.component.ids["copy_png"],
            self.component.ids["open_settings"],
            self.component.ids["download_html"],
            self.component.ids["download_component"],
            self.component.ids["save_png"],
            self.component.ids["clear"],
            self.component.ids["view_revision"],
            self.component.ids["custom_colors"],
        }
        self.assertTrue(expected.issubset(descendant_ids))
        self.assertEqual(self.component.ids["update"], "test-graph-update")

    def test_clear_action_contains_hard_plotly_recovery(self):
        script = _plotly_recovery_script("test-graph")

        self.assertIn('document.getElementById("test-graph")', script)
        self.assertIn("window.Plotly.purge(plot)", script)
        self.assertIn("classList.remove('dash-graph--pending')", script)

    def test_figure_builder_api_owns_graph_output(self):
        test_app = Dash("graph-workspace-figure-test")

        @self.component.figure_callback(test_app, Input("test-signal", "data"))
        def build_figure(_signal):
            return {}, []

        self.assertTrue(
            any("test-graph.figure" in key for key in test_app.callback_map)
        )

    def test_missing_field_control_is_rejected(self):
        with self.assertRaises(ValueError):
            GraphWorkspace(
                graph_id="broken-graph",
                chart_type_control=html.Div(),
                field_controls={},
            )

    def test_help_button_and_modal_are_rendered(self):
        descendant_ids = {
            getattr(component, "id", None) for component in self.components
        }
        for key in ("help", "help_modal", "help_type", "help_content", "help_close"):
            self.assertIn(self.component.ids[key], descendant_ids)


class GraphHelpTests(unittest.TestCase):
    def test_instructions_cover_every_chart_type_option(self):
        options = {item["value"] for item in dropdown_chart_type.data}
        self.assertEqual(set(GRAPH_INSTRUCTIONS), options)
        self.assertEqual(set(GRAPH_HELP_ORDER), options)

    def test_every_instruction_has_required_sections(self):
        for chart_type, info in GRAPH_INSTRUCTIONS.items():
            with self.subTest(chart_type=chart_type):
                self.assertTrue(info["title"])
                self.assertTrue(info["purpose"])
                self.assertTrue(info["fields"])

    def test_render_instruction_returns_content_for_each_type(self):
        for chart_type in GRAPH_HELP_ORDER:
            with self.subTest(chart_type=chart_type):
                children, title = render_instruction(chart_type)
                self.assertTrue(title)
                self.assertTrue(children)

    def test_unknown_type_falls_back_to_scatter(self):
        _, title = render_instruction("NoSuchType")
        self.assertEqual(title, GRAPH_INSTRUCTIONS["Scatter"]["title"])


class GraphSettingsPanelTests(unittest.TestCase):
    def setUp(self):
        self.controls = {
            key: html.Div(id=f"test-setting-{key}")
            for key in REQUIRED_CONTROLS
        }
        self.panel = GraphSettingsPanel(self.controls).render()
        self.components = list(walk_components(self.panel))

    def test_settings_are_grouped_in_single_level_tabs(self):
        tabs = next(
            component
            for component in self.components
            if getattr(component, "id", None) == "graph-settings-tabs"
        )
        self.assertEqual(tabs.value, "axes")

        panel_values = {
            getattr(component, "value", None)
            for component in self.components
            if component.__class__.__name__ == "TabsPanel"
        }
        self.assertEqual(panel_values, {"axes", "labels", "legend", "series"})

    def test_settings_panel_is_slide_out_overlay(self):
        self.assertEqual(getattr(self.panel, "id", None), "drawer-simple")
        self.assertIn("slide-panel", self.panel.className.split())
        self.assertIn("slide-panel--overlay", self.panel.className.split())
        component_ids = {
            getattr(component, "id", None)
            for component in self.components
        }
        for expected in ("drawer-simple-tab", "drawer-simple-open-state"):
            self.assertIn(expected, component_ids)

    def test_settings_panel_keeps_existing_callback_ids(self):
        component_ids = {
            getattr(component, "id", None)
            for component in self.components
        }
        expected_ids = {
            "InputSizePlot",
            "InputSizePlotW",
            "InputMaxSizeBubble",
            "font-size-xaxis",
            "font-size-yaxis",
            "font-size-title",
            "font-size-ticks",
            "tick-step-xaxis",
            "tick-step-yaxis",
            "graph-settings-reset",
            "test-setting-pie_aggregation",
        }
        self.assertTrue(expected_ids.issubset(component_ids))

    def test_internal_settings_ids_can_be_namespaced(self):
        panel = GraphSettingsPanel(
            self.controls,
            ids={
                "drawer-simple": "sales-settings-drawer",
                "InputSizePlot": "sales-graph-height",
            },
        ).render()
        component_ids = {
            getattr(component, "id", None)
            for component in walk_components(panel)
        }
        self.assertIn("sales-settings-drawer", component_ids)
        self.assertIn("sales-graph-height", component_ids)
        self.assertNotIn("drawer-simple", component_ids)

    def test_missing_settings_control_is_rejected(self):
        with self.assertRaises(ValueError):
            GraphSettingsPanel({})


if __name__ == "__main__":
    unittest.main()
