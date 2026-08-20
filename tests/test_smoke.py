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
    MAX_BAR_LABELS,
    MAX_BAR_POINTS,
    Y_ONLY_CHART_TYPES,
    _build_pie_figure,
    _build_ridge_figure,
    _graph_uirevision,
    _primary_axis_errors,
    update_main_graph,
)
from callbacks.pipeline import apply_filters
from components import (
    SwitchBubble,
    bar_aggregation_select,
    dropdown_chart_type,
    make_column_badge,
    mv_chart_type,
)
from config import APP_NAME, APP_TITLE, APP_VERSION
from correlation_workspace import _build_correlation_figures, compute_correlation
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
            "/assets/graph_settings_popover.js": "text/javascript",
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
        self.assertNotIn("correlation-matrix", correlation_ids)
        self.assertIn("mv-graph", correlation_ids)
        self.assertIn("mv-chart-type", correlation_ids)
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
        workspace = next(
            component for component in walk_components(correlation_page)
            if getattr(component, "id", None) == "correlation-workspace"
        )
        workspace_style = workspace.to_plotly_json()["props"]["style"]
        self.assertEqual(workspace_style["overflowX"], "clip")
        self.assertNotIn("overflowY", workspace_style)
        self.assertEqual(_normalize_main_chart_type("Correlation"), "Scatter")
        self.assertEqual(_normalize_main_chart_type("Pie"), "Pie")

    def test_filter_panel_is_global_and_not_below_the_graph(self):
        components = list(walk_components(app.layout))
        component_ids = {getattr(component, "id", None) for component in components}
        self.assertNotIn("filters-panel-toggle", component_ids)
        self.assertNotIn("filter-drop-target", component_ids)
        for component_id in (
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

    def test_only_dataset_and_filters_use_slide_panels(self):
        panels = {
            component.id: component
            for component in walk_components(app.layout)
            if getattr(component, "id", None) in {
                "dataset-drawer",
                "filters-drawer",
            }
        }
        self.assertEqual(set(panels), {"dataset-drawer", "filters-drawer"})
        for panel in panels.values():
            classes = panel.className.split()
            self.assertIn("slide-panel", classes)
        self.assertIn("slide-panel--left", panels["dataset-drawer"].className.split())
        self.assertIn("slide-panel--reflow", panels["dataset-drawer"].className.split())
        self.assertIn("slide-panel--right", panels["filters-drawer"].className.split())
        self.assertIn("slide-panel--reflow", panels["filters-drawer"].className.split())
        settings_popup = next(
            component
            for component in walk_components(app.layout)
            if getattr(component, "id", None) == "graph-graph-settings-popover"
        )
        self.assertIn("graph-settings-popover", settings_popup.className.split())
        self.assertNotIn("slide-panel", settings_popup.className.split())


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

    def test_compute_correlation_returns_matrix_for_multivariate_workspace(self):
        source = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
        })

        correlation, pair_counts, status, error = compute_correlation(
            source, ["a", "b"], "pearson", 2
        )

        self.assertIsNone(error)
        self.assertAlmostEqual(correlation.loc["a", "b"], -1.0)
        self.assertEqual(int(pair_counts.loc["a", "b"]), 4)
        self.assertIn("Метод: Пирсон", status)


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
                    _primary_axis_errors(chart_type, None, "value", None, ["value"]),
                    [],
                )

    def test_chart_without_x_or_y_is_rejected(self):
        self.assertEqual(
            _primary_axis_errors("Line", None, None, None, ["value"]),
            ["Не выбран столбец X"],
        )

    def test_density_chart_still_requires_x(self):
        self.assertEqual(
            _primary_axis_errors("DensityHeat", None, "value", None, ["value"]),
            ["Не выбран столбец X"],
        )

    def test_hierarchy_charts_accept_any_of_color_x_y(self):
        for chart_type in ("Sunburst", "Treemap"):
            with self.subTest(chart_type=chart_type):
                self.assertEqual(
                    _primary_axis_errors(chart_type, None, None, "region", ["region"]),
                    [],
                )
                self.assertEqual(
                    _primary_axis_errors(chart_type, None, None, None, ["region"]),
                    ["Для Sunburst/Treemap выберите хотя бы один столбец из Цвет/X/Y"],
                )

    def test_hierarchy_charts_render_with_color_field_only(self):
        source = pd.DataFrame({
            "region": ["A", "A", "B", "B"],
            "well": ["W1", "W2", "W3", "W3"],
            "value": [10.0, 20.0, 30.0, 40.0],
        })
        meta = {"numeric": ["value"], "categorical": ["region", "well"], "datetime": []}

        for chart_type, trace_type in (("Sunburst", "sunburst"), ("Treemap", "treemap")):
            with self.subTest(chart_type=chart_type):
                figure, notifications = update_main_graph(
                    n_clicks=1,
                    x_col=None,
                    y_col=None,
                    z_col=None,
                    color_col="region",
                    size_col=None,
                    text_col=None,
                    dropdown_text_pozition="top right",
                    chart_type=chart_type,
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
                    custom_colors=None,
                    tick_step_x=0,
                    tick_step_y=0,
                    legend_order="alphabetical",
                    legend_custom_order="",
                    meta=meta,
                    pie_aggregation="sum",
                )
                self.assertEqual(notifications, [])
                self.assertEqual([trace.type for trace in figure.data], [trace_type])

    def test_string_dtype_columns_do_not_break_numeric_checks(self):
        source = pd.DataFrame({
            "region": ["A", "A", "B", "B"],
            "well": ["W1", "W2", "W3", "W3"],
            "value": [10.0, 20.0, 30.0, 40.0],
        })
        # pandas 3 хранит текст в StringDtype — проверка «числовой ли столбец»
        # не должна падать на нём исключением.
        self.assertFalse(pd.api.types.is_numeric_dtype(source["region"]))
        meta = {"numeric": ["value"], "categorical": ["region", "well"], "datetime": []}
        kwargs = dict(
            n_clicks=1,
            z_col=None,
            size_col=None,
            text_col=None,
            dropdown_text_pozition="top right",
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
            custom_colors=None,
            tick_step_x=0,
            tick_step_y=0,
            legend_order="alphabetical",
            legend_custom_order="",
            meta=meta,
            pie_aggregation="sum",
        )

        # Sunburst: текстовый столбец в Y игнорируется, график строится.
        figure, notifications = update_main_graph(
            x_col="well", y_col="region", color_col=None,
            chart_type="Sunburst", **kwargs,
        )
        self.assertEqual(notifications, [])
        self.assertEqual([trace.type for trace in figure.data], ["sunburst"])

        # DensityHeat: текстовый X — понятная ошибка вместо исключения.
        _figure, notifications = update_main_graph(
            x_col="region", y_col="value", color_col=None,
            chart_type="DensityHeat", **kwargs,
        )
        self.assertEqual(len(notifications), 1)
        self.assertIn("должны быть числовыми", notifications[0]["message"])

    def test_hierarchy_charts_tolerate_empty_path_values(self):
        source = pd.DataFrame({
            "region": ["A", "A", None, "B"],
            "value": [10.0, 20.0, 30.0, 40.0],
        })
        meta = {"numeric": ["value"], "categorical": ["region"], "datetime": []}

        for chart_type, trace_type in (("Sunburst", "sunburst"), ("Treemap", "treemap")):
            with self.subTest(chart_type=chart_type):
                figure, notifications = update_main_graph(
                    n_clicks=1,
                    x_col=None,
                    y_col=None,
                    z_col=None,
                    color_col="region",
                    size_col=None,
                    text_col=None,
                    dropdown_text_pozition="top right",
                    chart_type=chart_type,
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
                    custom_colors=None,
                    tick_step_x=0,
                    tick_step_y=0,
                    legend_order="alphabetical",
                    legend_custom_order="",
                    meta=meta,
                    pie_aggregation="sum",
                )
                self.assertEqual(notifications, [])
                self.assertEqual([trace.type for trace in figure.data], [trace_type])
                # Пропуски помечаются «(пусто)», как в круговой диаграмме.
                self.assertIn("(пусто)", list(figure.data[0].labels))

    def test_bar_aggregates_duplicate_categories(self):
        source = pd.DataFrame({
            "cat": ["A", "A", "B"],
            "value": [10.0, 30.0, 5.0],
        })
        meta = {"numeric": ["value"], "categorical": ["cat"], "datetime": []}
        kwargs = dict(
            n_clicks=1,
            z_col=None,
            size_col=None,
            text_col=None,
            dropdown_text_pozition="top right",
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
            custom_colors=None,
            tick_step_x=0,
            tick_step_y=0,
            legend_order="alphabetical",
            legend_custom_order="",
            meta=meta,
            pie_aggregation="sum",
        )

        def build(mode):
            figure, notifications = update_main_graph(
                x_col="cat", y_col="value", color_col=None,
                chart_type="Bar", bar_aggregation=mode, **kwargs,
            )
            self.assertEqual(notifications, [])
            return figure

        summed = build("sum")
        self.assertEqual(list(summed.data[0].x), ["A", "B"])
        self.assertEqual(list(summed.data[0].y), [40.0, 5.0])
        self.assertIn(
            "Агрегация:</b> сумма «value» по «cat»",
            summed.layout.annotations[0].text,
        )

        mean = build("mean")
        self.assertEqual(list(mean.data[0].y), [20.0, 5.0])

        count = build("count")
        self.assertEqual(list(count.data[0].y), [2, 1])

        raw = build("none")
        self.assertEqual(len(raw.data[0].x), 3)

    def test_large_raw_bar_falls_back_to_safe_aggregation(self):
        row_count = MAX_BAR_POINTS + 25
        source = pd.DataFrame({
            "cat": ["A", "B"] * (row_count // 2) + (["A"] if row_count % 2 else []),
            "value": range(row_count),
        })
        meta = {"numeric": ["value"], "categorical": ["cat"], "datetime": []}
        figure, notifications = update_main_graph(
            n_clicks=1,
            x_col="cat",
            y_col="value",
            z_col=None,
            color_col=None,
            size_col=None,
            text_col=None,
            dropdown_text_pozition="top right",
            chart_type="Bar",
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
            custom_colors=None,
            tick_step_x=0,
            tick_step_y=0,
            legend_order="alphabetical",
            legend_custom_order="",
            meta=meta,
            pie_aggregation="sum",
            bar_aggregation="none",
        )

        self.assertEqual(list(figure.data[0].x), ["A", "B"])
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0]["id"], "bar-auto-aggregation")
        self.assertLessEqual(len(figure.data[0].x), MAX_BAR_LABELS)
        self.assertIn(
            "Автоагрегация:</b> сумма «value» по «cat»",
            figure.layout.annotations[0].text,
        )

    def test_too_many_unique_bars_returns_error_instead_of_freezing(self):
        row_count = MAX_BAR_POINTS + 1
        source = pd.DataFrame({"x": range(row_count), "value": range(row_count)})
        meta = {"numeric": ["x", "value"], "categorical": [], "datetime": []}
        kwargs = dict(
            n_clicks=1, x_col="x", y_col="value", z_col=None, color_col=None,
            size_col=None, text_col=None, dropdown_text_pozition="top right",
            chart_type="Bar", bubble=False, MaxSizeBubble=30, height=550,
            width=None, selected_style="plotly", bar_text_auto=True,
            view_revision=0, filtered_json=source.to_json(orient="split"),
            hover_cols=None, facet_row=None, facet_col=None, filters_state={},
            xaxis_font_size=14, yaxis_font_size=14, font_size_ticks=12,
            title_font_size=16, dropdown_sort_column="trace",
            axes_category="auto", dropdown_overlay="overlay",
            legend="top-right-outside", custom_colors=None, tick_step_x=0,
            tick_step_y=0, legend_order="alphabetical", legend_custom_order="",
            meta=meta, pie_aggregation="sum", bar_aggregation="none",
        )

        figure, notifications = update_main_graph(**kwargs)

        self.assertEqual(len(figure.data), 0)
        self.assertEqual(notifications[0]["color"], "red")

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
            self.component.ids["download_html"],
            self.component.ids["download_component"],
            self.component.ids["save_png"],
            self.component.ids["clear"],
            self.component.ids["view_revision"],
            self.component.ids["custom_colors"],
            self.component.ids["help"],
        }
        self.assertTrue(expected.issubset(descendant_ids))
        # Кнопка настроек рендерится только при наличии панели настроек.
        self.assertNotIn(self.component.ids["open_settings"], descendant_ids)
        self.assertEqual(self.component.ids["update"], "test-graph-update")

    def test_settings_are_bound_to_the_workspace_instance(self):
        fields = {
            field["target"]: dcc.Store(id=f"sales-{field['target']}")
            for field in DEFAULT_FIELDS
        }
        settings_controls = {
            key: html.Div(id=f"sales-setting-{key}")
            for key in REQUIRED_CONTROLS
        }
        settings = GraphSettingsPanel(settings_controls)
        workspace = GraphWorkspace(
            graph_id="sales",
            chart_type_control=html.Div(id="sales-chart-type"),
            field_controls=fields,
            settings_panel=settings,
        )
        tree = workspace.render()
        components = list(walk_components(tree))
        component_ids = {getattr(component, "id", None) for component in components}
        workspace_node = next(
            component for component in components
            if getattr(component, "id", None) == "sales-workspace"
        )
        workspace_props = workspace_node.to_plotly_json()["props"]

        self.assertIn("sales-graph-settings-popover", component_ids)
        self.assertIn("sales-InputSizePlot", component_ids)
        self.assertEqual(
            workspace_props["data-settings-popup-id"],
            "sales-graph-settings-popover",
        )
        self.assertEqual(
            workspace_props["data-action-open-specific-settings"],
            workspace.ids["open_settings"],
        )

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
        options |= {item["value"] for item in mv_chart_type.data}
        self.assertEqual(set(GRAPH_INSTRUCTIONS), options)
        self.assertEqual(set(GRAPH_HELP_ORDER), options)

    def test_multivariate_types_moved_to_correlation_page(self):
        main_options = {item["value"] for item in dropdown_chart_type.data}
        self.assertNotIn("ScatterMatrix", main_options)
        self.assertNotIn("Parcoords", main_options)
        self.assertEqual(
            {item["value"] for item in mv_chart_type.data},
            {"Correlogram", "ScatterMatrix", "Parcoords"},
        )
        self.assertEqual(mv_chart_type.value, "Correlogram")

        correlation_page = next(
            component for component in walk_components(app.layout)
            if getattr(component, "id", None) == "page-correlation"
        )
        correlation_ids = {
            getattr(component, "id", None)
            for component in walk_components(correlation_page)
        }
        self.assertIn("mv-graph", correlation_ids)
        self.assertIn("mv-chart-type", correlation_ids)
        self.assertNotIn("correlation-matrix", correlation_ids)
        self.assertTrue(
            any("mv-graph.figure" in key for key in app.callback_map)
        )

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
        self.assertEqual(panel_values, {"axes", "labels", "legend"})

    def test_settings_panel_is_local_popover_with_two_modes(self):
        self.assertEqual(getattr(self.panel, "id", None), "graph-settings-popover")
        self.assertIn("graph-settings-popover", self.panel.className.split())
        self.assertNotIn("slide-panel", self.panel.className.split())
        component_ids = {
            getattr(component, "id", None)
            for component in self.components
        }
        for expected in (
            "graph-settings-close",
            "graph-settings-common",
            "graph-settings-specific",
            "graph-settings-specific-points",
            "graph-settings-specific-bars",
            "graph-settings-specific-pie",
            "graph-settings-specific-empty",
            "graph-settings-close-on-outside",
        ):
            self.assertIn(expected, component_ids)
        self.assertTrue(
            any(
                "graph-settings-drag-handle" in (getattr(component, "className", "") or "")
                for component in self.components
            )
        )

    def test_series_controls_use_compact_defaults(self):
        self.assertEqual(SwitchBubble.label, "Bubbles")
        self.assertTrue(SwitchBubble.checked)
        self.assertIsNone(getattr(bar_aggregation_select, "description", None))

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
            "test-setting-bar_aggregation",
        }
        self.assertTrue(expected_ids.issubset(component_ids))

    def test_internal_settings_ids_can_be_namespaced(self):
        settings = GraphSettingsPanel(
            self.controls,
            ids={
                "InputSizePlot": "sales-graph-height",
            },
        ).bind_namespace("sales")
        panel = settings.render()
        component_ids = {
            getattr(component, "id", None)
            for component in walk_components(panel)
        }
        self.assertIn("sales-graph-settings-popover", component_ids)
        self.assertIn("sales-graph-height", component_ids)
        self.assertNotIn("graph-settings-popover", component_ids)

    def test_settings_panel_cannot_be_shared_by_different_workspaces(self):
        settings = GraphSettingsPanel(self.controls).bind_namespace("sales")
        with self.assertRaises(ValueError):
            settings.bind_namespace("inventory")

    def test_missing_settings_control_is_rejected(self):
        with self.assertRaises(ValueError):
            GraphSettingsPanel({})


if __name__ == "__main__":
    unittest.main()
