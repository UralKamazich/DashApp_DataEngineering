"""Fast smoke tests for application startup and core data serialization."""

import json
import unittest

import pandas as pd
from dash import dcc, html

from app import app
from callbacks.graph import (
    Y_ONLY_CHART_TYPES,
    _graph_uirevision,
    _primary_axis_errors,
    update_main_graph,
)
from components import make_column_badge
from graph_settings import GraphSettingsPanel, REQUIRED_CONTROLS
from graph_workspace import DEFAULT_FIELDS, GraphWorkspace
from utils import meta_from_df, read_df_from_store


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
        }
        for path, content_type in assets.items():
            with self.subTest(path=path):
                with self.client.get(path) as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertIn(content_type, response.content_type)


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
        self.assertEqual(badge_shell.className, "column-badge")

        badge = next(
            component
            for component in walk_components(badge_shell)
            if component.__class__.__name__ == "Badge"
        )
        self.assertEqual(badge.className, "column-badge-pill")
        self.assertTrue(badge.fullWidth)


class GraphAxisValidationTests(unittest.TestCase):
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
            corr_cols=None,
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
            corr_cols=None,
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
        self.workspace = GraphWorkspace(
            graph_id="test-graph",
            chart_type_control=html.Div(id="test-chart-type"),
            field_controls=controls,
        ).render()
        self.components = list(walk_components(self.workspace))

    def test_all_drop_targets_are_rendered(self):
        targets = {
            component.to_plotly_json()["props"].get("data-drop-target")
            for component in self.components
            if hasattr(component, "to_plotly_json")
        }
        targets.discard(None)
        self.assertEqual(targets, {field["target"] for field in DEFAULT_FIELDS})

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
        self.assertNotIn("update-graf", graph_descendants)
        self.assertNotIn("copy-png-button", graph_descendants)
        self.assertNotIn("context-menu-btn", graph_descendants)

    def test_paper_is_owned_and_sized_by_workspace(self):
        self.assertEqual(self.workspace.id, "test-graph-paper")
        self.assertEqual(self.workspace.style["height"], "750px")
        self.assertEqual(self.workspace.style["width"], "100%")

        descendant_ids = {
            getattr(component, "id", None)
            for component in walk_components(self.workspace)
        }
        self.assertIn("test-graph", descendant_ids)
        self.assertIn("test-graph-workspace", descendant_ids)

    def test_missing_field_control_is_rejected(self):
        with self.assertRaises(ValueError):
            GraphWorkspace(
                graph_id="broken-graph",
                chart_type_control=html.Div(),
                field_controls={},
            )


class GraphSettingsPanelTests(unittest.TestCase):
    def setUp(self):
        controls = {
            key: html.Div(id=f"test-setting-{key}")
            for key in REQUIRED_CONTROLS
        }
        self.panel = GraphSettingsPanel(controls).render()
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

    def test_settings_drawer_uses_compact_width(self):
        self.assertEqual(self.panel.size, "476px")

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
        }
        self.assertTrue(expected_ids.issubset(component_ids))

    def test_missing_settings_control_is_rejected(self):
        with self.assertRaises(ValueError):
            GraphSettingsPanel({})


if __name__ == "__main__":
    unittest.main()
