"""Fast smoke tests for application startup and core data serialization."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from dash import Dash, Input, dcc, html, no_update
import dash_mantine_components as dmc

from app import app
from callbacks.columns_sidebar import update_column_badges
from callbacks.dropdowns import _select_options, update_dropdown_options_all
from callbacks.file_handling import _clicked_sheet, on_excel_upload
from callbacks.modals import _normalize_main_chart_type
from callbacks.multivariate import build_multivariate_figure
from callbacks.filters import (
    _clean_filter_state,
    _default_filter_value,
    _filter_card,
    update_filter_draft_status,
    update_filter_summary,
)
from callbacks.graph import (
    MAX_BAR_LABELS,
    MAX_BAR_POINTS,
    Y_ONLY_CHART_TYPES,
    _build_pie_figure,
    _build_ridge_figure,
    build_main_figure,
    _graph_uirevision,
    _primary_axis_errors,
    _temporary_category_frame,
    update_main_graph,
)
from callbacks.pipeline import apply_filters
from components import (
    SwitchBubble,
    bar_aggregation_select,
    dropdown_chart_type,
    graph_render_mode,
    make_column_badge,
    mv_chart_type,
)
from config import APP_NAME, APP_TITLE, APP_VERSION
from correlation_workspace import (
    _build_correlation_figures,
    _rating_target_options,
    compute_correlation,
)
from graph_help import GRAPH_HELP_ORDER, GRAPH_INSTRUCTIONS, render_instruction
from graph_settings import (
    GraphSettingsPanel,
    REQUIRED_CONTROLS,
    SETTINGS_COMBOBOX_Z_INDEX,
)
from graph_workspace import DEFAULT_FIELDS, GraphWorkspace, _plotly_recovery_script
from utils import (
    _empty_fig,
    apply_filter_conditions,
    create_value_control,
    meta_from_df,
    read_df_from_store,
)


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
            "/assets/graph_help_window.js": "text/javascript",
            "/assets/graph_context_menu.js": "text/javascript",
            "/assets/graph_field_picker.js": "text/javascript",
            "/assets/graph_png.js": "text/javascript",
            "/assets/graph_modebar.js": "text/javascript",
            "/assets/graph_fullscreen.js": "text/javascript",
            "/assets/filter_panel.css": "text/css",
            "/assets/slide_panel.css": "text/css",
            "/assets/filter_panel.js": "text/javascript",
        }
        for path, content_type in assets.items():
            with self.subTest(path=path):
                with self.client.get(path) as response:
                    self.assertEqual(response.status_code, 200)
                self.assertIn(content_type, response.content_type)

    def test_settings_dropdown_portal_is_treated_as_part_of_popover(self):
        script_path = Path(__file__).resolve().parents[1] / "assets" / "graph_settings_popover.js"
        script = script_path.read_text(encoding="utf-8")
        self.assertIn("function ownsPortalTarget", script)
        self.assertIn("aria-controls", script)
        self.assertIn("ownsPortalTarget(popup, event.target)", script)

    def test_field_picker_moves_inside_native_fullscreen_host(self):
        script_path = Path(__file__).resolve().parents[1] / "assets" / "graph_field_picker.js"
        script = script_path.read_text(encoding="utf-8")
        self.assertIn("function pickerPortal(zone)", script)
        self.assertIn("document.fullscreenElement", script)
        self.assertIn("fullscreenHost?.contains(zone)", script)
        self.assertIn("portal.appendChild(picker)", script)

    def test_graph_context_menu_blocks_right_button_drag_before_plotly(self):
        script_path = Path(__file__).resolve().parents[1] / "assets" / "graph_context_menu.js"
        script = script_path.read_text(encoding="utf-8")
        self.assertIn('e.button !== 2', script)
        self.assertIn('e.stopImmediatePropagation()', script)
        self.assertIn('".graph-workspace-plot"', script)
        self.assertIn("fullscreenHost?.contains(_activeWorkspace)", script)
        self.assertIn("portal.appendChild(menu)", script)
        self.assertNotIn('menuItem("save-png"', script)

    def test_graph_modebar_owns_compact_workspace_controls(self):
        script_path = Path(__file__).resolve().parents[1] / "assets" / "graph_modebar.js"
        script = script_path.read_text(encoding="utf-8")
        self.assertIn('className = "modebar-chart-select"', script)
        self.assertIn('"modebar-settings-custom"', script)
        self.assertIn('"modebar-help-custom"', script)
        self.assertIn('reset.insertAdjacentElement("afterend", button)', script)

    def test_field_picker_has_temporary_categorical_switch(self):
        script_path = Path(__file__).resolve().parents[1] / "assets" / "graph_field_picker.js"
        script = script_path.read_text(encoding="utf-8")
        self.assertIn("field-picker-category-switch", script)
        self.assertIn("data-field-mode-store-id", script)
        self.assertIn("writeFieldMode", script)

    def test_graph_fullscreen_control_uses_scoped_host(self):
        script_path = Path(__file__).resolve().parents[1] / "assets" / "graph_fullscreen.js"
        script = script_path.read_text(encoding="utf-8")
        self.assertIn('const HOST = ".graph-fullscreen-host"', script)
        self.assertIn("host.requestFullscreen", script)
        self.assertIn("document.exitFullscreen", script)
        self.assertIn("window.Plotly.Plots.resize", script)

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

    def test_every_plot_group_reacts_to_theme_changes(self):
        output_markers = (
            ("..graph.figure...", ("filtered-data", "data")),
            ("mv-graph.figure", ("filtered-data", "data")),
            ("correlation-bar-primary.figure", ("filtered-data", "data")),
            ("cluster-elbow-graph.figure", ("cluster-metrics", "data")),
        )
        for marker, source_input in output_markers:
            callback = next(
                value for key, value in app.callback_map.items()
                if marker in key and source_input in {
                    (dependency["id"], dependency["property"])
                    for dependency in value["inputs"]
                }
            )
            inputs = {
                (dependency["id"], dependency["property"])
                for dependency in callback["inputs"]
            }
            with self.subTest(output=marker):
                self.assertIn(("dropdown_style", "value"), inputs)

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
        self.assertIn("correlation-bar-primary-target", correlation_ids)
        self.assertIn("correlation-bar-secondary-target", correlation_ids)
        self.assertIn("correlation-columns-drop", correlation_ids)
        for obsolete_id in (
            "mv-dropdown-x", "mv-dropdown-y", "mv-dropdown-z", "mv-dropdown-color"
        ):
            self.assertNotIn(obsolete_id, correlation_ids)
        multivariate_workspace = next(
            component for component in walk_components(correlation_page)
            if getattr(component, "id", None) == "mv-graph-workspace"
        )
        self.assertFalse(any(
            "graph-drop-zone" in (getattr(component, "className", "") or "")
            for component in walk_components(multivariate_workspace)
        ))
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

        rating_callback = next(
            callback for key, callback in app.callback_map.items()
            if "correlation-bar-primary.figure" in key
        )
        rating_inputs = {
            (dependency["id"], dependency["property"])
            for dependency in rating_callback["inputs"]
        }
        self.assertIn(("dropdown_corr_columns", "value"), rating_inputs)
        self.assertIn(("correlation-bar-primary-target", "value"), rating_inputs)
        self.assertIn(("correlation-bar-secondary-target", "value"), rating_inputs)
        self.assertFalse(any(
            component_id.startswith("mv-dropdown-")
            for component_id, _property in rating_inputs
        ))

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
            "filter-applied-logic",
            "filter-logic-mode",
            "apply-filters-btn",
            "revert-filters-btn",
            "filter-draft-status",
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
    def test_empty_figure_uses_selected_theme(self):
        figure = _empty_fig("plotly_dark")
        self.assertEqual(
            figure.layout.template.layout.plot_bgcolor,
            "rgb(17,17,17)",
        )

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
        self.assertEqual(meta["row_count"], 2)
        self.assertEqual(meta["column_count"], 3)
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

    def test_metadata_builds_options_without_deserializing_dataset(self):
        source = pd.DataFrame({"value": [1.0], "category": ["A"]})
        meta = meta_from_df(source)
        payload = source.to_json(orient="split")
        with patch(
            "callbacks.dropdowns.read_df_from_store",
            side_effect=AssertionError("dataset should not be deserialized"),
        ):
            result = update_dropdown_options_all(payload, meta)
        self.assertEqual(result[0], _select_options(["value", "category"]))

    def test_metadata_builds_badges_without_deserializing_dataset(self):
        source = pd.DataFrame({"value": [1.0], "category": ["A"]})
        meta = meta_from_df(source)
        payload = source.to_json(orient="split")
        with patch(
            "callbacks.columns_sidebar.read_df_from_store",
            side_effect=AssertionError("dataset should not be deserialized"),
        ):
            badges, _style = update_column_badges(payload, payload, meta)
        self.assertEqual(
            [badge.to_plotly_json()["props"]["data-column-name"] for badge in badges],
            ["value", "category"],
        )


class ExcelLoadingTests(unittest.TestCase):
    def test_single_sheet_is_parsed_from_the_already_open_workbook(self):
        frame = pd.DataFrame({"depth": [100.0, 200.0]})
        workbook = MagicMock()
        workbook.sheet_names = ["Data"]
        workbook.parse.return_value = frame
        manager = MagicMock()
        manager.__enter__.return_value = workbook

        with patch("callbacks.file_handling.pd.ExcelFile", return_value=manager) as excel_file:
            result = on_excel_upload("/tmp/example.xlsx", "example.xlsx")

        excel_file.assert_called_once_with("/tmp/example.xlsx", engine="openpyxl")
        workbook.parse.assert_called_once_with("Data")
        self.assertEqual(result[2], ["Data"])
        self.assertEqual(result[3], "Data")
        self.assertEqual(result[4], result[5])
        self.assertEqual(result[6]["row_count"], 2)

    def test_triggered_sheet_wins_when_other_buttons_were_clicked_before(self):
        ids = [
            {"type": "sheet-select", "index": "First"},
            {"type": "sheet-select", "index": "Second"},
        ]
        self.assertEqual(
            _clicked_sheet(
                [1, 1],
                ids,
                {"type": "sheet-select", "index": "Second"},
            ),
            "Second",
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
        self.assertIn("NumberInput", names)
        operator_selects = [
            component for component in walk_components(card)
            if isinstance(getattr(component, "id", None), dict)
            and component.id.get("type") == "filter-operator"
        ]
        self.assertEqual(operator_selects[0].value, "between")

        date_control = create_value_control("date", "date", None, self.source)
        category_control = create_value_control("category", "category", None, self.source)
        self.assertIn("DatePickerInput", {
            component.__class__.__name__ for component in walk_components(date_control)
        })
        self.assertIn("MultiSelect", {
            component.__class__.__name__ for component in walk_components(category_control)
        })
        category_select = next(
            component for component in walk_components(category_control)
            if component.__class__.__name__ == "MultiSelect"
        )
        self.assertTrue(category_select.withCheckIcon)
        self.assertEqual(category_select.checkIconPosition, "left")
        self.assertFalse(category_select.hidePickedOptions)

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

    def test_clean_filter_state_keeps_operators_and_drops_full_domain(self):
        state = {
            "1": {
                "column": "value",
                "operator": "between",
                "value": [1.0, 4.0],
                "domain": [1.0, 4.0],
            },
            "2": {"column": "category", "operator": "is_empty", "value": None},
            "3": {"column": "value", "operator": "gt", "value": 2.0},
        }
        self.assertEqual(
            _clean_filter_state(state),
            {
                "2": {"column": "category", "operator": "is_empty", "value": None},
                "3": {"column": "value", "operator": "gt", "value": 2.0},
            },
        )

    def test_typed_filter_operators(self):
        filtered = apply_filter_conditions(
            self.source,
            {
                "1": {"column": "value", "operator": "gt", "value": 1.0},
                "2": {"column": "category", "operator": "contains", "value": "a"},
            },
            self.meta,
            "and",
        )
        self.assertEqual(filtered["value"].tolist(), [2.0])

        excluded = apply_filter_conditions(
            self.source,
            {"1": {"column": "category", "operator": "not_in", "value": ["A"]}},
            self.meta,
            "and",
        )
        self.assertEqual(excluded["category"].tolist(), ["B", "C"])

        with_missing = pd.DataFrame({"category": ["A", "B", None, ""]})
        missing_meta = meta_from_df(with_missing)
        not_a = apply_filter_conditions(
            with_missing,
            {"1": {"column": "category", "operator": "not_in", "value": ["A"]}},
            missing_meta,
            "and",
        )
        self.assertEqual(not_a["category"].tolist(), ["B"])

    def test_draft_status_tracks_unapplied_operator_and_logic_changes(self):
        draft = {"1": {"column": "category", "operator": "contains", "value": ""}}
        status = update_filter_draft_status(draft, {}, "and", "and")
        self.assertEqual(status[0], "Есть неприменённые изменения")
        self.assertFalse(status[2])
        self.assertFalse(status[4])

        logic_status = update_filter_draft_status({}, {}, "or", "and")
        self.assertEqual(logic_status[0], "Есть неприменённые изменения")

    def test_filter_summary_includes_retained_percentage(self):
        applied = {"1": {"column": "value", "operator": "gt", "value": 2.0}}
        filtered = self.source.loc[self.source["value"] > 2].to_json(
            orient="split", date_format="iso"
        )
        summary, count, _ = update_filter_summary(applied, self.payload, filtered)
        self.assertEqual(summary, "2 из 4 · 50,0%")
        self.assertEqual(count, "1")

    def test_pipeline_listens_only_to_applied_logic(self):
        callback = next(
            value for key, value in app.callback_map.items()
            if "filtered-data.data" in key
            and "meta-columns.data" in key
            and ("filters-applied-state", "data") in {
                (dependency["id"], dependency["property"])
                for dependency in value["inputs"]
            }
            and ("filter-applied-logic", "data") in {
                (dependency["id"], dependency["property"])
                for dependency in value["inputs"]
            }
        )
        inputs = {
            (dependency["id"], dependency["property"])
            for dependency in callback["inputs"]
        }
        self.assertNotIn(("filter-logic-mode", "value"), inputs)

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
    def test_hidden_correlation_page_does_not_deserialize_loaded_data(self):
        with patch(
            "callbacks.multivariate.read_df_from_store",
            side_effect=AssertionError("hidden page should not read the dataset"),
        ):
            figure, notifications = build_multivariate_figure(
                "Correlogram", ["x", "y"], "pearson", 2,
                '{"large":"payload"}', "/", "plotly", {},
            )
        self.assertIs(figure, no_update)
        self.assertIs(notifications, no_update)

    def test_multivariate_graph_uses_only_shared_correlation_channels(self):
        source = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [3.0, 2.0, 1.0],
            "c": [2.0, 4.0, 8.0],
        })
        figure, notifications = build_multivariate_figure(
            "ScatterMatrix", ["c", "a", "b"], "pearson", 2,
            source.to_json(orient="split"), "/correlation", "plotly",
            meta_from_df(source),
        )

        self.assertEqual(notifications, [])
        self.assertEqual(
            [dimension.label for dimension in figure.data[0].dimensions],
            ["c", "a", "b"],
        )

    def test_rating_targets_are_limited_to_shared_channels(self):
        options, primary, secondary = _rating_target_options(
            ["gamma", "alpha", "beta"], "alpha", "removed"
        )
        self.assertEqual(
            options,
            [
                {"label": "gamma", "value": "gamma"},
                {"label": "alpha", "value": "alpha"},
                {"label": "beta", "value": "beta"},
            ],
        )
        self.assertEqual(primary, "alpha")
        self.assertEqual(secondary, "gamma")

    def test_rating_graphs_follow_explicit_target_channels(self):
        source = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 4.0, 2.0, 3.0],
        })
        _matrix, first, second, status, error = _build_correlation_figures(
            source, ["a", "b", "c"], "pearson", 2, "plotly", "c", "a"
        )

        self.assertIsNone(error)
        self.assertEqual(first.layout.title.text, "Корреляции с «c»")
        self.assertEqual(second.layout.title.text, "Корреляции с «a»")
        self.assertIn("Рейтинги: c / a", status)

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
        self.assertNotEqual(
            original,
            _graph_uirevision(
                "Scatter", "x", "y", None, None, None, 0, {"x": True}
            ),
        )

    def test_temporary_category_frame_is_shallow_and_does_not_mutate_source(self):
        source = pd.DataFrame({"depth": [100.0, 200.0, 100.0], "value": [1, 2, 3]})

        frame, active, error = _temporary_category_frame(
            source,
            {"x": True},
            {"x": "depth"},
        )

        self.assertIsNone(error)
        self.assertEqual(active, {"x"})
        self.assertTrue(isinstance(frame["depth"].dtype, pd.CategoricalDtype))
        self.assertEqual(list(frame["depth"]), ["100.0", "200.0", "100.0"])
        self.assertTrue(pd.api.types.is_float_dtype(source["depth"]))

    def test_temporary_category_frame_guards_expensive_group_counts(self):
        source = pd.DataFrame({"group": range(121)})

        frame, _active, error = _temporary_category_frame(
            source,
            {"color": True},
            {"color": "group"},
        )

        self.assertIsNone(frame)
        self.assertIn("121 уникальных", error)
        self.assertIn("не более 120", error)

    def test_numeric_x_can_render_as_a_temporary_category(self):
        source = pd.DataFrame({"marker": [10, 20, 10], "value": [1.0, 2.0, 3.0]})
        figure, notifications = build_main_figure(
            n_clicks=1,
            x_col="marker",
            y_col="value",
            z_col=None,
            color_col=None,
            size_col=None,
            text_col=None,
            dropdown_text_pozition="top right",
            chart_type="Scatter",
            bubble=False,
            MaxSizeBubble=30,
            height=550,
            width=None,
            selected_style="plotly",
            bar_text_auto=True,
            view_revision=0,
            filtered_json=source.to_json(orient="split"),
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
            meta={"numeric": ["marker", "value"], "categorical": [], "datetime": []},
            pie_aggregation="sum",
            bar_aggregation="sum",
            categorical_fields={"x": True},
        )

        self.assertEqual(notifications, [])
        self.assertEqual(list(figure.data[0].x), ["10", "20", "10"])
        self.assertEqual(figure.layout.xaxis.type, "category")

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

        figure, notifications = build_main_figure(**kwargs)

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

    def test_scatter_can_force_svg_and_apply_marker_size_in_pixels(self):
        row_count = 1200
        source = pd.DataFrame({"x": range(row_count), "y": range(row_count)})
        kwargs = dict(
            n_clicks=1, x_col="x", y_col="y", z_col=None, color_col=None,
            size_col=None, text_col=None, dropdown_text_pozition="middle center",
            chart_type="Scatter", bubble=True, MaxSizeBubble=30, height=550,
            width=None, selected_style="plotly", bar_text_auto=True,
            view_revision=0, filtered_json=source.to_json(orient="split"),
            hover_cols=[], facet_row=None, facet_col=None, filters_state={},
            xaxis_font_size=14, yaxis_font_size=14, font_size_ticks=12,
            title_font_size=16, dropdown_sort_column="trace",
            axes_category="auto", dropdown_overlay="overlay",
            legend="top-right-outside", custom_colors={}, tick_step_x=0,
            tick_step_y=0, legend_order="original", legend_custom_order=None,
            meta=meta_from_df(source), marker_size=9,
        )

        hybrid_figure, hybrid_notifications = build_main_figure(
            **kwargs, render_mode="hybrid"
        )
        svg_figure, svg_notifications = build_main_figure(
            **kwargs, render_mode="svg"
        )

        self.assertEqual(hybrid_notifications, [])
        self.assertEqual(svg_notifications, [])
        self.assertEqual(hybrid_figure.data[0].type, "scattergl")
        self.assertEqual(svg_figure.data[0].type, "scatter")
        self.assertEqual(svg_figure.data[0].marker.size, 9)

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
    @staticmethod
    def _settings_controls(prefix):
        selects = {
            key: dmc.Select(
                id=f"{prefix}-{key}",
                data=[{"value": "default", "label": "Default"}],
                value="default",
            )
            for key in REQUIRED_CONTROLS
            if key not in {"bubble", "bar_labels", "legend_custom_order"}
        }
        selects["bubble"] = dmc.Switch(id=f"{prefix}-bubble", checked=True)
        selects["bar_labels"] = dmc.Switch(id=f"{prefix}-bar-labels", checked=True)
        selects["legend_custom_order"] = dmc.TextInput(
            id=f"{prefix}-legend-custom-order",
            value="",
        )
        return selects

    @staticmethod
    def _field_controls(prefix, fields=DEFAULT_FIELDS):
        controls = {}
        for field in fields:
            component_id = f"{prefix}-{field['target']}"
            controls[field["target"]] = (
                dmc.MultiSelect(id=component_id, data=[], value=[])
                if field.get("mode") == "append"
                else dmc.Select(id=component_id, data=[], value=None)
            )
        return controls

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

    def test_z_zone_is_positioned_independently_from_secondary_fields(self):
        layer = next(
            component for component in self.components
            if getattr(component, "className", None) == "graph-drop-layer"
        )
        direct_classes = {
            getattr(component, "className", "") or ""
            for component in layer.children
        }
        self.assertTrue(any("graph-drop-zone--z" in value for value in direct_classes))

        secondary = next(
            component for component in layer.children
            if getattr(component, "className", None) == "graph-drop-secondary"
        )
        self.assertFalse(any(
            "graph-drop-zone--z" in (getattr(component, "className", "") or "")
            for component in walk_components(secondary)
        ))

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
        paper_ids = {
            getattr(component, "id", None)
            for component in walk_components(paper)
        }
        self.assertIn(self.component.ids["help_modal"], paper_ids)

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
            self.component.ids["field_modes"],
            self.component.ids["custom_colors"],
            self.component.ids["help"],
        }
        self.assertTrue(expected.issubset(descendant_ids))
        # Кнопка настроек рендерится только при наличии панели настроек.
        self.assertNotIn(self.component.ids["open_settings"], descendant_ids)
        self.assertNotIn(self.component.ids["change_colors"], descendant_ids)
        self.assertEqual(self.component.ids["update"], "test-graph-update")

        classes = {
            getattr(component, "className", "") or ""
            for component in self.components
        }
        self.assertNotIn("graph-workspace-toolbar", classes)
        self.assertIn("graph-modebar-state", classes)

        graph = next(
            component for component in self.components
            if getattr(component, "id", None) == "test-graph"
        )
        self.assertEqual(graph.config["displayModeBar"], "hover")
        self.assertTrue({
            "zoom2d", "pan2d", "zoomIn2d", "zoomOut2d", "autoScale2d",
            "select2d", "lasso2d",
        }.issubset(graph.config["modeBarButtonsToRemove"]))

        workspace_node = next(
            component for component in self.components
            if getattr(component, "id", None) == "test-graph-workspace"
        )
        self.assertNotIn(
            "data-action-change-colors",
            workspace_node.to_plotly_json()["props"],
        )

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
        paper = next(
            component for component in components
            if getattr(component, "id", None) == "sales-paper"
        )
        paper_ids = {
            getattr(component, "id", None)
            for component in walk_components(paper)
        }
        self.assertIn("sales-graph-settings-popover", paper_ids)
        self.assertIn(workspace.ids["help_modal"], paper_ids)
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

    def test_duplicate_semantic_field_key_is_rejected(self):
        fields = (
            {"key": "x", "label": "X", "target": "first", "zone": "axis-x"},
            {"key": "x", "label": "X2", "target": "second", "zone": "axis-y"},
        )
        with self.assertRaises(ValueError):
            GraphWorkspace(
                graph_id="duplicate-fields",
                chart_type_control=dmc.Select(id="duplicate-chart", data=[]),
                field_controls={
                    "first": dmc.Select(id="duplicate-first", data=[]),
                    "second": dmc.Select(id="duplicate-second", data=[]),
                },
                fields=fields,
            )

    def test_color_editor_cannot_be_enabled_without_settings(self):
        with self.assertRaises(ValueError):
            GraphWorkspace(
                graph_id="orphan-colors",
                chart_type_control=dmc.Select(id="orphan-chart", data=[]),
                field_controls=self._field_controls("orphan"),
                include_color_controls=True,
            )

    def test_export_uses_semantic_fields_not_legacy_target_names(self):
        fields = (
            {"key": "x", "label": "X", "target": "horizontal", "zone": "axis-x"},
            {"key": "y", "label": "Y", "target": "vertical", "zone": "axis-y"},
        )
        workspace = GraphWorkspace(
            graph_id="semantic",
            chart_type_control=dmc.Select(id="semantic-chart", data=[]),
            field_controls={
                "horizontal": dmc.Select(id="semantic-horizontal", data=[]),
                "vertical": dmc.Select(id="semantic-vertical", data=[]),
            },
            fields=fields,
            notifications_id="semantic-notifications",
        )
        test_app = Dash("semantic-fields-test")
        test_app.layout = html.Div([
            dmc.NotificationContainer(id="semantic-notifications"),
            workspace.render(),
        ])

        workspace.register_callbacks(test_app)

        self.assertEqual(workspace.field_id("x"), "semantic-horizontal")
        self.assertEqual(workspace.field_id("y"), "semantic-vertical")
        self.assertTrue(
            any("semantic-download.data" in key for key in test_app.callback_map)
        )

    def test_two_full_workspaces_register_without_id_collisions(self):
        def build_workspace(prefix):
            settings = GraphSettingsPanel(self._settings_controls(f"{prefix}-setting"))
            return GraphWorkspace(
                graph_id=prefix,
                chart_type_control=dmc.Select(
                    id=f"{prefix}-chart",
                    data=[{"value": "Scatter", "label": "Scatter"}],
                    value="Scatter",
                ),
                field_controls=self._field_controls(prefix),
                settings_panel=settings,
                notifications_id="dashboard-notifications",
            )

        sales = build_workspace("sales-dashboard")
        inventory = build_workspace("inventory-dashboard")
        test_app = Dash("two-workspaces-test")
        test_app.layout = html.Div([
            dmc.NotificationContainer(id="dashboard-notifications"),
            sales.render(),
            inventory.render(),
        ])

        sales.register_callbacks(test_app)
        inventory.register_callbacks(test_app)

        component_ids = [
            getattr(component, "id", None)
            for component in walk_components(test_app.layout)
            if isinstance(getattr(component, "id", None), str)
        ]
        self.assertEqual(len(component_ids), len(set(component_ids)))
        self.assertIn("sales-dashboard-graph-settings-popover", component_ids)
        self.assertIn("inventory-dashboard-graph-settings-popover", component_ids)
        self.assertTrue(
            any("sales-dashboard-paper.style" in key for key in test_app.callback_map)
        )
        self.assertTrue(
            any("inventory-dashboard-paper.style" in key for key in test_app.callback_map)
        )

    def test_help_button_and_nonmodal_window_are_rendered(self):
        descendant_ids = {
            getattr(component, "id", None) for component in self.components
        }
        for key in ("help", "help_modal", "help_title", "help_type", "help_content", "help_close"):
            self.assertIn(self.component.ids[key], descendant_ids)
        help_window = next(
            component for component in self.components
            if getattr(component, "id", None) == self.component.ids["help_modal"]
        )
        props = help_window.to_plotly_json()["props"]
        self.assertIn("graph-help-window", help_window.className.split())
        self.assertEqual(props["aria-modal"], "false")
        self.assertEqual(props["aria-hidden"], "true")

    def test_help_window_is_dragged_and_closed_only_explicitly(self):
        script_path = Path(__file__).resolve().parents[1] / "assets" / "graph_help_window.js"
        script = script_path.read_text(encoding="utf-8")
        self.assertIn('event.target.closest(".graph-help-window-close")', script)
        self.assertIn('document.addEventListener("pointermove"', script)
        self.assertIn("window.graphHelpWindow", script)
        self.assertNotIn('document.addEventListener("keydown"', script)


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
        multivariate_callback = next(
            callback for key, callback in app.callback_map.items()
            if "mv-graph.figure" in key
        )
        multivariate_inputs = {
            (dependency["id"], dependency["property"])
            for dependency in multivariate_callback["inputs"]
        }
        self.assertIn(("dropdown_corr_columns", "value"), multivariate_inputs)
        self.assertFalse(any(
            component_id.startswith("mv-dropdown-")
            for component_id, _property in multivariate_inputs
        ))

    def test_every_instruction_has_required_sections(self):
        for chart_type, info in GRAPH_INSTRUCTIONS.items():
            with self.subTest(chart_type=chart_type):
                self.assertTrue(info["title"])
                self.assertTrue(info["purpose"])
                self.assertTrue(info["fields"])
                self.assertTrue(info["reading"])

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
        self.assertEqual(graph_render_mode.value, "hybrid")

    def test_every_settings_select_opens_above_the_popover(self):
        settings_popup = next(
            component
            for component in walk_components(app.layout)
            if getattr(component, "id", None) == "graph-graph-settings-popover"
        )
        selects = {
            component.id: component
            for component in walk_components(settings_popup)
            if component.__class__.__name__ == "Select"
        }
        expected_ids = {
            "dropdown_style",
            "dropdown_text_pozition",
            "dropdown_axes_category",
            "dropdown_category_ascending",
            "dropdown_overlay",
            "bar-aggregation",
            "dropdown_pie_aggregation",
            "dropdown_legend",
            "dropdown_legend_order",
        }
        self.assertEqual(set(selects), expected_ids)
        for component_id, select in selects.items():
            with self.subTest(component_id=component_id):
                self.assertEqual(
                    select.comboboxProps.get("zIndex"),
                    SETTINGS_COMBOBOX_Z_INDEX,
                )
                self.assertFalse(select.comboboxProps.get("withinPortal"))

    def test_settings_panel_keeps_existing_callback_ids(self):
        component_ids = {
            getattr(component, "id", None)
            for component in self.components
        }
        expected_ids = {
            "InputSizePlot",
            "InputSizePlotW",
            "InputMaxSizeBubble",
            "InputMarkerSize",
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
        marker_size = next(
            component
            for component in self.components
            if getattr(component, "id", None) == "InputMarkerSize"
        )
        bubble_size = next(
            component
            for component in self.components
            if getattr(component, "id", None) == "InputMaxSizeBubble"
        )
        self.assertEqual(marker_size.label, "Размер маркера, px")
        self.assertEqual(bubble_size.label, "Макс. размер пузыря")

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
