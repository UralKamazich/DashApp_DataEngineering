"""Structural tests for the permanent, reusable Multi-Y workspace."""

import unittest
from unittest.mock import patch

from dash import Dash

from layout import MULTI_Y_WORKSPACE, NAV_LINKS, create_layout
from multi_axis_workspace import (
    MultiYAxisWorkspace,
    _FRAME_CACHE,
    _FRAME_CACHE_MAX_ENTRIES,
    _frame_for_payload,
    _set_shared_y_axis_mode,
    _sync_state_for_input,
    empty_multi_axis_state,
)


def walk_components(root):
    stack = [root]
    while stack:
        component = stack.pop()
        yield component
        children = getattr(component, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(reversed(children))
        elif children is not None and not isinstance(
            children, (str, int, float, bool)
        ):
            stack.append(children)


def component_by_id(root, component_id):
    return next(
        component
        for component in walk_components(root)
        if getattr(component, "id", None) == component_id
    )


class MultiYAxisWorkspaceTests(unittest.TestCase):
    def test_permanent_page_is_separate_from_the_legacy_graph(self):
        layout = create_layout()
        legacy_page = component_by_id(layout, "page-graph")
        multi_page = component_by_id(layout, "page-multi-y")
        legacy_ids = {getattr(item, "id", None) for item in walk_components(legacy_page)}
        multi_ids = {getattr(item, "id", None) for item in walk_components(multi_page)}

        self.assertIn("graph", legacy_ids)
        self.assertNotIn("multi-y-graph", legacy_ids)
        self.assertIn("multi-y-graph", multi_ids)
        self.assertNotIn("graph", multi_ids)
        self.assertIn({"label": "Multi‑Y", "href": "/multi-y"}, NAV_LINKS)

    def test_workspace_renders_left_right_y_and_shared_x_drop_zones(self):
        tree = MULTI_Y_WORKSPACE.render()
        zones = [
            item
            for item in walk_components(tree)
            if "multi-axis-drop-zone" in (
                getattr(item, "className", "") or ""
            ).split()
        ]
        self.assertEqual(len(zones), 3)
        props = [zone.to_plotly_json()["props"] for zone in zones]
        self.assertEqual(
            {(item.get("data-multi-axis-drop"), item.get("data-axis-side")) for item in props},
            {("x", None), ("y", "left"), ("y", "right")},
        )

    def test_local_picker_keeps_non_active_datasets_configurable(self):
        tree = MULTI_Y_WORKSPACE.render()
        component_ids = {
            getattr(item, "id", None)
            for item in walk_components(tree)
            if isinstance(getattr(item, "id", None), str)
        }
        self.assertIn(MULTI_Y_WORKSPACE.ids["shared_x"], component_ids)
        self.assertIn(MULTI_Y_WORKSPACE.ids["add_y"], component_ids)
        self.assertIn(MULTI_Y_WORKSPACE.ids["add_side"], component_ids)
        self.assertIn(MULTI_Y_WORKSPACE.ids["add_series"], component_ids)
        self.assertIn(MULTI_Y_WORKSPACE.ids["show_legend"], component_ids)
        self.assertIn(MULTI_Y_WORKSPACE.ids["legend_font_size"], component_ids)
        self.assertIn(MULTI_Y_WORKSPACE.ids["shared_y_axes"], component_ids)

    def test_legend_is_enabled_in_a_new_workspace_state(self):
        self.assertTrue(empty_multi_axis_state()["show_legend"])
        self.assertFalse(empty_multi_axis_state()["shared_y_axes"])

    def test_shared_y_mode_retains_private_axes_and_synchronizes_each_side(self):
        state = {
            "series": [
                {"id": "a", "axis_id": "axis-a"},
                {"id": "b", "axis_id": "axis-b"},
                {"id": "c", "axis_id": "axis-c"},
            ],
            "axes": [
                {"id": "axis-a", "side": "left", "type": "log"},
                {"id": "axis-b", "side": "left", "type": "linear"},
                {"id": "axis-c", "side": "right", "type": "linear"},
            ],
        }

        shared = _set_shared_y_axis_mode(state, True)
        restored = _set_shared_y_axis_mode(shared, False)

        self.assertTrue(shared["shared_y_axes"])
        self.assertEqual(
            [axis["id"] for axis in shared["axes"]],
            ["axis-a", "axis-b", "axis-c"],
        )
        self.assertEqual(
            [axis["type"] for axis in shared["axes"]],
            ["log", "log", "linear"],
        )
        self.assertFalse(restored["shared_y_axes"])
        self.assertEqual(
            [axis["id"] for axis in restored["axes"]],
            ["axis-a", "axis-b", "axis-c"],
        )

    def test_series_card_keeps_only_scale_as_advanced_axis_control(self):
        card = MULTI_Y_WORKSPACE._series_card(
            {
                "id": "series-1",
                "y": "pressure",
                "type": "line",
                "color": "#228be6",
                "axis_id": "axis-1",
            },
            {
                "id": "axis-1",
                "title": "Pressure",
                "side": "left",
                "type": "linear",
                "autorange": True,
                "visible": True,
            },
            [{"label": "pressure", "value": "pressure"}],
        )
        pattern_ids = {
            item_id.get("type")
            for item in walk_components(card)
            if isinstance((item_id := getattr(item, "id", None)), dict)
        }
        self.assertNotIn("x", MULTI_Y_WORKSPACE.pattern_types)
        self.assertNotIn("x-mode", MULTI_Y_WORKSPACE.pattern_types)
        self.assertNotIn(f"{MULTI_Y_WORKSPACE.graph_id}-series-x", pattern_ids)
        self.assertNotIn(f"{MULTI_Y_WORKSPACE.graph_id}-series-x-mode", pattern_ids)
        removed_axis_controls = {
            "axis-title", "axis-autorange", "range-min", "range-max", "axis-visible"
        }
        self.assertTrue(removed_axis_controls.isdisjoint(MULTI_Y_WORKSPACE.pattern_types))
        self.assertTrue(all(
            f"{MULTI_Y_WORKSPACE.graph_id}-series-{key}" not in pattern_ids
            for key in removed_axis_controls
        ))
        name_control = component_by_id(
            card,
            MULTI_Y_WORKSPACE.pattern_id("name", "series-1"),
        )
        scale_control = component_by_id(
            card,
            MULTI_Y_WORKSPACE.pattern_id("axis-scale", "series-1"),
        )
        smooth_control = component_by_id(
            card,
            MULTI_Y_WORKSPACE.pattern_id("smooth", "series-1"),
        )
        trace_control = component_by_id(
            card,
            MULTI_Y_WORKSPACE.pattern_id("trace-type", "series-1"),
        )
        self.assertEqual(name_control.label, "Подпись оси")
        self.assertEqual(
            {item["value"] for item in scale_control.data},
            {"linear", "log"},
        )
        self.assertFalse(smooth_control.checked)
        self.assertFalse(smooth_control.disabled)
        self.assertIn("box", {item["value"] for item in trace_control.data})

    def test_box_series_card_exposes_native_point_display_setting(self):
        card = MULTI_Y_WORKSPACE._series_card(
            {
                "id": "series-box",
                "y": "pressure",
                "type": "box",
                "box_points": "all",
                "color": "#228be6",
                "axis_id": "axis-box",
            },
            {"id": "axis-box", "side": "left", "type": "linear"},
            [{"label": "pressure", "value": "pressure"}],
        )
        control = component_by_id(
            card,
            MULTI_Y_WORKSPACE.pattern_id("box-points", "series-box"),
        )

        self.assertEqual(control.value, "all")
        self.assertEqual(control.style.get("display"), "block")

    def test_common_settings_are_collapsible_under_the_modebar_gear(self):
        tree = MULTI_Y_WORKSPACE.render()
        workspace = component_by_id(tree, MULTI_Y_WORKSPACE.workspace_id)
        settings = component_by_id(tree, MULTI_Y_WORKSPACE.ids["settings"])
        common_details = next(
            item
            for item in walk_components(settings)
            if "multi-axis-common-details" in (
                getattr(item, "className", "") or ""
            ).split()
        )
        detail_ids = {
            getattr(item, "id", None)
            for item in walk_components(common_details)
            if isinstance(getattr(item, "id", None), str)
        }
        self.assertTrue({
            MULTI_Y_WORKSPACE.ids["dataset"],
            MULTI_Y_WORKSPACE.ids["theme"],
            MULTI_Y_WORKSPACE.ids["render_mode"],
            MULTI_Y_WORKSPACE.ids["height"],
            MULTI_Y_WORKSPACE.ids["width"],
        }.issubset(detail_ids))
        settings_classes = {
            getattr(item, "className", "")
            for item in walk_components(settings)
        }
        self.assertFalse(any(
            "graph-settings-common" in (class_name or "").split()
            for class_name in settings_classes
        ))
        workspace_props = workspace.to_plotly_json()["props"]
        self.assertEqual(workspace_props["data-common-settings-in-specific"], "true")
        self.assertNotIn("data-action-change-colors", workspace_props)

    def test_state_sync_removes_legacy_per_series_x_and_axis_ui_state(self):
        legacy = {
            "dataset_id": "source",
            "scope": "filtered",
            "data_ref": "base-ref",
            "shared_x": "time",
            "series": [{
                "id": "old-series",
                "y": "pressure",
                "x": "measured_depth",
                "x_mode": "individual",
                "line_shape": "spline",
            }],
            "axes": [{
                "id": "axis-old-series",
                "title": "Старая подпись",
                "range": [1, 2],
                "min": 1,
                "max": 2,
                "autorange": False,
                "range_auto": False,
                "visible": False,
                "type": "log",
                "side": "right",
            }],
        }
        synced = _sync_state_for_input(
            legacy,
            "source",
            "filtered",
            {"dataset_id": "source", "data_ref": "base-ref"},
        )

        self.assertEqual(synced["shared_x"], "time")
        self.assertNotIn("x", synced["series"][0])
        self.assertNotIn("x_mode", synced["series"][0])
        self.assertNotIn("line_shape", synced["series"][0])
        self.assertTrue(synced["series"][0]["smooth"])
        self.assertEqual(synced["axes"][0]["type"], "log")
        self.assertEqual(synced["axes"][0]["side"], "right")
        for removed in (
            "title", "range", "min", "max", "autorange", "range_auto", "visible"
        ):
            self.assertNotIn(removed, synced["axes"][0])

    def test_instances_own_namespaced_state_and_callbacks(self):
        first = MultiYAxisWorkspace(graph_id="multi-a")
        second = MultiYAxisWorkspace(graph_id="multi-b")
        self.assertTrue(set(first.ids.values()).isdisjoint(second.ids.values()))

        app = Dash(__name__, suppress_callback_exceptions=True)
        app.layout = [first.render(), second.render()]
        first.register_callbacks(app)
        second.register_callbacks(app)

        outputs = set(app.callback_map)
        self.assertTrue(any("multi-a.figure" in output for output in outputs))
        self.assertTrue(any("multi-b.figure" in output for output in outputs))

    def test_dataframe_payload_is_not_duplicated_into_instance_store(self):
        tree = MULTI_Y_WORKSPACE.render()
        context = component_by_id(tree, MULTI_Y_WORKSPACE.ids["data_context"])
        state = component_by_id(tree, MULTI_Y_WORKSPACE.ids["state"])

        self.assertIsNone(context.to_plotly_json()["props"].get("data"))
        self.assertEqual(state.data["series"], [])
        self.assertNotIn("payload", state.data)

    def test_replacing_source_file_clears_old_series_but_filter_update_does_not(self):
        configured = {
            "dataset_id": "source",
            "scope": "filtered",
            "data_ref": "base-file-a",
            "shared_x": "depth",
            "series": [{"id": "s", "y": "pressure"}],
            "axes": [{"id": "a"}],
        }

        filtered = _sync_state_for_input(
            configured,
            "source",
            "filtered",
            {
                "dataset_id": "source",
                "data_ref": "base-file-a",
                "payload_ref": "filtered-version-2",
            },
        )
        replaced = _sync_state_for_input(
            configured,
            "source",
            "filtered",
            {
                "dataset_id": "source",
                "data_ref": "base-file-b",
                "payload_ref": "base-file-b",
            },
        )

        self.assertEqual(filtered["series"], configured["series"])
        self.assertEqual(filtered["shared_x"], "depth")
        self.assertEqual(replaced["series"], [])
        self.assertIsNone(replaced["shared_x"])
        self.assertEqual(replaced["data_ref"], "base-file-b")

    def test_stale_context_does_not_erase_a_series_added_during_dataset_switch(self):
        just_dropped = {
            "dataset_id": "dataset-2",
            "scope": "filtered",
            "data_ref": None,
            "shared_x": None,
            "series": [{"id": "new", "y": "temperature"}],
            "axes": [{"id": "axis-new"}],
        }
        result = _sync_state_for_input(
            just_dropped,
            "dataset-2",
            "filtered",
            {"dataset_id": "source", "data_ref": "old-source-ref"},
        )
        self.assertEqual(result["series"], just_dropped["series"])
        self.assertIsNone(result["data_ref"])

    def test_large_payload_is_decoded_once_per_registry_reference(self):
        _FRAME_CACHE.clear()
        decoded = object()
        with patch("multi_axis_workspace.read_df_from_store", return_value=decoded) as reader:
            first = _frame_for_payload("immutable-ref", "payload", {"numeric": []})
            second = _frame_for_payload("immutable-ref", "payload", {"numeric": []})

        self.assertIs(first, decoded)
        self.assertIs(second, decoded)
        reader.assert_called_once()

        with patch(
            "multi_axis_workspace.read_df_from_store",
            side_effect=lambda *_args, **_kwargs: object(),
        ):
            for index in range(_FRAME_CACHE_MAX_ENTRIES + 2):
                _frame_for_payload(f"other-ref-{index}", "payload", {})
        self.assertLessEqual(len(_FRAME_CACHE), _FRAME_CACHE_MAX_ENTRIES)

    def test_clear_action_includes_hard_plotly_recovery(self):
        workspace = MultiYAxisWorkspace(graph_id="recovery-probe")
        app = Dash(__name__, suppress_callback_exceptions=True)
        app.layout = workspace.render()
        workspace.register_callbacks(app)

        clear_scripts = [
            script
            for script in app._inline_scripts
            if "recovery-probe-clear" in script or "Plotly.purge" in script
        ]
        self.assertTrue(clear_scripts)
        self.assertTrue(any("Plotly.purge" in script for script in clear_scripts))


if __name__ == "__main__":
    unittest.main()
