"""Unit tests for the reusable multi-Y-axis figure engine."""

import copy
import unittest

import numpy as np
import pandas as pd

from multi_axis_engine import (
    build_multi_axis_figure,
    multi_axis_uirevision,
    normalize_multi_axis_state,
    required_columns,
)


class MultiAxisStateTests(unittest.TestCase):
    def test_normalization_assigns_stable_series_axis_and_plotly_ids(self):
        raw = {
            "x": "depth",
            "series": [
                {"y": "pressure", "side": "left"},
                {"y": "temperature", "side": "right"},
            ],
        }

        first = normalize_multi_axis_state(raw)
        second = normalize_multi_axis_state(first)

        self.assertEqual(first, second)
        self.assertEqual(first["version"], 3)
        self.assertEqual(first["shared_x"], "depth")
        self.assertEqual(
            [series["id"] for series in first["series"]],
            ["series-1", "series-2"],
        )
        self.assertEqual(
            [axis["id"] for axis in first["axes"]],
            ["axis-1", "axis-2"],
        )
        self.assertEqual(
            [axis["plotly_ref"] for axis in first["axes"]],
            ["y2", "y3"],
        )
        self.assertEqual([axis["side"] for axis in first["axes"]], ["left", "right"])

    def test_duplicate_ids_are_repaired_without_changing_existing_ids(self):
        state = {
            "series": [
                {"id": "flow", "y": "a", "axis_id": "scale"},
                {"id": "flow", "y": "b", "axis_id": "scale"},
                {"y": "c"},
            ],
            "axes": [{"id": "scale", "plotly_ref": "y8"}],
        }
        normalized = normalize_multi_axis_state(state)

        self.assertEqual(normalized["series"][0]["id"], "flow")
        self.assertEqual(normalized["series"][1]["id"], "series-1")
        self.assertEqual(normalized["series"][2]["id"], "series-2")
        self.assertEqual(normalized["axes"][0]["plotly_ref"], "y8")

    def test_legacy_individual_x_is_ignored_in_favor_of_shared_x(self):
        state = {
            "shared_x": "time",
            "axes": [{"id": "common", "side": "right"}],
            "series": [
                {"id": "a", "y": "pressure", "axis_id": "common"},
                {
                    "id": "b",
                    "y": "rate",
                    "x_mode": "individual",
                    "x": "measured_depth",
                    "axis_id": "common",
                },
            ],
        }
        normalized = normalize_multi_axis_state(state)

        self.assertEqual(len(normalized["axes"]), 1)
        self.assertEqual(normalized["series"][0]["axis_id"], "common")
        self.assertEqual(normalized["series"][1]["axis_id"], "common")
        self.assertNotIn("x", normalized["series"][1])
        self.assertNotIn("x_mode", normalized["series"][1])
        self.assertEqual(
            required_columns(normalized),
            ["time", "pressure", "rate"],
        )

    def test_shared_y_mode_collapses_axes_by_side_without_losing_series(self):
        normalized = normalize_multi_axis_state({
            "shared_y_axes": True,
            "series": [
                {"id": "left-a", "y": "a", "side": "left"},
                {"id": "left-b", "y": "b", "side": "left"},
                {"id": "right-a", "y": "c", "side": "right"},
            ],
        })

        self.assertTrue(normalized["shared_y_axes"])
        self.assertEqual(
            [series["axis_id"] for series in normalized["series"]],
            ["axis-shared-left", "axis-shared-left", "axis-shared-right"],
        )
        self.assertEqual(
            [(axis["id"], axis["plotly_ref"], axis["side"]) for axis in normalized["axes"]],
            [
                ("axis-shared-left", "y2", "left"),
                ("axis-shared-right", "y3", "right"),
            ],
        )

    def test_available_columns_do_not_destroy_stale_assignments(self):
        normalized = normalize_multi_axis_state(
            {"x": "time", "series": [{"y": "missing"}]},
            ["time", "other"],
        )
        self.assertEqual(normalized["series"][0]["y"], "missing")
        self.assertFalse(normalized["series"][0]["available"])

    def test_uirevision_ignores_cosmetics_but_tracks_coordinates(self):
        state = normalize_multi_axis_state({
            "x": "time",
            "series": [{"id": "s", "y": "value", "color": "#111111"}],
        })
        original = multi_axis_uirevision(state)

        cosmetic = copy.deepcopy(state)
        cosmetic["title"] = "New title"
        cosmetic["series"][0].update({
            "name": "Renamed",
            "color": "#abcdef",
            "type": "scatter",
            "marker_size": 19,
        })
        cosmetic["axes"][0]["side"] = "right"
        self.assertEqual(multi_axis_uirevision(cosmetic), original)

        structural = copy.deepcopy(state)
        structural["series"][0]["y"] = "another_value"
        self.assertNotEqual(multi_axis_uirevision(structural), original)


class MultiAxisFigureTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "time": np.arange(12),
            "line": np.linspace(0, 1, 12),
            "scatter": np.linspace(10, 20, 12),
            "line_markers": np.linspace(20, 30, 12),
            "step": np.linspace(30, 40, 12),
            "area": np.linspace(40, 50, 12),
            "box": np.tile([1.0, 2.0, 8.0], 4),
            "unused": ["large payload"] * 12,
        })

    def test_all_supported_series_types_have_expected_plotly_style(self):
        state = {
            "x": "time",
            "series": [
                {"y": "line", "type": "line"},
                {"y": "scatter", "type": "scatter"},
                {"y": "line_markers", "type": "line+markers"},
                {"y": "step", "type": "step"},
                {"y": "area", "type": "area", "color": "#123456"},
            ],
        }
        result = build_multi_axis_figure(self.frame, state)
        traces = result.figure.data

        self.assertEqual([trace.mode for trace in traces], [
            "lines", "markers", "lines+markers", "lines", "lines",
        ])
        self.assertEqual(traces[3].line.shape, "hv")
        self.assertEqual(traces[4].fill, "tozeroy")
        self.assertEqual(traces[4].fillcolor, "rgba(18,52,86,0.220)")

    def test_every_series_uses_shared_x_even_with_legacy_individual_state(self):
        frame = self.frame.assign(legacy_x=np.arange(len(self.frame)) * 100)
        result = build_multi_axis_figure(frame, {
            "shared_x": "time",
            "series": [{
                "y": "line",
                "x": "legacy_x",
                "x_mode": "individual",
            }],
        })

        self.assertEqual(list(result.figure.data[0].x), list(frame["time"]))
        self.assertEqual(result.metadata["selected_columns"], ["time", "line"])

    def test_render_mode_supports_hybrid_and_forced_svg(self):
        state = {"x": "time", "series": [{"y": "scatter", "type": "scatter"}]}
        hybrid = build_multi_axis_figure(self.frame, state, render_mode="hybrid")
        svg = build_multi_axis_figure(self.frame, state, render_mode="svg")

        self.assertEqual(hybrid.figure.data[0].type, "scattergl")
        self.assertEqual(svg.figure.data[0].type, "scatter")
        self.assertEqual(hybrid.metadata["render_mode"], "hybrid")

    def test_native_box_plot_has_transparent_body_and_requested_points(self):
        frame = pd.DataFrame({
            "group": ["A", "A", "A", "B", "B", "B"],
            "value": [1, 2, 10, 3, 4, 20],
        })
        figure = build_multi_axis_figure(frame, {
            "shared_x": "group",
            "series": [{
                "id": "box-a",
                "y": "value",
                "type": "box",
                "box_points": "all",
                "color": "#228be6",
            }],
        }).figure

        trace = figure.data[0]
        self.assertEqual(trace.type, "box")
        self.assertEqual(trace.boxpoints, "all")
        self.assertEqual(trace.fillcolor, "rgba(34,139,230,0.240)")
        self.assertEqual(trace.line.color, "#228be6")
        self.assertEqual(trace.yaxis, "y2")
        self.assertEqual(list(figure.layout.xaxis.ticktext), ["A", "B"])

    def test_native_boxes_on_different_axes_use_separate_horizontal_slots(self):
        frame = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "first": [1, 2, 3, 4],
            "second": [100, 120, 140, 160],
            "line": [10, 20, 30, 40],
        })
        figure = build_multi_axis_figure(frame, {
            "shared_x": "group",
            "series": [
                {"id": "first-box", "y": "first", "type": "box"},
                {"id": "second-box", "y": "second", "type": "box"},
                {"id": "ordinary-line", "y": "line", "type": "line"},
            ],
        }).figure

        first, second, line = figure.data
        self.assertEqual([first.type, second.type], ["box", "box"])
        self.assertNotEqual(first.x[0], second.x[0])
        self.assertLess(first.x[0], second.x[0])
        self.assertEqual(first.fillcolor, "rgba(34,139,230,0.240)")
        self.assertTrue(second.fillcolor.endswith(",0.240)"))
        self.assertEqual(figure.layout.plot_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(line.mode, "lines")
        self.assertEqual(list(line.y), list(frame["line"]))

    def test_every_axis_gets_a_distinct_position_outside_the_data_domain(self):
        figure = build_multi_axis_figure(self.frame, {
            "shared_x": "time",
            "series": [
                {"y": "line", "side": "left"},
                {"y": "scatter", "side": "left"},
                {"y": "step", "side": "right"},
                {"y": "area", "side": "right"},
            ],
        }).figure

        left_first = figure.layout.yaxis2
        left_second = figure.layout.yaxis3
        right_first = figure.layout.yaxis4
        right_second = figure.layout.yaxis5
        self.assertEqual(list(figure.layout.xaxis.domain), [0.06, 0.94])
        self.assertEqual((left_first.position, left_first.shift), (0.06, 0))
        self.assertEqual((left_second.position, left_second.shift), (0.0, 0))
        self.assertEqual((right_first.position, right_first.shift), (0.94, 0))
        self.assertEqual((right_second.position, right_second.shift), (1.0, 0))
        self.assertEqual(
            {left_first.anchor, left_second.anchor, right_first.anchor, right_second.anchor},
            {"free"},
        )
        self.assertEqual(left_first.title.standoff, 5)
        self.assertTrue(figure.layout.xaxis.showline)
        self.assertEqual(figure.layout.xaxis.ticks, "outside")
        self.assertEqual(figure.layout.xaxis.ticklen, 5)
        self.assertFalse(left_first.autoshift)
        self.assertFalse(right_first.autoshift)

    def test_box_without_shared_x_builds_one_native_distribution(self):
        figure = build_multi_axis_figure(self.frame, {
            "series": [{"y": "box", "type": "box", "box_points": "none"}],
        }).figure

        trace = figure.data[0]
        self.assertEqual(trace.type, "box")
        self.assertIsNone(trace.x)
        self.assertFalse(trace.boxpoints)
        self.assertEqual(list(trace.y), list(self.frame["box"]))

    def test_legend_visibility_is_owned_by_workspace_state(self):
        hidden = build_multi_axis_figure(
            self.frame,
            {
                "shared_x": "time",
                "show_legend": False,
                "series": [{"y": "line"}, {"y": "scatter"}],
            },
        ).figure
        visible = build_multi_axis_figure(
            self.frame,
            {
                "shared_x": "time",
                "show_legend": True,
                "series": [{"y": "line"}, {"y": "scatter"}],
            },
        ).figure

        self.assertFalse(hidden.layout.showlegend)
        self.assertTrue(visible.layout.showlegend)

    def test_series_name_is_the_default_axis_label(self):
        figure = build_multi_axis_figure(
            self.frame,
            {
                "shared_x": "time",
                "series": [{
                    "id": "pressure-series",
                    "axis_id": "axis-1",
                    "y": "line",
                    "name": "Подпись давления",
                }],
                "axes": [{
                    "id": "axis-1",
                    "title": "Устаревшая отдельная подпись",
                    "side": "left",
                    "type": "linear",
                }],
            },
        ).figure

        self.assertEqual(figure.layout.yaxis2.title.text, "Подпись давления")

    def test_shared_y_axes_have_black_scales_and_multicolor_composite_titles(self):
        result = build_multi_axis_figure(
            self.frame,
            {
                "shared_x": "time",
                "shared_y_axes": True,
                "series": [
                    {
                        "y": "line", "name": "Давление",
                        "color": "#ff0000", "side": "left",
                    },
                    {
                        "y": "scatter", "name": "Температура",
                        "color": "#0000ff", "side": "left",
                    },
                    {
                        "y": "area", "name": "Расход",
                        "color": "#00aa00", "side": "right",
                    },
                ],
            },
        )

        figure = result.figure
        self.assertEqual([trace.yaxis for trace in figure.data], ["y2", "y2", "y3"])
        self.assertEqual(set(result.metadata["axis_refs"]), {
            "axis-shared-left", "axis-shared-right",
        })
        left_title = figure.layout.yaxis2.title.text
        self.assertIn("Давление", left_title)
        self.assertIn("Температура", left_title)
        self.assertIn("#ff0000", left_title)
        self.assertIn("#0000ff", left_title)
        self.assertIn("; ", left_title)
        for axis in (figure.layout.yaxis2, figure.layout.yaxis3):
            self.assertEqual(axis.linecolor, "#000000")
            self.assertEqual(axis.tickcolor, "#000000")
            self.assertEqual(axis.tickfont.color, "#000000")
        self.assertEqual(figure.layout.xaxis.linecolor, "#000000")
        self.assertEqual(figure.layout.xaxis.tickcolor, "#000000")
        self.assertEqual(figure.layout.xaxis.tickfont.color, "#000000")

    def test_line_smoothing_uses_svg_spline_only_for_supported_types(self):
        result = build_multi_axis_figure(
            self.frame,
            {
                "shared_x": "time",
                "series": [
                    {"y": "line", "type": "line", "smooth": True},
                    {"y": "line_markers", "type": "line+markers", "smooth": True},
                    {"y": "area", "type": "area", "smooth": True},
                    {"y": "scatter", "type": "scatter", "smooth": True},
                    {
                        "y": "step",
                        "type": "step",
                        "smooth": True,
                        "step_shape": "spline",
                    },
                ],
            },
        )

        for trace in result.figure.data[:3]:
            self.assertEqual(trace.type, "scatter")
            self.assertEqual(trace.line.shape, "spline")
            self.assertEqual(trace.line.smoothing, 0.7)
        self.assertEqual(result.figure.data[1].mode, "lines+markers")
        self.assertEqual(result.figure.data[2].fill, "tozeroy")
        self.assertEqual(result.figure.data[3].type, "scattergl")
        self.assertNotEqual(result.figure.data[3].line.shape, "spline")
        self.assertEqual(result.figure.data[4].type, "scattergl")
        self.assertEqual(result.figure.data[4].line.shape, "hv")

    def test_string_false_does_not_enable_line_smoothing(self):
        normalized = normalize_multi_axis_state({
            "series": [{"y": "line", "smooth": "false"}],
        })

        self.assertFalse(normalized["series"][0]["smooth"])

    def test_figure_dimensions_follow_workspace_paper(self):
        state = {
            "x": "time",
            "height": 825,
            "width": 1320,
            "series": [{"y": "line"}],
        }
        fixed = build_multi_axis_figure(self.frame, state).figure
        fluid = build_multi_axis_figure(
            self.frame,
            {**state, "width": None},
        ).figure

        self.assertEqual(fixed.layout.height, 825)
        self.assertEqual(fixed.layout.width, 1320)
        self.assertFalse(fixed.layout.autosize)
        self.assertEqual(fluid.layout.height, 825)
        self.assertIsNone(fluid.layout.width)
        self.assertTrue(fluid.layout.autosize)

    def test_private_axis_follows_trace_color_after_cosmetic_change(self):
        original = normalize_multi_axis_state({
            "x": "time",
            "series": [{"id": "s", "y": "line", "color": "#228be6"}],
        })
        # This is the exact store lifecycle that used to leave a stale axis
        # color: the already-normalized state receives only a series update.
        original["series"][0]["color"] = "#e03131"
        result = build_multi_axis_figure(self.frame, original)

        self.assertEqual(result.figure.data[0].line.color, "#e03131")
        self.assertEqual(result.figure.layout.yaxis2.title.font.color, "#e03131")
        self.assertEqual(result.figure.layout.yaxis2.tickfont.color, "#e03131")
        self.assertEqual(result.figure.layout.yaxis2.linecolor, "#e03131")
        self.assertEqual(result.figure.layout.yaxis2.tickcolor, "#e03131")
        self.assertEqual(result.state["axes"][0]["color"], "#e03131")

    def test_log_axis_manual_range_uses_real_values_in_workspace_state(self):
        state = {
            "x": "time",
            "series": [{"y": "scatter", "axis_id": "log-axis"}],
            "axes": [{
                "id": "log-axis",
                "type": "log",
                "autorange": False,
                "range": [1, 1000],
            }],
        }
        result = build_multi_axis_figure(self.frame, state)
        layout_key = result.metadata["axis_refs"]["log-axis"]["layout"]
        layout_axis = getattr(result.figure.layout, layout_key)

        self.assertEqual(result.state["axes"][0]["range"], [1.0, 1000.0])
        self.assertEqual(list(layout_axis.range), [0.0, 3.0])
        self.assertFalse(layout_axis.autorange)

    def test_invalid_nonpositive_log_range_falls_back_to_autorange(self):
        state = {
            "x": "time",
            "series": [{"y": "scatter", "axis_id": "log-axis"}],
            "axes": [{
                "id": "log-axis",
                "type": "log",
                "autorange": False,
                "range": [-1, 100],
            }],
        }
        result = build_multi_axis_figure(self.frame, state)
        layout_key = result.metadata["axis_refs"]["log-axis"]["layout"]
        layout_axis = getattr(result.figure.layout, layout_key)

        self.assertIsNone(layout_axis.range)
        self.assertTrue(layout_axis.autorange)

    def test_secondary_axes_overlay_hidden_carrier_and_only_first_has_grid(self):
        state = {
            "x": "time",
            "series": [
                {"y": "line", "side": "left"},
                {"y": "scatter", "side": "right"},
                {"y": "step", "side": "left"},
            ],
        }
        figure = build_multi_axis_figure(self.frame, state).figure

        self.assertFalse(figure.layout.yaxis.visible)
        self.assertEqual([trace.yaxis for trace in figure.data], ["y2", "y3", "y4"])
        for axis in (figure.layout.yaxis2, figure.layout.yaxis3, figure.layout.yaxis4):
            self.assertEqual(axis.overlaying, "y")
            self.assertFalse(axis.autoshift)
            self.assertEqual(axis.anchor, "free")
        self.assertTrue(figure.layout.yaxis2.showgrid)
        self.assertFalse(figure.layout.yaxis3.showgrid)
        self.assertFalse(figure.layout.yaxis4.showgrid)
        self.assertEqual(figure.layout.yaxis3.side, "right")

    def test_plotly_axis_reference_survives_removal_and_reordering(self):
        normalized = normalize_multi_axis_state({
            "x": "time",
            "series": [
                {"id": "first", "y": "line"},
                {"id": "keeper", "y": "scatter"},
                {"id": "third", "y": "step"},
            ],
        })
        keeper_axis = normalized["series"][1]["axis_id"]
        keeper_ref = next(
            axis["plotly_ref"] for axis in normalized["axes"] if axis["id"] == keeper_axis
        )
        normalized["series"] = [normalized["series"][2], normalized["series"][1]]

        result = build_multi_axis_figure(self.frame, normalized)
        keeper_trace = next(trace for trace in result.figure.data if trace.meta["series_id"] == "keeper")
        self.assertEqual(keeper_trace.yaxis, keeper_ref)
        self.assertEqual(result.metadata["axis_refs"][keeper_axis]["trace"], keeper_ref)

    def test_browser_axis_id_has_stable_ref_even_before_state_is_normalized(self):
        raw = {
            "x": "time",
            "series": [
                {"id": "series-old", "y": "line", "axis_id": "axis-series-old"},
                {"id": "series-keeper", "y": "scatter", "axis_id": "axis-series-keeper"},
            ],
            "axes": [
                {"id": "axis-series-old"},
                {"id": "axis-series-keeper"},
            ],
        }
        before = normalize_multi_axis_state(raw)
        keeper_before = next(
            axis["plotly_ref"] for axis in before["axes"]
            if axis["id"] == "axis-series-keeper"
        )
        raw["series"] = [raw["series"][1]]
        raw["axes"] = [raw["axes"][1]]
        after = normalize_multi_axis_state(raw)

        self.assertEqual(after["axes"][0]["plotly_ref"], keeper_before)

    def test_shared_axis_creates_one_colored_scale(self):
        state = {
            "x": "time",
            "axes": [{"id": "shared", "side": "right", "color": "#555555"}],
            "series": [
                {"y": "line", "axis_id": "shared", "color": "#ff0000"},
                {"y": "scatter", "axis_id": "shared", "color": "#0000ff"},
            ],
        }
        result = build_multi_axis_figure(self.frame, state)

        self.assertEqual(result.figure.data[0].yaxis, result.figure.data[1].yaxis)
        self.assertEqual(len(result.metadata["axis_refs"]), 1)
        layout_key = result.metadata["axis_refs"]["shared"]["layout"]
        shared_layout = result.figure.layout[layout_key]
        self.assertEqual(shared_layout.side, "right")
        self.assertEqual(shared_layout.linecolor, "#555555")

    def test_build_selects_only_required_columns_and_skips_missing_series(self):
        state = {
            "x": "time",
            "series": [
                {"y": "line"},
                {"y": "missing"},
                {"y": "area", "visible": False},
            ],
        }
        result = build_multi_axis_figure(self.frame, state)

        self.assertEqual(result.metadata["selected_columns"], ["time", "line"])
        self.assertEqual(result.metadata["missing_columns"], ["missing"])
        self.assertNotIn("unused", result.metadata["selected_columns"])
        self.assertEqual(len(result.figure.data), 1)

    def test_visual_sampling_shares_budget_and_preserves_endpoints(self):
        rows = 100_001
        frame = pd.DataFrame({
            "x": np.arange(rows),
            "a": np.arange(rows),
            "b": np.arange(rows) * 2,
            "untouched": np.arange(rows) * 3,
        })
        original = frame.copy(deep=True)
        result = build_multi_axis_figure(
            frame,
            {"x": "x", "series": [{"y": "a"}, {"y": "b"}]},
            max_visual_points=1_000,
        )

        self.assertTrue(result.metadata["sampled"])
        self.assertEqual(result.metadata["displayed_rows"], 500)
        self.assertEqual(result.metadata["displayed_points"], 1_000)
        self.assertEqual(result.metadata["sampling_method"], "evenly_spaced")
        self.assertIsNotNone(result.metadata["visual_sample_message"])
        self.assertEqual(result.figure.data[0].x[0], 0)
        self.assertEqual(result.figure.data[0].x[-1], rows - 1)
        pd.testing.assert_frame_equal(frame, original)

    def test_absent_x_uses_source_row_numbers(self):
        result = build_multi_axis_figure(
            self.frame,
            {"series": [{"y": "line"}]},
            max_visual_points=5,
        )
        self.assertEqual(list(result.figure.data[0].x), [0, 2, 5, 8, 11])
        self.assertEqual(result.figure.layout.xaxis.title.text, "Номер строки")


if __name__ == "__main__":
    unittest.main()
