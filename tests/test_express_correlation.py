"""«Экспресс корр. анализ»: rating engine, cross-plot endpoint and page wiring."""

import json
import unittest

import numpy as np
import pandas as pd
import plotly

from app import app
from callbacks.express_correlation import build_rating, status_text
from express_correlation import crossplot_points, rank_correlations
from utils import frame_to_store, meta_from_df


def _frame(seed=1, n=2000):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"depth": np.arange(n, dtype=float), "name": "well"})
    for index in range(12):
        frame[f"c{index}"] = rng.normal(size=n)
    frame["target"] = 2 * frame["c0"] + rng.normal(size=n)
    frame["c5"] = frame["c5"].round(0)                       # ties
    for column in ("c1", "c2", "target"):
        frame.loc[rng.choice(n, 150, replace=False), column] = np.nan
    frame.loc[rng.choice(n, 120, replace=False), "c3"] = 0.0
    frame.loc[rng.choice(n, 80, replace=False), "target"] = 0.0
    frame["const"] = 1.0
    return frame


class RankCorrelationTests(unittest.TestCase):
    def _assert_matches_pandas(self, exclude_zero):
        frame = _frame()
        numeric = frame.drop(columns=["name"])
        if exclude_zero:
            numeric = numeric.replace(0, np.nan)
        result = rank_correlations(frame, "target", exclude_zero=exclude_zero)
        for method in ("pearson", "spearman"):
            expected = numeric.corr(method=method)["target"].drop("target").dropna()
            got = pd.Series({row["channel"]: row[method] for row in result["rows"]})
            self.assertEqual(set(got.index), set(expected.index), method)
            self.assertLess(float((got - expected[got.index]).abs().max()), 1e-5, method)

    def test_both_coefficients_match_pandas_without_zero_exclusion(self):
        self._assert_matches_pandas(exclude_zero=False)

    def test_both_coefficients_match_pandas_with_zero_exclusion(self):
        self._assert_matches_pandas(exclude_zero=True)

    def test_p_values_match_scipy(self):
        from scipy import stats

        frame = _frame()
        row = next(r for r in rank_correlations(frame, "target")["rows"] if r["channel"] == "c1")
        pair = frame[["target", "c1"]].replace(0, np.nan).dropna()
        self.assertAlmostEqual(row["p_pearson"], stats.pearsonr(pair.target, pair.c1).pvalue, places=8)
        self.assertAlmostEqual(row["p_spearman"], stats.spearmanr(pair.target, pair.c1).pvalue, places=8)

    def test_rows_are_sorted_by_absolute_pearson_and_constant_is_excluded(self):
        result = rank_correlations(_frame(), "target")
        strengths = [abs(row["pearson"]) for row in result["rows"]]
        self.assertEqual(strengths, sorted(strengths, reverse=True))
        self.assertEqual(result["rows"][0]["channel"], "c0")
        self.assertNotIn("const", [row["channel"] for row in result["rows"]])
        self.assertGreaterEqual(result["excluded"], 1)

    def test_start_after_limits_channels_to_the_right(self):
        result = rank_correlations(_frame(), "target", start_after="c8")
        self.assertEqual(
            sorted(row["channel"] for row in result["rows"]), ["c10", "c11", "c9"]
        )

    def test_unknown_columns_raise(self):
        with self.assertRaises(ValueError):
            rank_correlations(_frame(), "missing")
        with self.assertRaises(ValueError):
            rank_correlations(_frame(), "target", start_after="missing")


class CrossplotTests(unittest.TestCase):
    def test_points_are_thinned_and_zeros_dropped(self):
        frame = _frame(n=5000)
        data = crossplot_points(frame, "c3", "target", max_points=1000)
        self.assertEqual(data["shown"], 1000)
        self.assertEqual(len(data["x"]), 1000)
        self.assertNotIn(0.0, data["x"])
        with_zeros = crossplot_points(frame, "c3", "target", exclude_zero=False)
        self.assertGreater(with_zeros["total"], data["total"])

    def test_endpoint_returns_points_for_stored_dataset(self):
        frame = _frame()
        ref = frame_to_store(frame)
        client = app.server.test_client()
        response = client.get(
            "/api/express-corr/crossplot",
            query_string={"ref": ref, "x": "c0", "y": "target", "zero": "1"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertGreater(payload["pearson"], 0.8)
        self.assertEqual(len(payload["x"]), payload["shown"])

        missing = client.get(
            "/api/express-corr/crossplot",
            query_string={"ref": "dfref:gone", "x": "c0", "y": "target"},
        )
        self.assertEqual(missing.status_code, 410)


class ExpressPageTests(unittest.TestCase):
    def test_build_rating_returns_reference_and_status(self):
        frame = _frame()
        ref = frame_to_store(frame)
        result = build_rating(ref, meta_from_df(frame), "target", True, None)
        self.assertEqual(result["ref"], ref)
        self.assertTrue(result["rows"])
        self.assertIn("Пирсон и Спирмен", status_text(result))
        self.assertIn("spearman", result["rows"][0])
        self.assertIn("без нулей", status_text(result))
        self.assertIn("error", build_rating(None, {}, "target", True, None))

    def test_page_is_in_layout_and_navigation(self):
        layout = json.dumps(
            json.loads(json.dumps(app.layout, cls=plotly.utils.PlotlyJSONEncoder)),
            ensure_ascii=False,
        )
        for component_id in (
            "page-express-correlation", "xcorr-target",
            "xcorr-exclude-zero", "xcorr-start-after", "xcorr-run", "xcorr-table",
        ):
            self.assertIn(f'"{component_id}"', layout)
        self.assertIn("/express-correlation", layout)
        self.assertIn("Экспресс корр. анализ", layout)

    def test_exclude_zero_checkbox_is_on_by_default(self):
        layout = json.loads(json.dumps(app.layout, cls=plotly.utils.PlotlyJSONEncoder))

        def find(node):
            if isinstance(node, dict):
                props = node.get("props", {})
                if props.get("id") == "xcorr-exclude-zero":
                    return props
                for value in props.values():
                    found = find(value)
                    if found:
                        return found
            elif isinstance(node, list):
                for item in node:
                    found = find(item)
                    if found:
                        return found
            return None

        self.assertIs(find(layout)["checked"], True)
