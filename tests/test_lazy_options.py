"""Page-lazy column-list callbacks."""

import json
import unittest

import plotly
from dash.exceptions import PreventUpdate

from app import app
from lazy_options import LAZY_KEYS, page_matches


def _callback_for(output_fragment):
    for key, spec in app.callback_map.items():
        if output_fragment in key:
            return spec
    raise AssertionError(f"No callback writes {output_fragment}")


class LazyOptionTests(unittest.TestCase):
    def test_expected_pickers_are_lazy(self):
        for key in (
            "graph-options", "ml-columns", "rf-columns", "nn-columns",
            "ml-profile", "ml-profile-target", "cluster-cols",
        ):
            self.assertIn(key, LAZY_KEYS)
        self.assertEqual(sum(key.startswith("options-") for key in LAZY_KEYS), 4)

    def test_layout_contains_signature_store_for_every_lazy_callback(self):
        layout = json.dumps(app.layout, cls=plotly.utils.PlotlyJSONEncoder)
        for key in LAZY_KEYS:
            self.assertIn(f'"lazy-sig-{key}"', layout)

    def test_page_matching(self):
        self.assertTrue(page_matches(None, "/"))
        self.assertTrue(page_matches("/", "/"))
        self.assertFalse(page_matches("/dashboard", "/"))
        self.assertTrue(page_matches("/ml/catboost", "/ml/catboost"))
        self.assertFalse(page_matches("/ml", "/ml/catboost"))

    def test_wrapper_skips_hidden_page_and_unchanged_inputs(self):
        spec = _callback_for("lazy-sig-cluster-cols")
        wrapper = spec["callback"].__wrapped__
        registry = {"source": {"meta": {"numeric": ["a", "b"]}}}

        with self.assertRaises(PreventUpdate):
            wrapper("source", registry, "/", [], None)

        data, kept, signature = wrapper("source", registry, "/clustering", ["a"], None)
        self.assertEqual(kept, ["a"])
        self.assertTrue(signature)

        with self.assertRaises(PreventUpdate):
            wrapper("source", registry, "/clustering", ["a"], signature)

        changed = {"source": {"meta": {"numeric": ["a", "c"]}}}
        _, _, new_signature = wrapper("source", changed, "/clustering", ["a"], signature)
        self.assertNotEqual(new_signature, signature)
