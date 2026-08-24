import unittest

import numpy as np
import pandas as pd

from ml_data_profile import missingness_figure, profile_dataset, target_figure


class MLDataProfileTests(unittest.TestCase):
    def test_regression_profile_detects_quality_and_leakage_signals(self):
        rows = 100
        x = np.linspace(0, 20, rows)
        frame = pd.DataFrame({
            "x": x,
            "target": 2 * x + 3,
            "target_copy": 2 * x + 3,
            "constant": 1,
            "missing": [np.nan] * 40 + list(range(60)),
            "category": [f"item-{index}" for index in range(rows)],
        })
        original = frame.copy(deep=True)

        profile = profile_dataset(frame, target="target", task_mode="regression")

        self.assertEqual(profile["task"], "regression")
        self.assertEqual(profile["summary"]["rows"], rows)
        self.assertEqual(profile["summary"]["columns"], len(frame.columns))
        self.assertIn("target_copy", profile["leak_candidates"])
        self.assertEqual(profile["status"], "critical")
        signals = {row["Канал"]: row["Сигналы"] for row in profile["columns"]}
        self.assertIn("Константа", signals["constant"])
        self.assertIn("Много пропусков", signals["missing"])
        self.assertIn("Возможная утечка", signals["target_copy"])
        pd.testing.assert_frame_equal(frame, original)

    def test_auto_task_detects_imbalanced_classification(self):
        frame = pd.DataFrame({
            "feature": np.arange(100),
            "target": ["major"] * 96 + ["minor"] * 4,
        })
        profile = profile_dataset(frame, target="target", task_mode="auto")

        self.assertEqual(profile["task"], "classification")
        self.assertEqual(profile["target"]["unique"], 2)
        self.assertAlmostEqual(profile["target"]["minority_share"], 4.0)
        self.assertTrue(any(
            issue["title"] == "Сильный дисбаланс классов"
            for issue in profile["issues"]
        ))
        self.assertTrue(any(
            item["title"] == "Баланс классов"
            for item in profile["recommendations"]
        ))

    def test_profile_figures_cover_empty_and_populated_states(self):
        frame = pd.DataFrame({
            "feature": [1, np.nan, 3, np.nan, 5, 6],
            "target": ["A", "A", "B", "B", "B", "A"],
        })
        profile = profile_dataset(frame, target="target", task_mode="classification")

        self.assertGreater(len(missingness_figure(profile).data), 0)
        self.assertGreater(len(target_figure(frame, profile).data), 0)
        empty = profile_dataset(pd.DataFrame())
        self.assertEqual(len(missingness_figure(empty).layout.annotations), 1)


if __name__ == "__main__":
    unittest.main()
