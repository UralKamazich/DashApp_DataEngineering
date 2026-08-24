import unittest

import numpy as np
import pandas as pd

from dataset_registry import summarize_transformation_steps
from random_forest_engine import (
    run_random_forest_classification,
    run_random_forest_regression,
)


class RandomForestTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(25)
        rows = 72
        x = rng.normal(size=rows)
        category = np.where(np.arange(rows) % 3, "песчаник", "карбонат")
        self.frame = pd.DataFrame({
            "id": [f"row-{index}" for index in range(rows)],
            "x": x,
            "category": category,
            "target": 4 * x + (category == "песчаник") * 2 + rng.normal(0, .3, rows),
            "class": np.where(x + (category == "песчаник") * .7 > .25, "да", "нет"),
        })
        self.frame.loc[3, "x"] = np.nan
        self.frame.loc[5, "category"] = None

    def test_regression_supports_mixed_features_oob_and_aligned_outputs(self):
        result = run_random_forest_regression(
            self.frame, target="target", features=["x", "category"],
            id_column="id", n_estimators=30, max_samples=.8,
            prediction_column="rf prediction",
        )
        self.assertEqual(len(result.frame), len(self.frame))
        self.assertIn("rf prediction", result.frame)
        self.assertIn("Остаток target", result.frame)
        self.assertEqual(result.analysis["categorical_features"], ["category"])
        self.assertIsNotNone(result.analysis["oob_score"])
        self.assertIn(result.analysis["overfitting"]["status"], {"low", "moderate", "high"})
        self.assertAlmostEqual(
            sum(item["importance"] for item in result.analysis["feature_importance"]),
            1.0, places=6,
        )
        self.assertEqual(result.committed_step["operation"], "random_forest_regression")
        self.assertIn("Random Forest", summarize_transformation_steps([result.committed_step])[0])

    def test_classification_adds_confidence_and_metrics(self):
        result = run_random_forest_classification(
            self.frame, target="class", features=["x", "category"],
            id_column="id", n_estimators=30, max_samples=.8,
            prediction_column="rf class",
        )
        self.assertEqual(result.analysis["task"], "classification")
        self.assertIn("rf class", result.frame)
        self.assertIn("Уверенность Random Forest", result.frame)
        self.assertTrue(result.frame["Уверенность Random Forest"].between(0, 1).all())
        self.assertGreaterEqual(result.analysis["metrics"]["balanced_accuracy"], .5)
        self.assertEqual(
            result.committed_step["operation"], "random_forest_classification"
        )

    def test_cross_validation_and_group_control(self):
        frame = self.frame.copy()
        frame["well"] = [f"well-{index // 6}" for index in range(len(frame))]
        result = run_random_forest_regression(
            frame, target="target", features=["x", "category", "well"],
            method="group_cv", group_column="well", folds=3,
            n_estimators=20, max_samples=.8,
        )
        self.assertEqual(len(result.analysis["fold_metrics"]), 3)
        self.assertIn("Group OOF", result.analysis["evaluation_label"])
        self.assertNotIn("well", result.analysis["features"])

    def test_continuous_classification_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "похож на непрерывный"):
            run_random_forest_classification(
                self.frame, target="target", features=["x"], n_estimators=20,
            )


if __name__ == "__main__":
    unittest.main()
