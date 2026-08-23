import unittest

import numpy as np
import pandas as pd

from dataset_registry import summarize_transformation_steps
from ml_engine import cache_result, cached_result, ml_signature, run_catboost_regression


class CatBoostRegressionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        rows = 36
        x = np.linspace(0, 10, rows)
        category = np.where(np.arange(rows) % 2, "B", "A")
        self.frame = pd.DataFrame({
            "id": [f"row-{index}" for index in range(rows)],
            "x": x,
            "category": category,
            "target": 3 * x + (category == "B") * 4 + rng.normal(0, .2, rows),
        })

    def test_split_supports_mixed_features_and_aligned_outputs(self):
        result = run_catboost_regression(
            self.frame,
            target="target",
            features=["x", "category"],
            id_column="id",
            method="split",
            test_size=.25,
            iterations=35,
            early_stopping_rounds=8,
            prediction_column="prediction",
            compute_shap=True,
        )
        self.assertEqual(len(result.frame), len(self.frame))
        self.assertIn("prediction", result.frame)
        self.assertIn("Остаток target", result.frame)
        self.assertEqual(result.analysis["categorical_features"], ["category"])
        self.assertGreater(result.analysis["evaluation_rows"], 1)
        self.assertTrue(result.analysis["shap_importance"])
        self.assertEqual(result.committed_step["operation"], "catboost_regression")

    def test_cross_validation_uses_out_of_fold_predictions(self):
        result = run_catboost_regression(
            self.frame,
            target="target",
            features=["x", "category"],
            method="cv",
            folds=3,
            iterations=20,
            early_stopping_rounds=5,
            include_residual=False,
            compute_shap=False,
        )
        self.assertEqual(result.analysis["evaluation_rows"], len(self.frame))
        self.assertEqual(len(result.analysis["fold_metrics"]), 3)
        self.assertEqual(result.analysis["evaluation_label"], "OOF, 3 folds")
        self.assertEqual(len(result.analysis["outputs"]), 1)

    def test_cache_signature_and_dataset_summary(self):
        signature = ml_signature(target="target", features=["x"])
        result = run_catboost_regression(
            self.frame,
            target="target",
            features=["x"],
            iterations=10,
            early_stopping_rounds=3,
            compute_shap=False,
            signature=signature,
        )
        reference = cache_result(result)
        self.assertIs(cached_result(reference), result)
        summary = summarize_transformation_steps([result.committed_step])[0]
        self.assertIn("CatBoost", summary)
        self.assertIn("target", summary)

    def test_rejects_non_numeric_target(self):
        with self.assertRaisesRegex(ValueError, "пяти строк"):
            run_catboost_regression(
                self.frame,
                target="category",
                features=["x"],
                iterations=5,
                compute_shap=False,
            )


if __name__ == "__main__":
    unittest.main()
