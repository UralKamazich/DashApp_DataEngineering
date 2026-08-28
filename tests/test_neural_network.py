import unittest

import numpy as np
import pandas as pd

from dataset_registry import summarize_transformation_steps
from neural_network_engine import (
    run_neural_network_classification,
    run_neural_network_regression,
)


class NeuralNetworkTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(41)
        rows = 96
        x = rng.normal(size=rows)
        category = np.where(np.arange(rows) % 3, "песчаник", "карбонат").astype(object)
        self.frame = pd.DataFrame({
            "id": [f"row-{index}" for index in range(rows)],
            "x": x,
            "category": category,
            "target": 3 * x + (category == "песчаник") * 1.5 + rng.normal(0, .2, rows),
            "class": np.where(x + (category == "песчаник") * .6 > .15, "да", "нет"),
        })
        self.frame.loc[3, "x"] = np.nan
        self.frame.loc[5, "category"] = None

    def test_regression_scales_mixed_features_and_adds_outputs(self):
        result = run_neural_network_regression(
            self.frame, target="target", features=["x", "category"],
            id_column="id", hidden_layers="16, 8", max_iter=120,
            n_iter_no_change=10, permutation_repeats=1,
            prediction_column="nn prediction",
        )
        self.assertEqual(len(result.frame), len(self.frame))
        self.assertIn("nn prediction", result.frame)
        self.assertIn("Остаток target", result.frame)
        self.assertEqual(result.analysis["categorical_features"], ["category"])
        self.assertEqual(result.analysis["engine"], "PyTorch")
        self.assertIn(result.analysis["compute"]["resolved"], {"CPU", "MPS"})
        self.assertGreater(result.analysis["encoded_feature_estimate"], 1)
        self.assertGreater(result.analysis["epochs_run"], 0)
        self.assertEqual(len(result.analysis["feature_importance"]), 2)
        self.assertIn(result.analysis["overfitting"]["status"], {"low", "moderate", "high"})
        self.assertEqual(result.committed_step["operation"], "neural_network_regression")
        self.assertIn(
            "Neural Network",
            summarize_transformation_steps([result.committed_step])[0],
        )

    def test_classification_adds_class_confidence_and_metrics(self):
        result = run_neural_network_classification(
            self.frame, target="class", features=["x", "category"],
            id_column="id", hidden_layers="16", max_iter=140,
            n_iter_no_change=12, permutation_repeats=1,
            prediction_column="nn class",
        )
        self.assertEqual(result.analysis["task"], "classification")
        self.assertIn("nn class", result.frame)
        self.assertIn("Уверенность Neural Network", result.frame)
        self.assertTrue(result.frame["Уверенность Neural Network"].between(0, 1).all())
        self.assertGreaterEqual(result.analysis["metrics"]["balanced_accuracy"], .5)
        self.assertEqual(
            result.committed_step["operation"], "neural_network_classification"
        )

    def test_group_validation_keeps_control_column_out_of_features(self):
        frame = self.frame.copy()
        frame["well"] = [f"well-{index // 8}" for index in range(len(frame))]
        result = run_neural_network_regression(
            frame, target="target", features=["x", "category", "well"],
            method="group_cv", group_column="well", folds=3,
            hidden_layers="8", max_iter=60, early_stopping=False,
            permutation_repeats=0, engine="sklearn", compute_device="cpu",
        )
        self.assertEqual(len(result.analysis["fold_metrics"]), 3)
        self.assertIn("Group OOF", result.analysis["evaluation_label"])
        self.assertNotIn("well", result.analysis["features"])
        self.assertEqual(result.analysis["engine"], "sklearn MLP")

    def test_rejects_invalid_architecture(self):
        with self.assertRaisesRegex(ValueError, "целыми числами"):
            run_neural_network_regression(
                self.frame, target="target", features=["x"],
                hidden_layers="64, bad", max_iter=50,
            )


if __name__ == "__main__":
    unittest.main()
