import unittest
from threading import Event

import numpy as np
import pandas as pd

from ml_tuning import tune_catboost_parameters


class CatBoostTuningTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(14)
        rows = 72
        x = generator.normal(size=rows)
        category = np.where(np.arange(rows) % 2, "A", "B")
        self.frame = pd.DataFrame({
            "x": x,
            "category": category,
            "target": 2.5 * x + (category == "B") * 2 + generator.normal(0, .2, rows),
            "class": np.where(x + (category == "B") * .7 > .2, "yes", "no"),
        })

    def test_regression_search_is_deterministic_and_keeps_baseline_trial(self):
        arguments = dict(
            task="regression", target="target", features=["x", "category"],
            iterations=18, depth=5, learning_rate=.08, l2_leaf_reg=3,
            early_stopping_rounds=4, random_seed=9, trials=3,
        )
        first = tune_catboost_parameters(self.frame, **arguments)
        second = tune_catboost_parameters(self.frame, **arguments)
        self.assertEqual(first["metric_name"], "MAE")
        self.assertFalse(first["higher_is_better"])
        self.assertEqual(first["trials_count"], 3)
        self.assertEqual(first["trials"], second["trials"])
        self.assertEqual(first["trials"][0]["depth"], 5)
        self.assertIn("iterations", first["best_params"])

    def test_classification_search_optimizes_accuracy(self):
        result = tune_catboost_parameters(
            self.frame, task="classification", target="class",
            features=["x", "category"], iterations=18,
            loss_function="Auto", early_stopping_rounds=4,
            random_seed=4, trials=3,
        )
        self.assertEqual(result["metric_name"], "Accuracy")
        self.assertTrue(result["higher_is_better"])
        self.assertGreaterEqual(result["best_value"], 0)
        self.assertLessEqual(result["best_value"], 1)

    def test_search_honours_pre_cancel(self):
        cancelled = Event()
        cancelled.set()
        with self.assertRaisesRegex(RuntimeError, "отменено"):
            tune_catboost_parameters(
                self.frame, task="regression", target="target", features=["x"],
                iterations=5, trials=3, cancel_event=cancelled,
            )

    def test_trial_limits_are_guarded(self):
        with self.assertRaisesRegex(ValueError, "от 3 до 60"):
            tune_catboost_parameters(
                self.frame, task="regression", target="target", features=["x"],
                iterations=5, trials=2,
            )


if __name__ == "__main__":
    unittest.main()
