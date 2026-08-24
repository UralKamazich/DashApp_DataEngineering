import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dataset_registry import summarize_transformation_steps
from ml_engine import (
    _overfitting_summary,
    available_gpu_count,
    cache_result,
    cached_result,
    ml_signature,
    resolve_compute_device,
    run_catboost_classification,
    run_catboost_regression,
)
from ml_models import MODEL_ADAPTERS, get_model_adapter


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
            compute_device="cpu",
            use_best_iteration=True,
            compute_shap=True,
        )
        self.assertEqual(len(result.frame), len(self.frame))
        self.assertIn("prediction", result.frame)
        self.assertIn("Остаток target", result.frame)
        self.assertEqual(result.analysis["categorical_features"], ["category"])
        self.assertGreater(result.analysis["evaluation_rows"], 1)
        self.assertTrue(result.analysis["shap_importance"])
        self.assertEqual(result.analysis["compute"]["resolved"], "CPU")
        self.assertIn(result.analysis["overfitting"]["status"], {"low", "moderate", "high"})
        self.assertLessEqual(result.analysis["final_iterations"], 35)
        self.assertEqual(result.committed_step["operation"], "catboost_regression")

    def test_compute_device_resolution_and_clean_gpu_error(self):
        available_gpu_count.cache_clear()
        with patch("ml_engine.get_gpu_device_count", return_value=0):
            self.assertEqual(resolve_compute_device("auto")["resolved"], "CPU")
            self.assertEqual(resolve_compute_device("cpu")["resolved"], "CPU")
            with self.assertRaisesRegex(ValueError, "CUDA"):
                resolve_compute_device("gpu")
        available_gpu_count.cache_clear()

    def test_overfitting_summary_reports_generalization_gap(self):
        summary = _overfitting_summary(
            "regression", {"mae": 2.0}, {"mae": 5.0}
        )
        self.assertEqual(summary["status"], "high")
        self.assertAlmostEqual(summary["gap_percent"], 60.0)

        classification = _overfitting_summary(
            "classification",
            {"balanced_accuracy": .96},
            {"balanced_accuracy": .74},
        )
        self.assertEqual(classification["status"], "high")
        self.assertAlmostEqual(classification["gap_percent"], 22.0)

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

    def test_group_cross_validation_keeps_group_strategy_in_analysis(self):
        frame = self.frame.copy()
        frame["well"] = [f"well-{index // 6}" for index in range(len(frame))]
        result = run_catboost_regression(
            frame,
            target="target",
            features=["x", "category", "well"],
            method="group_cv",
            group_column="well",
            folds=3,
            iterations=15,
            early_stopping_rounds=4,
            compute_shap=False,
        )
        self.assertEqual(result.analysis["group_column"], "well")
        self.assertIn("Group OOF", result.analysis["evaluation_label"])
        self.assertNotIn("well", result.analysis["features"])
        self.assertEqual(result.analysis["evaluation_rows"], len(frame))

    def test_time_series_validation_sorts_by_order_channel(self):
        frame = self.frame.sample(frac=1, random_state=7).copy()
        frame["timestamp"] = pd.date_range("2025-01-01", periods=len(frame), freq="D")
        result = run_catboost_regression(
            frame,
            target="target",
            features=["x", "category", "timestamp"],
            method="time_cv",
            time_column="timestamp",
            folds=3,
            iterations=15,
            early_stopping_rounds=4,
            compute_shap=False,
        )
        self.assertEqual(result.analysis["time_column"], "timestamp")
        self.assertIn("Time series OOF", result.analysis["evaluation_label"])
        self.assertNotIn("timestamp", result.analysis["features"])
        self.assertLess(result.analysis["evaluation_rows"], len(frame))

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

    def test_model_adapter_registry_separates_available_and_planned_models(self):
        self.assertTrue(get_model_adapter("catboost").available)
        self.assertEqual(
            get_model_adapter("catboost").descriptor.tasks,
            ("regression", "classification"),
        )
        self.assertTrue(MODEL_ADAPTERS["random-forest"].available)
        self.assertFalse(MODEL_ADAPTERS["neural-networks"].available)


class CatBoostClassificationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        rows = 72
        x = rng.normal(size=rows)
        category = np.where(np.arange(rows) % 3, "песчаник", "карбонат")
        target = np.where(x + (category == "песчаник") * .8 > .35, "продуктивный", "сухой")
        self.frame = pd.DataFrame({
            "id": [f"sample-{index}" for index in range(rows)],
            "x": x,
            "lithology": category,
            "class": target,
        })

    def test_binary_split_adds_class_and_confidence(self):
        result = run_catboost_classification(
            self.frame,
            target="class",
            features=["x", "lithology"],
            id_column="id",
            method="split",
            test_size=.25,
            iterations=35,
            early_stopping_rounds=8,
            prediction_column="predicted class",
            include_confidence=True,
            compute_shap=True,
        )
        self.assertEqual(result.analysis["task"], "classification")
        self.assertEqual(result.analysis["class_count"], 2)
        self.assertEqual(len(result.frame), len(self.frame))
        self.assertIn("predicted class", result.frame)
        self.assertIn("Уверенность CatBoost", result.frame)
        self.assertTrue(result.frame["Уверенность CatBoost"].between(0, 1).all())
        self.assertGreaterEqual(result.analysis["metrics"]["accuracy"], .5)
        self.assertTrue(result.analysis["shap_importance"])
        self.assertIn("training_metrics", result.analysis)
        self.assertIn("overfitting", result.analysis)
        self.assertEqual(
            result.committed_step["operation"], "catboost_classification"
        )
        summary = summarize_transformation_steps([result.committed_step])[0]
        self.assertIn("classification", summary)

    def test_multiclass_cv_uses_stratified_oof(self):
        frame = self.frame.copy()
        frame["class"] = np.tile(["A", "B", "C"], len(frame) // 3)
        result = get_model_adapter("catboost").run(
            frame,
            task="classification",
            target="class",
            features=["x", "lithology"],
            method="cv",
            folds=3,
            iterations=20,
            early_stopping_rounds=5,
            include_confidence=False,
            compute_shap=True,
        )
        self.assertEqual(result.analysis["evaluation_rows"], len(frame))
        self.assertEqual(len(result.analysis["fold_metrics"]), 3)
        self.assertIn("Stratified OOF", result.analysis["evaluation_label"])
        self.assertEqual(result.analysis["class_labels"], ["A", "B", "C"])
        self.assertEqual(len(result.analysis["outputs"]), 1)
        self.assertTrue(result.analysis["shap_importance"])

    def test_rejects_continuous_target(self):
        frame = self.frame.copy()
        frame["continuous"] = np.arange(len(frame))
        with self.assertRaisesRegex(ValueError, "непрерывный"):
            run_catboost_classification(
                frame,
                target="continuous",
                features=["x"],
                iterations=5,
                compute_shap=False,
            )


if __name__ == "__main__":
    unittest.main()
