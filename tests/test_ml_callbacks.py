import unittest

from dash import no_update

from callbacks.ml import (
    apply_preset,
    configure_run_mode,
    configure_task,
    preset_options,
    remember_tuning_preset,
)


class MLCallbackContractTests(unittest.TestCase):
    def test_task_configuration_returns_complete_contract_for_both_tasks(self):
        regression = configure_task("regression", None)
        classification = configure_task("classification", None)
        self.assertEqual(len(regression), 19)
        self.assertEqual(len(classification), 19)
        self.assertEqual(regression[0], "CatBoost · регрессия")
        self.assertEqual(classification[0], "CatBoost · классификация")

    def test_auto_tuning_result_is_stored_separately_for_each_task(self):
        store = {
            "analysis": {
                "task": "regression", "target": "target",
                "created_at": "2026-08-24T12:00:00",
                "tuning": {
                    "metric_name": "MAE", "best_value": .42,
                    "best_params": {
                        "iterations": 320, "depth": 7, "learning_rate": .04,
                        "l2_leaf_reg": 2.5, "random_strength": .7,
                        "bagging_temperature": .3,
                    },
                },
            },
        }
        presets = remember_tuning_preset(store, {})
        regression_options, regression_value = preset_options(
            "regression", presets, "balanced"
        )
        classification_options, classification_value = preset_options(
            "classification", presets, "tuning_result"
        )
        self.assertIn("Результат автоподбора", [item["label"] for item in regression_options])
        self.assertNotIn(
            "tuning_result", [item["value"] for item in classification_options]
        )
        self.assertEqual(regression_value, "balanced")
        self.assertEqual(classification_value, "balanced")

        values = apply_preset("tuning_result", presets, "regression")
        self.assertEqual(values[:4], (320, 7, .04, 2.5))
        self.assertIs(values[4], no_update)
        self.assertEqual(values[5:], (.7, .3))

    def test_named_presets_still_fill_all_tunable_parameters(self):
        self.assertEqual(
            apply_preset("balanced", {}, "regression"),
            (800, 6, .05, 3, 80, 1, 1),
        )
        self.assertEqual(configure_run_mode("tune")[-1], "Автоподбор и обучение")


if __name__ == "__main__":
    unittest.main()
