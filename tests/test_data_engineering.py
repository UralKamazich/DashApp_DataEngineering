import unittest

import pandas as pd

from dataset_registry import (
    SOURCE_DATASET_ID,
    commit_result,
    create_source_registry,
    input_payload,
    payload_for_record,
    save_runtime_state,
    suggest_dataset_name,
)
from engineering_ops import (
    apply_binning,
    apply_group_aggregates,
    apply_text_copy,
    execute_pipeline,
)
from utils import meta_from_df


class DatasetRegistryTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"group": ["A", "A", "B"], "value": [1.0, 2.0, 8.0]})
        self.payload = self.frame.to_json(orient="split")
        self.registry = create_source_registry(
            self.payload,
            meta_from_df(self.frame),
            "source.xlsx",
        )

    def test_registry_keeps_payloads_out_of_client_metadata(self):
        source = self.registry[SOURCE_DATASET_ID]
        self.assertNotIn("data", source)
        self.assertNotIn("filtered_data", source)
        self.assertEqual(payload_for_record(source), self.payload)

    def test_filtered_and_base_inputs_are_independent(self):
        filtered = self.frame.iloc[:1].to_json(orient="split")
        updated = save_runtime_state(
            self.registry,
            SOURCE_DATASET_ID,
            filtered_data=filtered,
        )
        base_payload, _ = input_payload(updated, SOURCE_DATASET_ID, "base")
        filtered_payload, _ = input_payload(updated, SOURCE_DATASET_ID, "filtered")
        self.assertEqual(base_payload, self.payload)
        self.assertEqual(filtered_payload, filtered)

    def test_derived_dataset_preserves_lineage_without_mutating_parent(self):
        first_step = {"type": "text_copy", "label": "Text", "outputs": ["value_txt"]}
        first_frame, _, _ = apply_text_copy(self.frame.copy(), ["value"], "_txt", True)
        first_payload = first_frame.to_json(orient="split")
        updated, derived_id = commit_result(
            self.registry,
            SOURCE_DATASET_ID,
            first_payload,
            meta_from_df(first_frame),
            first_step,
            output_mode="new",
            output_name="Prepared",
        )
        self.assertEqual(payload_for_record(updated[SOURCE_DATASET_ID]), self.payload)
        self.assertEqual(updated[derived_id]["parent_id"], SOURCE_DATASET_ID)
        self.assertEqual(updated[derived_id]["name"], "Prepared")
        self.assertEqual(updated[derived_id]["steps"], [first_step])

    def test_suggested_name_describes_pipeline_scope_and_sequence(self):
        queued = [
            {"operation": "binning"},
            {"operation": "text_copy"},
            {"operation": "group_aggregates"},
        ]
        self.assertEqual(
            suggest_dataset_name(self.registry, queued, "filtered"),
            "Биннинг_Текст_Агрегат_После фильтров_1",
        )

        derived, _ = commit_result(
            self.registry,
            SOURCE_DATASET_ID,
            self.payload,
            meta_from_df(self.frame),
            {"type": "text_copy", "outputs": []},
            output_mode="new",
            output_name="Первый",
        )
        self.assertTrue(suggest_dataset_name(derived, queued, "base").endswith("_2"))


class EngineeringOperationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"group": ["A", "A", "B"], "value": [1.0, 3.0, 8.0]})

    def test_binning_and_text_copy_create_unique_channels(self):
        frame, bin_outputs, _ = apply_binning(self.frame.copy(), "value", "count", 2, "index")
        frame, text_outputs, _ = apply_text_copy(frame, ["value"], "_txt", True)
        self.assertEqual(bin_outputs, ["Группа(value)"])
        self.assertEqual(text_outputs, ["value_txt"])
        self.assertEqual(frame["Группа(value)"].tolist(), ["Группа 1", "Группа 1", "Группа 2"])

    def test_group_aggregate_is_row_aligned(self):
        frame, outputs, _ = apply_group_aggregates(
            self.frame.copy(), ["group"], ["value"], ["mean", "count"]
        )
        self.assertEqual(outputs, ["value_mean", "value_count"])
        self.assertEqual(frame["value_mean"].tolist(), [2.0, 2.0, 8.0])
        self.assertEqual(frame["value_count"].tolist(), [2, 2, 1])

    def test_zero_and_nan_policies_only_change_aggregate_calculation(self):
        source = pd.DataFrame({"group": ["A", "A", "A"], "value": [0.0, None, 2.0]})
        default, _, _ = apply_group_aggregates(
            source.copy(), ["group"], ["value"], ["mean"], False, True
        )
        no_zeros, _, _ = apply_group_aggregates(
            source.copy(), ["group"], ["value"], ["mean"], True, True
        )
        nan_as_zero, _, _ = apply_group_aggregates(
            source.copy(), ["group"], ["value"], ["mean"], False, False
        )
        self.assertEqual(len(default), len(source))
        self.assertEqual(default["value_mean"].iloc[0], 1.0)
        self.assertEqual(no_zeros["value_mean"].iloc[0], 2.0)
        self.assertAlmostEqual(nan_as_zero["value_mean"].iloc[0], 2 / 3)

    def test_pipeline_executes_multiple_steps_on_one_materialization(self):
        queued = [
            {
                "operation": "binning",
                "label": "Binning",
                "scope": "base",
                "params": {
                    "column": "value",
                    "method": "count",
                    "groups": 2,
                    "label_style": "index",
                },
            },
            {
                "operation": "text_copy",
                "label": "Text",
                "scope": "base",
                "params": {
                    "columns": ["Группа(value)"],
                    "suffix": "_txt",
                    "strip": True,
                },
            },
        ]
        result, outputs, committed = execute_pipeline(self.frame, queued)
        self.assertEqual(outputs, ["Группа(value)", "Группа(value)_txt"])
        self.assertEqual(len(committed), 2)
        self.assertIn("Группа(value)_txt", result.columns)
        self.assertEqual(list(self.frame.columns), ["group", "value"])

    def test_pipeline_failure_does_not_mutate_source(self):
        queued = [{"operation": "text_copy", "label": "Broken", "params": {"columns": ["missing"]}}]
        with self.assertRaisesRegex(ValueError, "Шаг 1"):
            execute_pipeline(self.frame, queued)
        self.assertEqual(list(self.frame.columns), ["group", "value"])


if __name__ == "__main__":
    unittest.main()
