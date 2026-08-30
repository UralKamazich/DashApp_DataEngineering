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
    summarize_transformation_steps,
)
from engineering_ops import (
    apply_binning,
    apply_group_aggregates,
    apply_long_to_wide,
    apply_text_copy,
    apply_wide_to_long,
    execute_pipeline,
)
from utils import classify_simple, meta_from_df, read_df_from_store


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

    def test_reshape_operations_are_named_and_summarized(self):
        queued = [
            {"operation": "long_to_wide"},
            {"operation": "wide_to_long"},
        ]
        self.assertEqual(
            suggest_dataset_name(self.registry, queued, "base"),
            "LongToWide_WideToLong_До фильтров_1",
        )
        summaries = summarize_transformation_steps([
            {
                "type": "long_to_wide",
                "params": {
                    "index": ["well"],
                    "names_from": "metric",
                    "values_from": ["value"],
                },
            },
            {
                "type": "wide_to_long",
                "params": {
                    "id_columns": ["well"],
                    "value_columns": ["oil", "gas"],
                },
            },
        ])
        self.assertIn("Long → Wide", summaries[0])
        self.assertIn("Wide → Long", summaries[1])

    def test_transformation_summary_uses_committed_step_details(self):
        steps = [
            {
                "type": "binning",
                "inputs": ["value"],
                "outputs": ["value_bin"],
                "params": {"groups": 4},
                "scope": "filtered",
            },
            {
                "type": "group_aggregates",
                "params": {"keys": ["group"], "metrics": ["mean", "count"]},
            },
            {
                "type": "clustering",
                "inputs": ["x", "y"],
                "outputs": ["cluster"],
                "params": {"algorithm": "kmeans", "k": 3},
            },
        ]
        self.assertEqual(
            summarize_transformation_steps(steps),
            [
                "Биннинг: value → value_bin (4 групп) · после фильтров",
                "Агрегация: по group · mean, count",
                "Кластеризация: kmeans · K=3 · x, y → cluster",
            ],
        )


class EngineeringOperationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"group": ["A", "A", "B"], "value": [1.0, 3.0, 8.0]})

    def test_binning_and_text_copy_create_unique_channels(self):
        frame, bin_outputs, _ = apply_binning(self.frame.copy(), "value", "count", 2, "index")
        frame, text_outputs, _ = apply_text_copy(frame, ["value"], "_txt", True)
        self.assertEqual(bin_outputs, ["Группа(value)"])
        self.assertEqual(text_outputs, ["value_txt"])
        self.assertEqual(frame["Группа(value)"].tolist(), ["Группа 1", "Группа 1", "Группа 2"])

    def test_text_copy_stays_categorical_after_json_roundtrip(self):
        frame, outputs, _ = apply_text_copy(
            self.frame.copy(), ["value"], "_txt", True
        )
        output = outputs[0]
        meta = meta_from_df(frame)
        restored = read_df_from_store(
            frame.to_json(date_format="iso", orient="split"), meta
        )

        numeric, categorical, _datetime = classify_simple(restored)
        self.assertEqual(str(restored[output].dtype), "string")
        self.assertIn(output, categorical)
        self.assertNotIn(output, numeric)
        self.assertEqual(restored[output].tolist(), ["1.0", "3.0", "8.0"])

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

    def test_long_to_wide_and_back_preserves_values(self):
        source = pd.DataFrame({
            "well": ["A", "A", "B", "B"],
            "metric": ["oil", "gas", "oil", "gas"],
            "value": [10.0, 2.0, 20.0, 4.0],
        })
        wide, outputs, committed = apply_long_to_wide(
            source, ["well"], "metric", ["value"], "error", "__"
        )
        self.assertEqual(list(wide.columns), ["well", "oil", "gas"])
        self.assertEqual(outputs, ["oil", "gas"])
        self.assertEqual(committed["type"], "long_to_wide")
        self.assertEqual(wide["oil"].tolist(), [10.0, 20.0])

        restored, restored_outputs, restored_step = apply_wide_to_long(
            wide, ["well"], ["oil", "gas"], "metric", "value", False
        )
        restored = restored.sort_values(["well", "metric"]).reset_index(drop=True)
        expected = source.sort_values(["well", "metric"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(restored, expected)
        self.assertEqual(restored_outputs, ["metric", "value"])
        self.assertEqual(restored_step["type"], "wide_to_long")

    def test_long_to_wide_requires_duplicate_policy(self):
        source = pd.DataFrame({
            "well": ["A", "A"],
            "metric": ["oil", "oil"],
            "value": [10.0, 5.0],
        })
        with self.assertRaisesRegex(ValueError, "повторяющейся комбинацией"):
            apply_long_to_wide(source, ["well"], "metric", ["value"], "error")
        wide, _, _ = apply_long_to_wide(
            source, ["well"], "metric", ["value"], "sum"
        )
        self.assertEqual(wide.loc[0, "oil"], 15.0)

    def test_wide_to_long_can_use_all_non_identifier_columns(self):
        source = pd.DataFrame({
            "well": ["A", "B"],
            "oil": [10.0, None],
            "gas": [2.0, 4.0],
        })
        long, _, _ = apply_wide_to_long(
            source, ["well"], [], "metric", "value", True
        )
        self.assertEqual(len(long), 3)
        self.assertEqual(set(long["metric"]), {"oil", "gas"})

    def test_reshape_steps_run_inside_atomic_pipeline(self):
        source = pd.DataFrame({
            "well": ["A", "A", "B", "B"],
            "metric": ["oil", "gas", "oil", "gas"],
            "value": [10.0, 2.0, 20.0, 4.0],
        })
        queued = [{
            "operation": "long_to_wide",
            "label": "Long → Wide",
            "scope": "base",
            "params": {
                "index_columns": ["well"],
                "names_from": "metric",
                "value_columns": ["value"],
                "aggregation": "error",
                "separator": "__",
            },
        }]
        result, outputs, committed = execute_pipeline(source, queued)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(outputs, ["oil", "gas"])
        self.assertEqual(committed[0]["scope"], "base")


if __name__ == "__main__":
    unittest.main()
