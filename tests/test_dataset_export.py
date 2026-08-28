import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from callbacks.download import export_registered_dataset
from callbacks.datasets import initialize_dataset_registry, render_dataset_controls
from dataset_export import (
    dataset_export_name,
    export_catboost_model,
    export_frame_to_excel,
    export_sklearn_model,
    model_export_name,
    source_directory,
)
from dataset_registry import commit_result, create_source_registry, get_record
from utils import meta_from_df


class DatasetExportTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"x": [1, 2], "label": ["a", "b"]})

    def test_name_uses_source_dataset_and_creation_time(self):
        name = dataset_export_name(
            "/tmp/Pesec_metrics.xlsx",
            "Pesec_metrics.xlsx",
            "CatBoost_До фильтров_1",
            "2026-08-23T19:42:11+03:00",
        )
        self.assertEqual(
            name,
            "Pesec_metrics - CatBoost_До фильтров_1 - 20260823_194211.xlsx",
        )

    def test_online_source_exports_to_downloads(self):
        with tempfile.TemporaryDirectory() as folder:
            downloads = Path(folder) / "Downloads"
            downloads.mkdir()
            with patch("dataset_export.Path.home", return_value=Path(folder)):
                resolved = source_directory(
                    "https://example.test/datasets/iris.csv"
                )
        self.assertEqual(resolved, downloads)

    def test_source_registry_preserves_import_metadata(self):
        payload = self.frame.to_json(orient="split")
        meta = meta_from_df(self.frame)
        meta["import"] = {
            "format": "CSV", "encoding": "utf-8", "delimiter": ",",
            "decimal": ".", "remote": True,
        }
        registry, active_id, active_data = initialize_dataset_registry(
            payload, meta, "iris.csv"
        )
        self.assertEqual(active_id, "source")
        self.assertEqual(active_data, payload)
        self.assertEqual(registry["source"]["meta"]["import"], meta["import"])

    def test_export_is_next_to_source_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.xlsx"
            source.touch()
            first = export_frame_to_excel(
                self.frame,
                source_path=source,
                source_name=source.name,
                dataset_name="D1",
                created_at="2026-08-23T12:00:00+03:00",
            )
            second = export_frame_to_excel(
                self.frame,
                source_path=source,
                source_name=source.name,
                dataset_name="D1",
                created_at="2026-08-23T12:00:00+03:00",
            )
            self.assertEqual(first.parent, source.parent.resolve())
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertNotEqual(first, second)
            restored = pd.read_excel(first)
            pd.testing.assert_frame_equal(restored, self.frame)

    def test_native_model_export_uses_cbm_and_never_pickle(self):
        class FakeCatBoostModel:
            def save_model(self, path, format):
                self.format = format
                Path(path).write_bytes(b"native-catboost")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.xlsx"
            source.touch()
            model = FakeCatBoostModel()
            target = export_catboost_model(
                model,
                source_path=source,
                source_name=source.name,
                experiment_name="CatBoost run",
                created_at="2026-08-23T12:00:00+03:00",
            )
            self.assertEqual(model.format, "cbm")
            self.assertEqual(target.suffix, ".cbm")
            self.assertEqual(target.read_bytes(), b"native-catboost")
            self.assertFalse(list(Path(directory).glob("*.pkl")))
            self.assertEqual(
                model_export_name(source, source.name, "CatBoost run", "2026-08-23T12:00:00+03:00"),
                "source - CatBoost run - model - 20260823_120000.cbm",
            )

    def test_sklearn_model_export_uses_joblib_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.xlsx"
            source.touch()
            target = export_sklearn_model(
                {"model": "random-forest"}, source_path=source,
                source_name=source.name, experiment_name="RF run",
                created_at="2026-08-23T12:00:00+03:00",
            )
            self.assertTrue(target.exists())
            self.assertEqual(target.suffix, ".joblib")
            self.assertFalse(list(Path(directory).glob("*.pkl")))

    def test_registry_records_creation_time_and_ml_tab_class(self):
        payload = self.frame.to_json(orient="split")
        registry = create_source_registry(payload, meta_from_df(self.frame), "source.xlsx")
        self.assertTrue(get_record(registry, "source").get("created_at"))
        step = {
            "operation": "catboost_regression",
            "inputs": ["x"],
            "outputs": ["prediction"],
            "params": {"target": "x", "features": []},
        }
        updated, dataset_id = commit_result(
            registry,
            "source",
            payload,
            meta_from_df(self.frame),
            step,
            output_mode="new",
            output_name="ML result",
        )
        self.assertTrue(get_record(updated, dataset_id).get("created_at"))
        rail, *_ = render_dataset_controls(updated, "source", [], [])
        button = rail[0].children
        self.assertIn("is-ml", button.className.split())

    def test_context_export_uses_requested_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = str(Path(directory) / "source.xlsx")
            Path(source_path).touch()
            source_payload = self.frame.to_json(orient="split")
            registry = create_source_registry(
                source_payload, meta_from_df(self.frame), "source.xlsx"
            )
            derived = self.frame.assign(prediction=[10, 20])
            registry, dataset_id = commit_result(
                registry,
                "source",
                derived.to_json(orient="split"),
                meta_from_df(derived),
                {"operation": "catboost_regression", "outputs": ["prediction"]},
                output_mode="new",
                output_name="ML result",
            )
            notification = export_registered_dataset(
                json.dumps({"dataset_id": dataset_id, "nonce": 1}),
                registry,
                source_path,
                "source.xlsx",
            )
            self.assertEqual(notification[0]["color"], "green")
            exported = list(Path(directory).glob("source - ML result - *.xlsx"))
            self.assertEqual(len(exported), 1)
            self.assertIn("prediction", pd.read_excel(exported[0]).columns)

    def test_context_menu_script_targets_source_and_derived_tabs(self):
        script = Path("assets/dataset_context_menu.js").read_text(encoding="utf-8")
        self.assertIn(".dataset-rail-tab, #dataset-side-tab", script)
        self.assertIn('return "source"', script)
        self.assertIn("dataset-export-request", script)


if __name__ == "__main__":
    unittest.main()
