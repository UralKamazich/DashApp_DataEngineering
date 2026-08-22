import unittest

import numpy as np
import pandas as pd

from clustering_engine import (
    cache_result,
    cached_result,
    clustering_signature,
    run_clustering,
)


class ClusteringEngineTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "x": [-3.2, -3.0, -2.8, -2.7, 2.7, 2.9, 3.1, 3.3],
            "y": [-2.9, -3.1, -2.7, -3.2, 3.2, 2.8, 3.0, 3.1],
            "name": list("abcdefgh"),
        })

    def test_clustering_preserves_rows_and_creates_aligned_channels(self):
        source = self.frame.copy()
        result = run_clustering(
            source,
            features=["x", "y"],
            k=2,
            output_column="Группа",
            include_id=True,
            include_distance=True,
            include_pca=True,
        )
        self.assertEqual(len(result.frame), len(source))
        self.assertEqual(list(source.columns), ["x", "y", "name"])
        self.assertEqual(result.analysis["used_rows"], len(source))
        self.assertEqual(result.analysis["excluded_rows"], 0)
        self.assertEqual(
            result.analysis["outputs"],
            ["Группа", "Группа_ID", "Группа_Расстояние", "Группа_PCA1", "Группа_PCA2"],
        )
        self.assertEqual(result.frame["Группа"].nunique(), 2)
        self.assertTrue((result.frame["Группа_ID"] >= 1).all())

    def test_drop_missing_keeps_original_row_with_empty_cluster(self):
        frame = self.frame.copy()
        frame.loc[2, "x"] = np.nan
        result = run_clustering(
            frame,
            features=["x", "y"],
            k=2,
            missing_policy="drop",
            include_id=False,
            include_distance=False,
            include_pca=False,
        )
        self.assertEqual(len(result.frame), len(frame))
        self.assertEqual(result.analysis["excluded_rows"], 1)
        self.assertTrue(pd.isna(result.frame.loc[2, "Кластер"]))

    def test_median_policy_assigns_rows_with_missing_values(self):
        frame = self.frame.copy()
        frame.loc[2, "x"] = np.nan
        result = run_clustering(
            frame,
            features=["x", "y"],
            k=2,
            missing_policy="median",
            include_id=False,
            include_distance=False,
            include_pca=False,
        )
        self.assertEqual(result.analysis["excluded_rows"], 0)
        self.assertTrue(result.frame["Кластер"].notna().all())

    def test_existing_output_names_are_not_overwritten(self):
        frame = self.frame.assign(Кластер="старое значение")
        result = run_clustering(
            frame,
            features=["x", "y"],
            k=2,
            output_column="Кластер",
            include_id=False,
            include_distance=False,
            include_pca=False,
        )
        self.assertIn("Кластер_2", result.frame.columns)
        self.assertTrue((result.frame["Кластер"] == "старое значение").all())

    def test_invalid_k_and_constant_features_fail_cleanly(self):
        with self.assertRaisesRegex(ValueError, "меньше количества строк"):
            run_clustering(self.frame, features=["x", "y"], k=len(self.frame))
        constant = self.frame.assign(x=1, y=1)
        with self.assertRaisesRegex(ValueError, "изменяющихся"):
            run_clustering(constant, features=["x", "y"], k=2)

    def test_cached_result_uses_lightweight_reference(self):
        signature = clustering_signature(features=["x", "y"], k=2)
        result = run_clustering(
            self.frame,
            features=["x", "y"],
            k=2,
            signature=signature,
        )
        reference = cache_result(result)
        self.assertIs(cached_result(reference), result)
        self.assertEqual(cached_result(reference).signature, signature)


if __name__ == "__main__":
    unittest.main()
