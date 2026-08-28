import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from neural_network_engine import run_neural_network_regression
from torch_tabular_engine import resolve_torch_device, torch_runtime


class TorchTabularTests(unittest.TestCase):
    def test_auto_resolves_to_available_mac_accelerator(self):
        runtime = torch_runtime()
        resolved = resolve_torch_device("auto")
        expected = "MPS" if runtime["mps_available"] else "CPU"
        self.assertEqual(resolved["resolved"], expected)

    def test_mps_training_and_joblib_roundtrip_when_available(self):
        if not torch_runtime()["mps_available"]:
            self.skipTest("Metal/MPS недоступен на этом компьютере")
        rng = np.random.default_rng(9)
        rows = 48
        x = rng.normal(size=rows)
        frame = pd.DataFrame({"x": x, "target": 1.8 * x + rng.normal(0, .1, rows)})
        result = run_neural_network_regression(
            frame, target="target", features=["x"], hidden_layers="8",
            max_iter=40, n_iter_no_change=6, permutation_repeats=0,
            engine="pytorch", compute_device="mps",
        )
        self.assertEqual(result.analysis["compute"]["resolved"], "MPS")
        expected = result.model.predict(frame[["x"]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "neural-network.joblib"
            joblib.dump(result.model, path)
            restored = joblib.load(path)
            actual = restored.predict(frame[["x"]])
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
