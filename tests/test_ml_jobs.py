import time
import unittest
from threading import Event

import numpy as np
import pandas as pd

from ml_engine import run_catboost_regression
from ml_jobs import (
    cancel_ml_job,
    ml_job_result,
    ml_job_snapshot,
    submit_ml_job,
    take_ml_job_result,
)


def wait_for(job_id, states, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = ml_job_snapshot(job_id)
        if snapshot and snapshot["status"] in states:
            return snapshot
        time.sleep(.01)
    raise AssertionError(f"Job {job_id} did not reach {states}")


class MLBackgroundJobTests(unittest.TestCase):
    def test_job_reports_progress_and_keeps_result_server_side(self):
        def runner(report, _cancel_event):
            report(42, "Проверка")
            return {"ready": True}

        job_id = submit_ml_job(runner)
        snapshot = wait_for(job_id, {"completed"})
        self.assertEqual(snapshot["progress"], 100)
        self.assertEqual(snapshot["message"], "Модель готова")
        self.assertEqual(take_ml_job_result(job_id), {"ready": True})
        self.assertIsNone(ml_job_result(job_id))

    def test_running_job_can_be_cancelled(self):
        started = Event()

        def runner(_report, cancel_event):
            started.set()
            cancel_event.wait(2)
            raise RuntimeError("cancelled")

        job_id = submit_ml_job(runner)
        self.assertTrue(started.wait(1))
        self.assertTrue(cancel_ml_job(job_id))
        snapshot = wait_for(job_id, {"cancelled"})
        self.assertEqual(snapshot["status"], "cancelled")
        self.assertIsNone(ml_job_result(job_id))

    def test_catboost_emits_stage_progress_and_honours_pre_cancel(self):
        x = np.linspace(0, 1, 24)
        frame = pd.DataFrame({"x": x, "target": 2 * x})
        updates = []
        result = run_catboost_regression(
            frame,
            target="target",
            features=["x"],
            iterations=12,
            early_stopping_rounds=3,
            compute_shap=False,
            progress_callback=lambda value, message: updates.append((value, message)),
            cancel_event=Event(),
        )
        self.assertTrue(result.analysis["metrics"])
        self.assertGreaterEqual(max(value for value, _message in updates), 99)
        self.assertTrue(any("Финальная модель" in message for _value, message in updates))

        cancelled = Event()
        cancelled.set()
        with self.assertRaisesRegex(RuntimeError, "отменено"):
            run_catboost_regression(
                frame,
                target="target",
                features=["x"],
                iterations=5,
                compute_shap=False,
                cancel_event=cancelled,
            )


if __name__ == "__main__":
    unittest.main()
