from __future__ import annotations

import unittest
from concurrent.futures import Future
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
if str(WORKFLOW_ROOT) not in __import__("sys").path:
    __import__("sys").path.append(str(WORKFLOW_ROOT))

from src.ga.job_queue import GAJobQueue


class FakeExecutor:
    def __init__(self) -> None:
        self.future = Future()

    def submit(self, fn, request):  # noqa: ANN001, D401
        return self.future


class JobQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = GAJobQueue(max_workers=1, queue_capacity=0)
        self.fake_executor = FakeExecutor()
        self.queue._executor = self.fake_executor  # type: ignore[attr-defined]

    def test_submit_rejects_when_capacity_is_full(self) -> None:
        job_id, error = self.queue.submit({"job_id": "job-1"})
        self.assertIsNone(error)
        self.assertEqual(job_id, "job-1")

        second_job_id, second_error = self.queue.submit({"job_id": "job-2"})
        self.assertIsNone(second_job_id)
        self.assertEqual(second_error, "All GA workers are busy. Please wait and try again.")

    def test_future_completion_releases_capacity(self) -> None:
        job_id, error = self.queue.submit({"job_id": "job-1"})
        self.assertIsNone(error)
        self.assertIsNotNone(job_id)

        self.fake_executor.future.set_result({"best_fitness": 0.7})

        record = self.queue.get_job(job_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.status, "completed")
        self.assertEqual(record.result, {"best_fitness": 0.7})

        next_job_id, next_error = self.queue.submit({"job_id": "job-2"})
        self.assertIsNone(next_error)
        self.assertEqual(next_job_id, "job-2")

    def test_submit_generates_job_id_when_missing(self) -> None:
        job_id, error = self.queue.submit({})
        self.assertIsNone(error)
        self.assertIsNotNone(job_id)
        assert job_id is not None
        self.assertEqual(len(job_id), 32)

        record = self.queue.get_job(job_id)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.job_id, job_id)

    def test_failed_future_sets_failed_status_and_error(self) -> None:
        job_id, error = self.queue.submit({"job_id": "job-err"})
        self.assertIsNone(error)
        self.assertEqual(job_id, "job-err")

        self.fake_executor.future.set_exception(RuntimeError("boom"))
        record = self.queue.get_job("job-err")

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.status, "failed")
        self.assertIsNotNone(record.error)
        self.assertIn("boom", record.error)


if __name__ == "__main__":
    unittest.main()