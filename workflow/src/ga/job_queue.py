from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock, Semaphore
from typing import Any
from uuid import uuid4

from src.ga.job_runner import run_ga_job


@dataclass
class GAJobRecord:
    job_id: str
    request: dict[str, Any]
    submitted_at: str
    status: str = "queued"
    result: dict[str, Any] | None = None
    error: str | None = None
    future: Future | None = field(default=None, repr=False)


class GAJobQueue:
    def __init__(self, max_workers: int = 4, queue_capacity: int = 4) -> None:
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        self._capacity = Semaphore(max_workers + queue_capacity)
        self._lock = RLock()
        self._jobs: dict[str, GAJobRecord] = {}

    def submit(self, request: dict[str, Any]) -> tuple[str | None, str | None]:
        if not self._capacity.acquire(blocking=False):
            return None, "All GA workers are busy. Please wait and try again."

        job_id = str(request.get("job_id") or uuid4().hex)
        record = GAJobRecord(
            job_id=job_id,
            request=dict(request),
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            future = self._executor.submit(run_ga_job, request)
        except Exception as exc:  # noqa: BLE001
            self._capacity.release()
            return None, str(exc)

        record.future = future
        with self._lock:
            self._jobs[job_id] = record

        future.add_done_callback(lambda fut, jid=job_id: self._finalize_job(jid, fut))
        return job_id, None

    def get_job(self, job_id: str) -> GAJobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None

            future = record.future
            if future is None:
                return record

            if future.done() and record.status not in {"completed", "failed"}:
                self._finalize_from_future(record, future)
            elif future.running():
                record.status = "running"
            elif record.status == "queued":
                record.status = "queued"
            return record

    def list_jobs(self) -> list[GAJobRecord]:
        with self._lock:
            job_ids = list(self._jobs.keys())
        return [self.get_job(job_id) for job_id in job_ids if self.get_job(job_id) is not None]

    def _finalize_job(self, job_id: str, future: Future) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                self._capacity.release()
                return
            self._finalize_from_future(record, future)
        self._capacity.release()

    def _finalize_from_future(self, record: GAJobRecord, future: Future) -> None:
        try:
            record.result = future.result()
            record.status = "completed"
            record.error = None
        except Exception as exc:  # noqa: BLE001
            record.result = None
            record.status = "failed"
            record.error = str(exc)


_GA_JOB_QUEUE: GAJobQueue | None = None


def get_ga_job_queue() -> GAJobQueue:
    global _GA_JOB_QUEUE
    if _GA_JOB_QUEUE is None:
        _GA_JOB_QUEUE = GAJobQueue(max_workers=4, queue_capacity=4)
    return _GA_JOB_QUEUE