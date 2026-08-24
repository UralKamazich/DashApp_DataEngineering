# -*- coding: utf-8 -*-
"""In-process background jobs for long-running ML calculations."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from threading import Event, RLock
from typing import Callable
from uuid import uuid4


FINAL_STATES = {"completed", "cancelled", "failed"}


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class MLJob:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "В очереди"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    result: object | None = None
    error: str = ""
    cancel_event: Event = field(default_factory=Event, repr=False)
    future: Future | None = field(default=None, repr=False)


_LOCK = RLock()
_JOBS: dict[str, MLJob] = {}
# CatBoost already uses all CPU cores. One worker prevents two heavy fits competing
# for memory and processor while still leaving the Dash request thread responsive.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dashapp-ml")


def _set_state(job_id, **changes):
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = _now()


def _prune_jobs(limit=50):
    with _LOCK:
        if len(_JOBS) <= limit:
            return
        removable = [
            job for job in _JOBS.values()
            if job.status in FINAL_STATES
        ]
        removable.sort(key=lambda job: job.updated_at)
        for job in removable[:max(0, len(_JOBS) - limit)]:
            _JOBS.pop(job.job_id, None)


def submit_ml_job(runner: Callable[[Callable, Event], object]) -> str:
    """Run ``runner(report_progress, cancel_event)`` outside the Dash callback."""
    job_id = uuid4().hex
    job = MLJob(job_id=job_id)
    with _LOCK:
        _JOBS[job_id] = job

    def report_progress(value, message="Обучение"):
        with _LOCK:
            current = _JOBS.get(job_id)
            if not current or current.status in FINAL_STATES:
                return
            current.progress = max(current.progress, min(100.0, float(value or 0)))
            current.message = str(message or "Обучение")
            current.updated_at = _now()

    def execute():
        _set_state(job_id, status="running", message="Подготовка данных", progress=1.0)
        try:
            result = runner(report_progress, job.cancel_event)
            if job.cancel_event.is_set():
                _set_state(job_id, status="cancelled", message="Обучение отменено")
                return
            _set_state(
                job_id, status="completed", progress=100.0,
                message="Модель готова", result=result,
            )
        except Exception as error:
            if job.cancel_event.is_set():
                _set_state(job_id, status="cancelled", message="Обучение отменено")
            else:
                _set_state(
                    job_id, status="failed", message="Ошибка обучения",
                    error=str(error),
                )
        finally:
            _prune_jobs()

    future = _EXECUTOR.submit(execute)
    with _LOCK:
        job.future = future
    return job_id


def cancel_ml_job(job_id) -> bool:
    with _LOCK:
        job = _JOBS.get(str(job_id or ""))
        if not job or job.status in FINAL_STATES:
            return False
        job.cancel_event.set()
        if job.future and job.future.cancel():
            job.status = "cancelled"
            job.message = "Обучение отменено"
        else:
            job.status = "cancelling"
            job.message = "Останавливаем обучение…"
        job.updated_at = _now()
        return True


def ml_job_snapshot(job_id) -> dict | None:
    with _LOCK:
        job = _JOBS.get(str(job_id or ""))
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": round(float(job.progress), 1),
            "message": job.message,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }


def ml_job_result(job_id):
    with _LOCK:
        job = _JOBS.get(str(job_id or ""))
        return job.result if job and job.status == "completed" else None


def take_ml_job_result(job_id):
    """Transfer a completed result to its durable cache without retaining two copies."""
    with _LOCK:
        job = _JOBS.get(str(job_id or ""))
        if not job or job.status != "completed":
            return None
        result = job.result
        job.result = None
        return result
