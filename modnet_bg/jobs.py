"""In-process job queue with progress and cancellation.

Deliberately not Celery. A single gunicorn process with a bounded thread
pool covers this workload, and it keeps the deployment to one container.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .errors import BGRemoverError, JobNotFoundError

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "error", "cancelled"]

_GENERIC_ERROR = "Processing failed. Please try a different file."


class JobCancelled(Exception):
    """Raised inside a worker when the job has been cancelled."""


@dataclass
class Job:
    id: str
    kind: str
    filename: str
    status: JobStatus = "queued"
    stage: str = ""
    frames_done: int = 0
    frames_total: int | None = None
    device_used: str | None = None
    result_path: Path | None = None
    result_name: str = ""
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def progress(self) -> float:
        if self.status == "done":
            return 1.0
        if not self.frames_total:
            return 0.0
        return min(1.0, self.frames_done / self.frames_total)

    def set_stage(self, label: str) -> None:
        """What the job is doing right now, for the UI to show verbatim.

        A percentage alone reads as stuck while a single slow step runs; a
        label tells the user the difference between working and hung.
        """
        with self._lock:
            self.stage = label

    def set_total(self, total: int | None) -> None:
        with self._lock:
            self.frames_total = total

    def advance(self, by: int = 1) -> None:
        with self._lock:
            self.frames_done += by

    def cancel(self) -> None:
        self.cancel_event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled

    def to_dict(self) -> dict:
        """JSON-safe view. Never exposes a filesystem path."""
        return {
            "id": self.id,
            "kind": self.kind,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "device_used": self.device_used,
            "result_name": self.result_name,
            "error": self.error,
        }


class JobRegistry:
    def __init__(self, *, max_workers: int, ttl_seconds: int) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="modnet")
        self._ttl = ttl_seconds

    def submit(
        self, fn: Callable[[Job], tuple[Path | None, str]], *, kind: str, filename: str
    ) -> Job:
        job = Job(id=secrets.token_urlsafe(24), kind=kind, filename=filename)
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Job], tuple[Path | None, str]]) -> None:
        if job.cancel_event.is_set():
            job.status = "cancelled"
            job.finished_at = time.time()
            return

        job.status = "running"
        try:
            result_path, result_name = fn(job)
            job.result_path = result_path
            job.result_name = result_name
            job.status = "done"
        except JobCancelled:
            job.status = "cancelled"
        except BGRemoverError as exc:
            # Our own errors carry messages written for users.
            job.status = "error"
            job.error = str(exc)
            logger.info("Job %s rejected: %s", job.id, exc)
        except Exception:
            # Anything else may contain paths or internals. Log it, do not ship it.
            job.status = "error"
            job.error = _GENERIC_ERROR
            logger.exception("Job %s failed", job.id)
        finally:
            job.finished_at = time.time()

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError("No such job, or it has expired.")
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.cancel()
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = time.time()
        return job

    def reap(self) -> int:
        """Delete finished jobs past their TTL, and their artefacts."""
        cutoff = time.time() - self._ttl
        removed = 0
        with self._lock:
            expired = [
                job
                for job in self._jobs.values()
                if job.finished_at is not None and job.finished_at <= cutoff
            ]
            for job in expired:
                if job.result_path is not None:
                    Path(job.result_path).unlink(missing_ok=True)
                del self._jobs[job.id]
                removed += 1
        return removed

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
