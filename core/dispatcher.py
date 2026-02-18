"""
Core Dispatcher — Unified Remote Execution.

Allows ANY skill to dispatch jobs to remote environments (SSH, Modal, etc.).
"""
import asyncio
import time
import uuid
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any, TypeVar, Generic
from pydantic import BaseModel, Field

logger = logging.getLogger("core.dispatcher")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RemoteJobSpec(BaseModel):
    """Base specification for any remote job."""
    job_type: str  # e.g. "lora_train", "deep_research"
    params: Dict[str, Any]
    resources: Dict[str, Any] = Field(default_factory=dict)  # gpu=1, ram="16gb"
    timeout: float = 3600.0


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    submitted_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    message: str = ""
    progress: float = 0.0


TResult = TypeVar("TResult")


class BaseDispatcher(ABC, Generic[TResult]):
    """Abstract Job Dispatcher."""

    @abstractmethod
    async def submit(self, job: RemoteJobSpec) -> str:
        """Submit a job. Returns job_id."""

    @abstractmethod
    async def poll(self, job_id: str) -> JobInfo:
        """Poll job status."""

    @abstractmethod
    async def collect(self, job_id: str) -> TResult:
        """Collect job results."""

    @abstractmethod
    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""

    async def wait_for_completion(self, job_id: str, poll_interval: float = 5.0,
                                  timeout: float = 7200.0) -> TResult:
        """Wait for job completion with exponential backoff polling."""
        start = time.time()
        interval = poll_interval

        while time.time() - start < timeout:
            info = await self.poll(job_id)

            if info.status == JobStatus.COMPLETED:
                return await self.collect(job_id)
            elif info.status == JobStatus.FAILED:
                raise RuntimeError(f"Job {job_id} failed: {info.message}")
            elif info.status == JobStatus.CANCELLED:
                raise RuntimeError(f"Job {job_id} was cancelled")

            await asyncio.sleep(interval)
            interval = min(interval * 1.5, 60.0)

        raise TimeoutError(f"Job {job_id} timed out after {timeout}s")


class LocalDispatcher(BaseDispatcher[Dict[str, Any]]):
    """
    Runs jobs locally (in-process or subprocess substitute).
    For 'lora_train', it invokes the TrainingSkill logic directly.
    """
    def __init__(self):
        self._jobs: Dict[str, JobInfo] = {}
        self._results: Dict[str, Any] = {}

    async def submit(self, job: RemoteJobSpec) -> str:
        job_id = f"local_{uuid.uuid4().hex[:8]}"
        self._jobs[job_id] = JobInfo(
            job_id=job_id,
            status=JobStatus.RUNNING,
            submitted_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        # Dispatch logic based on job_type
        # In a real system, this might use a task queue or subprocess
        try:
            result = await self._execute_local(job)
            self._results[job_id] = result
            self._jobs[job_id].status = JobStatus.COMPLETED
            self._jobs[job_id].progress = 1.0
        except Exception as e:
            self._jobs[job_id].status = JobStatus.FAILED
            self._jobs[job_id].message = str(e)
            
        self._jobs[job_id].completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        return job_id

    async def _execute_local(self, job: RemoteJobSpec) -> Dict[str, Any]:
        """Route to appropriate local handler."""
        if job.job_type == "lora_train":
            # Lazy import to avoid circular deps
            from skills.lora_trainer.training_skill import TrainingSkill, TrainingInput
            from skills.lora_trainer.experiment_spec import TrialConfig
            
            skill = TrainingSkill()
            # Map generic params back to TrainingInput
            # job.params is expected to match TrainingJobSpec fields
            p = job.params
            return await skill._run_training_logic(
                config=TrialConfig(**p.get("config", {})),
                dataset_path=p.get("dataset_path"),
                output_dir=p.get("output_dir"),
                resume_from=p.get("resume_from"),
                dry_run=p.get("dry_run", False)
            )
        else:
            raise ValueError(f"Unknown job_type: {job.job_type}")

    async def poll(self, job_id: str) -> JobInfo:
        return self._jobs.get(job_id, JobInfo(job_id=job_id, status=JobStatus.FAILED))

    async def collect(self, job_id: str) -> Dict[str, Any]:
        return self._results.get(job_id, {})

    async def cancel(self, job_id: str) -> bool:
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.CANCELLED
            return True
        return False


class SSHDispatcher(BaseDispatcher[Dict[str, Any]]):
    """
    Dispatches to remote server via SSH.
    """
    def __init__(self, host: str, user: str, remote_dir: str = "/tmp/sgr_jobs", key_path: str = None):
        self.host = host
        self.user = user
        self.remote_dir = remote_dir
        self.key_path = key_path
        self._jobs: Dict[str, JobInfo] = {}

    # ... (Implementation similar to previous, but generalized) ...
    async def submit(self, job: RemoteJobSpec) -> str:
        # Placeholder for full generic implementation
        job_id = f"ssh_{uuid.uuid4().hex[:8]}"
        self._jobs[job_id] = JobInfo(job_id=job_id, status=JobStatus.FAILED, message="Not fully implemented yet")
        return job_id
        
    async def poll(self, job_id: str) -> JobInfo:
        return self._jobs.get(job_id, JobInfo(job_id=job_id, status=JobStatus.FAILED))

    async def collect(self, job_id: str) -> Dict[str, Any]:
        return {}

    async def cancel(self, job_id: str) -> bool:
        return False


def get_dispatcher(backend: str = "local", **kwargs) -> BaseDispatcher:
    dispatchers = {
        "local": LocalDispatcher,
        "ssh": SSHDispatcher,
    }
    cls = dispatchers.get(backend)
    if not cls:
        raise ValueError(f"Unknown backend: {backend}")
    return cls(**kwargs)
