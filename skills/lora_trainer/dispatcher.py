"""
GPU Worker Dispatcher — abstraction layer for GPU job execution.

Backends:
- LocalDispatcher: runs training in-process (current MVP behavior)
- SSHDispatcher: dispatches to remote GPU server via SSH + rsync
- ModalDispatcher: serverless GPU via Modal.com
- RunpodDispatcher: GPU pods via Runpod.io

All dispatchers implement the same interface:
    submit(job) → job_id
    poll(job_id) → JobStatus
    collect(job_id) → TrainingResult
"""
import asyncio
import time
import uuid
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from skills.lora_trainer.training_skill import TrainingJobSpec, TrainingResult

logger = logging.getLogger("lora_trainer.dispatcher")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobInfo(BaseModel):
    job_id: str
    trial_id: int
    status: JobStatus = JobStatus.QUEUED
    submitted_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float = 0.0  # 0..1
    message: str = ""


class BaseDispatcher(ABC):
    """Abstract GPU job dispatcher."""

    @abstractmethod
    async def submit(self, job: TrainingJobSpec) -> str:
        """Submit a training job. Returns job_id."""

    @abstractmethod
    async def poll(self, job_id: str) -> JobInfo:
        """Poll job status."""

    @abstractmethod
    async def collect(self, job_id: str) -> TrainingResult:
        """Collect results of a completed job."""

    @abstractmethod
    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""

    async def wait_for_completion(self, job_id: str, poll_interval: float = 10.0,
                                  timeout: float = 7200.0) -> TrainingResult:
        """Wait for job completion with exponential backoff polling."""
        start = time.time()
        interval = poll_interval

        while time.time() - start < timeout:
            info = await self.poll(job_id)

            if info.status == JobStatus.COMPLETED:
                return await self.collect(job_id)
            elif info.status == JobStatus.FAILED:
                return TrainingResult(
                    trial_id=info.trial_id,
                    status="failed",
                    error=info.message or "Job failed on remote worker",
                )
            elif info.status == JobStatus.CANCELLED:
                return TrainingResult(
                    trial_id=info.trial_id,
                    status="failed",
                    error="Job was cancelled",
                )

            logger.info(f"Job {job_id}: {info.status.value} ({info.progress*100:.0f}%)")
            await asyncio.sleep(interval)
            interval = min(interval * 1.5, 60.0)  # Exponential backoff, cap at 60s

        return TrainingResult(
            trial_id=0,
            status="failed",
            error=f"Job timed out after {timeout}s",
        )


class LocalDispatcher(BaseDispatcher):
    """
    Runs training in the current process.
    This is the default MVP dispatcher — no remote infrastructure needed.
    """

    def __init__(self):
        self._jobs: dict[str, JobInfo] = {}
        self._results: dict[str, TrainingResult] = {}

    async def submit(self, job: TrainingJobSpec) -> str:
        job_id = f"local_{uuid.uuid4().hex[:8]}"
        self._jobs[job_id] = JobInfo(
            job_id=job_id,
            trial_id=job.trial_id,
            status=JobStatus.RUNNING,
            submitted_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Run training synchronously (in-process)
        try:
            from skills.lora_trainer.training_skill import TrainingSkill

            skill = TrainingSkill()
            from skills.lora_trainer.schema import TrainingInput
            result_step = await skill.execute(
                TrainingInput(
                    trial_config=job.config,
                    dataset_path=job.dataset_path,
                    output_dir=job.output_dir,
                    resume_from=job.resume_from,
                ),
                None,  # state
            )
            result_data = result_step.data if isinstance(result_step.data, dict) else {}
            result = TrainingResult(**result_data)
            self._results[job_id] = result
            self._jobs[job_id].status = JobStatus.COMPLETED
            self._jobs[job_id].progress = 1.0
        except Exception as e:
            self._results[job_id] = TrainingResult(
                trial_id=job.trial_id, status="failed", error=str(e),
            )
            self._jobs[job_id].status = JobStatus.FAILED
            self._jobs[job_id].message = str(e)

        self._jobs[job_id].completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        return job_id

    async def poll(self, job_id: str) -> JobInfo:
        return self._jobs.get(job_id, JobInfo(job_id=job_id, trial_id=0, status=JobStatus.FAILED))

    async def collect(self, job_id: str) -> TrainingResult:
        return self._results.get(job_id, TrainingResult(trial_id=0, status="failed", error="Job not found"))

    async def cancel(self, job_id: str) -> bool:
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.CANCELLED
            return True
        return False


class SSHDispatcher(BaseDispatcher):
    """
    Dispatches training to a remote GPU server via SSH.
    
    Workflow:
    1. rsync dataset + config to remote
    2. ssh run training script
    3. Poll via SSH (check PID / log file)
    4. rsync results back
    """

    def __init__(self, host: str, user: str, remote_dir: str = "/tmp/lora_jobs",
                 key_path: Optional[str] = None):
        self.host = host
        self.user = user
        self.remote_dir = remote_dir
        self.key_path = key_path
        self._jobs: dict[str, JobInfo] = {}

    def _ssh_cmd(self) -> list[str]:
        cmd = ["ssh"]
        if self.key_path:
            cmd.extend(["-i", self.key_path])
        cmd.append(f"{self.user}@{self.host}")
        return cmd

    async def submit(self, job: TrainingJobSpec) -> str:
        import subprocess
        import json

        job_id = f"ssh_{uuid.uuid4().hex[:8]}"
        remote_job_dir = f"{self.remote_dir}/{job_id}"

        # 1. Create remote directory
        ssh = self._ssh_cmd()
        subprocess.run(ssh + [f"mkdir -p {remote_job_dir}"], check=True)

        # 2. rsync dataset
        rsync_cmd = ["rsync", "-az", job.dataset_path, f"{self.user}@{self.host}:{remote_job_dir}/data/"]
        subprocess.run(rsync_cmd, check=True)

        # 3. Write job config
        config_json = json.dumps(job.model_dump(), default=str)
        subprocess.run(ssh + [f"echo '{config_json}' > {remote_job_dir}/job.json"], check=True)

        # 4. Launch training (nohup)
        train_cmd = (
            f"cd {remote_job_dir} && "
            f"nohup python -m lora_train --config job.json > train.log 2>&1 &"
        )
        subprocess.run(ssh + [train_cmd], check=True)

        self._jobs[job_id] = JobInfo(
            job_id=job_id, trial_id=job.trial_id, status=JobStatus.RUNNING,
            submitted_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        return job_id

    async def poll(self, job_id: str) -> JobInfo:
        import subprocess

        info = self._jobs.get(job_id)
        if not info:
            return JobInfo(job_id=job_id, trial_id=0, status=JobStatus.FAILED)

        # Check if process is still running
        ssh = self._ssh_cmd()
        remote_job_dir = f"{self.remote_dir}/{job_id}"

        try:
            result = subprocess.run(
                ssh + [f"test -f {remote_job_dir}/DONE && echo done || echo running"],
                capture_output=True, text=True, timeout=10,
            )
            if "done" in result.stdout:
                info.status = JobStatus.COMPLETED
                info.progress = 1.0
            else:
                info.status = JobStatus.RUNNING
        except Exception:
            pass

        return info

    async def collect(self, job_id: str) -> TrainingResult:
        import subprocess
        import json

        remote_job_dir = f"{self.remote_dir}/{job_id}"

        # rsync results back
        local_dir = f"./ssh_results/{job_id}/"
        rsync_cmd = ["rsync", "-az", f"{self.user}@{self.host}:{remote_job_dir}/output/", local_dir]
        subprocess.run(rsync_cmd, check=True)

        # Read result
        result_path = f"{local_dir}/result.json"
        try:
            with open(result_path) as f:
                return TrainingResult(**json.load(f))
        except Exception as e:
            return TrainingResult(trial_id=0, status="failed", error=f"Failed to collect: {e}")

    async def cancel(self, job_id: str) -> bool:
        import subprocess
        ssh = self._ssh_cmd()
        try:
            subprocess.run(ssh + [f"pkill -f 'lora_train.*{job_id}'"], timeout=5)
            if job_id in self._jobs:
                self._jobs[job_id].status = JobStatus.CANCELLED
            return True
        except Exception:
            return False


def create_dispatcher(backend: str = "local", **kwargs) -> BaseDispatcher:
    """Factory for creating dispatchers."""
    dispatchers = {
        "local": LocalDispatcher,
        "ssh": SSHDispatcher,
    }
    cls = dispatchers.get(backend)
    if not cls:
        raise ValueError(f"Unknown dispatcher backend: {backend}. Available: {list(dispatchers.keys())}")
    return cls(**kwargs)
