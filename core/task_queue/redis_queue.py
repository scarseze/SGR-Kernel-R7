import json
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import asyncio

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

from core.task_queue import TaskQueue, BackgroundTask, TaskStatus

logger = logging.getLogger("core.task_queue.redis")

class RedisTaskQueue(TaskQueue):
    """
    Redis-backed Task Queue implementation.
    
    Structure:
    - Hash `task:{task_id}` -> Task Metadata (JSON)
    - List `queue:pending` -> [task_id, task_id, ...]
    - Set `tasks:all` -> {task_id, ...} (optional, for listing)
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        if redis is None:
            raise ImportError("redis-py is required for RedisTaskQueue. Install with `pip install redis`")
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.queue_key = "queue:pending"
        self.task_prefix = "task:"

    async def enqueue(self, name: str, params: Dict[str, Any], max_retries: int = 3) -> BackgroundTask:
        task = BackgroundTask(name=name, params=params, max_retries=max_retries)
        
        # 1. Save Task Metadata
        key = f"{self.task_prefix}{task.task_id}"
        await self.redis.set(key, task.model_dump_json())
        
        # 2. Push to Queue
        await self.redis.rpush(self.queue_key, task.task_id)
        
        logger.info(f"Task enqueued (Redis): {task.task_id} ({name})")
        return task

    async def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        key = f"{self.task_prefix}{task_id}"
        data = await self.redis.get(key)
        if not data:
            return None
        return BackgroundTask.model_validate_json(data)

    async def update_status(self, task_id: str, status: TaskStatus, result: Any = None, error: str = None) -> Optional[BackgroundTask]:
        task = await self.get_task(task_id)
        if not task:
            return None
            
        task.status = status
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
            
        if status == TaskStatus.RUNNING:
            task.started_at = datetime.now()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            task.completed_at = datetime.now()
            
        # Atomic update not guaranteed here without Lua/Watch, but acceptable for simple use
        key = f"{self.task_prefix}{task_id}"
        await self.redis.set(key, task.model_dump_json())
        return task

    async def list_tasks(self, status: Optional[TaskStatus] = None, limit: int = 50) -> List[BackgroundTask]:
        # Redis is not great for listing by status without indices.
        # This is a naive implementation scanning keys or using a set if we maintained one.
        # For production, use RediSearch or maintain secondary indices (Sets per status).
        
        # Simple scan for now (WARNING: Slow on large DBs)
        tasks = []
        async for key in self.redis.scan_iter(match=f"{self.task_prefix}*"):
            if len(tasks) >= limit:
                break
            
            data = await self.redis.get(key)
            if data:
                task = BackgroundTask.model_validate_json(data)
                if status is None or task.status == status:
                    tasks.append(task)
                    
        return tasks

    async def claim_next_task(self) -> Optional[BackgroundTask]:
        """
        Atomically claim the next pending task.
        Using RPOPLPUSH pattern or simple LPOP since we just mark running.
        Structure: LPOP queue -> task_id. Update task status.
        Better reliability: RPOPLPUSH queue:pending queue:processing
        """
        # Simple LPOP for now
        task_id = await self.redis.lpop(self.queue_key)
        if not task_id:
            return None
            
        # Mark as RUNNING
        return await self.update_status(task_id, TaskStatus.RUNNING)
